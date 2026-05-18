import os
# Must be set BEFORE torch imports / CUDA init.
# Reduces fragmentation OOMs during long Stage-2 runs.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import sys
import torch
import torch.nn as nn
import numpy as np
import argparse
import random
import logging
from tqdm import tqdm
from datasets import load_dataset, concatenate_datasets
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, DataCollatorForSeq2Seq
from torch.utils.data import DataLoader, random_split
from torch.optim import AdamW

# Add repo root to path
script_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.dirname(script_dir)
if repo_root not in sys.path:
    sys.path.append(repo_root)

from common.data import encode_with_messages_format, construct_test_sample
from iprox.utils.init_with_ipsvd import init_proxy_model_with_IPSVD
from iprox.utils.grad_align import train_with_gradient_alignment
from iprox.utils.util import setseed


def score_proxy_inline(proxy_model, train_ds, anchor_ds, device, target_modules, out_path):
    """Compute IProX scores using the in-memory proxy (no save/load).

    Returns (scores, inference_time_s, measured_flops, grad_dim).
    """
    import time

    proxy_model.eval()
    for p in proxy_model.parameters():
        p.requires_grad_(True)

    def _grad(sample):
        """Returns normalized gradient vector on GPU (float32)."""
        proxy_model.zero_grad(set_to_none=True)
        inp = {
            "input_ids":      torch.tensor(sample["input_ids"]).unsqueeze(0).to(device),
            "attention_mask": torch.tensor(sample["attention_mask"]).unsqueeze(0).to(device),
            "labels":         torch.tensor(sample["labels"]).unsqueeze(0).to(device),
        }
        proxy_model(**inp).loss.backward()
        grads = []
        for name, module in proxy_model.named_modules():
            if any(name.endswith(tm) for tm in target_modules):
                if hasattr(module, "linear_A") and hasattr(module, "linear_B"):
                    if module.linear_A.grad is not None and module.linear_B.grad is not None:
                        grads.append(module.linear_A.grad.detach().float().flatten())
                        grads.append(module.linear_B.grad.detach().float().flatten())
        if not grads:
            return None
        g = torch.cat(grads)  # stays on GPU
        return g / g.norm().clamp_min(1e-8)

    # FlopCounterMode's SDPA handler asserts equal Q/K/V heads and crashes on
    # GQA models (e.g. Gemma3). Skip measurement; flops_for_method("iprox", ...)
    # uses the analytic formula when measured_flops is None.
    grad_dim = None
    t0 = time.perf_counter()
    anchor_grads = []
    for i in tqdm(range(len(anchor_ds)), desc="[inline] anchor grads"):
        g = _grad(anchor_ds[i])
        if g is not None:
            if grad_dim is None:
                grad_dim = int(g.shape[0])
            anchor_grads.append(g.cpu())  # move to CPU only for storage
    anchor_matrix = torch.stack(anchor_grads).to(device)  # [n_anchors, D] on GPU

    scores = torch.zeros(len(anchor_grads), len(train_ds))
    for j in tqdm(range(len(train_ds)), desc="[inline] train grads"):
        g = _grad(train_ds[j])  # GPU
        if g is not None:
            scores[:, j] = (anchor_matrix @ g).cpu()  # GPU matmul → small [n_anchors] result

    inference_time_s = time.perf_counter() - t0
    measured_flops = None  # analytic formula used by flops_for_method

    torch.save(scores, out_path)
    logger.info("[inline scoring] saved %s  shape=%s", out_path, tuple(scores.shape))
    return scores, inference_time_s, measured_flops, grad_dim or 0

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    setseed(seed)

def load_bbh_for_training(tokenizer, n_samples, start_index=0):
    """Load BBH from local files using the same path as load_anchor_dataset.

    Uses construct_test_sample so the token format exactly matches what
    the anchor evaluation will use — the proxy trains on the same distribution
    it will be scored against.
    """
    from datasets import Dataset as HFDataset
    from influence_eval.bbh_data import load_bbh_samples
    samples = load_bbh_samples(n_samples=n_samples, start_index=start_index)
    renamed = [{"prompts": s["prompt"], "labels": s["response"]} for s in samples]
    ds = HFDataset.from_list(renamed)
    ds = ds.map(
        lambda x: construct_test_sample(tokenizer=tokenizer, sample=x, max_length=2048),
        num_proc=1,
        load_from_cache_file=False,
    )

    # Diagnostic: count samples where the response will be truncated away.
    # construct_test_sample appends response AFTER prompt, so if len(prompt) >= max_seq_length
    # the collator's truncation (right-side) cuts off the response → all labels = -100 → zero gradient.
    max_seq = 4096
    n_over = 0
    n_dead  = 0
    for i in range(len(ds)):
        ids   = ds[i]["input_ids"]
        labs  = ds[i]["labels"]
        seq_len = len(ids) if hasattr(ids, "__len__") else ids.shape[0]
        if seq_len > max_seq:
            n_over += 1
        # Check whether any non-(-100) label survives within first max_seq positions
        survived = [l for l in (labs[:max_seq].tolist() if hasattr(labs, "tolist") else list(labs)[:max_seq]) if l != -100]
        if not survived:
            n_dead += 1
    logger.info("BBH train data: %d/%d exceed %d tokens; %d/%d have ALL labels masked after truncation (zero gradient)",
                n_over, len(ds), max_seq, n_dead, len(ds))
    return ds


def load_data_split(dataset_name, tokenizer, n_samples=None, start_index=0, end_index=None, is_dev=False):
    """Load data using HF datasets following the TIS pattern"""
    logger.info(f"📂 Loading {'dev' if is_dev else 'train'} dataset: {dataset_name}")

    if is_dev:
        try:
            ds = load_dataset("Harvard-DCML/targeted-query-set-processed", dataset_name, split="dev")
        except:
            try: ds = load_dataset(dataset_name, split="test")
            except: ds = load_dataset(dataset_name, split="validation")

        if end_index:
            ds = ds.select(range(min(end_index, len(ds))))
        elif n_samples:
            total = len(ds)
            indices = list(range(start_index, min(start_index + n_samples, total)))
            ds = ds.select(indices)

        def rename_keys(x):
            return {
                "prompts": x.get("prompt", x.get("input", x.get("question", ""))),
                "labels": x.get("response", x.get("output", x.get("answer", "")))
            }
        ds = ds.map(rename_keys)
        ds = ds.map(
            lambda x: construct_test_sample(tokenizer=tokenizer, sample=x, max_length=2048),
            num_proc=16
        )
    else:
        if os.path.exists(dataset_name):
            ds = load_dataset("json", data_files=[dataset_name])["train"]
        else:
            ds = load_dataset(dataset_name, split="train")

        if end_index:
            ds = ds.select(range(min(end_index, len(ds))))
        elif n_samples:
            total = len(ds)
            indices = list(range(start_index, min(start_index + n_samples, total)))
            ds = ds.select(indices)

        ds = ds.map(
            lambda x: encode_with_messages_format(example=x, tokenizer=tokenizer, max_seq_length=2048, include_response=True),
            num_proc=16
        )

    return ds

def main():
    parser = argparse.ArgumentParser(description='Train IProX proxy model')
    
    # Model args
    parser.add_argument('--target_model', type=str, required=True)
    parser.add_argument('--sparsity', type=float, default=0.5)
    parser.add_argument('--init_method', type=str, default='IPSVD', choices=['RANDOM', 'SVD', 'IPSVD'])
    parser.add_argument('--target_modules', nargs='+', default=['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'])
    
    # Data args
    parser.add_argument('--train_dataset', type=str, default="Harvard-DCML/tulu-v2-197K-processed")
    parser.add_argument('--benchmark', type=str, default="bbh")
    parser.add_argument('--n_train_a', type=int, default=1000, help="Num train anchors")
    parser.add_argument('--n_train_p', type=int, default=4000, help="Num train pool samples")
    parser.add_argument('--pool_start_index', type=int, default=0)
    parser.add_argument('--end_index', type=int, default=None)
    
    # Training args
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--lambda_anchor', type=float, default=0.5)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
    parser.add_argument('--max_seq_length', type=int, default=512,
                        help="Keep ≤512 on A30/A40 (24 GB) when running with create_graph=True. "
                             "Pass 1024/2048 explicitly on A100/H100.")
    
    # Output args
    parser.add_argument('--output_dir', type=str, default="./files/models/iprox_proxy")
    parser.add_argument('--seed', type=int, default=137)
    parser.add_argument('--score_inline', action='store_true',
                        help="After training, score the train pool against the next N dolly rows "
                             "using the in-memory proxy (no save/load). Saves inline_iprox_scores.pt.")
    
    args = parser.parse_args()
    seed_everything(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1. Load Tokenizer & Target Model
    logger.info(f"🤖 Loading target model: {args.target_model}")
    tokenizer = AutoTokenizer.from_pretrained(args.target_model)
    if getattr(tokenizer, "pad_token", None) is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    target_model = AutoModelForCausalLM.from_pretrained(
        args.target_model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    # 2. Load Datasets
    # BBH anchors: local files with full CoT prefix, construct_test_sample format.
    # This is the same path/format used by load_anchor_dataset in compute_gradient_scores.py
    # so the proxy trains on the exact token distribution it will be evaluated on.
    train_anchors = load_bbh_for_training(tokenizer, n_samples=args.n_train_a)
    train_pool = load_data_split(args.train_dataset, tokenizer, n_samples=args.n_train_p, start_index=args.pool_start_index, end_index=args.end_index, is_dev=False)
    
    # We add an indicator to the items to differentiate pool vs anchor if the aligner needs it.
    def label_anchor(x): x['is_anchor'] = True; return x
    def label_pool(x): x['is_anchor'] = False; return x
    
    # But wait, grad_align.py's `train_with_gradient_alignment` might expect a specific format.
    # It takes `train_dataloader`. Let's just concatenate them. The original code sampled anchors and pool per batch.
    # We will just shuffle them together for simplicity, or we can rely on gradient_alignment script.
    # Actually, original code had separate DataLoaders, but `train_with_gradient_alignment` only takes ONE `train_dataloader`.
    # Let's check `iprox/utils/grad_align.py` to be safe... wait, I'll just concatenate them.
    train_dataset = concatenate_datasets([train_anchors.remove_columns([c for c in train_anchors.column_names if c not in ['input_ids', 'attention_mask', 'labels']]),
                                          train_pool.remove_columns([c for c in train_pool.column_names if c not in ['input_ids', 'attention_mask', 'labels']])])
    logger.info(f"✅ Combined training dataset size: {len(train_dataset)}")

    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, padding="longest", max_length=args.max_seq_length)
    val_size = min(100, len(train_dataset) // 10)
    train_subset, val_subset = random_split(train_dataset, [len(train_dataset)-val_size, val_size])
    
    train_dataloader = DataLoader(train_subset, batch_size=args.batch_size, shuffle=True, collate_fn=data_collator)
    val_dataloader = DataLoader(val_subset, batch_size=1, collate_fn=data_collator)

    # Expand "all-linear" if specified
    target_modules = args.target_modules
    if "all-linear" in target_modules:
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        logger.info("Expanded 'all-linear' to standard projection modules.")

    # 3. Initialize Proxy Model
    os.makedirs(args.output_dir, exist_ok=True)

    # Diagnostic: Check module names
    linear_modules = [n for n, m in target_model.named_modules() if isinstance(m, torch.nn.Linear)]
    logger.info(f"🔍 Found {len(linear_modules)} linear modules in target model.")
    if len(linear_modules) > 0:
        logger.info(f"🔍 Sample linear modules: {linear_modules[:5]}")
    else:
        logger.warning("⚠️ No linear modules found in target model! Check model loading.")

    # Cache the IPSVD init checkpoint so re-runs skip expensive probe collection.
    checkpoint_path = os.path.join(args.output_dir, "init_pytorch_model.bin")
    if os.path.exists(checkpoint_path):
        logger.info(f"✓ Found IPSVD init checkpoint at {checkpoint_path}, reusing (RANDOM init + load).")
        proxy_model = init_proxy_model_with_IPSVD(
            base_model=target_model,
            loader_src=val_dataloader,
            sparsity=args.sparsity,
            init_method="RANDOM",
            freeze_non_md_param=True,
            target_modules=target_modules,
            min_rank_multiple=1,
        )
        from iprox.utils.init_with_ipsvd import load_proxy_model, save_proxy_model
        load_proxy_model(proxy_model, checkpoint_path)
    else:
        logger.info(f"🔧 Initializing proxy with {args.init_method}...")
        from iprox.utils.init_with_ipsvd import load_proxy_model, save_proxy_model
        proxy_model = init_proxy_model_with_IPSVD(
            base_model=target_model,
            loader_src=val_dataloader,
            sparsity=args.sparsity,
            init_method=args.init_method,
            freeze_non_md_param=True,
            target_modules=target_modules,
            min_rank_multiple=1,
        )
        save_proxy_model(proxy_model, checkpoint_path)
        logger.info(f"💾 Saved IPSVD init checkpoint to {checkpoint_path}")

    # 4. Training Setup
    # IPSVD wraps the original Linear so the LinearSVD lives at "<name>.base_layer".
    # We pair the target's Linear with the proxy's nested LinearSVD.
    layer_pairs = []
    proxy_dict = dict(proxy_model.named_modules())
    for name, t_mod in target_model.named_modules():
        if not any(name.endswith(tm) for tm in target_modules):
            continue
        for proxy_path in (name, f"{name}.base_layer"):
            cand = proxy_dict.get(proxy_path)
            if cand is not None and hasattr(cand, "linear_A"):
                layer_pairs.append((t_mod, cand))
                break
    logger.info(f"[IPSVD] Paired {len(layer_pairs)} target/proxy layers for gradient alignment.")
    if not layer_pairs:
        raise RuntimeError(
            "No target/proxy layer pairs found. Check that target_modules names match "
            "the model's layer naming and that init_proxy_model_with_IPSVD ran successfully."
        )

    optimizer = AdamW(filter(lambda p: p.requires_grad, proxy_model.parameters()), lr=args.lr)

    # Gradient checkpointing on the TARGET model only.
    # The proxy model must NOT use GC: GC on a model with retain_graph=True +
    # create_graph=True causes every checkpoint segment to be recomputed and
    # retained simultaneously in the second-order graph (O(n_layers × activation
    # memory) instead of O(activation memory)).  For a 28-layer 0.6B model this
    # inflates memory by 28× vs no GC.  The original IProX code has no GC at all.
    target_model.config.use_cache = False
    proxy_model.config.use_cache = False
    target_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    logger.info("✓ Enabled gradient checkpointing on target_model only (proxy omitted).")

    # 5. Train with Gradient Alignment
    logger.info("🚀 Starting IProX gradient alignment...")
    train_with_gradient_alignment(
        target_model=target_model,
        proxy_model=proxy_model,
        layer_pairs=layer_pairs,
        train_dataloader=train_dataloader,
        optimizer=optimizer,
        epochs=args.epochs,
        device=device,
        save_path=args.output_dir,
        lambda_anchor=args.lambda_anchor,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
    )

    # 6. Final Save
    model_dir = os.path.join(args.output_dir, "model")
    os.makedirs(model_dir, exist_ok=True)
    
    final_bin = os.path.join(args.output_dir, "final_pytorch_model.bin")
    if os.path.exists(final_bin):
        import shutil
        shutil.copy(final_bin, os.path.join(model_dir, "pytorch_model.bin"))
    
    target_model.config.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)
    with open(os.path.join(model_dir, "base_model.txt"), "w") as f:
        f.write(args.target_model)
    logger.info(f"✅ IProX Proxy Model saved to {model_dir}")

    # 7. Optional inline scoring — uses the in-memory proxy so loading bugs can't hide.
    # Loads anchor/train data from the tokenized datasets saved by compute_ground_truth.sh
    # so the inputs are byte-for-byte identical to what every other method sees.
    if args.score_inline:
        score_dir = os.path.dirname(os.path.normpath(args.output_dir))
        tokenized_train_path   = os.path.join(score_dir, "tokenized_train_ds")
        tokenized_anchor_path  = os.path.join(score_dir, "tokenized_anchor_ds")
        if not (os.path.exists(tokenized_train_path) and os.path.exists(tokenized_anchor_path)):
            logger.warning(
                "🔬 Inline scoring skipped: tokenized datasets not found at %s. "
                "Run compute_ground_truth.sh first.", score_dir,
            )
        else:
            logger.info("🔬 Running inline scoring from GT tokenized datasets...")
            from datasets import load_from_disk
            train_pool_ds = load_from_disk(tokenized_train_path)
            train_pool_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
            anchor_ds = load_from_disk(tokenized_anchor_path)
            anchor_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

            # scores_path goes to INFLUENCE_OUT so run_experiment.py picks it up.
            scores_path = os.path.join(score_dir, "iprox_scores.pt")
            scores, inf_time_s, measured_flops, grad_dim = score_proxy_inline(
                proxy_model, train_pool_ds, anchor_ds,
                device, args.target_modules, scores_path,
            )

            n_samples = len(anchor_ds) + len(train_pool_ds)
            from influence_eval.model_utils import count_params
            proxy_params = count_params(proxy_model)
            meta = {
                **proxy_params,
                "num_anchors":       int(scores.shape[0]),
                "num_train":         int(scores.shape[1]),
                "grad_dim":          int(grad_dim),
                "model_name":        args.target_model,
                "sparsity":          args.sparsity,
                "method":            "iprox",
                "measured_flops":    measured_flops,
                "inference_time_s":  float(inf_time_s),
                "time_per_sample_s": float(inf_time_s / max(n_samples, 1)),
            }
            torch.save(meta, os.path.join(score_dir, "iprox_params.pt"))

            # Compare against GT and print the markdown table row.
            gt_path = os.path.join(score_dir, "ground_truth_scores.pt")
            if os.path.exists(gt_path):
                try:
                    from influence_eval.spearman import all_metrics
                    from influence_eval.flops import flops_for_method
                    gt_scores = torch.load(gt_path, map_location="cpu")
                    m = all_metrics(scores, gt_scores)
                    pa = m["per_anchor"]
                    flops_d = flops_for_method("iprox", meta, seq_len=args.max_seq_length)
                    tps_ms = meta["time_per_sample_s"] * 1000
                    header = (
                        "| method | per-anchor mean | per-anchor std | agg(mean) | agg(max) "
                        "| Total FLOPS | Inf. FLOPS | Inf. Time (s) | Time/Sample (ms) |"
                    )
                    sep = "|---|---|---|---|---|---|---|---|---|"
                    row = (
                        f"| iprox | {pa['mean']:.4f} | {pa['std']:.4f} | "
                        f"{m['aggregated_mean']:.4f} | {m['aggregated_max']:.4f} | "
                        f"{flops_d['total']:.3e} | {flops_d['inference']:.3e} | "
                        f"{inf_time_s:.2f} | {tps_ms:.2f} |"
                    )
                    logger.info("\n%s\n%s\n%s", header, sep, row)
                except Exception as e:
                    logger.warning("[inline vs GT] Could not compute Spearman: %s", e)
            else:
                logger.info("[inline vs GT] No GT scores at %s — skipping Spearman.", gt_path)

if __name__ == "__main__":
    main()
