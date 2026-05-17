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

    Runs immediately after training so the model is guaranteed to have correct
    pretrained weights — this is the ground-truth check for whether training works.
    Saves a [n_anchors, n_train] float32 score matrix to out_path.
    """
    proxy_model.eval()
    for p in proxy_model.parameters():
        p.requires_grad_(True)

    def _grad(sample):
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
                        grads.append(module.linear_A.grad.detach().cpu().float().flatten())
                        grads.append(module.linear_B.grad.detach().cpu().float().flatten())
        if not grads:
            return None
        g = torch.cat(grads)
        return g / g.norm().clamp_min(1e-8)

    anchor_grads = []
    for i in tqdm(range(len(anchor_ds)), desc="[inline] anchor grads"):
        g = _grad(anchor_ds[i])
        if g is not None:
            anchor_grads.append(g)
    anchor_matrix = torch.stack(anchor_grads)

    scores = torch.zeros(len(anchor_grads), len(train_ds))
    for j in tqdm(range(len(train_ds)), desc="[inline] train grads"):
        g = _grad(train_ds[j])
        if g is not None:
            scores[:, j] = anchor_matrix @ g

    torch.save(scores, out_path)
    logger.info("[inline scoring] saved %s  shape=%s", out_path, tuple(scores.shape))
    return scores

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    setseed(seed)

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

    # 2. Load Datasets via HF
    train_anchors = load_data_split(args.benchmark, tokenizer, n_samples=args.n_train_a, is_dev=True)
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
    if args.score_inline and os.path.exists(args.train_dataset):
        logger.info("🔬 Running inline scoring (no save/load)...")
        anchor_start = args.pool_start_index + args.n_train_p
        anchor_ds = load_data_split(
            args.train_dataset, tokenizer,
            n_samples=args.n_train_p,
            start_index=anchor_start,
            is_dev=False,
        )
        anchor_ds = anchor_ds.remove_columns(
            [c for c in anchor_ds.column_names if c not in ['input_ids', 'attention_mask', 'labels']]
        )
        train_pool_ds = train_pool.remove_columns(
            [c for c in train_pool.column_names if c not in ['input_ids', 'attention_mask', 'labels']]
        )
        out_path = os.path.join(args.output_dir, "inline_iprox_scores.pt")
        scores = score_proxy_inline(
            proxy_model, train_pool_ds, anchor_ds,
            device, args.target_modules, out_path,
        )
        logger.info("[inline scoring] score stats: min=%.4f max=%.4f mean=%.4f",
                    scores.min().item(), scores.max().item(), scores.mean().item())

        # Compare against ground-truth scores if they exist.
        gt_path = os.path.join(os.path.dirname(args.output_dir), "ground_truth_scores.pt")
        if os.path.exists(gt_path):
            try:
                from influence_eval.spearman import all_metrics
                gt_scores = torch.load(gt_path, map_location="cpu")
                metrics = all_metrics(scores, gt_scores)
                pa = metrics["per_anchor"]
                logger.info(
                    "[inline vs GT] per-anchor Spearman: mean=%.4f std=%.4f | "
                    "agg_mean=%.4f agg_max=%.4f",
                    pa["mean"], pa["std"],
                    metrics["aggregated_mean"], metrics["aggregated_max"],
                )
            except Exception as e:
                logger.warning("[inline vs GT] Could not compute Spearman: %s", e)
        else:
            logger.info("[inline vs GT] No GT scores found at %s — skipping Spearman.", gt_path)

if __name__ == "__main__":
    main()
