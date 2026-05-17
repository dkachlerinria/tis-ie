import argparse
import logging
import os
import time
import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from influence_eval.flops_measure import flop_counter
from influence_eval.model_utils import count_params
from iprox.utils.init_with_ipsvd import init_proxy_model_with_IPSVD, load_proxy_model

logger = logging.getLogger(__name__)

def compute_proxy_gradient(model, batch, device, target_modules):
    model.zero_grad(set_to_none=True)

    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    labels = batch["labels"].to(device)

    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    loss = outputs.loss
    loss.backward()

    sample_grads = []
    for name, module in model.named_modules():
        if any(name.endswith(tm) for tm in target_modules):
            if hasattr(module, 'linear_A') and hasattr(module, 'linear_B'):
                if module.linear_A.grad is not None and module.linear_B.grad is not None:
                    grad_A = module.linear_A.grad.detach().cpu().float().flatten()
                    grad_B = module.linear_B.grad.detach().cpu().float().flatten()
                    sample_grads.append(torch.cat([grad_A, grad_B]))

    if not sample_grads:
        return None
    return torch.cat(sample_grads)

def safe_normalize(grad, eps=1e-8):
    norm = torch.norm(grad)
    return grad / max(norm, eps)

def compute_iprox_scores(
    proxy_path: str,
    target_model: str,
    save_dir: str,
    tokenized_train_path: str,
    num_anchors: int,
    tokenized_anchor_path: str = None,
    tulu_as_anchors: bool = False,
    sparsity: float = 0.9,
    target_modules: list = None,
    out_name: str = "iprox",
) -> None:
    os.makedirs(save_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if target_modules is None:
        target_modules = ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']

    # Tokenizer comes from the saved proxy dir (matches what training used).
    logger.info("🔤 Loading tokenizer from proxy dir: %s", proxy_path)
    tokenizer = AutoTokenizer.from_pretrained(proxy_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    logger.info("🤖 Loading base model: %s", target_model)
    base_model = AutoModelForCausalLM.from_pretrained(
        target_model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="eager",  # required: FlopCounterMode's SDPA handler crashes on GQA models
    )

    proxy_model = init_proxy_model_with_IPSVD(
        base_model=base_model,
        loader_src=None,
        sparsity=sparsity,
        init_method="RANDOM",
        target_modules=target_modules,
        min_rank_multiple=1
    )

    final_bin = os.path.join(proxy_path, "pytorch_model.bin")
    logger.info("📥 Loading trained LinearSVD factors from: %s", final_bin)
    load_proxy_model(proxy_model, final_bin)
    proxy_params = count_params(proxy_model)
    proxy_model.eval()
    for p in proxy_model.parameters():
        p.requires_grad_(True)

    from datasets import load_from_disk, Dataset as HFDataset
    from common.data import encode_with_messages_format as _emf

    logger.info("📥 Loading GT-tokenized train dataset from: %s", tokenized_train_path)
    train_ds = load_from_disk(tokenized_train_path)
    train_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    if tulu_as_anchors:
        # Diagnostic mode: use first num_anchors Tulu train samples as anchors.
        # Both anchors and train are now Tulu with encode_with_messages_format —
        # same domain, same format. High Spearman here confirms gradient alignment
        # is working; if still low, the proxy itself is broken.
        logger.info("🔄 DIAGNOSTIC: using first %d Tulu train samples as anchors", num_anchors)
        anchor_ds = train_ds.select(range(min(num_anchors, len(train_ds))))
    else:
        # Normal mode: BBH anchors with encode_with_messages_format (same format
        # as the Tulu train pool — consistent within IProX, unlike construct_test_sample).
        from influence_eval.bbh_data import load_bbh_samples
        logger.info("📥 Loading %d BBH anchors (encode_with_messages_format)...", num_anchors)
        anchor_samples = load_bbh_samples(n_samples=num_anchors, start_index=0)
        msgs = [{"messages": [{"role": "user", "content": s["prompt"]},
                               {"role": "assistant", "content": s["response"]}]}
                for s in anchor_samples]
        anchor_hf = HFDataset.from_list(msgs)
        anchor_hf = anchor_hf.map(
            lambda x: _emf(x, tokenizer, max_seq_length=2048, include_response=True),
            num_proc=1, load_from_cache_file=False,
        )
        anchor_hf.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
        anchor_ds = anchor_hf

    logger.info("Anchors: %d samples | Train: %d samples", len(anchor_ds), len(train_ds))

    # Compute gradients and score matrix
    logger.info("📊 Computing similarity matrix for %d training samples...", len(train_ds))
    scores = torch.zeros((len(anchor_ds), len(train_ds)), dtype=torch.float32)

    t0 = time.perf_counter()
    with flop_counter() as counter:
        dev_grads = []
        grad_dim = None
        for i in tqdm(range(len(anchor_ds)), desc="Anchor Gradients"):
            batch = {k: anchor_ds[i][k].unsqueeze(0) for k in ["input_ids", "attention_mask", "labels"]}
            g = compute_proxy_gradient(proxy_model, batch, device, target_modules)
            torch.cuda.empty_cache()  # free activations from backward immediately
            if g is not None:
                if grad_dim is None:
                    grad_dim = int(g.shape[0])
                # Store as float16 to halve CPU RAM usage for the dev_matrix
                dev_grads.append(safe_normalize(g).half())
            else:
                dev_grads.append(torch.zeros(1, dtype=torch.float16))

        # [n_anchors, grad_dim] in float16 — dot products cast back to float32
        dev_matrix = torch.stack(dev_grads)

        for j in tqdm(range(len(train_ds)), desc="Train Similarity"):
            batch = {k: train_ds[j][k].unsqueeze(0) for k in ["input_ids", "attention_mask", "labels"]}
            g_t = compute_proxy_gradient(proxy_model, batch, device, target_modules)
            torch.cuda.empty_cache()
            if g_t is not None:
                scores[:, j] = (dev_matrix @ safe_normalize(g_t).half()).float()

    inference_time_s = time.perf_counter() - t0
    measured_flops = int(counter.get_total_flops())
    n_samples = len(anchor_ds) + len(train_ds)
    time_per_sample_s = inference_time_s / max(n_samples, 1)

    # Save
    out_path = os.path.join(save_dir, f"{out_name}_scores.pt")
    torch.save(scores, out_path)
    logger.info("Saved score matrix: %s shape=%s", out_path, tuple(scores.shape))

    meta = {
        **proxy_params,
        "num_anchors": int(scores.shape[0]),
        "num_train": int(scores.shape[1]),
        "grad_dim": grad_dim,
        "model_name": target_model,
        "proxy_path": proxy_path,
        "sparsity": sparsity,
        "method": "iprox",
        "measured_flops": measured_flops,
        "inference_time_s": float(inference_time_s),
        "time_per_sample_s": float(time_per_sample_s),
    }
    torch.save(meta, os.path.join(save_dir, f"{out_name}_params.pt"))

def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("--proxy_path", required=True,
                   help="Directory with the saved LinearSVD factors (pytorch_model.bin) + tokenizer.")
    p.add_argument("--target_model", required=True,
                   help="Original base model (e.g. Qwen/Qwen3-0.6B).")
    p.add_argument("--save_dir", required=True)
    p.add_argument("--tokenized_train_path", required=True,
                   help="Path to tokenized_train_ds saved by compute_ground_truth.sh.")
    p.add_argument("--num_anchors", type=int, required=True)
    p.add_argument("--tokenized_anchor_path", type=str, default=None,
                   help="Path to tokenized_anchor_ds (unused when --tulu_as_anchors).")
    p.add_argument("--tulu_as_anchors", action="store_true",
                   help="Diagnostic: use first --num_anchors Tulu train samples as anchors.")
    p.add_argument("--sparsity", type=float, default=0.9)
    p.add_argument("--out_name", type=str, default="iprox")
    args = p.parse_args()

    compute_iprox_scores(
        proxy_path=args.proxy_path,
        target_model=args.target_model,
        save_dir=args.save_dir,
        tokenized_train_path=args.tokenized_train_path,
        num_anchors=args.num_anchors,
        tokenized_anchor_path=args.tokenized_anchor_path,
        tulu_as_anchors=args.tulu_as_anchors,
        sparsity=args.sparsity,
        out_name=args.out_name
    )

if __name__ == "__main__":
    main()
