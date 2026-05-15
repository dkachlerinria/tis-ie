import argparse
import logging
import os
import time
import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from influence_eval.bbh_data import load_bbh_samples
from influence_eval.flops_measure import flop_counter
from influence_eval.model_utils import count_params
from iprox.utils.init_with_ipsvd import init_proxy_model_with_IPSVD, load_proxy_model
from common.data import encode_with_messages_format, construct_test_sample

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
    save_dir: str,
    end_index: int,
    num_anchors: int,
    train_dataset_name: str,
    dev_dataset_name: str,
    sparsity: float = 0.9,
    target_modules: list = None,
    out_name: str = "iprox",
) -> None:
    os.makedirs(save_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if target_modules is None:
        target_modules = ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']

    logger.info("🤖 Loading Proxy Model: %s", proxy_path)
    tokenizer = AutoTokenizer.from_pretrained(proxy_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        proxy_path,
        dtype=torch.bfloat16,
        device_map="auto",
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
    load_proxy_model(proxy_model, final_bin)
    # count_params before requires_grad_ so "trainable" reflects SVD-factor params only
    proxy_params = count_params(proxy_model)
    proxy_model.eval()
    for p in proxy_model.parameters():
        p.requires_grad_(True)

    # Load Dev Data (Spearman Standard) — same seed=42 shuffle as all other methods.
    logger.info("📥 Loading dev samples...")
    anchor_samples = load_bbh_samples(n_samples=num_anchors, start_index=0)

    # Load Train Data
    from datasets import load_dataset
    if os.path.exists(train_dataset_name):
        ds = load_dataset("json", data_files=[train_dataset_name])["train"]
    else:
        ds = load_dataset(train_dataset_name, split="train")
    ds = ds.select(range(min(end_index, len(ds))))

    # Compute Train Gradients and Similarity
    logger.info("📊 Computing similarity matrix for %d training samples...", len(ds))
    scores = torch.zeros((num_anchors, len(ds)), dtype=torch.float32)

    t0 = time.perf_counter()
    with flop_counter() as counter:
        dev_grads = []
        grad_dim = None
        for sample in tqdm(anchor_samples, desc="Dev Gradients"):
            prompt, response = sample["prompt"], sample["response"]
            messages = [{"role": "user", "content": prompt}, {"role": "assistant", "content": response}]
            batch = encode_with_messages_format({"messages": messages}, tokenizer, max_seq_length=2048, include_response=True)
            for k in batch:
                if torch.is_tensor(batch[k]):
                    batch[k] = batch[k].unsqueeze(0)

            g = compute_proxy_gradient(proxy_model, batch, device, target_modules)
            if g is not None:
                if grad_dim is None:
                    grad_dim = int(g.shape[0])
                dev_grads.append(safe_normalize(g))
            else:
                dev_grads.append(torch.zeros(1))

        dev_matrix = torch.stack(dev_grads) # [n_dev, dim]

        for j in tqdm(range(len(ds)), desc="Train Similarity"):
            batch = encode_with_messages_format(ds[j], tokenizer, max_seq_length=2048, include_response=True)
            for k in batch:
                if torch.is_tensor(batch[k]):
                    batch[k] = batch[k].unsqueeze(0)

            g_t = compute_proxy_gradient(proxy_model, batch, device, target_modules)
            if g_t is not None:
                g_t_norm = safe_normalize(g_t)
                scores[:, j] = dev_matrix @ g_t_norm

            if j % 1000 == 0:
                torch.cuda.empty_cache()

    inference_time_s = time.perf_counter() - t0
    measured_flops = int(counter.get_total_flops())
    n_samples = len(anchor_samples) + len(ds)
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
        "model_name": proxy_path,
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
    p.add_argument("--proxy_path", required=True)
    p.add_argument("--save_dir", required=True)
    p.add_argument("--end_index", type=int, required=True)
    p.add_argument("--num_anchors", type=int, required=True)
    p.add_argument("--train_dataset_name", type=str, default="Harvard-DCML/tulu-v2-197K-processed")
    p.add_argument("--dev_dataset_name", type=str, default="bbh")
    p.add_argument("--sparsity", type=float, default=0.9)
    p.add_argument("--out_name", type=str, default="iprox")
    args = p.parse_args()

    compute_iprox_scores(
        proxy_path=args.proxy_path,
        save_dir=args.save_dir,
        end_index=args.end_index,
        num_anchors=args.num_anchors,
        train_dataset_name=args.train_dataset_name,
        dev_dataset_name=args.dev_dataset_name,
        sparsity=args.sparsity,
        out_name=args.out_name
    )

if __name__ == "__main__":
    main()
