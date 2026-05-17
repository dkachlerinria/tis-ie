import argparse
import json
import logging
import os
import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

from influence_eval.model_utils import count_params
from iprox.utils.init_with_ipsvd import init_proxy_model_with_IPSVD, load_proxy_model
from common.data import encode_with_messages_format

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

def _iter_tokenized_disk(path: str, device: str):
    """Yield single-sample batches from a HF dataset saved to disk."""
    from datasets import load_from_disk
    ds = load_from_disk(path)
    ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    for i in range(len(ds)):
        row = ds[i]
        yield {k: v.unsqueeze(0).to(device) for k, v in row.items()}


def compute_iprox_scores(
    proxy_path: str,
    save_dir: str,
    tokenized_train_path: str,
    tokenized_anchor_path: str,
    sparsity: float = 0.9,
    target_modules: list = None,
    out_name: str = "iprox",
) -> None:
    os.makedirs(save_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if target_modules is None:
        target_modules = ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']

    # Tokenizer lives in proxy_path (saved there by train_iprox.py).
    tokenizer = AutoTokenizer.from_pretrained(proxy_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # save_proxy_model stores ONLY the SVD A/B factors, so from_pretrained(proxy_path)
    # would leave all standard weights randomly initialized.  Instead, read the original
    # model name from the config.json that train_iprox.py already saves via
    # target_model.config.save_pretrained(model_dir) — it includes "_name_or_path".
    config_path = os.path.join(proxy_path, "config.json")
    with open(config_path) as f:
        base_model_name = json.load(f)["_name_or_path"]
    logger.info("🤖 Loading base model from config._name_or_path: %s", base_model_name)
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.bfloat16,
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

    # Overlay the trained SVD factors onto the correct pretrained base.
    final_bin = os.path.join(proxy_path, "pytorch_model.bin")
    load_proxy_model(proxy_model, final_bin)
    proxy_model.eval()
    for p in proxy_model.parameters():
        p.requires_grad_(True)

    # Anchor gradients — load the exact tokenized dataset saved by compute_gradient_scores.
    from datasets import load_from_disk
    anchor_ds = load_from_disk(tokenized_anchor_path)
    anchor_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    num_anchors = len(anchor_ds)
    logger.info("📥 Loading %d anchor samples from %s", num_anchors, tokenized_anchor_path)

    dev_grads = []
    for i in tqdm(range(num_anchors), desc="Dev Gradients"):
        batch = {k: anchor_ds[i][k].unsqueeze(0).to(device) for k in ["input_ids", "attention_mask", "labels"]}
        g = compute_proxy_gradient(proxy_model, batch, device, target_modules)
        if g is not None:
            dev_grads.append(safe_normalize(g))
        else:
            dev_grads.append(torch.zeros(1))

    dev_matrix = torch.stack(dev_grads)  # [n_anchors, dim]

    # Train gradients — same pre-saved dataset the ground truth scored.
    train_ds = load_from_disk(tokenized_train_path)
    train_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    n_train = len(train_ds)
    logger.info("📊 Computing similarity matrix for %d training samples from %s", n_train, tokenized_train_path)
    scores = torch.zeros((num_anchors, n_train), dtype=torch.float32)

    for j in tqdm(range(n_train), desc="Train Similarity"):
        batch = {k: train_ds[j][k].unsqueeze(0).to(device) for k in ["input_ids", "attention_mask", "labels"]}
        g_t = compute_proxy_gradient(proxy_model, batch, device, target_modules)
        if g_t is not None:
            g_t_norm = safe_normalize(g_t)
            scores[:, j] = dev_matrix @ g_t_norm

        if j % 1000 == 0:
            torch.cuda.empty_cache()

    # Save
    out_path = os.path.join(save_dir, f"{out_name}_scores.pt")
    torch.save(scores, out_path)
    logger.info("Saved score matrix: %s shape=%s", out_path, tuple(scores.shape))

    proxy_params = count_params(proxy_model)
    meta = {
        **proxy_params,
        "num_anchors": int(scores.shape[0]),
        "num_train": int(scores.shape[1]),
        "model_name": proxy_path,
        "sparsity": sparsity,
        "method": "iprox",
        "measured_flops": None,
    }
    torch.save(meta, os.path.join(save_dir, f"{out_name}_params.pt"))

def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("--proxy_path", required=True,
                   help="Directory containing pytorch_model.bin, config.json, and tokenizer (output of train_iprox.py).")
    p.add_argument("--save_dir", required=True)
    p.add_argument("--tokenized_train_path", required=True,
                   help="Path to HF dataset saved by compute_gradient_scores (tokenized_train_ds).")
    p.add_argument("--tokenized_anchor_path", required=True,
                   help="Path to HF dataset saved by compute_gradient_scores (tokenized_anchor_ds).")
    p.add_argument("--sparsity", type=float, default=0.9)
    p.add_argument("--out_name", type=str, default="iprox")
    args = p.parse_args()

    compute_iprox_scores(
        proxy_path=args.proxy_path,
        save_dir=args.save_dir,
        tokenized_train_path=args.tokenized_train_path,
        tokenized_anchor_path=args.tokenized_anchor_path,
        sparsity=args.sparsity,
        out_name=args.out_name
    )

if __name__ == "__main__":
    main()
