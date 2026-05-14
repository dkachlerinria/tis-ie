import argparse
import logging
import os
import gc
import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from influence_eval.bbh_data import load_bbh_samples
from influence_eval.model_utils import count_params
from iprox.utils.init_with_ipsvd import init_proxy_model_with_IPSVD, load_proxy_model
from common.data import encode_with_messages_format, construct_test_sample

logger = logging.getLogger(__name__)


def compute_proxy_gradient(model, batch, device, target_modules):
    """
    Compute per-sample gradient of the proxy model's loss and return it as a
    flat numpy array (CPU, float32). Returns None if no target modules matched.
    Matching the original IProX implementation: batch values are plain Python
    lists/ints (from an HF dataset item) and are converted to tensors here.
    """
    model.zero_grad(set_to_none=True)

    input_ids = torch.tensor(batch["input_ids"]).unsqueeze(0).to(device)
    attention_mask = torch.tensor(batch["attention_mask"]).unsqueeze(0).to(device)
    labels = torch.tensor(batch["labels"]).unsqueeze(0).to(device)

    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    loss = outputs.loss
    loss.backward()

    sample_grads = []
    for name, module in model.named_modules():
        if any(name.endswith(tm) for tm in target_modules):
            if hasattr(module, 'linear_A') and hasattr(module, 'linear_B'):
                if module.linear_A.grad is not None and module.linear_B.grad is not None:
                    grad_A = module.linear_A.grad.detach().cpu().float().flatten().numpy()
                    grad_B = module.linear_B.grad.detach().cpu().float().flatten().numpy()
                    sample_grads.append(np.concatenate([grad_A, grad_B]))

    # Free GPU gradient buffers immediately after extraction.
    model.zero_grad(set_to_none=True)

    if not sample_grads:
        return None
    return np.concatenate(sample_grads)


def safe_normalize(grad: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    norm = np.linalg.norm(grad)
    return grad / max(norm, eps)


def compute_iprox_scores(
    proxy_path: str,
    save_dir: str,
    end_index: int,
    num_anchors: int,
    train_dataset_name: str,
    dev_dataset_name: str,
    sparsity: float = 0.9,
    proj_dim: int = None,
    target_modules: list = None,
    out_name: str = "iprox",
) -> None:
    """
    Compute IProX influence scores.

    When proj_dim is None (default), behaviour matches the original IProX
    implementation exactly: raw proxy gradients are stored and compared.  At
    sparsity=0.9 the gradient vectors are ~94 MB each; at sparsity=0.5 they
    are ~840 MB each.  If RAM is insufficient, pass proj_dim=4096 to apply a
    JL random projection and reduce each vector to ~16 KB.
    """
    os.makedirs(save_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if target_modules is None:
        target_modules = ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']

    logger.info("Loading Proxy Model: %s", proxy_path)
    tokenizer = AutoTokenizer.from_pretrained(proxy_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        proxy_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    proxy_model = init_proxy_model_with_IPSVD(
        base_model=base_model,
        loader_src=None,
        sparsity=sparsity,
        init_method="RANDOM",
        target_modules=target_modules,
        min_rank_multiple=1,
    )

    final_bin = os.path.join(proxy_path, "pytorch_model.bin")
    load_proxy_model(proxy_model, final_bin)
    proxy_model.eval()
    for p in proxy_model.parameters():
        p.requires_grad_(True)

    # Optional JL projection matrix — created lazily on first gradient.
    # Shape: (proj_dim, grad_dim), float32 on CPU.
    proj_matrix: np.ndarray = None

    def _maybe_project(g: np.ndarray) -> np.ndarray:
        nonlocal proj_matrix
        if proj_dim is None:
            return g
        if proj_matrix is None:
            rng = np.random.RandomState(42)
            proj_matrix = rng.randn(proj_dim, g.shape[0]).astype(np.float32)
            logger.info("JL projection: (%d, %d) — %.0f MB → %.0f KB per sample",
                        proj_dim, g.shape[0],
                        g.nbytes / 1e6,
                        proj_dim * 4 / 1e3)
        return (proj_matrix @ g) / (g.shape[0] ** 0.5)

    # --- Dev (anchor) gradients ---
    # Pre-tokenise dev samples matching the original IProX approach.
    logger.info("Loading %d dev samples...", num_anchors)
    anchor_samples = load_bbh_samples(n_samples=num_anchors, start_index=0)

    tokenised_dev = []
    for sample in anchor_samples:
        prompt, response = sample["prompt"], sample["response"]
        ex = {"prompts": prompt, "labels": response}
        tokenised_dev.append(construct_test_sample(tokenizer=tokenizer, sample=ex, max_length=1024))

    dev_grads = []
    for item in tqdm(tokenised_dev, desc="Dev Gradients"):
        g = compute_proxy_gradient(proxy_model, item, device, target_modules)
        if g is not None:
            dev_grads.append(safe_normalize(_maybe_project(g)))
        else:
            fallback_dim = proj_dim if proj_dim is not None else 1
            dev_grads.append(np.zeros(fallback_dim, dtype=np.float32))
        gc.collect()
        torch.cuda.empty_cache()

    dev_matrix = np.stack(dev_grads)  # (num_anchors, grad_dim_or_proj_dim)
    del dev_grads
    logger.info("dev_matrix: %s, %.1f MB", dev_matrix.shape, dev_matrix.nbytes / 1e6)

    # --- Train data ---
    from datasets import load_dataset
    if os.path.exists(train_dataset_name):
        ds = load_dataset("json", data_files=[train_dataset_name])["train"]
    else:
        ds = load_dataset(train_dataset_name, split="train")
    ds = ds.select(range(min(end_index, len(ds))))

    # Pre-tokenise train samples (same pattern as original IProX).
    logger.info("Tokenising %d train samples...", len(ds))
    ds = ds.map(
        lambda x: encode_with_messages_format(example=x, tokenizer=tokenizer, max_seq_length=1024, include_response=True),
        desc="Tokenising train",
    )

    logger.info("Computing similarity matrix for %d training samples...", len(ds))
    scores = np.zeros((num_anchors, len(ds)), dtype=np.float32)

    for j in tqdm(range(len(ds)), desc="Train Similarity"):
        g_t = compute_proxy_gradient(proxy_model, ds[j], device, target_modules)
        if g_t is not None:
            g_t_norm = safe_normalize(_maybe_project(g_t))
            scores[:, j] = dev_matrix @ g_t_norm

        if j % 100 == 0:
            gc.collect()
            torch.cuda.empty_cache()

    # --- Save ---
    out_path = os.path.join(save_dir, f"{out_name}_scores.pt")
    torch.save(torch.from_numpy(scores), out_path)
    logger.info("Saved score matrix: %s shape=%s", out_path, scores.shape)

    proxy_params = count_params(proxy_model)
    meta = {
        **proxy_params,
        "num_anchors": int(scores.shape[0]),
        "num_train": int(scores.shape[1]),
        "model_name": proxy_path,
        "sparsity": sparsity,
        "proj_dim": proj_dim,
        "method": "iprox",
        "measured_flops": None,
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
    p.add_argument("--sparsity", type=float, default=0.9,
                   help="Sparsity used when the proxy was trained. Must match the checkpoint.")
    p.add_argument("--proj_dim", type=int, default=None,
                   help="Optional JL random-projection dimension. None = no projection (original "
                        "IProX behaviour). Set to e.g. 4096 when RAM is insufficient for raw "
                        "proxy gradients (~840 MB each at sparsity=0.5, ~94 MB at sparsity=0.9).")
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
        proj_dim=args.proj_dim,
        out_name=args.out_name,
    )


if __name__ == "__main__":
    main()
