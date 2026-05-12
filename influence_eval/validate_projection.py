"""One-time sanity check: does TRAK projection at our chosen ground-truth
proj_dim preserve Spearman rankings vs UNPROJECTED LoRA gradients?

If yes (Spearman >= ~0.95), we can trust the projected matrix as ground
truth. If no, we need a higher proj_dim.

Run on a TINY config (small num_anchors + small end_index) because the
unprojected branch concatenates the full LoRA-grad vector per sample,
which is ~hundreds of MB per sample for a 4B model with rank-128 LoRA.
"""

import argparse
import json
import logging
import os
from typing import Dict, List

import torch
from scipy.stats import spearmanr
from tqdm import tqdm
from transformers import AutoTokenizer

from common.data import construct_test_sample, encode_with_messages_format
from influence_eval.model_utils import load_base_with_fresh_lora
from representation.helper import batch_cosine_similarity
from representation.less.compute_less_embeds import (
    collect_grads,
    normalize_embeddings_in_chunks,
)

logger = logging.getLogger(__name__)


def _collect_unprojected_grads(dataloader, model) -> torch.Tensor:
    """Compute and concatenate full LoRA gradients per sample (no projection).

    Returns (N, P_lora) on CPU in float16 to save memory.
    """
    model.train()
    out = []
    for batch in tqdm(dataloader, desc="unprojected grads"):
        for k in batch:
            batch[k] = batch[k].to(next(model.parameters()).device)
        loss = model(**batch).loss
        loss.backward()
        g = torch.cat(
            [p.grad.detach().view(-1) for p in model.parameters() if p.grad is not None]
        )
        out.append(g.to(dtype=torch.float16, device="cpu"))
        model.zero_grad()
        del g
        torch.cuda.empty_cache()
    return torch.stack(out)


def _collect_projected_grads(dataloader, model, proj_dim: int) -> torch.Tensor:
    grads = collect_grads(
        dataloader=dataloader,
        model=model,
        proj_dim=proj_dim,
        adam_optimizer_state=None,
        gradient_type="sgd",
        project_interval=8,
    )
    return normalize_embeddings_in_chunks(grads, chunk_size=10000, dim=1).float()


def _normalize_rows(x: torch.Tensor) -> torch.Tensor:
    return x / x.norm(dim=1, keepdim=True).clamp_min(1e-12)


def _spearman_matrix(a: torch.Tensor, b: torch.Tensor) -> Dict:
    """Compute per-anchor + aggregated Spearman between two (n_anchors, n_train) matrices."""
    a_np = a.cpu().float().numpy()
    b_np = b.cpu().float().numpy()
    per_anchor = []
    for i in range(a_np.shape[0]):
        res = spearmanr(a_np[i], b_np[i])
        per_anchor.append(0.0 if res.statistic != res.statistic else float(res.statistic))
    agg_mean = spearmanr(a_np.mean(0), b_np.mean(0))
    return {
        "per_anchor_mean": float(sum(per_anchor) / len(per_anchor)),
        "per_anchor_min": float(min(per_anchor)),
        "aggregated_mean": float(agg_mean.statistic) if agg_mean.statistic == agg_mean.statistic else 0.0,
    }


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name", required=True, type=str)
    p.add_argument("--save_dir", required=True, type=str)
    p.add_argument("--end_index", type=int, default=50)
    p.add_argument("--num_anchors", type=int, default=10)
    p.add_argument("--dev_dataset_name", type=str, default="bbh")
    p.add_argument("--proj_dims", type=int, nargs="+", default=[8192, 16384, 32768, 65536])
    p.add_argument("--lora_rank", type=int, default=128)
    p.add_argument("--lora_alpha", type=int, default=512)
    p.add_argument("--lora_dropout", type=float, default=0.1)
    p.add_argument("--lora_target_modules", type=str, default="all-linear")
    p.add_argument("--lora_seed", type=int, default=0)
    return p.parse_args()


def _load_data(tokenizer, end_index: int, num_anchors: int, dev_dataset_name: str):
    from datasets import load_dataset

    train_ds = load_dataset("Harvard-DCML/tulu-v2-197K-processed", split="train").select(
        range(end_index)
    )
    train_ds = train_ds.map(
        lambda x: encode_with_messages_format(
            example=x, tokenizer=tokenizer, max_seq_length=2048, include_response=True
        ),
        num_proc=1,
        load_from_cache_file=False,
    )
    train_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    anchor_ds = load_dataset(
        "Harvard-DCML/targeted-query-set-processed", dev_dataset_name, split="dev"
    ).select(range(min(num_anchors, 10_000)))
    anchor_ds = anchor_ds.select(range(min(num_anchors, len(anchor_ds))))
    anchor_ds = anchor_ds.map(
        lambda x: construct_test_sample(tokenizer=tokenizer, sample=x, max_length=2048),
        num_proc=1,
        load_from_cache_file=False,
    )
    anchor_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    train_dl = torch.utils.data.DataLoader(train_ds, batch_size=1, shuffle=False)
    anchor_dl = torch.utils.data.DataLoader(anchor_ds, batch_size=1, shuffle=False)
    return train_dl, anchor_dl


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = load_base_with_fresh_lora(
        model_name=args.model_name,
        tokenizer=tokenizer,
        lora_target_modules=args.lora_target_modules,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        seed=args.lora_seed,
    )

    train_dl, anchor_dl = _load_data(
        tokenizer, args.end_index, args.num_anchors, args.dev_dataset_name
    )

    # Unprojected reference
    logger.info("Computing UNPROJECTED LoRA gradients (reference)")
    train_grads_full = _collect_unprojected_grads(train_dl, model)
    anchor_grads_full = _collect_unprojected_grads(anchor_dl, model)
    logger.info(
        "Unprojected shapes: train=%s anchor=%s",
        tuple(train_grads_full.shape),
        tuple(anchor_grads_full.shape),
    )
    train_grads_full = _normalize_rows(train_grads_full.float())
    anchor_grads_full = _normalize_rows(anchor_grads_full.float())
    ref_scores = batch_cosine_similarity(
        anchor_grads_full, train_grads_full, chunk_size=64, normalize=False
    ).float()
    del train_grads_full, anchor_grads_full
    torch.cuda.empty_cache()

    # Projected at each requested dim
    matrices = {"unprojected": ref_scores}
    for pd in args.proj_dims:
        logger.info("Computing PROJECTED grads at proj_dim=%d", pd)
        train_grads = _collect_projected_grads(train_dl, model, pd)
        anchor_grads = _collect_projected_grads(anchor_dl, model, pd)
        matrices[f"proj_{pd}"] = batch_cosine_similarity(
            anchor_grads, train_grads, chunk_size=256, normalize=False
        ).float()
        del train_grads, anchor_grads
        torch.cuda.empty_cache()

    # Pairwise comparisons, focusing on each proj vs unprojected
    report = {
        "config": vars(args),
        "vs_unprojected": {},
        "pairwise": {},
    }
    keys = list(matrices.keys())
    for k in keys:
        if k == "unprojected":
            continue
        report["vs_unprojected"][k] = _spearman_matrix(matrices[k], matrices["unprojected"])
    for i, k1 in enumerate(keys):
        for k2 in keys[i + 1 :]:
            report["pairwise"][f"{k1}__vs__{k2}"] = _spearman_matrix(
                matrices[k1], matrices[k2]
            )

    out_path = os.path.join(args.save_dir, "validation_report.json")
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    logger.info("Wrote %s", out_path)

    print()
    print("=== Spearman vs unprojected reference ===")
    for k, m in report["vs_unprojected"].items():
        print(
            f"{k}: per-anchor mean={m['per_anchor_mean']:.4f}, "
            f"min={m['per_anchor_min']:.4f}, agg_mean={m['aggregated_mean']:.4f}"
        )


if __name__ == "__main__":
    main()
