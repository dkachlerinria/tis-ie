"""Compute (num_anchors, num_train) score matrix from gradient-based influence.

Used for BOTH:
  - the ground-truth matrix (high proj_dim, e.g. 65536)
  - the LESS-method matrix (proj_dim=8192)

The only differences between the two cases are `--proj_dim` and `--out_name`.
Everything else (model, LoRA seed, dataset slices, gradient type) is held
constant so the two matrices are directly comparable.

Plain SGD gradients throughout (no Adam state because there is no warmup
checkpoint). LoRA-only: gradients are taken only on the LoRA adapter
parameters, which is what gets trained during SFT.
"""

import argparse
import logging
import os
from typing import Tuple

import torch
from datasets import load_dataset
from transformers import AutoTokenizer

from datasets import Dataset as HFDataset

from common.data import construct_test_sample, encode_with_messages_format
from influence_eval.bbh_data import load_bbh_samples
from influence_eval.model_utils import count_params, load_base_with_fresh_lora
from representation.helper import batch_cosine_similarity
from representation.less.compute_less_embeds import (
    collect_grads,
    normalize_embeddings_in_chunks,
)

logger = logging.getLogger(__name__)


def load_train_dataset(
    tokenizer: AutoTokenizer,
    end_index: int,
    max_seq_length: int = 2048,
) -> torch.utils.data.Dataset:
    ds = load_dataset("Harvard-DCML/tulu-v2-197K-processed", split="train")
    end_index = min(end_index, len(ds))
    ds = ds.select(range(0, end_index))
    ds = ds.map(
        lambda x: encode_with_messages_format(
            example=x,
            tokenizer=tokenizer,
            max_seq_length=max_seq_length,
            include_response=True,
        ),
        num_proc=1,
        load_from_cache_file=False,
    )
    ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    logger.info("Loaded %d train samples", len(ds))
    return ds


def load_anchor_dataset(
    tokenizer: AutoTokenizer,
    dev_dataset_name: str,
    num_anchors: int,
    max_length: int = 2048,
) -> torch.utils.data.Dataset:
    # Load from local BBH files (same seed=42 shuffle as gradient_stocking.py)
    # Spearman eval anchors are always [0:num_anchors]
    samples = load_bbh_samples(n_samples=num_anchors, start_index=0)
    # construct_test_sample expects "prompts"/"labels" keys
    renamed = [{"prompts": s["prompt"], "labels": s["response"]} for s in samples]
    ds = HFDataset.from_list(renamed)
    ds = ds.map(
        lambda x: construct_test_sample(
            tokenizer=tokenizer, sample=x, max_length=max_length
        ),
        num_proc=1,
        load_from_cache_file=False,
    )
    ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    logger.info("Loaded %d anchor samples from local BBH [0:%d]", len(ds), num_anchors)
    return ds


def _collect_and_normalize(
    dataloader,
    model,
    proj_dim: int,
    project_interval: int,
) -> torch.Tensor:
    grads = collect_grads(
        dataloader=dataloader,
        model=model,
        proj_dim=proj_dim,
        adam_optimizer_state=None,
        gradient_type="sgd",
        project_interval=project_interval,
    )
    return normalize_embeddings_in_chunks(
        grads, chunk_size=10000, dim=1, eps=1e-12, in_place=False
    )


def compute_scores(
    model,
    tokenizer,
    save_dir: str,
    out_name: str,
    end_index: int,
    num_anchors: int,
    proj_dim: int,
    dev_dataset_name: str,
    project_interval: int,
    save_grads: bool,
) -> Tuple[str, dict]:
    os.makedirs(save_dir, exist_ok=True)

    # Train grads
    train_ds = load_train_dataset(tokenizer, end_index)
    train_dl = torch.utils.data.DataLoader(train_ds, batch_size=1, shuffle=False)
    logger.info("Computing %d train gradients (proj_dim=%d)", len(train_ds), proj_dim)
    train_grads = _collect_and_normalize(train_dl, model, proj_dim, project_interval)

    # Anchor grads
    anchor_ds = load_anchor_dataset(tokenizer, dev_dataset_name, num_anchors)
    anchor_dl = torch.utils.data.DataLoader(anchor_ds, batch_size=1, shuffle=False)
    logger.info("Computing %d anchor gradients (proj_dim=%d)", len(anchor_ds), proj_dim)
    anchor_grads = _collect_and_normalize(anchor_dl, model, proj_dim, project_interval)

    if save_grads:
        torch.save(train_grads, os.path.join(save_dir, f"{out_name}_train_grads.pt"))
        torch.save(anchor_grads, os.path.join(save_dir, f"{out_name}_anchor_grads.pt"))

    # Score matrix
    logger.info("Computing cosine similarity matrix")
    scores = batch_cosine_similarity(
        dev_reps=anchor_grads,
        train_reps=train_grads,
        chunk_size=256,
        normalize=False,
    ).float()

    out_path = os.path.join(save_dir, f"{out_name}_scores.pt")
    torch.save(scores, out_path)
    logger.info("Saved score matrix: %s shape=%s", out_path, tuple(scores.shape))

    meta = {
        "num_anchors": int(scores.shape[0]),
        "num_train": int(scores.shape[1]),
        "proj_dim": int(proj_dim),
    }
    return out_path, meta


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name", required=True, type=str)
    p.add_argument("--save_dir", required=True, type=str)
    p.add_argument(
        "--out_name",
        required=True,
        type=str,
        help="Prefix for output artifacts. e.g. 'ground_truth' or 'less'.",
    )
    p.add_argument("--end_index", type=int, required=True)
    p.add_argument("--num_anchors", type=int, required=True)
    p.add_argument("--proj_dim", type=int, required=True)
    p.add_argument("--dev_dataset_name", type=str, default="bbh")
    p.add_argument("--lora_target_modules", type=str, default="all-linear")
    p.add_argument("--lora_rank", type=int, default=128)
    p.add_argument("--lora_alpha", type=int, default=512)
    p.add_argument("--lora_dropout", type=float, default=0.1)
    p.add_argument("--lora_seed", type=int, default=0)
    p.add_argument("--project_interval", type=int, default=8)
    p.add_argument("--save_grads", action="store_true")
    return p.parse_args()


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    args = parse_args()

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

    out_path, meta = compute_scores(
        model=model,
        tokenizer=tokenizer,
        save_dir=args.save_dir,
        out_name=args.out_name,
        end_index=args.end_index,
        num_anchors=args.num_anchors,
        proj_dim=args.proj_dim,
        dev_dataset_name=args.dev_dataset_name,
        project_interval=args.project_interval,
        save_grads=args.save_grads,
    )

    # Save param counts alongside scores so FLOPS can read them later
    params = count_params(model)
    params_path = os.path.join(args.save_dir, f"{args.out_name}_params.pt")
    torch.save({**params, **meta, "model_name": args.model_name}, params_path)
    logger.info("Saved param/meta info: %s", params_path)


if __name__ == "__main__":
    main()
