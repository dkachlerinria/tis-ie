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
from influence_eval.flops_measure import flop_counter
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
    local_train_dataset: str = None,
) -> torch.utils.data.Dataset:
    if local_train_dataset:
        # Local file (e.g. dolly): load the first end_index rows in file order, no shuffle.
        ds = load_dataset("json", data_files=[local_train_dataset])["train"]
        n = min(end_index, len(ds))
        ds = ds.select(range(n))
    else:
        ds = load_dataset("Harvard-DCML/tulu-v2-197K-processed", split="train")
        ds = ds.shuffle(seed=42)
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
    # Use construct_test_sample (prompt + response tokenized separately) so the
    # response is always present even when the CoT prefix is thousands of tokens.
    # Joint encoding with max_seq_length=2048 would truncate the answer away for
    # long CoT prompts, making all same-task samples identical → constant scores.
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

    # Diagnostic: detect samples whose response will be invisible to the model.
    n_dead = 0
    for i in range(len(ds)):
        labs = ds[i]["labels"]
        survived = (labs != -100).sum().item()
        if survived == 0:
            n_dead += 1
    if n_dead:
        logger.warning(
            "BBH anchors: %d/%d samples have ALL labels=-100 (response truncated or missing). "
            "These contribute zero gradient — increase max_length or shorten prompts.",
            n_dead, num_anchors,
        )
    else:
        logger.info("BBH anchors: all %d samples have at least one non-masked label token.", num_anchors)
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
    local_train_dataset: str = None,
    local_anchor_offset: bool = False,
) -> Tuple[str, dict]:
    os.makedirs(save_dir, exist_ok=True)

    # Load datasets outside FlopCounterMode (data loading is not GPU work).
    train_ds = load_train_dataset(
        tokenizer, end_index,
        local_train_dataset=local_train_dataset,
    )
    train_dl = torch.utils.data.DataLoader(train_ds, batch_size=1, shuffle=False)

    if local_anchor_offset and local_train_dataset:
        # Anchors = next slice of the local file after the train pool: [end_index : end_index+num_anchors].
        logger.info("Loading anchors from %s [%d:%d]", local_train_dataset, end_index, end_index + num_anchors)
        raw = load_dataset("json", data_files=[local_train_dataset])["train"]
        raw = raw.select(range(end_index, min(end_index + num_anchors, len(raw))))
        anchor_ds = raw.map(
            lambda x: encode_with_messages_format(
                example=x, tokenizer=tokenizer, max_seq_length=2048, include_response=True,
            ),
            num_proc=1, load_from_cache_file=False,
        )
        anchor_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    else:
        anchor_ds = load_anchor_dataset(tokenizer, dev_dataset_name, num_anchors)
    anchor_dl = torch.utils.data.DataLoader(anchor_ds, batch_size=1, shuffle=False)

    logger.info("Computing %d train gradients (proj_dim=%d)", len(train_ds), proj_dim)
    logger.info("Computing %d anchor gradients (proj_dim=%d)", len(anchor_ds), proj_dim)

    # Save tokenized datasets so downstream methods (IProX, proxy variants) can
    # load the EXACT same input_ids/attention_mask/labels the GT used, eliminating
    # any tokenization format discrepancy.
    anchor_save = os.path.join(save_dir, "tokenized_anchor_ds")
    train_save = os.path.join(save_dir, "tokenized_train_ds")
    anchor_ds.save_to_disk(anchor_save)
    train_ds.save_to_disk(train_save)
    logger.info("Saved tokenized anchor_ds → %s", anchor_save)
    logger.info("Saved tokenized train_ds  → %s", train_save)

    import time
    t0 = time.perf_counter()
    with flop_counter() as counter:
        train_grads = _collect_and_normalize(train_dl, model, proj_dim, project_interval)
        anchor_grads = _collect_and_normalize(anchor_dl, model, proj_dim, project_interval)
        scores = batch_cosine_similarity(
            dev_reps=anchor_grads,
            train_reps=train_grads,
            chunk_size=256,
            normalize=False,
        ).float()
    inference_time_s = time.perf_counter() - t0
    measured_flops = int(counter.get_total_flops())
    n_samples = len(anchor_ds) + len(train_ds)
    time_per_sample_s = inference_time_s / max(n_samples, 1)
    logger.info("Measured FLOPs: %.3e | Wall-clock: %.2fs (%.2fms/sample)",
                measured_flops, inference_time_s, 1000 * time_per_sample_s)

    if save_grads:
        torch.save(train_grads, os.path.join(save_dir, f"{out_name}_train_grads.pt"))
        torch.save(anchor_grads, os.path.join(save_dir, f"{out_name}_anchor_grads.pt"))

    out_path = os.path.join(save_dir, f"{out_name}_scores.pt")
    torch.save(scores, out_path)
    logger.info("Saved score matrix: %s shape=%s", out_path, tuple(scores.shape))

    meta = {
        "num_anchors": int(scores.shape[0]),
        "num_train": int(scores.shape[1]),
        "proj_dim": int(proj_dim),
        "measured_flops": measured_flops,
        "inference_time_s": float(inference_time_s),
        "time_per_sample_s": float(time_per_sample_s),
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
    p.add_argument("--local_train_dataset", type=str, default=None,
                   help="Path to local JSONL file (e.g. dolly/dolly_data.jsonl). Uses first --end_index rows in file order instead of tulu.")
    p.add_argument("--local_anchor_offset", action="store_true",
                   help="When set (with --local_train_dataset), use rows [end_index:end_index+num_anchors] of the local file as anchors instead of BBH.")
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
        local_train_dataset=args.local_train_dataset,
        local_anchor_offset=args.local_anchor_offset,
    )

    # Save param counts alongside scores so FLOPS can read them later
    params = count_params(model)
    params_path = os.path.join(args.save_dir, f"{args.out_name}_params.pt")
    torch.save({**params, **meta, "model_name": args.model_name}, params_path)
    logger.info("Saved param/meta info: %s", params_path)


if __name__ == "__main__":
    main()
