"""Compute (num_anchors, num_train) score matrix via LoGRA.

LoGRA uses per-sample gradients from LoRA-B layers with Fisher Information
Matrix (FIM) preconditioning on anchor embeddings.

Order matters: train must be encoded first (FIM accumulation, is_test=False),
then anchors (FIM-preconditioned, is_test=True).
"""

import argparse
import logging
import os
import sys

import torch

# Add logra/ to path so `from less.utils.modeling_logra import LoGra` resolves
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_LOGRA_DIR = os.path.join(_REPO_ROOT, "logra")
if _LOGRA_DIR not in sys.path:
    sys.path.insert(0, _LOGRA_DIR)

from less.utils.modeling_logra import LoGra

from influence_eval.compute_gradient_scores import load_anchor_dataset, load_train_dataset

logger = logging.getLogger(__name__)


def compute_logra_scores(
    model_name: str,
    save_dir: str,
    end_index: int,
    num_anchors: int,
    dev_dataset_name: str,
    logra_rank: int,
    mlp_only: bool,
    batch_size: int,
) -> str:
    os.makedirs(save_dir, exist_ok=True)

    logra = LoGra.from_pretrained(
        model_name=model_name,
        rank=logra_rank,
        mlp_only=mlp_only,
    )
    tokenizer = logra.tokenizer
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_ds = load_train_dataset(tokenizer, end_index)
    anchor_ds = load_anchor_dataset(tokenizer, dev_dataset_name, num_anchors)

    logger.info("Encoding %d train samples (accumulating FIM)...", len(train_ds))
    train_embeds = logra.encode(train_ds, batch_size=batch_size, is_test=False)
    logger.info("Train embeds shape: %s", train_embeds.shape)

    logger.info("Encoding %d anchors (FIM-preconditioned)...", len(anchor_ds))
    anchor_embeds = logra.encode(anchor_ds, batch_size=batch_size, is_test=True)
    logger.info("Anchor embeds shape: %s", anchor_embeds.shape)

    sim = logra.similarity(anchor_embeds, train_embeds, mode="cosine")
    scores = torch.from_numpy(sim).float()

    out_path = os.path.join(save_dir, "logra_scores.pt")
    torch.save(scores, out_path)
    logger.info("Saved logra scores: %s shape=%s", out_path, tuple(scores.shape))

    total = sum(p.numel() for p in logra.model.parameters())
    trainable = sum(p.numel() for p in logra.model.parameters() if p.requires_grad)
    params = {
        "total": total,
        "trainable": trainable,
        "num_anchors": int(scores.shape[0]),
        "num_train": int(scores.shape[1]),
        "model_name": model_name,
    }
    params_path = os.path.join(save_dir, "logra_params.pt")
    torch.save(params, params_path)
    logger.info("Saved logra params: %s", params_path)

    return out_path


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name", required=True)
    p.add_argument("--save_dir", required=True)
    p.add_argument("--end_index", type=int, required=True)
    p.add_argument("--num_anchors", type=int, required=True)
    p.add_argument("--dev_dataset_name", type=str, default="bbh")
    p.add_argument("--logra_rank", type=int, default=8)
    p.add_argument("--no_mlp_only", action="store_true")
    p.add_argument("--batch_size", type=int, default=1)
    return p.parse_args()


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    args = parse_args()
    compute_logra_scores(
        model_name=args.model_name,
        save_dir=args.save_dir,
        end_index=args.end_index,
        num_anchors=args.num_anchors,
        dev_dataset_name=args.dev_dataset_name,
        logra_rank=args.logra_rank,
        mlp_only=not args.no_mlp_only,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
