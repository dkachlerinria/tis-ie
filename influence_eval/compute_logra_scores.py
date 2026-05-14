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
from influence_eval.flops_measure import flop_counter

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
    out_suffix: str = "",
) -> None:
    """Writes BOTH variants in one run (loads the model once):
      - logra_raw_scores.pt: cosine on raw B-gradients (no FIM)
      - logra_fim_scores.pt: cosine on FIM-preconditioned anchor gradients
    """
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

    # Shared encoding cost (train + anchors + FIM accumulation).
    with flop_counter() as enc_counter:
        logger.info("Encoding %d train samples (accumulating FIM)...", len(train_ds))
        train_embeds = logra.encode(train_ds, batch_size=batch_size, is_test=False)
        logger.info("Train embeds shape: %s", train_embeds.shape)

        # Save training FIM before anchor encoding overwrites it
        train_fim = logra.fim

        logger.info("Encoding %d anchors (raw, is_test=False)...", len(anchor_ds))
        raw_anchor_embeds = logra.encode(anchor_ds, batch_size=batch_size, is_test=False)
        logger.info("Raw anchor embeds shape: %s", raw_anchor_embeds.shape)
    shared_flops = int(enc_counter.get_total_flops())

    # logra_raw marginal cost: raw similarity only.
    with flop_counter() as raw_counter:
        logger.info("Computing logra_raw scores (no FIM)...")
        raw_sim = logra.similarity(raw_anchor_embeds, train_embeds, mode="cosine")
        raw_scores = torch.from_numpy(raw_sim).float()
    raw_marginal_flops = int(raw_counter.get_total_flops())

    # logra_fim marginal cost: FIM preconditioning + FIM similarity.
    with flop_counter() as fim_counter:
        logra.fim = train_fim
        precond_anchor_embeds = (
            logra._precondition(torch.from_numpy(raw_anchor_embeds))
            .float()
            .numpy()
        )
        logger.info("FIM-preconditioned anchor embeds shape: %s", precond_anchor_embeds.shape)
        logger.info("Computing logra_fim scores (FIM-preconditioned)...")
        fim_sim = logra.similarity(precond_anchor_embeds, train_embeds, mode="cosine")
        fim_scores = torch.from_numpy(fim_sim).float()
    fim_marginal_flops = int(fim_counter.get_total_flops())

    raw_flops = shared_flops + raw_marginal_flops
    fim_flops = shared_flops + fim_marginal_flops
    logger.info(
        "FLOPs — shared=%.3e, raw_total=%.3e, fim_total=%.3e",
        shared_flops, raw_flops, fim_flops,
    )

    raw_path = os.path.join(save_dir, f"logra_raw{out_suffix}_scores.pt")
    fim_path = os.path.join(save_dir, f"logra_fim{out_suffix}_scores.pt")
    torch.save(raw_scores, raw_path)
    torch.save(fim_scores, fim_path)
    logger.info("Saved %s shape=%s", raw_path, tuple(raw_scores.shape))
    logger.info("Saved %s shape=%s", fim_path, tuple(fim_scores.shape))

    total = sum(p.numel() for p in logra.model.parameters())
    trainable = sum(p.numel() for p in logra.model.parameters() if p.requires_grad)
    base_params = {
        "total": total,
        "trainable": trainable,
        "num_anchors": int(raw_scores.shape[0]),
        "num_train": int(raw_scores.shape[1]),
        "model_name": model_name,
    }
    torch.save(
        {**base_params, "measured_flops": raw_flops},
        os.path.join(save_dir, f"logra_raw{out_suffix}_params.pt"),
    )
    torch.save(
        {**base_params, "measured_flops": fim_flops},
        os.path.join(save_dir, f"logra_fim{out_suffix}_params.pt"),
    )
    logger.info("Saved params for both variants")


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
    p.add_argument("--out_suffix", type=str, default="",
                   help="Suffix appended to output filenames, e.g. '_small' → logra_raw_small_scores.pt")
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
        out_suffix=args.out_suffix,
    )


if __name__ == "__main__":
    main()
