"""Stage-2 entry point: SFT on each Tulu subset, evaluate per-example loss on BBH dev.

For each (subset, dev) pair from stage 1:
  - SFT a small causal LM on the Tulu subset examples.
  - Evaluate per-example loss on the BBH dev examples.
  - Write `loss-{pair_id}.json` with the per-example losses.
GPU work is wrapped in `flop_counter()`; the total is accumulated into
`_flops_sft.json` under `--output_dir` (consumed by compute_airrep_scores.py).

Shard with `--start_idx`/`--num_gpus` to run multiple ranks in parallel:
each rank handles pair indices `i` where `i % num_gpus == start_idx`.
"""

import argparse
import json
import logging
import os
import sys

import torch
from datasets import load_dataset

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from airrep.sft_trainer import SFTTrainer  # noqa: E402
from influence_eval.bbh_data import load_bbh_samples  # noqa: E402
from influence_eval.flops_measure import add_phase_flops, flop_counter  # noqa: E402

logger = logging.getLogger(__name__)


MODES = {
    "tiny":   {"epochs": 1, "batch_size": 4,  "lr": 2e-5},
    "quick":  {"epochs": 2, "batch_size": 8,  "lr": 2e-5},
    "small":  {"epochs": 2, "batch_size": 8,  "lr": 2e-5},
    "medium": {"epochs": 2, "batch_size": 16, "lr": 2e-5},
    "full":   {"epochs": 2, "batch_size": 16, "lr": 2e-5},
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--pairs_path", required=True, type=str)
    p.add_argument("--output_dir", required=True, type=str)
    p.add_argument("--model_name", default="Qwen/Qwen3-0.6B-Base", type=str)
    p.add_argument("--run_mode", default="tiny", choices=list(MODES.keys()))
    p.add_argument("--max_length", type=int, default=1024)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--start_idx", type=int, default=0)
    p.add_argument("--num_gpus", type=int, default=1)
    p.add_argument("--use_flash_attn", action="store_true")
    return p.parse_args()


def _load_pairs(path: str):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def _load_pairs_config(pairs_path: str):
    cfg_path = os.path.join(os.path.dirname(pairs_path), "pairs_config.json")
    with open(cfg_path) as f:
        return json.load(f)


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    args = parse_args()
    cfg = MODES[args.run_mode]
    epochs = args.epochs or cfg["epochs"]
    batch_size = args.batch_size or cfg["batch_size"]
    lr = args.lr if args.lr is not None else cfg["lr"]

    os.makedirs(args.output_dir, exist_ok=True)
    flops_path = os.path.join(args.output_dir, "_flops_sft.json")

    pairs = _load_pairs(args.pairs_path)
    pairs_cfg = _load_pairs_config(args.pairs_path)
    logger.info(
        "Loaded %d pairs (subset_size=%d, dev_size=%d, num_tulu=%d, dev_pool_size=%d, num_anchors=%d)",
        len(pairs),
        pairs_cfg["subset_size"], pairs_cfg["dev_size"],
        pairs_cfg["num_tulu"], pairs_cfg["dev_pool_size"], pairs_cfg["num_anchors"],
    )

    # Shard
    shard = [p for i, p in enumerate(pairs) if i % args.num_gpus == args.start_idx]
    logger.info("Rank %d/%d processing %d pairs", args.start_idx, args.num_gpus, len(shard))

    logger.info("Loading Tulu train pool...")
    tulu = load_dataset("Harvard-DCML/tulu-v2-197K-processed", split="train")
    tulu = tulu.select(range(pairs_cfg["num_tulu"]))

    logger.info("Loading BBH dev pool (start=%d, n=%d)...",
                pairs_cfg["num_anchors"], pairs_cfg["dev_pool_size"])
    bbh_dev = load_bbh_samples(
        n_samples=pairs_cfg["dev_pool_size"],
        start_index=pairs_cfg["num_anchors"],
    )

    trainer = SFTTrainer(
        model_name=args.model_name,
        batch_size=batch_size,
        epochs=epochs,
        lr=lr,
        max_length=args.max_length,
        use_flash_attn=args.use_flash_attn,
    )

    for pair in shard:
        out_path = os.path.join(args.output_dir, f"loss-{pair['id']}.json")
        if os.path.exists(out_path):
            logger.info("Skipping %s (exists)", out_path)
            continue

        select_examples = [tulu[i] for i in pair["select"]]
        dev_samples = [bbh_dev[j] for j in pair["dev"]]

        logger.info(
            "Pair %s: training on %d Tulu, evaluating on %d BBH",
            pair["id"], len(select_examples), len(dev_samples),
        )

        with flop_counter() as counter:
            model = trainer.train(select_examples)
            losses = trainer.evaluate(model, dev_samples)
        pair_flops = int(counter.get_total_flops())
        add_phase_flops(flops_path, pair_flops)

        with open(out_path, "w") as f:
            json.dump({
                "pair_id": pair["id"],
                "dev_id": pair["dev_id"],
                "losses": losses,
                "mean_loss": float(sum(losses) / len(losses)) if losses else 0.0,
                "flops": pair_flops,
            }, f)
        logger.info("Wrote %s (FLOPs=%.3e, mean loss=%.4f)",
                    out_path, pair_flops, float(sum(losses) / max(1, len(losses))))

        # Free GPU memory between pairs.
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None


if __name__ == "__main__":
    main()
