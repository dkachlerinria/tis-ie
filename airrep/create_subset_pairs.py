"""Stage-1 entry point: create (Tulu-subset, BBH-dev) pairs and persist to JSONL.

Tulu pool indices are 0..(num_tulu-1) within the Tulu-v2-197K dataset.
BBH dev indices are *local* offsets into the slice
`load_bbh_samples(start_index=NUM_ANCHORS, n_samples=dev_pool_size)`; stage-2 and
stage-3 must use the same NUM_ANCHORS and dev_pool_size to resolve them.

Usage:
  python -m airrep.create_subset_pairs \
      --output_path out/airrep/pairs.jsonl \
      --run_mode tiny --num_tulu 197000 --num_anchors 200 --dev_pool_size 1024
"""

import argparse
import json
import logging
import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from airrep.data_sampler import SubsetDevSampler  # noqa: E402

logger = logging.getLogger(__name__)


MODES = {
    "tiny":   {"n_splits": 1,   "subsets_per_split": 4,   "subset_size": 64,   "dev_size": 16},
    "quick":  {"n_splits": 2,   "subsets_per_split": 20,  "subset_size": 256,  "dev_size": 64},
    "small":  {"n_splits": 4,   "subsets_per_split": 50,  "subset_size": 512,  "dev_size": 128},
    "medium": {"n_splits": 8,   "subsets_per_split": 100, "subset_size": 1024, "dev_size": 256},
    "full":   {"n_splits": 100, "subsets_per_split": 100, "subset_size": 1000, "dev_size": 1000},
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--output_path", required=True, type=str)
    p.add_argument("--run_mode", default="tiny", choices=list(MODES.keys()))
    p.add_argument("--num_tulu", type=int, default=197000,
                   help="Size of Tulu train pool to draw subsets from.")
    p.add_argument("--num_anchors", type=int, required=True,
                   help="Number of BBH anchors reserved for Spearman eval; AirRep dev "
                        "indices are offsets into bbh[NUM_ANCHORS:NUM_ANCHORS+dev_pool_size].")
    p.add_argument("--dev_pool_size", type=int, default=1024,
                   help="Size of the BBH slice reserved as AirRep's dev pool.")
    p.add_argument("--seed", type=int, default=42)
    # Overrides for individual MODES fields.
    p.add_argument("--n_splits", type=int, default=None)
    p.add_argument("--subsets_per_split", type=int, default=None)
    p.add_argument("--subset_size", type=int, default=None)
    p.add_argument("--dev_size", type=int, default=None)
    return p.parse_args()


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    args = parse_args()
    cfg = MODES[args.run_mode]
    n_splits = args.n_splits or cfg["n_splits"]
    subsets_per_split = args.subsets_per_split or cfg["subsets_per_split"]
    subset_size = args.subset_size or cfg["subset_size"]
    dev_size = args.dev_size or cfg["dev_size"]

    logger.info(
        "mode=%s n_splits=%d subsets/split=%d subset_size=%d dev_size=%d "
        "num_tulu=%d num_anchors=%d dev_pool_size=%d",
        args.run_mode, n_splits, subsets_per_split, subset_size, dev_size,
        args.num_tulu, args.num_anchors, args.dev_pool_size,
    )

    sampler = SubsetDevSampler(
        train_pool_size=args.num_tulu,
        dev_pool_size=args.dev_pool_size,
        subset_size=subset_size,
        dev_size=dev_size,
        n_splits=n_splits,
        n_subsets_per_split=subsets_per_split,
        seed=args.seed,
    )
    pairs = sampler.sample()

    os.makedirs(os.path.dirname(os.path.abspath(args.output_path)) or ".", exist_ok=True)
    with open(args.output_path, "w") as f:
        for p in pairs:
            f.write(json.dumps(p) + "\n")

    # Also persist the dev-pool config so downstream stages can re-derive the same BBH slice.
    cfg_path = os.path.join(os.path.dirname(args.output_path), "pairs_config.json")
    with open(cfg_path, "w") as f:
        json.dump({
            "num_anchors": args.num_anchors,
            "dev_pool_size": args.dev_pool_size,
            "num_tulu": args.num_tulu,
            "subset_size": subset_size,
            "dev_size": dev_size,
            "run_mode": args.run_mode,
            "seed": args.seed,
        }, f, indent=2)

    logger.info("Wrote %d pairs to %s (config: %s)", len(pairs), args.output_path, cfg_path)


if __name__ == "__main__":
    main()
