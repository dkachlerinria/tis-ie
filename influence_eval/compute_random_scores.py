"""Compute (num_anchors, num_train) score matrix with seeded uniform random.

This is the noise floor: any method that does worse than this in Spearman
is actively anti-correlated with true influence.
"""

import argparse
import logging
import os

import torch

logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--save_dir", required=True, type=str)
    p.add_argument("--end_index", type=int, required=True)
    p.add_argument("--num_anchors", type=int, required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out_name", type=str, default="random")
    return p.parse_args()


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    g = torch.Generator().manual_seed(args.seed)
    scores = torch.rand(
        (args.num_anchors, args.end_index), generator=g, dtype=torch.float32
    )

    out_path = os.path.join(args.save_dir, f"{args.out_name}_scores.pt")
    torch.save(scores, out_path)
    logger.info("Saved random score matrix: %s shape=%s", out_path, tuple(scores.shape))

    # Random is trivial RNG; one draw per matrix cell. FlopCounterMode doesn't
    # meaningfully count RNG ops, so we record the exact count directly.
    meta = {
        "num_anchors": int(scores.shape[0]),
        "num_train": int(scores.shape[1]),
        "seed": int(args.seed),
        "measured_flops": int(scores.shape[0]) * int(scores.shape[1]),
    }
    torch.save(meta, os.path.join(args.save_dir, f"{args.out_name}_params.pt"))


if __name__ == "__main__":
    main()
