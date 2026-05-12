"""Glue: load each method's score matrix, compute Spearman metrics, look up
analytic FLOPS, write results.json and print a markdown table.

Expects `{method}_scores.pt` and `{method}_params.pt` to exist in `--out_dir`
for each method named in `--methods` (default: less, embedding, random). The
ground-truth artifacts are expected under the name `ground_truth`.
"""

import argparse
import json
import logging
import os
from typing import Dict, List

import torch

from influence_eval.flops import flops_for_method
from influence_eval.spearman import all_metrics

logger = logging.getLogger(__name__)


def _load(out_dir: str, name: str):
    scores_path = os.path.join(out_dir, f"{name}_scores.pt")
    params_path = os.path.join(out_dir, f"{name}_params.pt")
    if not os.path.exists(scores_path):
        raise FileNotFoundError(scores_path)
    if not os.path.exists(params_path):
        raise FileNotFoundError(params_path)
    return torch.load(scores_path, map_location="cpu"), torch.load(
        params_path, map_location="cpu"
    )


def _markdown_table(results: dict) -> str:
    lines = [
        "| method | per-anchor mean | per-anchor std | agg(mean) | agg(max) | FLOPS |",
        "|---|---|---|---|---|---|",
    ]
    for method, r in results["methods"].items():
        lines.append(
            f"| {method} | {r['per_anchor']['mean']:.4f} | "
            f"{r['per_anchor']['std']:.4f} | "
            f"{r['aggregated_mean']:.4f} | "
            f"{r['aggregated_max']:.4f} | "
            f"{r['flops']:.3e} |"
        )
    return "\n".join(lines)


def run(out_dir: str, methods: List[str], seq_len: int, gt_name: str = "ground_truth") -> Dict:
    gt_scores, gt_params = _load(out_dir, gt_name)
    logger.info("Loaded ground truth: shape=%s", tuple(gt_scores.shape))

    results = {
        "config": {
            "out_dir": out_dir,
            "gt_name": gt_name,
            "gt_model": gt_params.get("model_name"),
            "num_anchors": gt_params["num_anchors"],
            "num_train": gt_params["num_train"],
            "gt_proj_dim": gt_params.get("proj_dim"),
            "gt_num_params_total": gt_params.get("total"),
            "gt_num_params_trainable": gt_params.get("trainable"),
            "seq_len_for_flops": seq_len,
        },
        "methods": {},
    }

    for method in methods:
        scores, params = _load(out_dir, method)
        if scores.shape != gt_scores.shape:
            raise ValueError(
                f"{method} shape {tuple(scores.shape)} != GT {tuple(gt_scores.shape)}"
            )
        metrics = all_metrics(scores, gt_scores)
        flops = flops_for_method(method, params, seq_len=seq_len)
        results["methods"][method] = {
            **metrics,
            "flops": int(flops) if flops is not None else None,
            "params_meta": {
                k: int(v) if isinstance(v, (int, float)) and k != "model_name" else v
                for k, v in params.items()
            },
        }
        logger.info(
            "%s: per-anchor mean=%.4f agg_mean=%.4f agg_max=%.4f flops=%.3e",
            method,
            metrics["per_anchor"]["mean"],
            metrics["aggregated_mean"],
            metrics["aggregated_max"],
            flops or 0,
        )

    # Also report ground-truth's own self-FLOPS for context
    gt_flops = flops_for_method("ground_truth", gt_params, seq_len=seq_len)
    results["config"]["gt_flops"] = int(gt_flops) if gt_flops else None

    return results


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir", required=True, type=str)
    p.add_argument(
        "--methods",
        nargs="+",
        default=["less", "embedding", "random", "influcoder"],
        help="Method names; each expects {method}_scores.pt + {method}_params.pt",
    )
    p.add_argument("--gt_name", type=str, default="ground_truth")
    p.add_argument("--seq_len", type=int, default=2048, help="Used for FLOPS accounting only")
    args = p.parse_args()

    results = run(args.out_dir, args.methods, args.seq_len, args.gt_name)

    out_path = os.path.join(args.out_dir, "results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Wrote %s", out_path)

    print()
    print(_markdown_table(results))
    print()


if __name__ == "__main__":
    main()
