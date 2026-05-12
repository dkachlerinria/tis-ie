"""Spearman correlation metrics between a method's score matrix and ground truth.

All inputs are (num_anchors, num_train) tensors.

Three metrics:
  - per_anchor: for each anchor row, Spearman(row_method, row_gt); return mean,
    std, and the per-anchor vector
  - aggregated_mean: Spearman(method.mean(0), gt.mean(0)) — one scalar
  - aggregated_max:  Spearman(method.max(0), gt.max(0)) — one scalar
"""

from typing import Dict

import numpy as np
import torch
from scipy.stats import spearmanr


def _to_numpy(x: torch.Tensor) -> np.ndarray:
    return x.detach().cpu().float().numpy()


def per_anchor_spearman(method: torch.Tensor, gt: torch.Tensor) -> Dict:
    m = _to_numpy(method)
    g = _to_numpy(gt)
    if m.shape != g.shape:
        raise ValueError(f"shape mismatch: method={m.shape} gt={g.shape}")

    rhos = []
    for i in range(m.shape[0]):
        res = spearmanr(m[i], g[i])
        rho = float(res.statistic) if not np.isnan(res.statistic) else 0.0
        rhos.append(rho)

    rhos_arr = np.array(rhos)
    return {
        "mean": float(rhos_arr.mean()),
        "std": float(rhos_arr.std()),
        "min": float(rhos_arr.min()),
        "max": float(rhos_arr.max()),
        "per_anchor": rhos,
    }


def aggregated_spearman_mean(method: torch.Tensor, gt: torch.Tensor) -> float:
    m = _to_numpy(method).mean(axis=0)
    g = _to_numpy(gt).mean(axis=0)
    res = spearmanr(m, g)
    return float(res.statistic) if not np.isnan(res.statistic) else 0.0


def aggregated_spearman_max(method: torch.Tensor, gt: torch.Tensor) -> float:
    m = _to_numpy(method).max(axis=0)
    g = _to_numpy(gt).max(axis=0)
    res = spearmanr(m, g)
    return float(res.statistic) if not np.isnan(res.statistic) else 0.0


def all_metrics(method: torch.Tensor, gt: torch.Tensor) -> Dict:
    return {
        "per_anchor": per_anchor_spearman(method, gt),
        "aggregated_mean": aggregated_spearman_mean(method, gt),
        "aggregated_max": aggregated_spearman_max(method, gt),
    }
