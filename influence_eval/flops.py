"""Analytic FLOPS formulas per selection method.

Closed-form so they're reproducible across hardware. Includes a
__main__ profiler sanity check that compares analytic vs measured
FLOPS for one forward+backward pass.

Conventions:
  - Forward through dense linear layers: ~ 2 * P * L FLOPS where P is
    parameter count and L is sequence length.
  - Backward: ~ 2x forward (the standard rule of thumb for autograd
    through dense linear graphs), so ~ 4 * P * L. Total fwd+bwd ~ 6 P L.
  - This counts the full forward+backward through the WHOLE network
    even when only LoRA params are trainable, because PyTorch still
    computes activation grads through frozen layers.
"""

import argparse
import json
from typing import Optional


def flops_forward_backward(num_params_total: int, seq_len: int) -> int:
    return 6 * int(num_params_total) * int(seq_len)


def flops_forward_only(num_params: int, seq_len: int) -> int:
    return 2 * int(num_params) * int(seq_len)


def flops_trak_projection(num_params_trainable: int, proj_dim: int) -> int:
    # One matmul: gradient vector of size P_trainable times sketch matrix
    return 2 * int(num_params_trainable) * int(proj_dim)


def flops_cosine_matrix(num_anchors: int, num_train: int, dim: int) -> int:
    return 2 * int(num_anchors) * int(num_train) * int(dim)


def flops_less(
    *,
    num_params_total: int,
    num_params_lora: int,
    seq_len: int,
    num_anchors: int,
    num_train: int,
    proj_dim: int,
) -> int:
    """LESS / gradient-based methods (also ground truth at higher proj_dim)."""
    per_sample = flops_forward_backward(num_params_total, seq_len) + flops_trak_projection(
        num_params_lora, proj_dim
    )
    return (num_anchors + num_train) * per_sample + flops_cosine_matrix(
        num_anchors, num_train, proj_dim
    )


def flops_embedding(
    *,
    num_params_encoder: int,
    seq_len: int,
    num_anchors: int,
    num_train: int,
    emb_dim: int,
) -> int:
    """Sentence-transformer encoder: forward only, no backward."""
    per_sample = flops_forward_only(num_params_encoder, seq_len)
    return (num_anchors + num_train) * per_sample + flops_cosine_matrix(
        num_anchors, num_train, emb_dim
    )


def flops_logra(
    *,
    num_params_total: int,
    num_params_logra_b: int,
    seq_len: int,
    num_anchors: int,
    num_train: int,
) -> int:
    """LoGRA: fwd+bwd per sample (base + LoRA-A/B/C), cosine on LoRA-B grad vectors.

    grad_dim = num_params_logra_b = num_lora_modules * rank^2.
    FIM inversion cost O(n_modules * rank^6) is negligible for rank<=8.
    """
    per_sample = flops_forward_backward(num_params_total, seq_len)
    return (num_anchors + num_train) * per_sample + flops_cosine_matrix(
        num_anchors, num_train, num_params_logra_b
    )


def flops_random(*, num_anchors: int, num_train: int) -> int:
    # One RNG draw per cell; negligible vs the others but nonzero.
    return int(num_anchors) * int(num_train)


def flops_influcoder(
    *,
    num_params_total: int,
    num_params_linear: int,
    seq_len: int,
    n_stock_anchors: int,
    n_stock_pool: int,
    proj_dim: int,
    num_params_encoder: int,
    num_anchors: int,
    num_train: int,
    emb_dim: int,
) -> int:
    """Influcoder: gradient stocking (LESS-style) + encoder inference.

    Stocking cost: fwd+bwd on (n_stock_anchors + n_stock_pool) samples with
    TRAK projection onto proj_dim, using all-linear target modules.
    Encoder inference cost: forward-only for the Spearman eval set.
    Encoder training cost is negligible vs stocking and omitted.
    """
    stocking = flops_less(
        num_params_total=num_params_total,
        num_params_lora=num_params_linear,
        seq_len=seq_len,
        num_anchors=n_stock_anchors,
        num_train=n_stock_pool,
        proj_dim=proj_dim,
    )
    inference = flops_embedding(
        num_params_encoder=num_params_encoder,
        seq_len=seq_len,
        num_anchors=num_anchors,
        num_train=num_train,
        emb_dim=emb_dim,
    )
    return stocking + inference


def flops_for_method(method: str, params: dict, seq_len: int = 2048) -> Optional[int]:
    """Dispatch helper for run_experiment.py.

    `params` is the dict saved by each compute_*_scores.py as
    `{method}_params.pt`.
    """
    if method in ("ground_truth", "less"):
        return flops_less(
            num_params_total=params["total"],
            num_params_lora=params["trainable"],
            seq_len=seq_len,
            num_anchors=params["num_anchors"],
            num_train=params["num_train"],
            proj_dim=params["proj_dim"],
        )
    if method == "embedding":
        return flops_embedding(
            num_params_encoder=params["total"],
            seq_len=seq_len,
            num_anchors=params["num_anchors"],
            num_train=params["num_train"],
            emb_dim=params["emb_dim"],
        )
    if method in ("logra_raw", "logra_fim"):
        # FIM inversion cost is negligible vs fwd+bwd; both variants share FLOPS.
        return flops_logra(
            num_params_total=params["total"],
            num_params_logra_b=params["trainable"],
            seq_len=seq_len,
            num_anchors=params["num_anchors"],
            num_train=params["num_train"],
        )
    if method == "random":
        return flops_random(
            num_anchors=params["num_anchors"], num_train=params["num_train"]
        )
    if method == "influcoder":
        return flops_influcoder(
            num_params_total=params["grad_model_total"],
            num_params_linear=params["grad_model_linear"],
            seq_len=seq_len,
            n_stock_anchors=params["n_stock_anchors"],
            n_stock_pool=params["n_stock_pool"],
            proj_dim=params["proj_dim"],
            num_params_encoder=params["total"],
            num_anchors=params["num_anchors"],
            num_train=params["num_train"],
            emb_dim=params["emb_dim"],
        )
    return None


def _profiler_sanity_check(model_name: str, seq_len: int = 512):
    """Compare analytic fwd+bwd FLOPS against torch.profiler's flop counter.

    Loads the model with a fresh LoRA adapter, runs one forward+backward on
    a dummy input, and prints predicted vs measured FLOPS.
    """
    import torch
    from torch.utils.flop_counter import FlopCounterMode
    from transformers import AutoTokenizer

    from influence_eval.model_utils import count_params, load_base_with_fresh_lora

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = load_base_with_fresh_lora(
        model_name=model_name,
        tokenizer=tokenizer,
        lora_target_modules="all-linear",
        lora_rank=128,
        lora_alpha=512,
        lora_dropout=0.1,
        seed=0,
    )
    model.train()

    params = count_params(model)
    predicted = flops_forward_backward(params["total"], seq_len)

    device = next(model.parameters()).device
    input_ids = torch.randint(0, tokenizer.vocab_size, (1, seq_len), device=device)
    labels = input_ids.clone()
    attention_mask = torch.ones_like(input_ids)

    with FlopCounterMode(display=False) as counter:
        loss = model(
            input_ids=input_ids, attention_mask=attention_mask, labels=labels
        ).loss
        loss.backward()
    measured = counter.get_total_flops()

    print(
        json.dumps(
            {
                "model_name": model_name,
                "seq_len": seq_len,
                "num_params_total": params["total"],
                "num_params_trainable": params["trainable"],
                "predicted_fwd_bwd_flops": predicted,
                "measured_fwd_bwd_flops": int(measured),
                "ratio_measured_over_predicted": measured / predicted
                if predicted > 0
                else None,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FLOPS sanity check via torch profiler")
    parser.add_argument("--model_name", required=True, type=str)
    parser.add_argument("--seq_len", type=int, default=512)
    args = parser.parse_args()
    _profiler_sanity_check(args.model_name, args.seq_len)
