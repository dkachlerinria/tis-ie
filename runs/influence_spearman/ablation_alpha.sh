#!/bin/bash
set -euo pipefail
# Ablation over --alpha (Pearson weight in InBatchLoss) for train_influence_encoder.py.
# Trains three times — alpha ∈ {0.0, 0.5, 1.0} — then prints a side-by-side table.
#
# Usage:
#   bash runs/influence_spearman/ablation_alpha.sh
#   bash runs/influence_spearman/ablation_alpha.sh runs/influence_spearman/config_influence_tiny.sh

CFG="${1:-runs/influence_spearman/config_influence.sh}"
source "$CFG"

ABLATION_DIR="${INFLUCODER_ENCODER_DIR}_ablation_alpha"
mkdir -p "$ABLATION_DIR"

ALPHAS=(0.0 0.5 1.0)

for ALPHA in "${ALPHAS[@]}"; do
    OUT_DIR="${ABLATION_DIR}/alpha_${ALPHA}"
    LOG_FILE="${ABLATION_DIR}/alpha_${ALPHA}.log"
    mkdir -p "$OUT_DIR"

    echo ""
    echo "=========================================="
    echo "  alpha=${ALPHA}  →  ${OUT_DIR}"
    echo "=========================================="

    python influcoder/train_influence_encoder.py \
        --anchor_train_prefix "${INFLUCODER_DB_DIR}/train_anchors" \
        --anchor_eval_prefix  "${INFLUCODER_DB_DIR}/eval_anchors" \
        --pool_train_prefix   "${INFLUCODER_DB_DIR}/pool" \
        --pool_eval_prefix    "${INFLUCODER_DB_DIR}/eval_pool" \
        --encoder_model       "${ENCODER_MODEL}" \
        --gradient_model      "${INFLUENCE_MODEL}" \
        --run_mode            "${INFLUCODER_RUN_MODE}" \
        --output_dir          "$OUT_DIR" \
        --lora_rank           "${LORA_RANK}" \
        --lora_alpha          "${LORA_ALPHA}" \
        --lora_dropout        "${LORA_DROPOUT}" \
        --lora_seed           "${LORA_SEED}" \
        --lora_target_modules "${LORA_TARGET_MODULES}" \
        --gt_proj_dim         "${GT_PROJ_DIM}" \
        --project_interval    "${PROJECT_INTERVAL}" \
        --alpha               "$ALPHA" \
        2>&1 | tee "$LOG_FILE"
done

# ── Comparison table ────────────────────────────────────────────────────────
echo ""
echo "=========================================="
echo "  Alpha Ablation — Final Eval Summary"
echo "=========================================="

python3 - "$ABLATION_DIR" <<'PYEOF'
import json, os, sys

ablation_dir = sys.argv[1]
alphas = ["0.0", "0.5", "1.0"]

rows = []
for alpha in alphas:
    meta_path = os.path.join(ablation_dir, f"alpha_{alpha}", "metadata.json")
    if not os.path.exists(meta_path):
        rows.append([alpha, "MISSING", "-", "-", "-"])
        continue
    with open(meta_path) as f:
        meta = json.load(f)
    m = meta.get("metrics", {})
    tr    = m.get("trained",    {})
    tr_gt = m.get("trained_gt", {})
    rows.append([
        alpha,
        f"{tr.get('agg_spearman', 0.0):.4f}",
        f"{tr.get('per_anchor_spearman_mean', 0.0):.4f}",
        f"{tr_gt.get('agg_spearman', 0.0):.4f}",
        f"{tr_gt.get('per_anchor_spearman_mean', 0.0):.4f}",
    ])

col_w = [7, 16, 14, 12, 10]
header = ["alpha", "Agg ρ (sketch)", "PA ρ (sketch)", "Agg ρ (GT)", "PA ρ (GT)"]
sep    = "  ".join("-" * w for w in col_w)
fmt    = "  ".join(f"{h:>{w}}" for h, w in zip(header, col_w))
print(fmt)
print(sep)
for r in rows:
    print("  ".join(f"{v:>{w}}" for v, w in zip(r, col_w)))
PYEOF
