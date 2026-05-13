#!/bin/bash
set -euo pipefail
# Usage:
#   bash runs/influence_spearman/train_iprox_proxy.sh runs/influence_spearman/config_influence_tiny.sh

CFG="${1:-runs/influence_spearman/config_influence.sh}"
source "$CFG"

# IProX specific output dir
IPROX_PROXY_DIR="${INFLUENCE_OUT}/iprox_proxy"
mkdir -p "${IPROX_PROXY_DIR}"

echo "🚀 Training IProX Proxy (Mode: ${INFLUCODER_RUN_MODE})..."

python iprox/train_iprox_gradients.py \
    --run_mode "${INFLUCODER_RUN_MODE}" \
    --anchor_train_db "${INFLUCODER_DB_DIR}/train_anchors.sqlite" \
    --anchor_eval_db  "${INFLUCODER_DB_DIR}/eval_anchors.sqlite" \
    --pool_train_db   "${INFLUCODER_DB_DIR}/pool.sqlite" \
    --pool_eval_db    "${INFLUCODER_DB_DIR}/eval_pool.sqlite" \
    --target_model    "${INFLUENCE_MODEL}" \
    --sparsity        0.5 \
    --output_dir      "${IPROX_PROXY_DIR}" \
    --gradient_accumulation_steps "${GRAD_ACC}" \
    --lr              1e-4 \
    --seed            137
