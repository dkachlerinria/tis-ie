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

# Force a 100% fresh start by deleting the cached SVD factorization
rm -f "${IPROX_PROXY_DIR}/init_pytorch_model.bin"

python iprox/train_iprox.py \
    --target_model      "${INFLUENCE_MODEL}" \
    --train_dataset     "${TRAIN_DATASET}" \
    --n_train_p         8000 \
    --pool_start_index  "${END_INDEX}" \
    --sparsity          0.9 \
    --batch_size        1 \
    --gradient_accumulation_steps 1 \
    --lambda_anchor     0.0 \
    --epochs            1 \
    --max_seq_length    2048 \
    --output_dir        "${IPROX_PROXY_DIR}" \
    --lr                1e-4 \
    --seed              137
