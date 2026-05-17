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

# Force a 100% fresh start: remove all proxy artifacts from previous runs
rm -f  "${IPROX_PROXY_DIR}/init_pytorch_model.bin"
rm -f  "${IPROX_PROXY_DIR}/final_pytorch_model.bin"
rm -rf "${IPROX_PROXY_DIR}/model"

python iprox/train_iprox.py \
    --target_model                 "${INFLUENCE_MODEL}" \
    --train_dataset                "dolly/dolly_data.jsonl" \
    --n_train_p                    100 \
    --pool_start_index             0 \
    --init_method                  IPSVD \
    --sparsity                     "${IPROX_SPARSITY}" \
    --batch_size                   4 \
    --gradient_accumulation_steps  1 \
    --lambda_anchor                0.01 \
    --epochs                       1 \
    --max_seq_length               2048 \
    --lr                           1e-4 \
    --seed                         42 \
    --output_dir                   "${IPROX_PROXY_DIR}" \
    --score_inline

# Test
