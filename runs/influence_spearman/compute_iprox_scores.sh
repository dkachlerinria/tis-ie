#!/bin/bash
set -euo pipefail
# Usage:
#   bash runs/influence_spearman/compute_iprox_scores.sh runs/influence_spearman/config_influence_tiny.sh

CFG="${1:-runs/influence_spearman/config_influence.sh}"
source "$CFG"

IPROX_PROXY_DIR="${INFLUENCE_OUT}/iprox_proxy/model"

if [ ! -d "$IPROX_PROXY_DIR" ]; then
    echo "❌ Error: IProX proxy not found at $IPROX_PROXY_DIR. Run train_iprox_proxy.sh first."
    exit 1
fi

echo "🧮 Computing IProX similarity matrix for Spearman evaluation..."

python3 -m influence_eval.compute_iprox_scores \
    --proxy_path         "$IPROX_PROXY_DIR" \
    --save_dir           "${INFLUENCE_OUT}" \
    --train_dataset_name "dolly/dolly_data.jsonl" \
    --end_index          100 \
    --num_anchors        100 \
    --sparsity           "${IPROX_SPARSITY}" \
    --out_name           "iprox"
