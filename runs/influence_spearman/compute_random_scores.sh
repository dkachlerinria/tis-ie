#!/bin/bash
set -euo pipefail
CFG="${1:-runs/influence_spearman/config_influence.sh}"
source "$CFG"

mkdir -p "$INFLUENCE_OUT"

python3 -m influence_eval.compute_random_scores \
    --save_dir    "${INFLUENCE_OUT}" \
    --out_name    "random" \
    --end_index   100 \
    --num_anchors 100 \
    --seed        "${RANDOM_SEED}"
