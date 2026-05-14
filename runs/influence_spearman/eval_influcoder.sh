#!/bin/bash
set -euo pipefail
# Usage:
#   bash runs/influence_spearman/eval_influcoder.sh [config_file]

CFG="${1:-runs/influence_spearman/config_influence.sh}"
source "$CFG"

echo "----------------------------------------------------------------------"
echo "🔍 EVALUATING INFLUCODER"
echo "   Out Dir: ${INFLUENCE_OUT}"
echo "----------------------------------------------------------------------"

# 1. Compute scores if missing (or re-run to be sure)
bash runs/influence_spearman/compute_influcoder_scores.sh "$CFG"

# 2. Run final evaluation script for just this method
python3 -m influence_eval.run_experiment \
    --out_dir "${INFLUENCE_OUT}" \
    --methods influcoder \
    --gt_name ground_truth \
    --seq_len "${FLOPS_SEQ_LEN}"
