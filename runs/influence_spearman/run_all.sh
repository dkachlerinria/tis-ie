#!/bin/bash
set -euo pipefail
# Usage:
#   bash runs/influence_spearman/run_all.sh                                     # full config
#   bash runs/influence_spearman/run_all.sh runs/influence_spearman/config_influence_tiny.sh
CFG="${1:-runs/influence_spearman/config_influence.sh}"
source "$CFG"

echo "=========================================="
echo "Influence-Spearman pipeline"
echo "  TRAINING_MODEL = ${TRAINING_MODEL}"
echo "  BENCHMARK      = ${BENCHMARK}"
echo "  NUM_ANCHORS    = ${NUM_ANCHORS}"
echo "  END_INDEX      = ${END_INDEX}"
echo "  GT_PROJ_DIM    = ${GT_PROJ_DIM}"
echo "  LESS_PROJ_DIM  = ${LESS_PROJ_DIM}"
echo "  INFLUENCE_OUT  = ${INFLUENCE_OUT}"
echo "=========================================="

mkdir -p "$INFLUENCE_OUT"

bash runs/influence_spearman/compute_ground_truth.sh "$CFG"
bash runs/influence_spearman/compute_less_scores.sh "$CFG"
bash runs/influence_spearman/compute_embedding_scores.sh "$CFG"
bash runs/influence_spearman/compute_random_scores.sh "$CFG"

python3 -m influence_eval.run_experiment \
    --out_dir "${INFLUENCE_OUT}" \
    --methods less embedding random \
    --gt_name ground_truth \
    --seq_len "${FLOPS_SEQ_LEN}"

echo
echo "Done. Results in: ${INFLUENCE_OUT}/results.json"
