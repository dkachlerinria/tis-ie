#!/bin/bash
set -euo pipefail
# Usage:
#   bash runs/influence_spearman/run_all.sh                                     # full config
#   bash runs/influence_spearman/run_all.sh runs/influence_spearman/config_influence_tiny.sh
CFG="${1:-runs/influence_spearman/config_influence.sh}"
source "$CFG"

echo "=========================================="
echo "Influence-Spearman pipeline"
echo "  INFLUENCE_MODEL = ${INFLUENCE_MODEL}"
echo "  BENCHMARK      = ${BENCHMARK}"
echo "  NUM_ANCHORS    = ${NUM_ANCHORS}"
echo "  END_INDEX      = ${END_INDEX}"
echo "  GT_PROJ_DIM    = ${GT_PROJ_DIM}"
echo "  LESS_PROJ_DIM  = ${LESS_PROJ_DIM}"
echo "  INFLUENCE_OUT  = ${INFLUENCE_OUT}"
echo "=========================================="

mkdir -p "$INFLUENCE_OUT"

# Wipe all stale score/params artifacts so run_experiment only sees results
# from this run — not leftover files from a previous invocation.
rm -f "${INFLUENCE_OUT}"/*_scores.pt "${INFLUENCE_OUT}"/*_params.pt

bash runs/influence_spearman/compute_influcoder_scores.sh "$CFG"       || echo "Influcoder failed, skipping"
#bash runs/influence_spearman/compute_iprox_scores.sh "$CFG"           || echo "IProX failed, skipping"
bash runs/influence_spearman/compute_ground_truth.sh "$CFG"           || echo "Ground Truth failed, skipping"
#bash runs/influence_spearman/compute_less_scores.sh "$CFG"           || echo "LESS failed, skipping"
#bash runs/influence_spearman/compute_less_small_scores.sh "$CFG"     || echo "LESS-small failed, skipping"
bash runs/influence_spearman/compute_embedding_scores.sh "$CFG"      || echo "Embedding failed, skipping"
#bash runs/influence_spearman/compute_random_scores.sh "$CFG"         || echo "Random failed, skipping"
bash runs/influence_spearman/compute_logra_scores.sh "$CFG"          || echo "LoGRA failed, skipping"
bash runs/influence_spearman/compute_logra_small_scores.sh "$CFG"    || echo "LoGRA-small failed, skipping"
# Force a fresh evaluation summary
rm -f "${INFLUENCE_OUT}/results.json"

python3 -m influence_eval.run_experiment \
    --out_dir "${INFLUENCE_OUT}" \
    --methods less less_small embedding random logra_raw logra_fim logra_raw_small logra_fim_small influcoder iprox \
    --gt_name ground_truth \
    --seq_len "${FLOPS_SEQ_LEN}"

echo
echo "Done. Results in: ${INFLUENCE_OUT}/results.json"
