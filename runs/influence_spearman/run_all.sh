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
rm -f "${INFLUENCE_OUT}/results.json"

ALL_METHODS="less less_small less_proxy embedding random logra_raw logra_fim logra_raw_small logra_fim_small logra_raw_proxy logra_fim_proxy influcoder iprox"

# Print intermediate results after each method completes.
# run_experiment silently skips methods whose files don't exist yet, so this
# grows naturally as more methods finish.
show_results() {
    echo ""
    python3 -m influence_eval.run_experiment \
        --out_dir  "${INFLUENCE_OUT}" \
        --methods  ${ALL_METHODS} \
        --gt_name  ground_truth \
        --seq_len  "${FLOPS_SEQ_LEN}" 2>/dev/null || true
}

# ── Ground truth (must run first — required for all Spearman scores) ──────────
bash runs/influence_spearman/compute_ground_truth.sh "$CFG"           || echo "Ground Truth failed, skipping"

# ── Methods (each prints a running table immediately on completion) ────────────
bash runs/influence_spearman/compute_iprox_scores.sh "$CFG"          && show_results || echo "IProX failed, skipping"

bash runs/influence_spearman/compute_less_scores.sh "$CFG"           && show_results || echo "LESS failed, skipping"
#bash runs/influence_spearman/compute_less_small_scores.sh "$CFG"    && show_results || echo "LESS-small failed, skipping"
bash runs/influence_spearman/compute_embedding_scores.sh "$CFG"      && show_results || echo "Embedding failed, skipping"
#bash runs/influence_spearman/compute_random_scores.sh "$CFG"        && show_results || echo "Random failed, skipping"
#bash runs/influence_spearman/compute_logra_scores.sh "$CFG"         && show_results || echo "LoGRA failed, skipping"
#bash runs/influence_spearman/compute_logra_small_scores.sh "$CFG"   && show_results || echo "LoGRA-small failed, skipping"
#bash runs/influence_spearman/compute_less_proxy_scores.sh "$CFG"     && show_results || echo "LESS-proxy failed, skipping"
#bash runs/influence_spearman/compute_logra_proxy_scores.sh "$CFG"   && show_results || echo "LoGRA-proxy failed, skipping"
#bash runs/influence_spearman/compute_influcoder_scores.sh "$CFG"    && show_results || echo "Influcoder failed, skipping"

# ── Final canonical write ─────────────────────────────────────────────────────
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "FINAL RESULTS"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
python3 -m influence_eval.run_experiment \
    --out_dir  "${INFLUENCE_OUT}" \
    --methods  ${ALL_METHODS} \
    --gt_name  ground_truth \
    --seq_len  "${FLOPS_SEQ_LEN}"

echo ""
echo "Done. Results in: ${INFLUENCE_OUT}/results.json"
