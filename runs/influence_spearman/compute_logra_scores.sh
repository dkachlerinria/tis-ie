#!/bin/bash
set -euo pipefail
CFG="${1:-runs/influence_spearman/config_influence.sh}"
source "$CFG"

mkdir -p "$INFLUENCE_OUT"

python3 -m influence_eval.compute_logra_scores \
    --model_name "${INFLUENCE_MODEL}" \
    --save_dir "${INFLUENCE_OUT}" \
    --end_index "${END_INDEX}" \
    --num_anchors "${NUM_ANCHORS}" \
    --dev_dataset_name "${BENCHMARK}" \
    --logra_rank "${LOGRA_RANK}" \
    --batch_size "${LOGRA_BATCH_SIZE}" \
    ${LOGRA_ALL_LINEAR:+--no_mlp_only}
