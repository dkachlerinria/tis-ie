#!/bin/bash
set -euo pipefail
CFG="${1:-runs/influence_spearman/config_influence.sh}"
source "$CFG"

mkdir -p "$INFLUENCE_OUT"

python3 -m influence_eval.compute_embedding_scores \
    --encoder_model      "${ENCODER_MODEL}" \
    --save_dir           "${INFLUENCE_OUT}" \
    --out_name           "embedding" \
    --end_index          "${END_INDEX}" \
    --num_anchors        "${NUM_ANCHORS}" \
    --dev_dataset_name   "${BENCHMARK}" \
    --local_train_dataset "dolly/dolly_data.jsonl" \
    --batch_size         32
