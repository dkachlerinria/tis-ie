#!/bin/bash
set -euo pipefail
CFG="${1:-runs/influence_spearman/config_influence.sh}"
source "$CFG"

mkdir -p "$INFLUENCE_OUT"

python3 -m influence_eval.compute_influcoder_scores \
    --encoder_dir     "${INFLUCODER_ENCODER_DIR}/model" \
    --gradient_model  "${INFLUENCE_MODEL}" \
    --save_dir        "${INFLUENCE_OUT}" \
    --end_index       "${END_INDEX}" \
    --num_anchors     "${NUM_ANCHORS}" \
    --dev_dataset_name "${BENCHMARK}" \
    --n_stock_anchors "$((INFLUCODER_N_TRAIN_A + INFLUCODER_N_EVAL_A))" \
    --n_stock_pool    "$((INFLUCODER_N_TRAIN_P + INFLUCODER_N_EVAL_P))" \
    --proj_dim        "${INFLUCODER_PROJ_DIM}" \
    --stocking_flops_path  "${INFLUCODER_DB_DIR}/_flops.json" \
    --training_flops_path  "${INFLUCODER_ENCODER_DIR}/_flops.json" \
    --stocking_timing_path "${INFLUCODER_DB_DIR}/_timing.json" \
    --training_timing_path "${INFLUCODER_ENCODER_DIR}/_timing.json"
