#!/bin/bash
set -euo pipefail
# Config can be overridden by passing path as $1
CFG="${1:-runs/influence_spearman/config_influence.sh}"
source "$CFG"

mkdir -p "$INFLUENCE_OUT"

python3 -m influence_eval.compute_gradient_scores \
    --model_name "${INFLUENCE_MODEL}" \
    --save_dir "${INFLUENCE_OUT}" \
    --out_name "ground_truth" \
    --end_index "${END_INDEX}" \
    --num_anchors "${NUM_ANCHORS}" \
    --proj_dim "${GT_PROJ_DIM}" \
    --dev_dataset_name "${BENCHMARK}" \
    --lora_target_modules "${LORA_TARGET_MODULES}" \
    --lora_rank "${LORA_RANK}" \
    --lora_alpha "${LORA_ALPHA}" \
    --lora_dropout "${LORA_DROPOUT}" \
    --lora_seed "${LORA_SEED}" \
    --project_interval "${PROJECT_INTERVAL}" \
    --eval_on_train 100 \
    --tulu_as_anchors   # DIAGNOSTIC: comment out to restore BBH anchors

