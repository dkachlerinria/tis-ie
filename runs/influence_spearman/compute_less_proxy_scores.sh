#!/bin/bash
set -euo pipefail
CFG="${1:-runs/influence_spearman/config_influence.sh}"
source "$CFG"

mkdir -p "$INFLUENCE_OUT"

python3 -m influence_eval.compute_gradient_scores \
    --model_name          "${PROXY_MODEL}" \
    --save_dir            "${INFLUENCE_OUT}" \
    --out_name            "less_proxy" \
    --end_index           "${END_INDEX}" \
    --num_anchors         "${NUM_ANCHORS}" \
    --proj_dim            "${LESS_PROJ_DIM}" \
    --dev_dataset_name    "${BENCHMARK}" \
    --lora_target_modules "${LORA_TARGET_MODULES}" \
    --lora_rank           "${LORA_RANK}" \
    --lora_alpha          "${LORA_ALPHA}" \
    --lora_dropout        "${LORA_DROPOUT}" \
    --lora_seed           "${LORA_SEED}" \
    --project_interval    "${PROJECT_INTERVAL}" \
    --local_train_dataset   "dolly/dolly_data.jsonl" \
    --tokenized_train_path  "${INFLUENCE_OUT}/tokenized_train_ds" \
    --tokenized_anchor_path "${INFLUENCE_OUT}/tokenized_anchor_ds"
