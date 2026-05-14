#!/bin/bash
set -euo pipefail
CFG="${1:-runs/influence_spearman/config_influence.sh}"
source "$CFG"

mkdir -p "${INFLUCODER_ENCODER_DIR}"

python influcoder/train_influence_encoder.py \
    --anchor_train_db    "${INFLUCODER_DB_DIR}/train_anchors.sqlite" \
    --anchor_eval_db     "${INFLUCODER_DB_DIR}/eval_anchors.sqlite" \
    --pool_train_db      "${INFLUCODER_DB_DIR}/pool.sqlite" \
    --pool_eval_db       "${INFLUCODER_DB_DIR}/eval_pool.sqlite" \
    --encoder_model      "${ENCODER_MODEL}" \
    --gradient_model     "${INFLUENCE_MODEL}" \
    --run_mode           "${INFLUCODER_RUN_MODE}" \
    --output_dir         "${INFLUCODER_ENCODER_DIR}" \
    --lora_rank          "${LORA_RANK}" \
    --lora_alpha         "${LORA_ALPHA}" \
    --lora_dropout       "${LORA_DROPOUT}" \
    --lora_seed          "${LORA_SEED}" \
    --lora_target_modules ${LORA_TARGET_MODULES} \
    --gt_proj_dim        "${GT_PROJ_DIM}" \
    --project_interval   "${PROJECT_INTERVAL}"
