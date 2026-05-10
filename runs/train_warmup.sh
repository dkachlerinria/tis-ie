#!/bin/bash
set -e

# Load configuration
source runs/config.sh

# Warmup specific config
WARMUP_DATASET="Harvard-DCML/tulu-v2-10K-warmup-processed"

echo "Starting LESS Warmup Training..."
echo "Model: ${WARMUP_MODEL}"
echo "Dataset: ${WARMUP_DATASET}"
echo "Output Directory: ${CKPT_DIR}"

python3 -m training.train_sft \
    --model_name "${WARMUP_MODEL}" \
    --train_dataset_name "${WARMUP_DATASET}" \
    --num_samples "${WARMUP_SAMPLES}" \
    --output_dir "${CKPT_DIR}" \
    --per_device_train_batch_size "${BATCH_SIZE}" \
    --gradient_accumulation_steps "${GRAD_ACC}" \
    --num_train_epochs "${WARMUP_EPOCHS}" \
    --learning_rate "${LR}" \
    --seed "${SEED}" \
    --warmup_ratio "${WARMUP_RATIO}" \
    --lr_scheduler_type "${LR_SCHEDULER}" \
    --weight_decay "${WEIGHT_DECAY}" \
    --bf16 "${BF16}" \
    --save_strategy epoch \
    --logging_steps 1 \
    --use_lora "${USE_LORA}" \
    --lora_rank "${LORA_RANK}" \
    --lora_alpha "${LORA_ALPHA}" \
    --lora_dropout "${LORA_DROPOUT}" \
    --lora_target_modules "${LORA_TARGET_MODULES}" \
    --report_to "none" \
    --overwrite_output_dir True
