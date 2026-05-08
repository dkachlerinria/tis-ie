#!/bin/bash
set -e

# Load configuration
source runs/config.sh

# Warmup specific config
WARMUP_DATASET="Harvard-DCML/tulu-v2-10K-warmup-processed"
WARMUP_MODEL="Qwen/Qwen2.5-0.5B" # Usually a smaller model is used for warmup

echo "Starting LESS Warmup Training..."
echo "Model: ${WARMUP_MODEL}"
echo "Dataset: ${WARMUP_DATASET}"
echo "Output Directory: ${CKPT_DIR}"

python3 -m training.train_sft \
    --model_name "${WARMUP_MODEL}" \
    --train_dataset_name "${WARMUP_DATASET}" \
    --output_dir "${CKPT_DIR}" \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 128 \
    --num_train_epochs 4 \
    --learning_rate 2e-5 \
    --seed ${SEED} \
    --warmup_ratio 0.03 \
    --lr_scheduler_type linear \
    --weight_decay 0.0 \
    --save_strategy epoch \
    --logging_steps 1 \
    --use_lora True \
    --lora_r 128 \
    --lora_alpha 512 \
    --lora_dropout 0.1 \
    --report_to "none" \
    --overwrite_output_dir True
