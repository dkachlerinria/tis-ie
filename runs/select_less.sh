#!/bin/bash
set -e

# Load configuration
source runs/config.sh

METHOD="less"
DATASET_DIR="${DATASET_ROOT}/${METHOD}_${BENCHMARK}_subset_${END_INDEX}"
MODEL_DIR="${MODEL_ROOT}/trained_model_${METHOD}_${BENCHMARK}_top${NUM_SAMPLES}_subset_${END_INDEX}"
RESULTS_DIR="${RESULTS_ROOT}/trained_model_${METHOD}_${BENCHMARK}_top${NUM_SAMPLES}_subset_${END_INDEX}"
LESS_DIR="$(pwd)/files/less/${MODEL_SLUG}"

# Performance Tuning (A30 optimized)
PROJ_DIM=8192
PROJECT_INTERVAL=16

echo "Starting LESS Selection Pipeline..."

# Resolve "latest" checkpoint if specified
if [ "${CKPT_STEPS}" = "latest" ]; then
    echo "Finding latest checkpoint in ${CKPT_DIR}..."
    LATEST_CKPT=$(ls -d ${CKPT_DIR}/checkpoint-* 2>/dev/null | sort -V | tail -n 1)
    if [ -z "${LATEST_CKPT}" ]; then
        echo "Error: No checkpoints found in ${CKPT_DIR}. Did you run the warmup?"
        exit 1
    fi
    CKPT_STEPS=$(basename ${LATEST_CKPT} | sed 's/checkpoint-//')
    echo "Using latest checkpoint: checkpoint-${CKPT_STEPS}"
fi

# Step 1: Extract Gradients for all checkpoints
# Train grads use adam (EMA-corrected with warmup optimizer state).
# Dev/query grads use sgd (raw gradients, no optimizer state).
# These must be separate calls so the files are named correctly for compute_less_similarity.
echo "Step 1: Extracting gradients for all checkpoints..."
for step in ${CKPT_STEPS}; do
    echo "Processing checkpoint-${step} — train gradients (adam)..."
    python3 -m representation.less.compute_less_embeds \
        --ckpt_path "${CKPT_DIR}/checkpoint-${step}" \
        --ckpt_step "${step}" \
        --save_dir "${LESS_DIR}" \
        --compute_train_grads \
        --gradient_type adam \
        --end_index "${END_INDEX}" \
        --proj_dim "${PROJ_DIM}" \
        --project_interval "${PROJECT_INTERVAL}"

    echo "Processing checkpoint-${step} — dev/query gradients (sgd)..."
    python3 -m representation.less.compute_less_embeds \
        --ckpt_path "${CKPT_DIR}/checkpoint-${step}" \
        --ckpt_step "${step}" \
        --dev_dataset_name "${BENCHMARK}" \
        --save_dir "${LESS_DIR}" \
        --compute_dev_grads \
        --gradient_type sgd \
        --proj_dim "${PROJ_DIM}" \
        --project_interval "${PROJECT_INTERVAL}"
done

# Step 2: Compute LESS Influence/Similarity Matrix
echo "Step 2: Computing LESS similarity matrix..."
python3 -m representation.less.compute_less_similarity \
    --train_dataset_name "${TRAIN_DATASET}" \
    --dev_dataset_name "${BENCHMARK}" \
    --output_dir "${LESS_DIR}" \
    --ckpt_dir "${CKPT_DIR}" \
    --checkpoint_steps ${CKPT_STEPS} \
    --proj_dim ${PROJ_DIM} \
    --num_epochs 1

# Step 2: Perform Data Selection
# LESS saves to {output_dir}/{dev_dataset_name}_cossim.npy
LESS_MATRIX="${LESS_DIR}/${BENCHMARK}_cossim.npy"

echo "Step 2: Performing data selection using ${SELECTION_METHOD}..."
python3 -m selection.sim_subset \
    --selection_method "${SELECTION_METHOD}" \
    --subset_dataset_dir "${DATASET_DIR}" \
    --similarity_matrix_path "${LESS_MATRIX}" \
    --train_dataset_name "${TRAIN_DATASET}" \
    --dev_dataset_name "${BENCHMARK}" \
    --sizes ${NUM_SAMPLES}

SELECTED_DATA="${DATASET_DIR}/${BENCHMARK}_subset_top${NUM_SAMPLES}.jsonl"

# Step 3: Train the model
echo "Step 3: Training ${TRAINING_MODEL} on ${SELECTED_DATA}..."
python3 -m training.train_sft \
    --model_name "${TRAINING_MODEL}" \
    --train_dataset_path "${SELECTED_DATA}" \
    --output_dir "${MODEL_DIR}" \
    --per_device_train_batch_size ${BATCH_SIZE} \
    --gradient_accumulation_steps ${GRAD_ACC} \
    --num_train_epochs ${EPOCHS} \
    --learning_rate ${LR} \
    --seed ${SEED} \
    --warmup_ratio ${WARMUP_RATIO} \
    --lr_scheduler_type ${LR_SCHEDULER} \
    --weight_decay ${WEIGHT_DECAY} \
    --bf16 ${BF16} \
    --use_lora ${USE_LORA} \
    --lora_rank ${LORA_RANK} \
    --lora_alpha ${LORA_ALPHA} \
    --lora_dropout ${LORA_DROPOUT} \
    --lora_target_modules ${LORA_TARGET_MODULES} \
    --save_strategy no \
    --report_to "none"

# Step 4: Evaluate
echo "Step 4: Evaluating on ${BENCHMARK}..."
python3 -m evaluation.run_eval \
    --model_name_or_path "${MODEL_DIR}" \
    --eval_dataset "${BENCHMARK}" \
    --save_dir "${RESULTS_DIR}" \
    --use_vllm
