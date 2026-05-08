#!/bin/bash
set -e

# Load configuration
source runs/config.sh

METHOD="random"
DATASET_DIR="${DATASET_ROOT}/${METHOD}_${BENCHMARK}_subset_${END_INDEX}"
MODEL_DIR="${MODEL_ROOT}/trained_model_${METHOD}_${BENCHMARK}_top${NUM_SAMPLES}_subset_${END_INDEX}"
RESULTS_DIR="${RESULTS_ROOT}/trained_model_${METHOD}_${BENCHMARK}_top${NUM_SAMPLES}_subset_${END_INDEX}"

echo "Starting Random Selection Pipeline..."

# Step 1: Perform Random Data Selection
echo "Step 1: Performing random data selection..."
python3 -m selection.random \
    --train_dataset "${TRAIN_DATASET}" \
    --subset_dataset_dir "${DATASET_DIR}" \
    --sizes ${NUM_SAMPLES} \
    --seed ${SEED}

SELECTED_DATA="${DATASET_DIR}/subset_size_${NUM_SAMPLES}_seed_${SEED}.jsonl"

# Step 2: Train the model
echo "Step 2: Training ${TRAINING_MODEL} on ${SELECTED_DATA}..."
python3 -m training.train_sft \
    --model_name "${TRAINING_MODEL}" \
    --train_dataset_path "${SELECTED_DATA}" \
    --output_dir "${MODEL_DIR}" \
    --per_device_train_batch_size ${BATCH_SIZE} \
    --gradient_accumulation_steps ${GRAD_ACC} \
    --num_train_epochs ${EPOCHS} \
    --learning_rate ${LR} \
    --seed ${SEED} \
    --use_lora ${USE_LORA} \
    --lora_rank ${LORA_RANK} \
    --lora_alpha ${LORA_ALPHA} \
    --lora_dropout ${LORA_DROPOUT} \
    --save_strategy no \
    --report_to "none"

# Step 3: Evaluate
echo "Step 3: Evaluating on ${BENCHMARK}..."
python3 -m evaluation.run_eval \
    --model_name_or_path "${MODEL_DIR}" \
    --eval_dataset "${BENCHMARK}" \
    --save_dir "${RESULTS_DIR}" \
    --use_vllm
