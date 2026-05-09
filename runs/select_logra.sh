#!/bin/bash
set -e

# Prerequisite: runs/train_warmup.sh must have been run first (produces CKPT_DIR).
source runs/config.sh

METHOD="logra"
DATASET_DIR="${DATASET_ROOT}/${METHOD}_${BENCHMARK}_subset_${END_INDEX}"
MODEL_DIR="${MODEL_ROOT}/trained_model_${METHOD}_${BENCHMARK}_top${NUM_SAMPLES}_subset_${END_INDEX}"
RESULTS_DIR="${RESULTS_ROOT}/trained_model_${METHOD}_${BENCHMARK}_top${NUM_SAMPLES}_subset_${END_INDEX}"

# LoGra-specific config
LOGRA_DIR="$(pwd)/files/logra/${MODEL_SLUG}"
LOGRA_RANK=8

echo "Starting LoGra Selection Pipeline..."
mkdir -p "${LOGRA_DIR}"

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

# Step 1: Compute LoGra similarity matrix
echo "Step 1: Computing LoGra similarity matrix..."
python3 logra/run_logra.py \
    --ckpt_path "${CKPT_DIR}/checkpoint-${CKPT_STEPS}" \
    --train_dataset_name "${TRAIN_DATASET}" \
    --dev_dataset_name "${BENCHMARK}" \
    --output_dir "${LOGRA_DIR}" \
    --rank "${LOGRA_RANK}" \
    --end_index "${END_INDEX}"

# Step 2: Perform Data Selection
# LoGra saves to {output_dir}/{dev_dataset_name}_cossim.npy
LOGRA_MATRIX="${LOGRA_DIR}/${BENCHMARK}_cossim.npy"

echo "Step 2: Performing data selection using ${SELECTION_METHOD}..."
python3 -m selection.sim_subset \
    --selection_method "${SELECTION_METHOD}" \
    --subset_dataset_dir "${DATASET_DIR}" \
    --similarity_matrix_path "${LOGRA_MATRIX}" \
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
