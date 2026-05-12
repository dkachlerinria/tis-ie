#!/bin/bash
set -e

# Prerequisite: runs/train_warmup.sh must have been run first (produces CKPT_DIR).
source runs/config.sh

METHOD="iprox"
DATASET_DIR="${DATASET_ROOT}/${METHOD}_${BENCHMARK}_subset_${END_INDEX}"
MODEL_DIR="${MODEL_ROOT}/trained_model_${METHOD}_${BENCHMARK}_top${NUM_SAMPLES}_subset_${END_INDEX}"
RESULTS_DIR="${RESULTS_ROOT}/trained_model_${METHOD}_${BENCHMARK}_top${NUM_SAMPLES}_subset_${END_INDEX}"

# --- IProX-specific config ---
IPROX_RUN_MODE="small"
IPROX_SPARSITY=0.5
IPROX_LAMBDA=0.5

# Data Partitioning (Standardized with Influcoder)
N_TRAIN_ANCHORS=1000      # From BENCHMARK dataset
N_EVAL_ANCHORS=100        # From BENCHMARK dataset
N_TRAIN_POOL=4000         # From TRAIN_DATASET (tulu)
N_EVAL_POOL=200           # From TRAIN_DATASET (tulu)

IPROX_DB_DIR="$(pwd)/files/index/iprox_gradients"
TRAINED_PROXY_DIR="$(pwd)/files/models/iprox_proxy"
IPROX_SCORES_DIR="$(pwd)/files/index/iprox_scores_${END_INDEX}"

# --- Execution Toggles ---
FORCE_RECOMPUTE=true
RECOMPUTE_FLAG=""
if [ "$FORCE_RECOMPUTE" = true ]; then
    RECOMPUTE_FLAG="--force_recompute"
fi

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

LESS_WARMUP_CKPT="${CKPT_DIR}/checkpoint-${CKPT_STEPS}"
echo "  Warmup checkpoint: ${LESS_WARMUP_CKPT}"

## Step 1: Train the IProX Proxy
# Note: IProX trains a small proxy model to mimic the gradients of the target model.
echo "Step 1: Training IProX proxy..."
python iprox/train_iprox.py \
    --target_model "${LESS_WARMUP_CKPT}" \
    --benchmark "${BENCHMARK}" \
    --train_dataset "${TRAIN_DATASET}" \
    --n_train_a "${N_TRAIN_ANCHORS}" \
    --n_train_p "${N_TRAIN_POOL}" \
    --sparsity "${IPROX_SPARSITY}" \
    --lambda_anchor "${IPROX_LAMBDA}" \
    --output_dir "${TRAINED_PROXY_DIR}" \
    --epochs "${EPOCHS}" \
    --lr "${LR}" \
    --gradient_accumulation_steps "${GRAD_ACC}"

# Step 2: Compute Similarity Matrix
# Note: This uses the trained proxy to compute a (dev x train) similarity matrix.
echo "Step 2: Computing IProX similarity matrix..."
python iprox/compute_iprox_scores.py \
    --proxy_path "${TRAINED_PROXY_DIR}/model" \
    --benchmark "${BENCHMARK}" \
    --train_dataset_name "${TRAIN_DATASET}" \
    --output_dir "${IPROX_SCORES_DIR}" \
    --end_index "${END_INDEX}"

# Step 3: Perform Selection
IPROX_MATRIX="${IPROX_SCORES_DIR}/${BENCHMARK}_cossim.npy"

echo "Step 3: Performing data selection with sim_subset..."
python3 -m selection.sim_subset \
    --selection_method "${SELECTION_METHOD}" \
    --subset_dataset_dir "${DATASET_DIR}" \
    --similarity_matrix_path "${IPROX_MATRIX}" \
    --train_dataset_name "${TRAIN_DATASET}" \
    --dev_dataset_name "${BENCHMARK}" \
    --sizes "${NUM_SAMPLES}" \
    --end_index "${END_INDEX}"

SELECTED_DATA="${DATASET_DIR}/${BENCHMARK}_subset_top${NUM_SAMPLES}.jsonl"

# Step 4: Fine-tune
echo "Step 4: Fine-tuning ${TRAINING_MODEL} on ${SELECTED_DATA}..."
python3 -m training.train_sft \
    --model_name_or_path "${TRAINING_MODEL}" \
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

# Step 5: Evaluate
echo "Step 5: Evaluating on ${BENCHMARK}..."
python3 -m evaluation.run_eval \
    --model_name_or_path "${MODEL_DIR}" \
    --eval_dataset "${BENCHMARK}" \
    --save_dir "${RESULTS_DIR}" \
    --use_vllm
