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

# Step 1: Stock Gradients (Reuse the strict 4-way partitioning from Influcoder)
# Note: IProX trains on these gradients to align its proxy model.
mkdir -p "${IPROX_DB_DIR}"

# Step 1a: Stock train_anchors
echo "Step 1a: Stocking train_anchors gradients (using ${BENCHMARK})..."
python influcoder/gradient_stocking.py \
    --dataset "${BENCHMARK}" \
    --train_dataset_name "${TRAIN_DATASET}" \
    --split train_anchors \
    --model_name "${TRAINING_MODEL}" \
    --proj_dim "${PROJ_DIM}" \
    --n_samples "${N_TRAIN_ANCHORS}" \
    --start_index 0 \
    --load_warmup_path "${LESS_WARMUP_CKPT}" \
    ${RECOMPUTE_FLAG} \
    --target_modules ${LORA_TARGET_MODULES} \
    --output_name "${IPROX_DB_DIR}/train_anchors"

# Step 1b: Stock eval_anchors
echo "Step 1b: Stocking eval_anchors gradients (using ${BENCHMARK})..."
python influcoder/gradient_stocking.py \
    --dataset "${BENCHMARK}" \
    --train_dataset_name "${TRAIN_DATASET}" \
    --split eval_anchors \
    --model_name "${TRAINING_MODEL}" \
    --proj_dim "${PROJ_DIM}" \
    --n_samples "${N_EVAL_ANCHORS}" \
    --start_index "${N_TRAIN_ANCHORS}" \
    --load_warmup_path "${LESS_WARMUP_CKPT}" \
    ${RECOMPUTE_FLAG} \
    --target_modules ${LORA_TARGET_MODULES} \
    --output_name "${IPROX_DB_DIR}/eval_anchors"

# Step 1c: Stock pool
echo "Step 1c: Stocking pool gradients (using tulu)..."
python influcoder/gradient_stocking.py \
    --dataset tulu \
    --train_dataset_name "${TRAIN_DATASET}" \
    --split pool \
    --model_name "${TRAINING_MODEL}" \
    --proj_dim "${PROJ_DIM}" \
    --n_samples "${N_TRAIN_POOL}" \
    --start_index 0 \
    --load_warmup_path "${LESS_WARMUP_CKPT}" \
    ${RECOMPUTE_FLAG} \
    --target_modules ${LORA_TARGET_MODULES} \
    --output_name "${IPROX_DB_DIR}/pool"

# Step 1d: Stock eval_pool
echo "Step 1d: Stocking eval_pool gradients (using tulu)..."
python influcoder/gradient_stocking.py \
    --dataset tulu \
    --train_dataset_name "${TRAIN_DATASET}" \
    --split eval_pool \
    --model_name "${TRAINING_MODEL}" \
    --proj_dim "${PROJ_DIM}" \
    --n_samples "${N_EVAL_POOL}" \
    --start_index "${N_TRAIN_POOL}" \
    --load_warmup_path "${LESS_WARMUP_CKPT}" \
    ${RECOMPUTE_FLAG} \
    --target_modules ${LORA_TARGET_MODULES} \
    --output_name "${IPROX_DB_DIR}/eval_pool"

# Step 2: Train the IProX Proxy
echo "Step 2: Training IProX proxy..."
python iprox/train_iprox_gradients.py \
    --run_mode "${IPROX_RUN_MODE}" \
    --anchor_train_db "${IPROX_DB_DIR}/train_anchors.sqlite" \
    --anchor_eval_db  "${IPROX_DB_DIR}/eval_anchors.sqlite" \
    --pool_train_db   "${IPROX_DB_DIR}/pool.sqlite" \
    --pool_eval_db    "${IPROX_DB_DIR}/eval_pool.sqlite" \
    --target_model "${TRAINING_MODEL}" \
    --sparsity "${IPROX_SPARSITY}" \
    --lambda_anchor "${IPROX_LAMBDA}" \
    --output_dir "${TRAINED_PROXY_DIR}"

# Step 3: Compute Influence Scores for Data Selection
# Note: This uses the trained proxy to score the FULL training dataset.
echo "Step 3: Computing influence scores with IProX proxy..."
python iprox/compute_iprox_scores.py \
    --proxy_path "${TRAINED_PROXY_DIR}/model" \
    --benchmark "${BENCHMARK}" \
    --train_dataset_name "${TRAIN_DATASET}" \
    --output_dir "${IPROX_SCORES_DIR}"

# Step 4: Perform Selection
echo "Step 4: Selecting top ${NUM_SAMPLES} samples..."
python selection/select_data.py \
    --score_path "${IPROX_SCORES_DIR}/scores.pt" \
    --num_samples "${NUM_SAMPLES}" \
    --output_dir "${DATASET_DIR}"

# Step 5: Fine-tune
echo "Step 5: Fine-tuning on selected data..."
python3 -m training.train_sft \
    --model_name_or_path "${TRAINING_MODEL}" \
    --dataset_path "${DATASET_DIR}" \
    --output_dir "${MODEL_DIR}" \
    --bf16 ${BF16} \
    --use_lora ${USE_LORA} \
    --lora_rank ${LORA_RANK} \
    --lora_alpha ${LORA_ALPHA} \
    --lora_dropout ${LORA_DROPOUT} \
    --lora_target_modules ${LORA_TARGET_MODULES} \
    --save_strategy no \
    --report_to "none"

# Step 6: Evaluate
echo "Step 6: Evaluating on ${BENCHMARK}..."
python3 -m evaluation.run_eval \
    --model_name_or_path "${MODEL_DIR}" \
    --eval_dataset "${BENCHMARK}" \
    --save_dir "${RESULTS_DIR}" \
    --use_vllm
