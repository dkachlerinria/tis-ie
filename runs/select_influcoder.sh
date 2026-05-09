#!/bin/bash
set -e

# Prerequisite: runs/train_warmup.sh must have been run first (produces CKPT_DIR).
source runs/config.sh

METHOD="influcoder"
DATASET_DIR="${DATASET_ROOT}/${METHOD}_${BENCHMARK}_subset_${END_INDEX}"
MODEL_DIR="${MODEL_ROOT}/trained_model_${METHOD}_${BENCHMARK}_top${NUM_SAMPLES}_subset_${END_INDEX}"
RESULTS_DIR="${RESULTS_ROOT}/trained_model_${METHOD}_${BENCHMARK}_top${NUM_SAMPLES}_subset_${END_INDEX}"

# --- Influcoder-specific config ---
INFLUCODER_RUN_MODE="small"
INFLUCODER_PROJ_DIM=131072

# Data Partitioning
N_TRAIN_ANCHORS=200      # From BENCHMARK dataset
N_EVAL_ANCHORS=100       # From BENCHMARK dataset
N_TRAIN_POOL=200        # From TRAIN_DATASET (tulu)
N_EVAL_POOL=100          # From TRAIN_DATASET (tulu)

INFLUCODER_DB_DIR="$(pwd)/files/index/influcoder_gradients"
TRAINED_ENCODER_DIR="$(pwd)/files/models/influence_encoder"
INFLUCODER_EMBEDS_DIR="$(pwd)/files/index/influcoder_embeds_${END_INDEX}"

# --- Execution Toggles ---
FORCE_RECOMPUTE=true
RECOMPUTE_FLAG=""
if [ "$FORCE_RECOMPUTE" = true ]; then
    RECOMPUTE_FLAG="--force_recompute"
fi

echo "Starting Influcoder Selection Pipeline..."

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
mkdir -p "${INFLUCODER_DB_DIR}" "${INFLUCODER_EMBEDS_DIR}"

# Step 1a: Stock train_anchors (from BENCHMARK dataset, e.g., BBH)
echo "Step 1a: Stocking train_anchors from ${BENCHMARK}..."
python influcoder/gradient_stocking.py \
    --dataset tulu \
    --train_dataset_name "${BENCHMARK}" \
    --split train_anchors \
    --model_name "${TRAINING_MODEL}" \
    --proj_dim "${INFLUCODER_PROJ_DIM}" \
    --n_samples "${N_TRAIN_ANCHORS}" \
    --start_index 0 \
    --load_warmup_path "${LESS_WARMUP_CKPT}" \
    ${RECOMPUTE_FLAG} \
    --target_modules ${LORA_TARGET_MODULES} \
    --output_name "${INFLUCODER_DB_DIR}/train_anchors"

# Step 1b: Stock eval_anchors (from BENCHMARK dataset)
START_EVAL_A=${N_TRAIN_ANCHORS}
echo "Step 1b: Stocking eval_anchors from ${BENCHMARK}..."
python influcoder/gradient_stocking.py \
    --dataset tulu \
    --train_dataset_name "${BENCHMARK}" \
    --split eval_anchors \
    --model_name "${TRAINING_MODEL}" \
    --proj_dim "${INFLUCODER_PROJ_DIM}" \
    --n_samples "${N_EVAL_ANCHORS}" \
    --start_index "${START_EVAL_A}" \
    --load_warmup_path "${LESS_WARMUP_CKPT}" \
    ${RECOMPUTE_FLAG} \
    --target_modules ${LORA_TARGET_MODULES} \
    --output_name "${INFLUCODER_DB_DIR}/eval_anchors"

# Step 1c: Stock pool (from TRAIN_DATASET, e.g., tulu-v2-197K)
echo "Step 1c: Stocking pool from ${TRAIN_DATASET}..."
python influcoder/gradient_stocking.py \
    --dataset tulu \
    --train_dataset_name "${TRAIN_DATASET}" \
    --split pool \
    --model_name "${TRAINING_MODEL}" \
    --proj_dim "${INFLUCODER_PROJ_DIM}" \
    --n_samples "${N_TRAIN_POOL}" \
    --start_index 0 \
    --load_warmup_path "${LESS_WARMUP_CKPT}" \
    ${RECOMPUTE_FLAG} \
    --target_modules ${LORA_TARGET_MODULES} \
    --output_name "${INFLUCODER_DB_DIR}/pool"

# Step 1d: Stock eval_pool (from TRAIN_DATASET)
START_EVAL_P=${N_TRAIN_POOL}
echo "Step 1d: Stocking eval_pool from ${TRAIN_DATASET}..."
python influcoder/gradient_stocking.py \
    --dataset tulu \
    --train_dataset_name "${TRAIN_DATASET}" \
    --split eval_pool \
    --model_name "${TRAINING_MODEL}" \
    --proj_dim "${INFLUCODER_PROJ_DIM}" \
    --n_samples "${N_EVAL_POOL}" \
    --start_index "${START_EVAL_P}" \
    --load_warmup_path "${LESS_WARMUP_CKPT}" \
    ${RECOMPUTE_FLAG} \
    --target_modules ${LORA_TARGET_MODULES} \
    --output_name "${INFLUCODER_DB_DIR}/eval_pool"

# Step 2: Train the influence encoder
echo "Step 2: Training influence encoder (mode: ${INFLUCODER_RUN_MODE})..."
python influcoder/train_influence_encoder.py \
    --anchor_train_db "${INFLUCODER_DB_DIR}/train_anchors.sqlite" \
    --anchor_eval_db  "${INFLUCODER_DB_DIR}/eval_anchors.sqlite" \
    --pool_train_db   "${INFLUCODER_DB_DIR}/pool.sqlite" \
    --pool_eval_db    "${INFLUCODER_DB_DIR}/eval_pool.sqlite" \
    --encoder_model   "${ENCODER_MODEL}" \
    --gradient_model  "${TRAINING_MODEL}" \
    --run_mode        "${INFLUCODER_RUN_MODE}" \
    --output_dir      "${TRAINED_ENCODER_DIR}"

# Step 3: Compute embeddings using the trained influence encoder
echo "Step 3: Computing embeddings with trained influence encoder..."
python3 -m representation.embed.compute_sentence_embeds \
    --model_name "${TRAINED_ENCODER_DIR}/model" \
    --train_dataset_name "${TRAIN_DATASET}" \
    --train_index_path "${INFLUCODER_EMBEDS_DIR}/train_embeds.pt" \
    --dev_dataset_name "${BENCHMARK}" \
    --dev_index_path "${INFLUCODER_EMBEDS_DIR}/${BENCHMARK}_dev_embeds.pt" \
    --save_dir "${INFLUCODER_EMBEDS_DIR}" \
    --batch_size 32 \
    --end_index "${END_INDEX}"

# Step 4: Perform data selection
echo "Step 4: Performing data selection using ${SELECTION_METHOD}..."
python3 -m selection.sim_subset \
    --selection_method "${SELECTION_METHOD}" \
    --subset_dataset_dir "${DATASET_DIR}" \
    --similarity_matrix_path "${INFLUCODER_EMBEDS_DIR}/${BENCHMARK}_cossim_0_${END_INDEX}.npy" \
    --train_dataset_name "${TRAIN_DATASET}" \
    --dev_dataset_name "${BENCHMARK}" \
    --sizes ${NUM_SAMPLES}

SELECTED_DATA="${DATASET_DIR}/${BENCHMARK}_subset_top${NUM_SAMPLES}.jsonl"

# Step 5: Train the model
echo "Step 5: Training ${TRAINING_MODEL} on ${SELECTED_DATA}..."
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

# Step 6: Evaluate
echo "Step 6: Evaluating on ${BENCHMARK}..."
python3 -m evaluation.run_eval \
    --model_name_or_path "${MODEL_DIR}" \
    --eval_dataset "${BENCHMARK}" \
    --save_dir "${RESULTS_DIR}" \
    --use_vllm
