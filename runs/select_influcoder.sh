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
N_ANCHOR_SAMPLES=50
N_POOL_SAMPLES=100
INFLUCODER_DB_DIR="$(pwd)/files/index/${MODEL_SLUG}_influcoder_gradients"
TRAINED_ENCODER_DIR="$(pwd)/files/models/${MODEL_SLUG}_influence_encoder"
INFLUCODER_EMBEDS_DIR="$(pwd)/files/index/${MODEL_SLUG}_influcoder_embeds_${END_INDEX}"

# --- Execution Toggles ---
FORCE_RECOMPUTE=true  # Set to true to delete existing DBs and re-extract gradients
RECOMPUTE_FLAG=""
if [ "$FORCE_RECOMPUTE" = true ]; then
    RECOMPUTE_FLAG="--force_recompute"
fi

# Shared warmup args for gradient_stocking.py to ensure consistency with SFT
INFLUCODER_WARMUP_ARGS="--lr ${LR} --batch_size ${BATCH_SIZE} --grad_acc ${GRAD_ACC} --epochs ${WARMUP_EPOCHS} --lora_rank ${LORA_RANK} --lora_alpha ${LORA_ALPHA} --lora_dropout ${LORA_DROPOUT} --target_modules ${LORA_TARGET_MODULES}"

# Use the same warmup checkpoint as select_less
LESS_WARMUP_CKPT="${CKPT_DIR}/checkpoint-${CKPT_STEPS}"

export PYTHONPATH="$(pwd)/influcoder:${PYTHONPATH}"

echo "Starting Influcoder Selection Pipeline..."
echo "  Warmup checkpoint: ${LESS_WARMUP_CKPT}"
mkdir -p "${INFLUCODER_DB_DIR}" "${INFLUCODER_EMBEDS_DIR}"

# Step 1a: Stock train_anchors
echo "Step 1a: Stocking train_anchors gradients (using ${BENCHMARK})..."
python influcoder/gradient_stocking.py \
    --dataset "${BENCHMARK}" \
    --train_dataset_name "${TRAIN_DATASET}" \
    --split train_anchors \
    --model_name "${TRAINING_MODEL}" \
    --proj_dim "${INFLUCODER_PROJ_DIM}" \
    --n_samples "${N_ANCHOR_SAMPLES}" \
    --anchor_size "${N_ANCHOR_SAMPLES}" \
    --pool_size "${N_POOL_SAMPLES}" \
    --load_warmup_path "${LESS_WARMUP_CKPT}" \
    ${RECOMPUTE_FLAG} \
    ${INFLUCODER_WARMUP_ARGS} \
    --output_name "${INFLUCODER_DB_DIR}/train_anchors"

# Step 1b: Stock eval_anchors
echo "Step 1b: Stocking eval_anchors gradients (using ${BENCHMARK})..."
python influcoder/gradient_stocking.py \
    --dataset "${BENCHMARK}" \
    --train_dataset_name "${TRAIN_DATASET}" \
    --split eval_anchors \
    --model_name "${TRAINING_MODEL}" \
    --proj_dim "${INFLUCODER_PROJ_DIM}" \
    --n_samples "${N_ANCHOR_SAMPLES}" \
    --anchor_size "${N_ANCHOR_SAMPLES}" \
    --pool_size "${N_POOL_SAMPLES}" \
    --load_warmup_path "${LESS_WARMUP_CKPT}" \
    ${RECOMPUTE_FLAG} \
    ${INFLUCODER_WARMUP_ARGS} \
    --output_name "${INFLUCODER_DB_DIR}/eval_anchors"

# Step 1c: Stock pool
echo "Step 1c: Stocking pool gradients (using tulu)..."
python influcoder/gradient_stocking.py \
    --dataset tulu \
    --train_dataset_name "${TRAIN_DATASET}" \
    --split pool \
    --model_name "${TRAINING_MODEL}" \
    --proj_dim "${INFLUCODER_PROJ_DIM}" \
    --n_samples "${N_POOL_SAMPLES}" \
    --anchor_size "${N_ANCHOR_SAMPLES}" \
    --pool_size "${N_POOL_SAMPLES}" \
    --load_warmup_path "${LESS_WARMUP_CKPT}" \
    ${RECOMPUTE_FLAG} \
    ${INFLUCODER_WARMUP_ARGS} \
    --output_name "${INFLUCODER_DB_DIR}/pool"

# Step 1d: Stock eval_pool
echo "Step 1d: Stocking eval_pool gradients (using tulu)..."
python influcoder/gradient_stocking.py \
    --dataset tulu \
    --train_dataset_name "${TRAIN_DATASET}" \
    --split eval_pool \
    --model_name "${TRAINING_MODEL}" \
    --proj_dim "${INFLUCODER_PROJ_DIM}" \
    --n_samples "${N_POOL_SAMPLES}" \
    --anchor_size "${N_ANCHOR_SAMPLES}" \
    --pool_size "${N_POOL_SAMPLES}" \
    --load_warmup_path "${LESS_WARMUP_CKPT}" \
    ${RECOMPUTE_FLAG} \
    ${INFLUCODER_WARMUP_ARGS} \
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
