#!/bin/bash
set -e

source runs/config.sh

METHOD="influcoder"
DATASET_DIR="${DATASET_ROOT}/${METHOD}_${BENCHMARK}_subset_${END_INDEX}"
MODEL_DIR="${MODEL_ROOT}/trained_model_${METHOD}_${BENCHMARK}_top${NUM_SAMPLES}_subset_${END_INDEX}"
RESULTS_DIR="${RESULTS_ROOT}/trained_model_${METHOD}_${BENCHMARK}_top${NUM_SAMPLES}_subset_${END_INDEX}"

# --- Influcoder-specific config ---
INFLUCODER_RUN_MODE="small"
INFLUCODER_PROJ_DIM=131072
N_ANCHOR_SAMPLES=5000
N_POOL_SAMPLES=10000
N_WARMUP_SAMPLES=5000
INFLUCODER_DB_DIR="$(pwd)/files/index/influcoder_gradients"
WARMUP_SAVE_DIR="$(pwd)/files/models/influcoder_warmup"
TRAINED_ENCODER_DIR="$(pwd)/files/models/influence_encoder"
INFLUCODER_EMBEDS_DIR="$(pwd)/files/index/influcoder_embeds_${END_INDEX}"

export PYTHONPATH="$(pwd)/influcoder:${PYTHONPATH}"

echo "Starting Influcoder Selection Pipeline..."
mkdir -p "${INFLUCODER_DB_DIR}" "${INFLUCODER_EMBEDS_DIR}"

# Step 1a: Stock train_anchors (runs internal warmup, saves merged model)
echo "Step 1a: Stocking train_anchors gradients (+ warmup)..."
python influcoder/gradient_stocking.py \
    --dataset tulu \
    --train_dataset_name "${TRAIN_DATASET}" \
    --split train_anchors \
    --model_name "${TRAINING_MODEL}" \
    --proj_dim "${INFLUCODER_PROJ_DIM}" \
    --n_samples "${N_ANCHOR_SAMPLES}" \
    --n_warmup "${N_WARMUP_SAMPLES}" \
    --anchor_size "${N_ANCHOR_SAMPLES}" \
    --pool_size "${N_POOL_SAMPLES}" \
    --save_warmup_path "${WARMUP_SAVE_DIR}" \
    --output_name "${INFLUCODER_DB_DIR}/train_anchors"

# Step 1b: Stock eval_anchors (reuses warmed model)
echo "Step 1b: Stocking eval_anchors gradients..."
python influcoder/gradient_stocking.py \
    --dataset tulu \
    --train_dataset_name "${TRAIN_DATASET}" \
    --split eval_anchors \
    --model_name "${TRAINING_MODEL}" \
    --proj_dim "${INFLUCODER_PROJ_DIM}" \
    --anchor_size "${N_ANCHOR_SAMPLES}" \
    --pool_size "${N_POOL_SAMPLES}" \
    --load_warmup_path "${WARMUP_SAVE_DIR}" \
    --output_name "${INFLUCODER_DB_DIR}/eval_anchors"

# Step 1c: Stock pool
echo "Step 1c: Stocking pool gradients..."
python influcoder/gradient_stocking.py \
    --dataset tulu \
    --train_dataset_name "${TRAIN_DATASET}" \
    --split pool \
    --model_name "${TRAINING_MODEL}" \
    --proj_dim "${INFLUCODER_PROJ_DIM}" \
    --n_samples "${N_POOL_SAMPLES}" \
    --anchor_size "${N_ANCHOR_SAMPLES}" \
    --pool_size "${N_POOL_SAMPLES}" \
    --load_warmup_path "${WARMUP_SAVE_DIR}" \
    --output_name "${INFLUCODER_DB_DIR}/pool"

# Step 1d: Stock eval_pool
echo "Step 1d: Stocking eval_pool gradients..."
python influcoder/gradient_stocking.py \
    --dataset tulu \
    --train_dataset_name "${TRAIN_DATASET}" \
    --split eval_pool \
    --model_name "${TRAINING_MODEL}" \
    --proj_dim "${INFLUCODER_PROJ_DIM}" \
    --anchor_size "${N_ANCHOR_SAMPLES}" \
    --pool_size "${N_POOL_SAMPLES}" \
    --load_warmup_path "${WARMUP_SAVE_DIR}" \
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
    --use_lora ${USE_LORA} \
    --lora_rank ${LORA_RANK} \
    --lora_alpha ${LORA_ALPHA} \
    --lora_dropout ${LORA_DROPOUT} \
    --save_strategy no \
    --report_to "none"

# Step 6: Evaluate
echo "Step 6: Evaluating on ${BENCHMARK}..."
python3 -m evaluation.run_eval \
    --model_name_or_path "${MODEL_DIR}" \
    --eval_dataset "${BENCHMARK}" \
    --save_dir "${RESULTS_DIR}" \
    --use_vllm
