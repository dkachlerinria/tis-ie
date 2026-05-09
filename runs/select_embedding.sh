#!/bin/bash
set -e

# Load configuration
source runs/config.sh

METHOD="embedding"
DATASET_DIR="${DATASET_ROOT}/${METHOD}_${BENCHMARK}_subset_${END_INDEX}"
MODEL_DIR="${MODEL_ROOT}/trained_model_${METHOD}_${BENCHMARK}_top${NUM_SAMPLES}_subset_${END_INDEX}"
RESULTS_DIR="${RESULTS_ROOT}/trained_model_${METHOD}_${BENCHMARK}_top${NUM_SAMPLES}_subset_${END_INDEX}"

echo "Starting Embedding Selection Pipeline..."

# Step 1: Compute Embeddings and Similarity Matrix
echo "Step 1: Computing embeddings with ${ENCODER_MODEL}..."
python3 -m representation.embed.compute_sentence_embeds \
    --model_name "${ENCODER_MODEL}" \
    --train_dataset_name "${TRAIN_DATASET}" \
    --train_index_path "${INDEX_DIR}/train_embeds.pt" \
    --dev_dataset_name "${BENCHMARK}" \
    --dev_index_path "${INDEX_DIR}/${BENCHMARK}_dev_embeds.pt" \
    --save_dir "${INDEX_DIR}" \
    --batch_size 32 \
    --end_index "${END_INDEX}"

# Step 2: Perform Data Selection
echo "Step 2: Performing data selection using ${SELECTION_METHOD}..."
python3 -m selection.sim_subset \
    --selection_method "${SELECTION_METHOD}" \
    --subset_dataset_dir "${DATASET_DIR}" \
    --similarity_matrix_path "${INDEX_DIR}/${BENCHMARK}_cossim_0_${END_INDEX}.npy" \
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
