#!/bin/bash
set -e

# Configuration
BENCHMARK="bbh"
ENCODER_MODEL="jhu-clsp/ettin-encoder-150m"
TRAINING_MODEL="Qwen/Qwen2.5-0.5B"
NUM_SAMPLES=5000
END_INDEX=20000

# Paths
INDEX_DIR="$(pwd)/files/index/ettin_subset_${END_INDEX}"
DATASET_DIR="$(pwd)/files/datasets/ettin_rr_${BENCHMARK}_subset_${END_INDEX}"
MODEL_DIR="$(pwd)/files/models/qwen2.5-0.5b_ettin_${BENCHMARK}_top${NUM_SAMPLES}_subset_${END_INDEX}"
RESULTS_DIR="$(pwd)/files/results/ettin_rr_${BENCHMARK}_top${NUM_SAMPLES}_subset_${END_INDEX}"

# Step 0: Ensure data is downloaded
if [ ! -d "data/eval/${BENCHMARK}" ]; then
    echo "Downloading evaluation data..."
    sh download_eval.sh
fi

# Step 1: Compute Embeddings and Similarity Matrix
echo "Step 1: Computing embeddings with ${ENCODER_MODEL}..."
python3 -m representation.embed.compute_sentence_embeds \
    --model_name "${ENCODER_MODEL}" \
    --train_dataset_name "Harvard-DCML/tulu-v2-197K-processed" \
    --train_index_path "${INDEX_DIR}/train_embeds.pt" \
    --dev_dataset_name "${BENCHMARK}" \
    --dev_index_path "${INDEX_DIR}/${BENCHMARK}_dev_embeds.pt" \
    --save_dir "${INDEX_DIR}" \
    --batch_size 128 \
    --end_index "${END_INDEX}"

# Step 2: Perform Data Selection (Round Robin)
echo "Step 2: Performing data selection..."
python3 -m selection.sim_subset \
    --selection_method "round_robin" \
    --subset_dataset_dir "${DATASET_DIR}" \
    --similarity_matrix_path "${INDEX_DIR}/${BENCHMARK}_cossim.npy" \
    --train_dataset_name "Harvard-DCML/tulu-v2-197K-processed" \
    --dev_dataset_name "${BENCHMARK}" \
    --sizes ${NUM_SAMPLES}

# Step 3: Train the model
SELECTED_DATA="${DATASET_DIR}/${BENCHMARK}_subset_top${NUM_SAMPLES}.jsonl"
echo "Step 3: Training ${TRAINING_MODEL} on ${SELECTED_DATA}..."
python3 -m training.train_sft \
    --model_name "${TRAINING_MODEL}" \
    --train_dataset_path "${SELECTED_DATA}" \
    --output_dir "${MODEL_DIR}" \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 128 \
    --num_train_epochs 2 \
    --learning_rate 2e-5 \
    --seed 0 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type linear \
    --weight_decay 0.0 \
    --save_strategy no \
    --logging_steps 1 \
    --report_to "none"

# Step 4: Evaluate
echo "Step 4: Evaluating on ${BENCHMARK}..."
python3 -m evaluation.run_eval \
    --model_name_or_path "${MODEL_DIR}" \
    --eval_dataset "${BENCHMARK}" \
    --save_dir "${RESULTS_DIR}" \
    --use_vllm
