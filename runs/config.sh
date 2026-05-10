#!/bin/bash
# Shared configuration for all selection methods

# General Config
export BENCHMARK="bbh"
export TRAINING_MODEL="Qwen/Qwen3-1.7B-Base"
export WARMUP_MODEL="${TRAINING_MODEL}"

# Slug for file paths
export MODEL_SLUG=$(echo "${TRAINING_MODEL}" | tr '[:upper:]' '[:lower:]' | sed 's|.*/||')

export NUM_SAMPLES=1000
export END_INDEX=20000
export SEED=42

# Selection Config
export ENCODER_MODEL="jhu-clsp/ettin-encoder-150m"
export TRAIN_DATASET="Harvard-DCML/tulu-v2-197K-processed"
export SELECTION_METHOD="doubly_greedy" # Default for embedding/less

# LESS Specific Config
export CKPT_DIR="$(pwd)/files/checkpoints/${MODEL_SLUG}_warmup"
export CKPT_STEPS="latest" # Use "latest" to auto-detect, or space-separated steps (e.g. "50 100")
export PROJ_DIM=512

# Paths
export INDEX_DIR="$(pwd)/files/index/${MODEL_SLUG}_ettin_subset_${END_INDEX}"
export DATASET_ROOT="$(pwd)/files/datasets/${MODEL_SLUG}"
export MODEL_ROOT="$(pwd)/files/models/${MODEL_SLUG}"
export RESULTS_ROOT="$(pwd)/files/results/${MODEL_SLUG}"

# Training Hyperparameters
export BATCH_SIZE=1
export GRAD_ACC=128
export EPOCHS=2
export LR=2e-5
export WARMUP_RATIO=0.03
export WEIGHT_DECAY=0.0
export LR_SCHEDULER="linear"
export BF16=True

# Warmup specific (only samples and epochs should differ)
export WARMUP_SAMPLES=500
export WARMUP_EPOCHS=1

# LoRA Config
export USE_LORA=True
export LORA_RANK=16
export LORA_ALPHA=32
export LORA_DROPOUT=0.05
export LORA_TARGET_MODULES="all-linear"
