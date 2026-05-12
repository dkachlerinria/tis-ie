#!/bin/bash
# Tiny configuration for quick debugging/testing

# General Config
export BENCHMARK="bbh"
export TRAINING_MODEL="Qwen/Qwen3-4B"
export WARMUP_MODEL="${TRAINING_MODEL}"

# Slug for file paths
export MODEL_SLUG=$(echo "${TRAINING_MODEL}" | tr '[:upper:]' '[:lower:]' | sed 's|.*/||')

export NUM_SAMPLES=10
export END_INDEX=100
export SEED=0

# Selection Config
export ENCODER_MODEL="jhu-clsp/ettin-encoder-150m"
export TRAIN_DATASET="Harvard-DCML/tulu-v2-197K-processed"
export SELECTION_METHOD="round_robin"

# LESS Specific Config
export CKPT_DIR="$(pwd)/files/checkpoints/${MODEL_SLUG}_warmup"
export CKPT_STEPS="latest"
export PROJ_DIM=64 # Small projection for speed

# Paths
export INDEX_DIR="$(pwd)/files/index/${MODEL_SLUG}_debug_subset_${END_INDEX}"
export DATASET_ROOT="$(pwd)/files/datasets/${MODEL_SLUG}_debug"
export MODEL_ROOT="$(pwd)/files/models/${MODEL_SLUG}_debug"
export RESULTS_ROOT="$(pwd)/files/results/${MODEL_SLUG}_debug"

# Training Hyperparameters (Tiny for debugging)
export BATCH_SIZE=1
export GRAD_ACC=1
export EPOCHS=1
export LR=2e-5
export WARMUP_RATIO=0.03
export WEIGHT_DECAY=0.0
export LR_SCHEDULER="linear"
export BF16=True

# Warmup specific
export WARMUP_SAMPLES=100
export WARMUP_EPOCHS=1

# LoRA Config
export USE_LORA=True
export LORA_RANK=8
export LORA_ALPHA=16
export LORA_DROPOUT=0.1
export LORA_TARGET_MODULES="all-linear"
