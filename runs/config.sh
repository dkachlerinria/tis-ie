#!/bin/bash
# Shared configuration for all selection methods

# General Config
export BENCHMARK="bbh"
export TRAINING_MODEL="Qwen/Qwen2.5-1.5B"
export NUM_SAMPLES=1000
export END_INDEX=20000
export SEED=42

# Selection Config
export ENCODER_MODEL="jhu-clsp/ettin-encoder-150m"
export TRAIN_DATASET="Harvard-DCML/tulu-v2-197K-processed"
export SELECTION_METHOD="doubly_greedy" # Default for embedding/less

# LESS Specific Config
export CKPT_DIR="$(pwd)/files/checkpoints/qwen2.5-0.5b_warmup"
export CKPT_STEPS="10 20 30 40 50" # Steps for 500 samples, 1 epoch, grad_acc 10
export PROJ_DIM=8192

# Paths
export INDEX_DIR="$(pwd)/files/index/ettin_subset_${END_INDEX}"
export DATASET_ROOT="$(pwd)/files/datasets"
export MODEL_ROOT="$(pwd)/files/models"
export RESULTS_ROOT="$(pwd)/files/results"

# Training Hyperparameters
export BATCH_SIZE=1
export GRAD_ACC=128
export EPOCHS=2
export LR=2e-5

# LoRA Config
export USE_LORA=True
export LORA_RANK=16
export LORA_ALPHA=32
export LORA_DROPOUT=0.05
