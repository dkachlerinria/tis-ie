#!/bin/bash
# Shared config for the influence-Spearman experiment.
# Sources runs/config.sh for TRAINING_MODEL, paths, LoRA defaults, etc.
# then layers experiment-specific knobs on top.

source runs/config.sh

# Model for influence calculations
export INFLUENCE_MODEL="Qwen/Qwen3-0.6B"
INFLUENCE_MODEL_SLUG=$(echo "${INFLUENCE_MODEL}" | tr '[:upper:]' '[:lower:]' | sed 's|.*/||')

# Anchor count (BBH dev split sliced [0:NUM_ANCHORS])
export NUM_ANCHORS=100

# Projection dimensions
export GT_PROJ_DIM=65536
export LESS_PROJ_DIM=8192

# Fresh-LoRA seed (must be identical across GT and LESS for apples-to-apples)
export LORA_SEED=0

# Gradient accumulation before projection (tune down to save memory)
export PROJECT_INTERVAL=1

# Smaller LoRA for A30
export LORA_RANK=16
export LORA_ALPHA=32

# Random baseline seed
export RANDOM_SEED=0

# Sequence length used for analytic FLOPS accounting (training max_seq_length)
export FLOPS_SEQ_LEN=2048

# Output directory (model-scoped so model swaps don't collide)
export RUN_ID="${INFLUENCE_MODEL_SLUG}_anchors${NUM_ANCHORS}_train${END_INDEX}"
export INFLUENCE_OUT="${RESULTS_ROOT}/influence_spearman/${RUN_ID}"
