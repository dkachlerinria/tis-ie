#!/bin/bash
# Shared config for the influence-Spearman experiment.
# Sources runs/config.sh for TRAINING_MODEL, paths, LoRA defaults, etc.
# then layers experiment-specific knobs on top.

source runs/config.sh

# Anchor count (BBH dev split sliced [0:NUM_ANCHORS])
export NUM_ANCHORS=100

# Projection dimensions
export GT_PROJ_DIM=65536
export LESS_PROJ_DIM=8192

# Fresh-LoRA seed (must be identical across GT and LESS for apples-to-apples)
export LORA_SEED=0

# Random baseline seed
export RANDOM_SEED=0

# Sequence length used for analytic FLOPS accounting (training max_seq_length)
export FLOPS_SEQ_LEN=2048

# Output directory (model-scoped so model swaps don't collide)
export RUN_ID="${MODEL_SLUG}_anchors${NUM_ANCHORS}_train${END_INDEX}"
export INFLUENCE_OUT="${RESULTS_ROOT}/influence_spearman/${RUN_ID}"
