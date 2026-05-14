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

# LoGRA settings (rank=8 matches paper default)
export LOGRA_RANK=8
export LOGRA_BATCH_SIZE=1
# Target all-linear layers (like LESS) for fair comparison, not just MLP
export LOGRA_ALL_LINEAR=1

# Random baseline seed
export RANDOM_SEED=0

# Sequence length used for analytic FLOPS accounting (training max_seq_length)
export FLOPS_SEQ_LEN=2048

# Influcoder settings
export INFLUCODER_PROJ_DIM=8192
export INFLUCODER_RUN_MODE="small"
export INFLUCODER_N_TRAIN_A=2000        # BBH anchors for encoder training (start at NUM_ANCHORS)
export INFLUCODER_N_EVAL_A=500          # BBH anchors for encoder eval (start after train_anchors)
export INFLUCODER_N_TRAIN_P=2000        # Tulu pool for encoder training (start at END_INDEX)
export INFLUCODER_N_EVAL_P=2000          # Tulu pool for encoder eval (start at END_INDEX+N_TRAIN_P, disjoint from [0:END_INDEX])

# Output directory (model-scoped, completely separate from SFT paths)
export INFLUENCE_OUT="files/influence_models/${INFLUENCE_MODEL_SLUG}"

export INFLUCODER_DB_DIR="${INFLUENCE_OUT}/influcoder_db"
export INFLUCODER_ENCODER_DIR="${INFLUENCE_OUT}/influcoder_encoder"
