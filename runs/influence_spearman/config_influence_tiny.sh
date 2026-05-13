#!/bin/bash
# Tiny override for quick smoke tests of the influence-Spearman pipeline.

source runs/influence_spearman/config_influence.sh

# Override sizes (very small for memory constraints)
export NUM_ANCHORS=100
export END_INDEX=1000

# Smaller ground-truth projection (still high relative to LESS@8192).
# Use validate_projection.sh once to confirm this is fine.
export GT_PROJ_DIM=262144

# Project after every sample to minimize gradient accumulation in memory
export PROJECT_INTERVAL=1

export RUN_ID="${INFLUENCE_MODEL_SLUG}_tiny_anchors${NUM_ANCHORS}_train${END_INDEX}"
export INFLUENCE_OUT="${RESULTS_ROOT}/influence_spearman/${RUN_ID}"

# Tiny influcoder overrides
export INFLUCODER_RUN_MODE="small"
export INFLUCODER_N_TRAIN_A=1000
export INFLUCODER_N_EVAL_A=100
export INFLUCODER_N_TRAIN_P=2000
export INFLUCODER_N_EVAL_P=500

export INFLUCODER_DB_DIR="${INFLUENCE_OUT}/influcoder_db"
export INFLUCODER_ENCODER_DIR="/home/dkachler/working_folder/tis-ie/tis-ie/files/results/qwen3-1.7b-base/influence_spearman/qwen3-1.7b-base_tiny_anchors20_train200/influcoder_encoder"
#${INFLUENCE_OUT}/influcoder_encoder"
