#!/bin/bash
# Tiny override for quick smoke tests of the influence-Spearman pipeline.

source runs/influence_spearman/config_influence.sh

# Override sizes (very small for memory constraints)
export NUM_ANCHORS=50
export END_INDEX=50

# Smaller ground-truth projection (still high relative to LESS@8192).
# Use validate_projection.sh once to confirm this is fine.
export GT_PROJ_DIM=262144

# Project after every sample to minimize gradient accumulation in memory
export PROJECT_INTERVAL=1

# We no longer override INFLUENCE_OUT here; it safely points to files/influence_models/...

# Tiny influcoder overrides
export INFLUCODER_RUN_MODE="small"
export INFLUCODER_N_TRAIN_A=500
export INFLUCODER_N_EVAL_A=100
export INFLUCODER_N_TRAIN_P=1000
export INFLUCODER_N_EVAL_P=100


