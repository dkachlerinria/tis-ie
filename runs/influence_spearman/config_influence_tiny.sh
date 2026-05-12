#!/bin/bash
# Tiny override for quick smoke tests of the influence-Spearman pipeline.

source runs/influence_spearman/config_influence.sh

# Override sizes (very small for memory constraints)
export NUM_ANCHORS=20
export END_INDEX=200

# Smaller ground-truth projection (still high relative to LESS@8192).
# Use validate_projection.sh once to confirm this is fine.
export GT_PROJ_DIM=16384

# Project after every sample to minimize gradient accumulation in memory
export PROJECT_INTERVAL=1

export RUN_ID="${MODEL_SLUG}_tiny_anchors${NUM_ANCHORS}_train${END_INDEX}"
export INFLUENCE_OUT="${RESULTS_ROOT}/influence_spearman/${RUN_ID}"
