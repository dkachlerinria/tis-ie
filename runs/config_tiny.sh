#!/bin/bash
# Tiny override for pipeline validation — sources config.sh then reduces all sizes.
# Usage: source runs/config_tiny.sh && bash runs/select_less.sh

source runs/config.sh

export WARMUP_SAMPLES=50
export END_INDEX=200
export NUM_SAMPLES=50
