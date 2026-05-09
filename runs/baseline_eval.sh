#!/bin/bash
set -e

# Source configuration
source runs/config.sh

# Baseline doesn't have a selection method or subset index
METHOD="baseline"
RESULTS_DIR="${RESULTS_ROOT}/base_model_${BENCHMARK}"

echo "=========================================================="
echo "Starting Baseline Evaluation (Zero-Training)"
echo "Model:    ${TRAINING_MODEL}"
echo "Dataset:  ${BENCHMARK}"
echo "Results:  ${RESULTS_DIR}"
echo "=========================================================="

mkdir -p "${RESULTS_DIR}"

# Run evaluation directly on the base model
python3 -m evaluation.run_eval \
    --model_name_or_path "${TRAINING_MODEL}" \
    --eval_dataset "${BENCHMARK}" \
    --save_dir "${RESULTS_DIR}" \
    --use_vllm

echo "Baseline evaluation complete. Results saved to ${RESULTS_DIR}"
