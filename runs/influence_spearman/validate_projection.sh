#!/bin/bash
set -euo pipefail
# One-time sanity: does TRAK projection at GT_PROJ_DIM agree with UNPROJECTED?
# Uses tiny sizes since the unprojected branch holds full LoRA-grad vectors.
CFG="${1:-runs/influence_spearman/config_influence_tiny.sh}"
source "$CFG"

VAL_OUT="${INFLUENCE_OUT}/projection_validation"
mkdir -p "$VAL_OUT"

python3 -m influence_eval.validate_projection \
    --model_name "${INFLUENCE_MODEL}" \
    --save_dir "${VAL_OUT}" \
    --end_index 50 \
    --num_anchors 10 \
    --dev_dataset_name "${BENCHMARK}" \
    --proj_dims 8192 16384 32768 65536 \
    --lora_target_modules "${LORA_TARGET_MODULES}" \
    --lora_rank "${LORA_RANK}" \
    --lora_alpha "${LORA_ALPHA}" \
    --lora_dropout "${LORA_DROPOUT}" \
    --lora_seed "${LORA_SEED}"

echo "Validation report: ${VAL_OUT}/validation_report.json"
