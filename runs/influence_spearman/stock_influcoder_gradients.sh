#!/bin/bash
set -euo pipefail
CFG="${1:-runs/influence_spearman/config_influence.sh}"
source "$CFG"

if [ ! -d "data/eval/bbh" ]; then
    echo "ERROR: data/eval/bbh not found. Run ./download_eval.sh first."
    exit 1
fi

rm -rf "${INFLUCODER_DB_DIR}"
mkdir -p "${INFLUCODER_DB_DIR}"

# LoRA/projection args shared across splits — match compute_gradient_scores.py exactly
COMMON_ARGS=(
    --model_name           "${INFLUENCE_MODEL}"
    --proj_dim             "${INFLUCODER_PROJ_DIM}"
    --proj_seed            42
    --project_interval     "${PROJECT_INTERVAL}"
    --lora_target_modules  "${LORA_TARGET_MODULES}"
    --lora_rank            "${LORA_RANK}"
    --lora_alpha           "${LORA_ALPHA}"
    --lora_dropout         "${LORA_DROPOUT}"
    --lora_seed            "${LORA_SEED}"
    --output_dir           "${INFLUCODER_DB_DIR}"
)

# train_anchors: BBH local [NUM_ANCHORS : NUM_ANCHORS+N_TRAIN_A]
# Starts AFTER the Spearman eval range [0:NUM_ANCHORS] to avoid contamination
echo "Stocking train_anchors (BBH [${NUM_ANCHORS}:$((NUM_ANCHORS+INFLUCODER_N_TRAIN_A))])..."
python influcoder/gradient_stocking_EXACT.py \
    "${COMMON_ARGS[@]}" \
    --split        train_anchors \
    --n_samples    "${INFLUCODER_N_TRAIN_A}" \
    --start_index  "${NUM_ANCHORS}" \
    --output_name  train_anchors

# eval_anchors: BBH local [NUM_ANCHORS+N_TRAIN_A : NUM_ANCHORS+N_TRAIN_A+N_EVAL_A]
echo "Stocking eval_anchors (BBH [$((NUM_ANCHORS+INFLUCODER_N_TRAIN_A)):$((NUM_ANCHORS+INFLUCODER_N_TRAIN_A+INFLUCODER_N_EVAL_A))])..."
python influcoder/gradient_stocking_EXACT.py \
    "${COMMON_ARGS[@]}" \
    --split        eval_anchors \
    --n_samples    "${INFLUCODER_N_EVAL_A}" \
    --start_index  "$((NUM_ANCHORS + INFLUCODER_N_TRAIN_A))" \
    --output_name  eval_anchors

# train_pool: Tulu [END_INDEX : END_INDEX+N_TRAIN_P]
echo "Stocking pool (Tulu [${END_INDEX}:$((END_INDEX+INFLUCODER_N_TRAIN_P))])..."
python influcoder/gradient_stocking_EXACT.py \
    "${COMMON_ARGS[@]}" \
    --split        pool \
    --n_samples    "${INFLUCODER_N_TRAIN_P}" \
    --start_index  "${END_INDEX}" \
    --output_name  pool

# eval_pool: Tulu [END_INDEX+N_TRAIN_P : END_INDEX+N_TRAIN_P+N_EVAL_P]
# Disjoint from both train_pool and the Spearman eval range [0:END_INDEX]
echo "Stocking eval_pool (Tulu [$((END_INDEX+INFLUCODER_N_TRAIN_P)):$((END_INDEX+INFLUCODER_N_TRAIN_P+INFLUCODER_N_EVAL_P))])..."
python influcoder/gradient_stocking_EXACT.py \
    "${COMMON_ARGS[@]}" \
    --split        eval_pool \
    --n_samples    "${INFLUCODER_N_EVAL_P}" \
    --start_index  "$((END_INDEX + INFLUCODER_N_TRAIN_P))" \
    --output_name  eval_pool

echo "Gradient stocking complete. Files in: ${INFLUCODER_DB_DIR}"
ls -la "${INFLUCODER_DB_DIR}"
