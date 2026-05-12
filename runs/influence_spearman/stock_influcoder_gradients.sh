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

# train_anchors: BBH local eval [0:N_TRAIN_A]
echo "Stocking train_anchors (BBH local [0:${INFLUCODER_N_TRAIN_A}])..."
python influcoder/gradient_stocking.py \
    --dataset bbh \
    --split train_anchors \
    --model_name "${INFLUENCE_MODEL}" \
    --proj_dim "${INFLUCODER_PROJ_DIM}" \
    --n_samples "${INFLUCODER_N_TRAIN_A}" \
    --start_index 0 \
    --load_warmup_path "${INFLUENCE_MODEL}" \
    --target_modules ${LORA_TARGET_MODULES} \
    --output_name "${INFLUCODER_DB_DIR}/train_anchors"

# eval_anchors: BBH local eval [N_TRAIN_A : N_TRAIN_A+N_EVAL_A]
echo "Stocking eval_anchors (BBH local [${INFLUCODER_N_TRAIN_A}:$((INFLUCODER_N_TRAIN_A+INFLUCODER_N_EVAL_A))])..."
python influcoder/gradient_stocking.py \
    --dataset bbh \
    --split eval_anchors \
    --model_name "${INFLUENCE_MODEL}" \
    --proj_dim "${INFLUCODER_PROJ_DIM}" \
    --n_samples "${INFLUCODER_N_EVAL_A}" \
    --start_index "${INFLUCODER_N_TRAIN_A}" \
    --load_warmup_path "${INFLUENCE_MODEL}" \
    --target_modules ${LORA_TARGET_MODULES} \
    --output_name "${INFLUCODER_DB_DIR}/eval_anchors"

# train_pool: Tulu [END_INDEX : END_INDEX+N_TRAIN_P]
echo "Stocking train_pool (Tulu [${END_INDEX}:$((END_INDEX+INFLUCODER_N_TRAIN_P))])..."
python influcoder/gradient_stocking.py \
    --dataset tulu \
    --split pool \
    --model_name "${INFLUENCE_MODEL}" \
    --proj_dim "${INFLUCODER_PROJ_DIM}" \
    --n_samples "${INFLUCODER_N_TRAIN_P}" \
    --start_index "${END_INDEX}" \
    --load_warmup_path "${INFLUENCE_MODEL}" \
    --target_modules ${LORA_TARGET_MODULES} \
    --output_name "${INFLUCODER_DB_DIR}/pool"

# eval_pool: Tulu [0:N_EVAL_P]
echo "Stocking eval_pool (Tulu [0:${INFLUCODER_N_EVAL_P}])..."
python influcoder/gradient_stocking.py \
    --dataset tulu \
    --split eval_pool \
    --model_name "${INFLUENCE_MODEL}" \
    --proj_dim "${INFLUCODER_PROJ_DIM}" \
    --n_samples "${INFLUCODER_N_EVAL_P}" \
    --start_index 0 \
    --load_warmup_path "${INFLUENCE_MODEL}" \
    --target_modules ${LORA_TARGET_MODULES} \
    --output_name "${INFLUCODER_DB_DIR}/eval_pool"

echo "Gradient stocking complete. DBs in: ${INFLUCODER_DB_DIR}"
