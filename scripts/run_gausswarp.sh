#!/usr/bin/env bash
set -euo pipefail

DATASET="${1:?Usage: bash scripts/run_gausswarp.sh <P12|MIMIC_III|USHCN|HumanActivity> [SEED] [GPU]}"
SEED="${2:-2024}"
GPU="${3:-0}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python}"
TSDM_ROOT="${GAUSSWARP_TSDM_ROOT:-${ROOT}/storage/tsdm}"
RESULTS_ROOT="${GAUSSWARP_RESULTS_ROOT:-${ROOT}/storage/results}"
TRAIN_EPOCHS="${GAUSSWARP_TRAIN_EPOCHS:-200}"
MIN_TRAIN_EPOCHS="${GAUSSWARP_MIN_TRAIN_EPOCHS:-40}"
PATIENCE="${GAUSSWARP_PATIENCE:-20}"
NUM_WORKERS="${GAUSSWARP_NUM_WORKERS:-4}"

command -v "${PYTHON}" >/dev/null 2>&1 || {
  echo "Python executable not found: ${PYTHON}" >&2
  exit 1
}

mkdir -p "${TSDM_ROOT}" "${RESULTS_ROOT}"

case "${DATASET}" in
  P12)
    DATA_ROOT="${TSDM_ROOT}/rawdata/Physionet2012"
    GEOMETRY=(--seq_len 36 --pred_len 3 --enc_in 36)
    NUM_WORKERS="${GAUSSWARP_NUM_WORKERS:-2}"
    PROTOCOL=(--batch_size 16 --dropout 0.05 --train_val_loader_drop_last 0)
    ;;
  MIMIC_III)
    DATA_ROOT="${TSDM_ROOT}/rawdata/MIMIC_III_DeBrouwer2019"
    GEOMETRY=(--seq_len 72 --pred_len 3 --enc_in 96)
    PROTOCOL=(--batch_size 8 --dropout 0.10)
    ;;
  USHCN)
    DATA_ROOT="${TSDM_ROOT}/rawdata/USHCN_DeBrouwer2019"
    GEOMETRY=(--seq_len 150 --pred_len 3 --enc_in 5)
    PROTOCOL=(--batch_size 16 --dropout 0.05)
    ;;
  HumanActivity)
    DATA_ROOT="${TSDM_ROOT}/rawdata/HumanActivity"
    GEOMETRY=(--seq_len 3000 --pred_len 300 --enc_in 12)
    PROTOCOL=(--batch_size 32 --dropout 0.05)
    ;;
  *)
    echo "Unknown dataset: ${DATASET}" >&2
    exit 2
    ;;
esac

mkdir -p "${DATA_ROOT}"

if [[ "${DATASET}" == "MIMIC_III" && ! -f "${DATA_ROOT}/complete_tensor.csv" ]]; then
  echo "MIMIC-III preprocessing file not found:" >&2
  echo "${DATA_ROOT}/complete_tensor.csv" >&2
  echo "Generate complete_tensor.csv with the preprocessing workflow from https://github.com/edebrouwer/gru_ode_bayes and place it at the path above." >&2
  exit 1
fi

cd "${ROOT}"
export GAUSSWARP_TSDM_ROOT="${TSDM_ROOT}"
export GAUSSWARP_RESULTS_ROOT="${RESULTS_ROOT}"
export CUDA_VISIBLE_DEVICES="${GPU}"
export FIX_SEED_LIST="${SEED}"

MODEL_ID="${GAUSSWARP_MODEL_ID:-GaussWarp_${DATASET}_seed${SEED}}"

exec "${PYTHON}" main.py \
  --is_training 1 --model_name GaussWarp --model_id "${MODEL_ID}" --seed "${SEED}" \
  --dataset_name "${DATASET}" --dataset_root_path "${DATA_ROOT}" \
  --features M --loss GaussWarpObjective --checkpoints "${RESULTS_ROOT}" \
  --train_epochs "${TRAIN_EPOCHS}" --min_train_epochs "${MIN_TRAIN_EPOCHS}" \
  --patience "${PATIENCE}" \
  --val_interval 1 --test_after_train 1 --itr 1 \
  --optimizer AdamW --learning_rate 0.0002 --weight_decay 0.0001 \
  --lr_scheduler WarmupCosineLR --lr_warmup_epochs 5 \
  --gradient_clip_norm 1.0 --validate_before_training 1 \
  --num_workers "${NUM_WORKERS}" --d_model 64 \
  --gausswarp_dimension 64 --gausswarp_num_heads 4 --gausswarp_num_slots 12 \
  --gausswarp_num_layers 1 --gausswarp_num_lag_scales 6 \
  --gausswarp_auxiliary_weight 0.50 \
  --gausswarp_residual_limit 4.0 --gausswarp_slope_limit 40.0 \
  --gausswarp_pseudo_horizon_ratio 0.10 \
  --gpu_id 0 --use_multi_gpu 0 \
  "${GEOMETRY[@]}" "${PROTOCOL[@]}"
