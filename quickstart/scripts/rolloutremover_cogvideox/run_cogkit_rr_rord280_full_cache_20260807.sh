#!/usr/bin/env bash
set -euo pipefail

RUN_TAG="cache-20260807-140340"
GPU_ID="6"
WORKDIR="/mnt/cpfs/jiachengliu/code/CogKit"
PYTHON="/mnt/cpfs/jiachengliu/envs/cogkit-py310-cu124/bin/python"
SCRIPT="/mnt/cpfs/jiachengliu/code/cogvideox_rollout_v2/cache_cogvideox_keyframes.py"
BASE_MODEL="/mnt/cpfs/jiachengliu/pretrained_models/CogVideoX-5b"
DATASET_ROOT="/mnt/cpfs/jiachengliu/dataset"
MANIFEST="/mnt/cpfs/jiachengliu/dataset/CORNE/cache/cogkit_rr_rord280_full_v1/manifests/rord280_full.jsonl"
OUTPUT_DIR="/mnt/cpfs/jiachengliu/dataset/CORNE/cache/cogkit_rr_rord280_full_v1/latents_mode_480x720"
LOG_DIR="/mnt/cpfs/jiachengliu/dataset/CORNE/cache/cogkit_rr_rord280_full_v1/logs"
LOG_PATH="${LOG_DIR}/${RUN_TAG}.log"

mkdir -p "${LOG_DIR}"
if [[ -e "${OUTPUT_DIR}" ]]; then
  echo "Refusing to overwrite existing output: ${OUTPUT_DIR}" >&2
  exit 2
fi

cd "${WORKDIR}"
export CUDA_VISIBLE_DEVICES="${GPU_ID}"
{
  echo "run_tag=${RUN_TAG}"
  echo "started_at=$(date -Is)"
  echo "host=$(hostname)"
  echo "gpu_id=${GPU_ID}"
  echo "workdir=${WORKDIR}"
  sha256sum "${SCRIPT}" "${MANIFEST}"
  "${PYTHON}" "${SCRIPT}" \
    --pretrained_model_name_or_path "${BASE_MODEL}" \
    --dataset_root "${DATASET_ROOT}" \
    --manifest "${MANIFEST}" \
    --output_dir "${OUTPUT_DIR}" \
    --preset wan14b_maskpred_c4r5_g1p2_14f \
    --height 480 \
    --width 720 \
    --seed 42 \
    --posterior_mode mode \
    --device cuda \
    --dtype bfloat16
  echo "completed_at=$(date -Is)"
} 2>&1 | tee "${LOG_PATH}"
