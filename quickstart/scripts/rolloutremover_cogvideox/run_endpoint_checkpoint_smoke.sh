#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 3 || ! "$1" =~ ^[1-9][0-9]*$ ]]; then
  echo "Usage: $0 <checkpoint-step> [physical-gpu=7] [sample-id=080001]" >&2
  exit 2
fi

STEP="$1"
GPU="${2:-7}"
SAMPLE_ID="${3:-080001}"
[[ "${GPU}" =~ ^[0-7]$ ]] || { echo "GPU must be a physical ID from 0 to 7" >&2; exit 2; }
[[ "${SAMPLE_ID}" =~ ^[A-Za-z0-9._-]+$ ]] || { echo "Unsafe sample ID" >&2; exit 2; }
[[ "${SAMPLE_ID}" != "." && "${SAMPLE_ID}" != ".." ]] || { echo "Unsafe sample ID" >&2; exit 2; }

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CODE_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
VENV="/mnt/cpfs/jiachengliu/envs/cogkit-py310-cu124"
MODEL="/mnt/cpfs/jiachengliu/pretrained_models/CogVideoX-5b"
TRAIN_ROOT="/mnt/cpfs/jiachengliu/dataset/CORNE/ckpt/cogkit_endpointremover_cogvideox5b_480x720_seed42_v1"
TRAINING_CONTRACT="${TRAIN_ROOT}/endpoint_training_contract.json"
LORA_DIR="/mnt/cpfs/jiachengliu/dataset/CORNE/ckpt/cogkit_endpointremover_cogvideox5b_480x720_seed42_v1_lora_exports/checkpoint-${STEP}"
MANIFEST="/mnt/cpfs/jiachengliu/dataset/CORNE/cache/cogvideox_rollout_5b_14f_mode_480x720_v1/manifests/manifest_64505.jsonl"
CACHE_DIR="/mnt/cpfs/jiachengliu/dataset/CORNE/cache/cogvideox_rollout_5b_14f_mode_480x720_v1/latents"
RUN_ID="run-$(date -u +%Y%m%d-%H%M%S)-endpoint${STEP}-${SAMPLE_ID}"
OUTPUT_DIR="/mnt/cpfs/jiachengliu/dataset/CORNE/inference/cogkit_endpoint_cached_smoke_v1/${RUN_ID}"
LOG_ROOT="/mnt/cpfs/jiachengliu/dataset/CORNE/logs/cogkit_endpoint_cached_smoke_v1"
LOG_PATH="${LOG_ROOT}/${RUN_ID}.log"

[[ -d "${MODEL}" ]] || { echo "Missing model: ${MODEL}" >&2; exit 1; }
[[ -s "${LORA_DIR}/adapter_model.safetensors" ]] || { echo "Missing exported LoRA: ${LORA_DIR}" >&2; exit 1; }
[[ -f "${TRAINING_CONTRACT}" ]] || { echo "Missing contract: ${TRAINING_CONTRACT}" >&2; exit 1; }
[[ -f "${MANIFEST}" ]] || { echo "Missing manifest: ${MANIFEST}" >&2; exit 1; }
[[ -f "${CACHE_DIR}/${SAMPLE_ID}.pt" ]] || { echo "Missing sample cache: ${SAMPLE_ID}.pt" >&2; exit 1; }
[[ -f "${CACHE_DIR}/${SAMPLE_ID}.json" ]] || { echo "Missing sample metadata: ${SAMPLE_ID}.json" >&2; exit 1; }
[[ ! -e "${OUTPUT_DIR}" ]] || { echo "Refusing existing output: ${OUTPUT_DIR}" >&2; exit 1; }

if nvidia-smi -i "${GPU}" --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | grep -Eq '^[0-9]+$'; then
  echo "Physical GPU ${GPU} already has a compute process; refusing to launch" >&2
  exit 1
fi

mkdir -p "${LOG_ROOT}"
source "${VENV}/bin/activate"
export PYTHONPATH="${CODE_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
cd "${CODE_ROOT}"

CUDA_VISIBLE_DEVICES="${GPU}" python "${SCRIPT_DIR}/infer_endpoint.py" \
  --model_path "${MODEL}" \
  --lora_path "${LORA_DIR}" \
  --training_contract "${TRAINING_CONTRACT}" \
  --manifest "${MANIFEST}" \
  --cache_dir "${CACHE_DIR}" \
  --output_dir "${OUTPUT_DIR}" \
  --sample_ids "${SAMPLE_ID}" \
  --mask_condition_kind mask_check \
  --height 480 \
  --width 720 \
  --num_inference_steps 50 \
  --scheduler dpm \
  --guidance_scale 6.0 \
  --seed 42 \
  --dtype bf16 \
  --device cuda \
  2>&1 | tee "${LOG_PATH}"

echo "Endpoint inference smoke completed: ${OUTPUT_DIR}"
