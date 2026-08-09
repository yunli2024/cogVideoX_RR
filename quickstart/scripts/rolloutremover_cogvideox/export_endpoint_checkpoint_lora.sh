#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 || ! "$1" =~ ^[1-9][0-9]*$ ]]; then
  echo "Usage: $0 <checkpoint-step>" >&2
  exit 2
fi

STEP="$1"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CODE_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
VENV="/mnt/cpfs/jiachengliu/envs/cogkit-py310-cu124"
TRAIN_ROOT="/mnt/cpfs/jiachengliu/dataset/CORNE/ckpt/cogkit_endpointremover_cogvideox5b_480x720_seed42_v1"
CHECKPOINT_DIR="${TRAIN_ROOT}/checkpoint-${STEP}"
EXPORT_ROOT="/mnt/cpfs/jiachengliu/dataset/CORNE/ckpt/cogkit_endpointremover_cogvideox5b_480x720_seed42_v1_lora_exports"
OUTPUT_DIR="${EXPORT_ROOT}/checkpoint-${STEP}"
LOG_ROOT="/mnt/cpfs/jiachengliu/dataset/CORNE/logs/cogkit_endpoint_lora_export_v1"
LOG_PATH="${LOG_ROOT}/checkpoint-${STEP}.log"
CONFIG="${CODE_ROOT}/quickstart/scripts/rolloutremover_cogvideox/config_endpoint_train_480x720_seed42_v1.yaml"
TRAINING_CONTRACT="${TRAIN_ROOT}/endpoint_training_contract.json"

[[ -d "${CHECKPOINT_DIR}" ]] || { echo "Missing checkpoint: ${CHECKPOINT_DIR}" >&2; exit 1; }
[[ -s "${CHECKPOINT_DIR}/.metadata" ]] || { echo "Checkpoint metadata is missing or empty" >&2; exit 1; }
[[ ! -e "${OUTPUT_DIR}" ]] || { echo "Refusing existing export: ${OUTPUT_DIR}" >&2; exit 1; }
[[ -f "${CONFIG}" ]] || { echo "Missing config: ${CONFIG}" >&2; exit 1; }
[[ -f "${TRAINING_CONTRACT}" ]] || { echo "Missing contract: ${TRAINING_CONTRACT}" >&2; exit 1; }

mkdir -p "${LOG_ROOT}"
source "${VENV}/bin/activate"
cd "${CODE_ROOT}"

python "${SCRIPT_DIR}/export_dcp_lora.py" \
  --checkpoint-dir "${CHECKPOINT_DIR}" \
  --output-dir "${OUTPUT_DIR}" \
  --code-root "${CODE_ROOT}" \
  --expected-shards 7 \
  --expected-tensor-count 336 \
  --expected-numel 132120576 \
  --provenance-file "${CONFIG}" \
  --provenance-file "${TRAINING_CONTRACT}" \
  2>&1 | tee "${LOG_PATH}"

echo "Endpoint LoRA export completed: ${OUTPUT_DIR}"
