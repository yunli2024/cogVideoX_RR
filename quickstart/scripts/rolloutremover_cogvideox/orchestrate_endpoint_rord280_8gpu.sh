#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "usage: $0 <run-id> <protocol-id> <checkpoint-step>" >&2
  exit 2
fi

RUN_ID="$1"
PROTOCOL_ID="$2"
CHECKPOINT="$3"
[[ "${RUN_ID}" =~ ^run-[0-9]{8}-[0-9]{6}-[a-f0-9]{8}$ ]] || { echo "unsafe run id" >&2; exit 2; }
[[ "${PROTOCOL_ID}" =~ ^orp-[a-f0-9]{16}$ ]] || { echo "unsafe protocol id" >&2; exit 2; }
[[ "${CHECKPOINT}" =~ ^[1-9][0-9]*$ ]] || { echo "invalid checkpoint" >&2; exit 2; }

CWD=/mnt/cpfs/jiachengliu/code/CogKit-endpoint-inference
SCRIPT_ROOT=${CWD}/quickstart/scripts/rolloutremover_cogvideox
RUNNER=${SCRIPT_ROOT}/run_endpoint_rord280_8gpu_shard.sh
AUDIT=${SCRIPT_ROOT}/audit_endpoint_rord280_8gpu.py
PYTHON=/mnt/cpfs/jiachengliu/envs/cogkit-py310-cu124/bin/python
RUN_ROOT=/mnt/cpfs/jiachengliu/dataset/CORNE/inference/cogkit_endpoint_rord280_8gpu_v1/${RUN_ID}
PIPELINE_LOG=/mnt/cpfs/jiachengliu/dataset/CORNE/logs/cogkit_endpoint_rord280_8gpu_v1/${RUN_ID}_inference_pipeline.log
MIN_FREE_MIB=45000

test -x "${PYTHON}"
test -f "${RUNNER}"
test -f "${AUDIT}"
test ! -e "${RUN_ROOT}"
test ! -e "${PIPELINE_LOG}"
for shard in 0 1 2 3 4 5 6 7; do
  test "$(wc -l < "${SCRIPT_ROOT}/rord280_full_8way_shard${shard}_ids_v1.txt")" = 35
done
for gpu in 0 1 2 3 4 5 6 7; do
  gpu_free=$(nvidia-smi -i "${gpu}" --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')
  test "${gpu_free}" -ge "${MIN_FREE_MIB}"
done

mkdir -p "$(dirname "${PIPELINE_LOG}")"
exec > >(tee "${PIPELINE_LOG}") 2>&1

pids=()
finish() {
  status=$?
  trap - EXIT INT TERM
  if [[ "${status}" -ne 0 ]]; then
    echo "nonzero pipeline exit; signalling only exact child PIDs"
    for pid in "${pids[@]-}"; do
      if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then kill -INT "${pid}" 2>/dev/null || true; fi
    done
    for pid in "${pids[@]-}"; do if [[ -n "${pid}" ]]; then wait "${pid}" 2>/dev/null || true; fi; done
  fi
  echo "inference_pipeline_exit_status=${status} at $(date -Is)"
  exit "${status}"
}
trap finish EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

echo "RUN_ID=${RUN_ID}"
echo "PROTOCOL_ID=${PROTOCOL_ID}"
echo "CHECKPOINT=${CHECKPOINT}"
echo "GPU_IDS=0,1,2,3,4,5,6,7"
echo "inference_pipeline_started_at=$(date -Is)"
echo "host=$(hostname)"

for shard in 0 1 2 3 4 5 6 7; do
  echo "starting shard=${shard} on physical_gpu=${shard} at $(date -Is)"
  bash "${RUNNER}" "${RUN_ID}" "${PROTOCOL_ID}" "${CHECKPOINT}" "${shard}" &
  pids[${shard}]=$!
done

failed=0
for shard in 0 1 2 3 4 5 6 7; do
  if wait "${pids[${shard}]}"; then shard_status=0; else shard_status=$?; failed=1; fi
  echo "shard=${shard} exit_status=${shard_status} completed_at=$(date -Is)"
done
if [[ "${failed}" -ne 0 ]]; then exit 1; fi

"${PYTHON}" "${AUDIT}" \
  --run-root "${RUN_ROOT}" \
  --run-id "${RUN_ID}" \
  --output "${RUN_ROOT}/audits/inference_audit.json"
echo "inference_artifact_audit=passed at $(date -Is)"
echo "inference_pipeline_completed_at=$(date -Is)"
