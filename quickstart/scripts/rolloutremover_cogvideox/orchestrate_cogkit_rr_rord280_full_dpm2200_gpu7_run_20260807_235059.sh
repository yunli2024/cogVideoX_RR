#!/usr/bin/env bash
set -euo pipefail

RUN_ID=run-20260807-235059-17ff7603
PROTOCOL_ID=orp-8003d3911e53dc05
CHECKPOINT=2200
GPU_ID=7
MIN_FREE_MIB=45000
CWD=/mnt/cpfs/jiachengliu/code/CogKit
SCRIPT_ROOT=${CWD}/quickstart/scripts/rolloutremover_cogvideox
RUNNER=${SCRIPT_ROOT}/run_cogkit_rr_rord280_full_dpm2200_gpu7_run_20260807_235059.sh
INFER=${SCRIPT_ROOT}/infer_explicit_scheduler_v2.py
BASE_AUDIT_SCRIPT=${SCRIPT_ROOT}/audit_cogkit_rr_rord280_full_20260807.py
AUDIT_SCRIPT=${SCRIPT_ROOT}/audit_cogkit_rr_rord280_full_run_20260807_235059.py
PYTHON=/mnt/cpfs/jiachengliu/envs/cogkit-py310-cu124/bin/python
BASE=/mnt/cpfs/jiachengliu/pretrained_models/CogVideoX-5b
LORA=/mnt/cpfs/jiachengliu/dataset/CORNE/ckpt/cogkit_rolloutremover_cogvideox5b_480x720_c4r5g1p2_seed42_v1_lora_exports/checkpoint-2200
CONTRACT=/mnt/cpfs/jiachengliu/dataset/CORNE/ckpt/cogkit_rolloutremover_cogvideox5b_480x720_c4r5g1p2_seed42_v1/rollout_training_contract.json
MANIFEST=/mnt/cpfs/jiachengliu/dataset/CORNE/cache/cogkit_rr_rord280_full_v1/manifests/rord280_full.jsonl
CACHE=/mnt/cpfs/jiachengliu/dataset/CORNE/cache/cogkit_rr_rord280_full_v1/latents_mode_480x720
RUN_ROOT=/mnt/cpfs/jiachengliu/dataset/CORNE/inference/cogkit_rr_rord280_full_dpm_v1/${RUN_ID}
INFERENCE_ROOT=${RUN_ROOT}/inference
PIPELINE_LOG=/mnt/cpfs/jiachengliu/dataset/CORNE/logs/cogkit_rr_rord280_full_dpm_v1/${RUN_ID}_gpu7_inference_pipeline.log
SMOKE_ROOT=/mnt/cpfs/jiachengliu/dataset/CORNE/inference/cogkit_rr_rord280_gpu7_shared_smoke_v1/${RUN_ID}-sample00000211
SMOKE_LOG=/mnt/cpfs/jiachengliu/dataset/CORNE/logs/cogkit_rr_rord280_gpu7_shared_smoke_v1/${RUN_ID}-sample00000211.log

test -x "${PYTHON}"
test -f "${RUNNER}"
test -f "${INFER}"
test -f "${BASE_AUDIT_SCRIPT}"
test -f "${AUDIT_SCRIPT}"
test -f "${LORA}/adapter_model.safetensors"
test -f "${CONTRACT}"
test -f "${MANIFEST}"
test -d "${CACHE}"
test ! -e "${RUN_ROOT}"
test ! -e "${PIPELINE_LOG}"
test ! -e "${SMOKE_ROOT}"
test ! -e "${SMOKE_LOG}"
test "$(sha256sum "${RUNNER}" | awk '{print $1}')" = 113faabe262ce384471e1b83dc9775af78c9b3301773c477affbc1fb9d0d0f96
test "$(sha256sum "${INFER}" | awk '{print $1}')" = a19c6c096fcc0e61428a9b16151e7afc320cb578e2c4b6ef34811835654e2cba
test "$(sha256sum "${BASE_AUDIT_SCRIPT}" | awk '{print $1}')" = fb5b37cae073856089bc5be80b7d1ef26022e8f661cb433e8947b46d661bd126
test "$(sha256sum "${AUDIT_SCRIPT}" | awk '{print $1}')" = 8501a4be24559b62a2b38643467c75393bc4c55d9927262d6d55a728605bdcc2
test "$(sha256sum "${LORA}/adapter_model.safetensors" | awk '{print $1}')" = db46680166757682b2d33edc7102bb0d33d8e17228ab0e4e6877453c301a0cb1
test "$(sha256sum "${CONTRACT}" | awk '{print $1}')" = 4064edfc0436a0003b5f2fbbf4a5da1c9044b50ed56021190f12daa8cf6dc9f1
test "$(sha256sum "${MANIFEST}" | awk '{print $1}')" = fda189ab9ce7e2ae85c02d58987181b089a3cf55caf3dcf81d33d164bff0306a

mkdir -p "$(dirname "${PIPELINE_LOG}")" "$(dirname "${SMOKE_LOG}")"
exec > >(tee "${PIPELINE_LOG}") 2>&1

finish() {
  status=$?
  trap - EXIT
  echo "inference_pipeline_exit_status=${status} at $(date -Is)"
  exit "${status}"
}
trap finish EXIT

read_gpu_state() {
  local state
  state=$(nvidia-smi -i "${GPU_ID}" --query-gpu=memory.total,memory.used,memory.free,utilization.gpu --format=csv,noheader,nounits)
  IFS=',' read -r gpu_total gpu_used gpu_free gpu_utilization <<< "${state}"
  gpu_total=${gpu_total// /}
  gpu_used=${gpu_used// /}
  gpu_free=${gpu_free// /}
  gpu_utilization=${gpu_utilization// /}
}

wait_for_headroom() {
  while true; do
    read_gpu_state
    test "${gpu_total}" -ge 79000
    echo "gpu7 total_mib=${gpu_total} used_mib=${gpu_used} free_mib=${gpu_free} util_pct=${gpu_utilization} at $(date -Is)"
    if [[ "${gpu_free}" -ge "${MIN_FREE_MIB}" ]]; then
      return 0
    fi
    echo "waiting for gpu7 free memory >= ${MIN_FREE_MIB} MiB; teammate process is not modified"
    sleep 30
  done
}

echo "RUN_ID=${RUN_ID}"
echo "PROTOCOL_ID=${PROTOCOL_ID}"
echo "CHECKPOINT=${CHECKPOINT}"
echo "GPU_ID=${GPU_ID} shared_with_teammate_preprocessing=true"
echo "inference_pipeline_started_at=$(date -Is)"
echo "host=$(hostname)"
nvidia-smi -i "${GPU_ID}" --query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu --format=csv,noheader,nounits
nvidia-smi --query-compute-apps=gpu_uuid,pid,used_gpu_memory --format=csv,noheader,nounits || true

wait_for_headroom
echo "starting shared-GPU one-case smoke at $(date -Is)"
cd "${CWD}"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="${GPU_ID}"
"${PYTHON}" "${INFER}" \
  --model_path "${BASE}" \
  --lora_path "${LORA}" \
  --training_contract "${CONTRACT}" \
  --manifest "${MANIFEST}" \
  --cache_dir "${CACHE}" \
  --output_dir "${SMOKE_ROOT}" \
  --sample_ids 00000211 \
  --mask_condition_kind mask_check \
  --height 480 \
  --width 720 \
  --num_inference_steps 50 \
  --scheduler dpm \
  --guidance_scale 6 \
  --no-use_dynamic_cfg \
  --seed 42 \
  --dtype bf16 \
  --device cuda \
  --no-save_trajectory \
  --no-save_raw_latents \
  >"${SMOKE_LOG}" 2>&1 &
smoke_pid=$!
peak_used_mib=0
while kill -0 "${smoke_pid}" 2>/dev/null; do
  read_gpu_state
  if [[ "${gpu_used}" -gt "${peak_used_mib}" ]]; then
    peak_used_mib=${gpu_used}
  fi
  sleep 2
done
wait "${smoke_pid}"
test -f "${SMOKE_ROOT}/inference_contract.json"
test -f "${SMOKE_ROOT}/inference_manifest.jsonl"
test "$(wc -l < "${SMOKE_ROOT}/inference_manifest.jsonl")" = 1
test -f "${SMOKE_ROOT}/00000211/prediction.png"
test -f "${SMOKE_ROOT}/00000211/metadata.json"
echo "shared_gpu_smoke=passed peak_total_gpu_used_mib=${peak_used_mib} at $(date -Is)"

for shard in 0 1 2 3; do
  wait_for_headroom
  echo "starting shard=${shard} on physical_gpu=${GPU_ID} at $(date -Is)"
  bash "${RUNNER}" "${shard}"
  echo "shard=${shard} exit_status=0 completed_at=$(date -Is)"
done

for shard in 0 1 2 3; do
  shard_root=${INFERENCE_ROOT}/shard${shard}
  test -f "${shard_root}/inference_contract.json"
  test -f "${shard_root}/inference_manifest.jsonl"
  test "$(wc -l < "${shard_root}/inference_manifest.jsonl")" = 70
  test "$(find "${shard_root}" -mindepth 2 -maxdepth 2 -type f -name prediction.png | wc -l)" = 70
  test "$(find "${shard_root}" -mindepth 2 -maxdepth 2 -type f -name metadata.json | wc -l)" = 70
  test "$(find "${shard_root}" -type f -path '*/trajectory/*.png' | wc -l)" = 0
  test "$(find "${shard_root}" -type f -name raw_f7_f13_latents.pt | wc -l)" = 0
done
test "$(find "${INFERENCE_ROOT}" -mindepth 3 -maxdepth 3 -type f -name prediction.png | wc -l)" = 280
"${PYTHON}" "${AUDIT_SCRIPT}" \
  --run-root "${RUN_ROOT}" \
  --stage inference \
  --output "${RUN_ROOT}/audits/inference_audit.json"
echo "inference_artifact_audit=passed at $(date -Is)"
echo "inference_pipeline_completed_at=$(date -Is)"
