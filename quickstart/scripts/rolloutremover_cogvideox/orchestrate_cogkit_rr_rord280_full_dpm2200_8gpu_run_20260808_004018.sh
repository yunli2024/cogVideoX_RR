#!/usr/bin/env bash
set -euo pipefail

RUN_ID=run-20260808-004018-149f154b
PROTOCOL_ID=orp-de76c87ed64fded9
CHECKPOINT=2200
MIN_FREE_MIB=45000
CWD=/mnt/cpfs/jiachengliu/code/CogKit
SCRIPT_ROOT=${CWD}/quickstart/scripts/rolloutremover_cogvideox
RUNNER=${SCRIPT_ROOT}/run_cogkit_rr_rord280_full_dpm2200_8gpu_run_20260808_004018.sh
AUDIT_SCRIPT=${SCRIPT_ROOT}/audit_cogkit_rr_rord280_full_8gpu_run_20260808_004018.py
INFER=${SCRIPT_ROOT}/infer_explicit_scheduler_v2.py
CORE=${CWD}/src/cogkit/finetune/diffusion/models/cogvideo/cogvideox_rollout_t2v/inference.py
PYTHON=/mnt/cpfs/jiachengliu/envs/cogkit-py310-cu124/bin/python
BASE=/mnt/cpfs/jiachengliu/pretrained_models/CogVideoX-5b
LORA=/mnt/cpfs/jiachengliu/dataset/CORNE/ckpt/cogkit_rolloutremover_cogvideox5b_480x720_c4r5g1p2_seed42_v1_lora_exports/checkpoint-2200
CONTRACT=/mnt/cpfs/jiachengliu/dataset/CORNE/ckpt/cogkit_rolloutremover_cogvideox5b_480x720_c4r5g1p2_seed42_v1/rollout_training_contract.json
MANIFEST=/mnt/cpfs/jiachengliu/dataset/CORNE/cache/cogkit_rr_rord280_full_v1/manifests/rord280_full.jsonl
CACHE=/mnt/cpfs/jiachengliu/dataset/CORNE/cache/cogkit_rr_rord280_full_v1/latents_mode_480x720
RUN_ROOT=/mnt/cpfs/jiachengliu/dataset/CORNE/inference/cogkit_rr_rord280_full_dpm_8gpu_v1/${RUN_ID}
INFERENCE_ROOT=${RUN_ROOT}/inference
PIPELINE_LOG=/mnt/cpfs/jiachengliu/dataset/CORNE/logs/cogkit_rr_rord280_full_dpm_8gpu_v1/${RUN_ID}_inference_pipeline.log

test -x "${PYTHON}"
test -f "${RUNNER}"
test -f "${AUDIT_SCRIPT}"
test -f "${INFER}"
test -f "${CORE}"
test -f "${LORA}/adapter_model.safetensors"
test -f "${CONTRACT}"
test -f "${MANIFEST}"
test -d "${CACHE}"
test ! -e "${RUN_ROOT}"
test ! -e "${PIPELINE_LOG}"
test "$(sha256sum "${RUNNER}" | awk '{print $1}')" = 93ee1ab4640f01335a8d38dde710b7a61003d7b0eb2cf3da3584c247132544f8
test "$(sha256sum "${AUDIT_SCRIPT}" | awk '{print $1}')" = 4e63b6131b86c19be07a815ebc1e6f1bb8e08285b61e36265a32d25ba4a538be
test "$(sha256sum "${INFER}" | awk '{print $1}')" = a19c6c096fcc0e61428a9b16151e7afc320cb578e2c4b6ef34811835654e2cba
test "$(sha256sum "${CORE}" | awk '{print $1}')" = 3ecf83947db43c60d20ff7c871baaac683f1d1ac81643bb58e63822ad2750b85
test "$(sha256sum "${LORA}/adapter_model.safetensors" | awk '{print $1}')" = db46680166757682b2d33edc7102bb0d33d8e17228ab0e4e6877453c301a0cb1
test "$(sha256sum "${CONTRACT}" | awk '{print $1}')" = 4064edfc0436a0003b5f2fbbf4a5da1c9044b50ed56021190f12daa8cf6dc9f1
test "$(sha256sum "${MANIFEST}" | awk '{print $1}')" = fda189ab9ce7e2ae85c02d58987181b089a3cf55caf3dcf81d33d164bff0306a
test "$(sha256sum "${BASE}/scheduler/scheduler_config.json" | awk '{print $1}')" = 247ecd6635dae7bf889a7ec69ba951c44d746cb4edef13b9c3b1b16bfeeedba5
test "$(find "${CACHE}" -maxdepth 1 -type f -name '*.pt' | wc -l)" = 280
test "$(find "${CACHE}" -maxdepth 1 -type f -name '*.json' | wc -l)" = 280

for shard in 0 1 2 3 4 5 6 7; do
  shard_file=${SCRIPT_ROOT}/rord280_full_8way_shard${shard}_ids_v1.txt
  test -f "${shard_file}"
  test "$(wc -l < "${shard_file}")" = 35
  case "${shard}" in
    0) expected_sha=5785d03146405e44661f6dac4191727ebeaf7154163a4503b0e445e6a84c78ca ;;
    1) expected_sha=40a4b2356c9d8df35d1aea53fda14595d9cee40f76544b3febe64b09434bed42 ;;
    2) expected_sha=43c1035a6cc3eeb38d5173440a36bb8b689136ed5b7b3f4c443ea3c10316671a ;;
    3) expected_sha=0c470924366b1d029b24b0f14a2033442accb18e499569ebe9f3430e97504922 ;;
    4) expected_sha=b7d6a51eeeb0bc2ff8123b81e47de2d1a56b808199e560808a98b40a6e99eee2 ;;
    5) expected_sha=220e7ffda0aaa331cbaa32079437a280184e563dacef91f37d7d756df8fe6584 ;;
    6) expected_sha=8309957c21ba7a7200a674904631c7c43d5bbd0db4c8d4107873de2d92cb61b3 ;;
    7) expected_sha=9525a046febce6adcbff70ded028362e4b9ed559d48d8fa9b72d663ef306cf1c ;;
  esac
  test "$(sha256sum "${shard_file}" | awk '{print $1}')" = "${expected_sha}"
done

mkdir -p "$(dirname "${PIPELINE_LOG}")"
exec > >(tee "${PIPELINE_LOG}") 2>&1

for gpu in 0 1 2 3 4 5 6 7; do
  gpu_state=$(nvidia-smi -i "${gpu}" --query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu --format=csv,noheader,nounits)
  IFS=',' read -r gpu_index gpu_name gpu_total gpu_used gpu_free gpu_utilization <<< "${gpu_state}"
  gpu_total=${gpu_total// /}
  gpu_used=${gpu_used// /}
  gpu_free=${gpu_free// /}
  gpu_utilization=${gpu_utilization// /}
  test "${gpu_total}" -ge 79000
  test "${gpu_free}" -ge "${MIN_FREE_MIB}"
  echo "preflight gpu=${gpu} name=${gpu_name} total_mib=${gpu_total} used_mib=${gpu_used} free_mib=${gpu_free} util_pct=${gpu_utilization}"
done

pids=()
finish() {
  status=$?
  trap - EXIT INT TERM
  if [[ "${status}" -ne 0 ]]; then
    echo "nonzero pipeline exit; signalling only exact inference child PIDs"
    for pid in "${pids[@]-}"; do
      if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
        kill -INT "${pid}" 2>/dev/null || true
      fi
    done
    for pid in "${pids[@]-}"; do
      if [[ -n "${pid}" ]]; then
        wait "${pid}" 2>/dev/null || true
      fi
    done
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
echo "GPU_IDS=0,1,2,3,4,5,6,7 shared_with_teammate_preprocessing=true"
echo "evaluation_enabled=false"
echo "inference_pipeline_started_at=$(date -Is)"
echo "host=$(hostname)"
nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu --format=csv,noheader,nounits
nvidia-smi --query-compute-apps=gpu_uuid,pid,used_gpu_memory --format=csv,noheader,nounits || true
pgrep -af preprocess_singleturn_cache.py || true

for shard in 0 1 2 3 4 5 6 7; do
  echo "starting shard=${shard} on physical_gpu=${shard} at $(date -Is)"
  bash "${RUNNER}" "${shard}" &
  pids[${shard}]=$!
  echo "shard=${shard} child_pid=${pids[${shard}]}"
done

failed=0
for shard in 0 1 2 3 4 5 6 7; do
  if wait "${pids[${shard}]}"; then
    shard_status=0
  else
    shard_status=$?
    failed=1
  fi
  echo "shard=${shard} exit_status=${shard_status} completed_at=$(date -Is)"
done
if [[ "${failed}" -ne 0 ]]; then
  echo "at least one inference shard failed; audit will not run" >&2
  exit 1
fi

for shard in 0 1 2 3 4 5 6 7; do
  shard_root=${INFERENCE_ROOT}/shard${shard}
  test -f "${shard_root}/inference_contract.json"
  test -f "${shard_root}/inference_manifest.jsonl"
  test "$(wc -l < "${shard_root}/inference_manifest.jsonl")" = 35
  test "$(find "${shard_root}" -mindepth 2 -maxdepth 2 -type f -name prediction.png | wc -l)" = 35
  test "$(find "${shard_root}" -mindepth 2 -maxdepth 2 -type f -name metadata.json | wc -l)" = 35
  test "$(find "${shard_root}" -type f -path '*/trajectory/*.png' | wc -l)" = 0
  test "$(find "${shard_root}" -type f -name raw_f7_f13_latents.pt | wc -l)" = 0
done
test "$(find "${INFERENCE_ROOT}" -mindepth 3 -maxdepth 3 -type f -name prediction.png | wc -l)" = 280
"${PYTHON}" "${AUDIT_SCRIPT}" \
  --run-root "${RUN_ROOT}" \
  --output "${RUN_ROOT}/audits/inference_audit.json"
echo "inference_artifact_audit=passed at $(date -Is)"
echo "inference_pipeline_completed_at=$(date -Is)"
echo "evaluation_deferred=true; exiting without geometry restore or evaluation"
