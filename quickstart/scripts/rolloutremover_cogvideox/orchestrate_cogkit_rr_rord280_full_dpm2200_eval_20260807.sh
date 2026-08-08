#!/usr/bin/env bash
set -euo pipefail

RUN_ID=run-20260807-221601-77609ae1
PROTOCOL_ID=orp-8003d3911e53dc05
CHECKPOINT=2200
RUNNER=/mnt/cpfs/jiachengliu/code/CogKit/quickstart/scripts/rolloutremover_cogvideox/run_cogkit_rr_rord280_full_dpm2200_20260807.sh
PREP_SCRIPT=/mnt/cpfs/jiachengliu/code/CogKit/quickstart/scripts/rolloutremover_cogvideox/prepare_rord280_eval_v1.py
AUDIT_SCRIPT=/mnt/cpfs/jiachengliu/code/CogKit/quickstart/scripts/rolloutremover_cogvideox/audit_cogkit_rr_rord280_full_20260807.py
COGKIT_PYTHON=/mnt/cpfs/jiachengliu/envs/cogkit-py310-cu124/bin/python
EVAL_PYTHON=/mnt/cpfs/jiachengliu/envs/evaluation/bin/python
EVAL_CWD=/mnt/cpfs/jiachengliu/code/object_removal/Evaluation_content_crop
EVALUATOR=${EVAL_CWD}/unify_eval.py
MODEL_ENV=${EVAL_CWD}/model_env.sh
DATASET_ROOT=/mnt/cpfs/jiachengliu/dataset/eval_dataset/RORD-280-indexed-v1
INDEX_MANIFEST=${DATASET_ROOT}/index_manifest.json
RUN_ROOT=/mnt/cpfs/jiachengliu/dataset/CORNE/inference/cogkit_rr_rord280_full_dpm_v1/${RUN_ID}
INFERENCE_ROOT=${RUN_ROOT}/inference
EVAL_READY=${RUN_ROOT}/eval_ready
EVAL_OUTPUT=${RUN_ROOT}/eval_outputs/all8_full_frame
PIPELINE_LOG=/mnt/cpfs/jiachengliu/dataset/CORNE/logs/cogkit_rr_rord280_full_dpm_v1/${RUN_ID}_pipeline.log
EVAL_LOG=${RUN_ROOT}/logs/eval_all8.log

test -f "${RUNNER}"
test -x "${COGKIT_PYTHON}"
test -x "${EVAL_PYTHON}"
test -f "${PREP_SCRIPT}"
test -f "${AUDIT_SCRIPT}"
test -f "${EVALUATOR}"
test -f "${MODEL_ENV}"
test -f "${INDEX_MANIFEST}"
test ! -e "${RUN_ROOT}"
test ! -e "${PIPELINE_LOG}"
test "$(sha256sum "${RUNNER}" | awk '{print $1}')" = 84769cff996555e83cfce2f0f6b13bbdaac4e73a47ce83ed3ae2db2cbc6ff177
test "$(sha256sum "${PREP_SCRIPT}" | awk '{print $1}')" = 062c6dd4876d82503d7f92c7bd7feb5625b063127428c71e968cebe57b90e483
test "$(sha256sum "${AUDIT_SCRIPT}" | awk '{print $1}')" = fb5b37cae073856089bc5be80b7d1ef26022e8f661cb433e8947b46d661bd126
test "$(sha256sum "${EVALUATOR}" | awk '{print $1}')" = 386c7054ab9aaee1bb3450250ea7292bdb8c142ff689dae4a35f19f8b2e35adf
test "$(sha256sum "${MODEL_ENV}" | awk '{print $1}')" = 476635030c778a9c871649ccbde17d774c9148f2e987e09769c139a976150b7c
test "$(sha256sum "${EVAL_CWD}/AestheticScore.py" | awk '{print $1}')" = a9a321917c6c29887df8ec89038c46344f29f372ed9af6dde6dc1aa99d39558f
test "$(sha256sum "${EVAL_CWD}/CFD.py" | awk '{print $1}')" = e3a1f1e2addd70c0629b54a9c83cde5a642b385aabfe39236ab7581c9e1a9776
test "$(sha256sum "${EVAL_CWD}/CMMD.py" | awk '{print $1}')" = 3d81ae41970db3a3e2231484d0c375aae5a54805b75b504120d0754db87c2b9e
test "$(sha256sum "${EVAL_CWD}/FID.py" | awk '{print $1}')" = f8c214c01ecf716bf105c5e0638417811a8b898a2fd00e1a5985eade1776ff1c
test "$(sha256sum "${EVAL_CWD}/LPIPS.py" | awk '{print $1}')" = 10a75f8e088036c841f00711da78bc24a8d9d5b56b96dbdfccc6dd7f373bd5c1
test "$(sha256sum "${EVAL_CWD}/PSNR.py" | awk '{print $1}')" = 9b4e1c1fe28a29bdcf3db5d8278d1d3a496ce9ab07420478912606219668e4d4
test "$(sha256sum "${EVAL_CWD}/ReMOVE.py" | awk '{print $1}')" = 4d53a8a24e09d53d743964534fecded3a067ea8a2834372d59b5dec45fbb5051
test "$(sha256sum "${EVAL_CWD}/SSIM.py" | awk '{print $1}')" = 4a4784b09a39eb5ed54f352e45a3483de48efcda739d7e538144a03501191edc
test "$(sha256sum "${EVAL_CWD}/spatial_protocol.py" | awk '{print $1}')" = fd04b2301d6a15b62a96a6cbf38b4e627a88d66af1a6a44dc438cbed8c9f0f31
test "$(sha256sum "${INDEX_MANIFEST}" | awk '{print $1}')" = 841c5b6f24ed0f1b03420dace35843c356fd803ab24ca1f6f3ef0bc52bbc1de4

mkdir -p "$(dirname "${PIPELINE_LOG}")"
exec > >(tee "${PIPELINE_LOG}") 2>&1

echo "RUN_ID=${RUN_ID}"
echo "PROTOCOL_ID=${PROTOCOL_ID}"
echo "CHECKPOINT=${CHECKPOINT}"
echo "pipeline_started_at=$(date -Is)"
echo "host=$(hostname)"
tmux list-sessions -F '#{session_name} #{session_windows} #{session_attached}'
nvidia-smi --query-compute-apps=gpu_uuid,pid,used_gpu_memory --format=csv,noheader,nounits || true

gpu_idle() {
  local gpu_id=$1
  local state memory utilization
  state=$(nvidia-smi -i "${gpu_id}" --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits)
  memory=${state%%,*}
  utilization=${state##*,}
  memory=${memory// /}
  utilization=${utilization// /}
  [[ "${memory}" = 0 && "${utilization}" = 0 ]]
}

gpu_without_compute_process() {
  local gpu_id=$1
  local gpu_uuid
  gpu_uuid=$(nvidia-smi -i "${gpu_id}" --query-gpu=uuid --format=csv,noheader,nounits)
  ! nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader,nounits | grep -Fxq "${gpu_uuid}"
}

gpus=(4 5 6 7)
started=(0 0 0 0)
idle_streak=(0 0 0 0)
declare -a pids
launched=0

echo "Waiting for two consecutive verified-idle checks on each assigned GPU; no occupied GPU will be used."
while [[ "${launched}" -lt 4 ]]; do
  for shard in 0 1 2 3; do
    if [[ "${started[${shard}]}" -eq 1 ]]; then
      continue
    fi
    gpu_id=${gpus[${shard}]}
    if gpu_idle "${gpu_id}" && gpu_without_compute_process "${gpu_id}"; then
      idle_streak[${shard}]=$((idle_streak[${shard}] + 1))
    else
      idle_streak[${shard}]=0
    fi
    if [[ "${idle_streak[${shard}]}" -ge 2 ]]; then
      echo "launching shard=${shard} gpu=${gpu_id} at $(date -Is)"
      bash "${RUNNER}" "${shard}" &
      pids[${shard}]=$!
      started[${shard}]=1
      launched=$((launched + 1))
    fi
  done
  if [[ "${launched}" -lt 4 ]]; then
    nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits
    nvidia-smi --query-compute-apps=gpu_uuid,pid,used_gpu_memory --format=csv,noheader,nounits || true
    sleep 30
  fi
done

inference_failed=0
for shard in 0 1 2 3; do
  if wait "${pids[${shard}]}"; then
    echo "shard=${shard} exit_status=0 completed_at=$(date -Is)"
  else
    shard_status=$?
    echo "shard=${shard} exit_status=${shard_status} failed_at=$(date -Is)" >&2
    inference_failed=1
  fi
done
test "${inference_failed}" = 0

for shard in 0 1 2 3; do
  shard_root=${INFERENCE_ROOT}/shard${shard}
  test -f "${shard_root}/inference_contract.json"
  test -f "${shard_root}/inference_manifest.jsonl"
  test "$(wc -l < "${shard_root}/inference_manifest.jsonl")" = 70
  test "$(find "${shard_root}" -mindepth 2 -maxdepth 2 -type f -name prediction.png | wc -l)" = 70
  test "$(find "${shard_root}" -mindepth 2 -maxdepth 2 -type f -name metadata.json | wc -l)" = 70
  test "$(find "${shard_root}" -type f -path '*/trajectory/*.png' | wc -l)" = 0
  test "$(find "${shard_root}" -type f -name raw_f7_f13_latents.pt | wc -l)" = 0
  grep -q '"class_name": "CogVideoXDPMScheduler"' "${shard_root}/inference_contract.json"
  grep -q '"config_sha256": "247ecd6635dae7bf889a7ec69ba951c44d746cb4edef13b9c3b1b16bfeeedba5"' "${shard_root}/inference_contract.json"
  grep -q '"save_trajectory": false' "${shard_root}/inference_contract.json"
  grep -q '"save_raw_latents": false' "${shard_root}/inference_contract.json"
done
test "$(find "${INFERENCE_ROOT}" -mindepth 3 -maxdepth 3 -type f -name prediction.png | wc -l)" = 280
"${COGKIT_PYTHON}" "${AUDIT_SCRIPT}" \
  --run-root "${RUN_ROOT}" \
  --stage inference \
  --output "${RUN_ROOT}/audits/inference_audit.json"
echo "inference_artifact_audit=passed at $(date -Is)"

test ! -e "${EVAL_READY}"
"${COGKIT_PYTHON}" "${PREP_SCRIPT}" \
  --inference-root "${INFERENCE_ROOT}" \
  --dataset-root "${DATASET_ROOT}" \
  --index-manifest "${INDEX_MANIFEST}" \
  --output-root "${EVAL_READY}"
test -f "${EVAL_READY}/prepare_audit.json"
test "$(find "${EVAL_READY}/prediction" -maxdepth 1 -type f -name '*.png' | wc -l)" = 280
test "$(find "${EVAL_READY}/gt" -maxdepth 1 -name '*.png' | wc -l)" = 280
test "$(find "${EVAL_READY}/mask" -maxdepth 1 -name '*.png' | wc -l)" = 280
grep -q '"status": "passed"' "${EVAL_READY}/prepare_audit.json"
grep -q '"count": 280' "${EVAL_READY}/prepare_audit.json"
test -z "$(comm -3 <(find "${EVAL_READY}/prediction" -maxdepth 1 -name '*.png' -printf '%f\n' | sort) <(find "${EVAL_READY}/gt" -maxdepth 1 -name '*.png' -printf '%f\n' | sort))"
test -z "$(comm -3 <(find "${EVAL_READY}/prediction" -maxdepth 1 -name '*.png' -printf '%f\n' | sort) <(find "${EVAL_READY}/mask" -maxdepth 1 -name '*.png' -printf '%f\n' | sort))"
"${COGKIT_PYTHON}" "${AUDIT_SCRIPT}" \
  --run-root "${RUN_ROOT}" \
  --stage prepared \
  --output "${RUN_ROOT}/audits/prepared_audit.json"
echo "eval_prepare_audit=passed at $(date -Is)"

eval_gpu=""
eval_idle_streak=(0 0 0 0 0 0 0 0)
while [[ -z "${eval_gpu}" ]]; do
  for gpu_id in 7 6 5 4; do
    if gpu_idle "${gpu_id}" && gpu_without_compute_process "${gpu_id}"; then
      eval_idle_streak[${gpu_id}]=$((eval_idle_streak[${gpu_id}] + 1))
      if [[ "${eval_idle_streak[${gpu_id}]}" -ge 2 ]]; then
        eval_gpu=${gpu_id}
        break
      fi
    else
      eval_idle_streak[${gpu_id}]=0
    fi
  done
  if [[ -z "${eval_gpu}" ]]; then
    sleep 30
  fi
done

test ! -e "${EVAL_OUTPUT}"
test ! -e "${EVAL_LOG}"
mkdir -p "$(dirname "${EVAL_OUTPUT}")" "$(dirname "${EVAL_LOG}")"
echo "launching all-eight eval on physical GPU ${eval_gpu} at $(date -Is)"
cd "${EVAL_CWD}"
source "${MODEL_ENV}"
CUDA_VISIBLE_DEVICES="${eval_gpu}" "${EVAL_PYTHON}" "${EVALUATOR}" \
  "${EVAL_READY}/prediction" \
  "${EVAL_READY}/gt" \
  --mask_dir "${EVAL_READY}/mask" \
  --output_dir "${EVAL_OUTPUT}" \
  --metrics psnr,ssim,lpips,fid,cmmd,as,cfd,remove \
  --crop_border 0 \
  --spatial_protocol full_frame \
  --fid_batch_size 50 \
  --cmmd_batch_size 32 \
  --device cuda:0 \
  >"${EVAL_LOG}" 2>&1

test -f "${EVAL_OUTPUT}/summary.json"
test -f "${EVAL_OUTPUT}/pairwise_metrics.csv"
test "$(wc -l < "${EVAL_OUTPUT}/pairwise_metrics.csv")" = 281
test "$(grep -c '"status": "ok"' "${EVAL_OUTPUT}/summary.json")" = 8
"${COGKIT_PYTHON}" "${AUDIT_SCRIPT}" \
  --run-root "${RUN_ROOT}" \
  --stage evaluation \
  --output "${RUN_ROOT}/audits/evaluation_audit.json"
echo "evaluation_artifact_audit=passed at $(date -Is)"
echo "pipeline_completed_at=$(date -Is)"
