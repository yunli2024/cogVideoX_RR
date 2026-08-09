#!/usr/bin/env bash
set -euo pipefail

RUN_ID=run-20260809-164745-15f36857
PROTOCOL_ID=orp-5cfabd443376be2e
CHECKPOINT=2200
PHYSICAL_GPU=7
COGKIT_CWD=/mnt/cpfs/jiachengliu/code/CogKit-endpoint-inference
SCRIPT_ROOT=${COGKIT_CWD}/quickstart/scripts/rolloutremover_cogvideox
INFERENCE_AUDIT=${SCRIPT_ROOT}/audit_endpoint_rord280_8gpu.py
BASE_AUDIT=${SCRIPT_ROOT}/audit_cogkit_rr_rord280_full_20260807.py
EVAL_AUDIT=${SCRIPT_ROOT}/audit_endpoint_rord280_eval.py
PREP_SCRIPT=${SCRIPT_ROOT}/prepare_rord280_eval_v1.py
COGKIT_PYTHON=/mnt/cpfs/jiachengliu/envs/cogkit-py310-cu124/bin/python
EVAL_PYTHON=/mnt/cpfs/jiachengliu/envs/evaluation/bin/python
EVAL_CWD=/mnt/cpfs/jiachengliu/code/object_removal/Evaluation_content_crop
EVALUATOR=${EVAL_CWD}/unify_eval.py
MODEL_ENV=${EVAL_CWD}/model_env.sh
DATASET_ROOT=/mnt/cpfs/jiachengliu/dataset/eval_dataset/RORD-280-indexed-v1
INDEX_MANIFEST=${DATASET_ROOT}/index_manifest.json
RUN_ROOT=/mnt/cpfs/jiachengliu/dataset/CORNE/inference/cogkit_endpoint_rord280_8gpu_v1/${RUN_ID}
INFERENCE_ROOT=${RUN_ROOT}/inference
EVAL_READY=${RUN_ROOT}/eval_ready
EVAL_OUTPUT=${RUN_ROOT}/eval_outputs/all8_full_frame
PIPELINE_LOG=/mnt/cpfs/jiachengliu/dataset/CORNE/logs/cogkit_endpoint_rord280_8gpu_v1/${RUN_ID}_eval_pipeline.log
EVAL_LOG=${RUN_ROOT}/logs/eval_all8.log
INFERENCE_REAUDIT=${RUN_ROOT}/audits/inference_reaudit_before_eval.json
PREPARED_AUDIT=${RUN_ROOT}/audits/prepared_audit.json
EVALUATION_AUDIT=${RUN_ROOT}/audits/evaluation_audit.json

test -d "${RUN_ROOT}"
test -d "${INFERENCE_ROOT}"
test -f "${RUN_ROOT}/audits/inference_audit.json"
test -x "${COGKIT_PYTHON}"
test -x "${EVAL_PYTHON}"
test -f "${INFERENCE_AUDIT}"
test -f "${BASE_AUDIT}"
test -f "${EVAL_AUDIT}"
test -f "${PREP_SCRIPT}"
test -f "${EVALUATOR}"
test -f "${MODEL_ENV}"
test -f "${INDEX_MANIFEST}"
test ! -e "${INFERENCE_REAUDIT}"
test ! -e "${EVAL_READY}"
test ! -e "${PREPARED_AUDIT}"
test ! -e "${EVAL_OUTPUT}"
test ! -e "${EVAL_LOG}"
test ! -e "${EVALUATION_AUDIT}"
test ! -e "${PIPELINE_LOG}"
test "$(sha256sum "${INFERENCE_AUDIT}" | awk '{print $1}')" = a5ad25927fa398e64f0c2315a9786d8ca32aa04d7de96dec5d0395ba2ddcf85a
test "$(sha256sum "${BASE_AUDIT}" | awk '{print $1}')" = fb5b37cae073856089bc5be80b7d1ef26022e8f661cb433e8947b46d661bd126
test "$(sha256sum "${EVAL_AUDIT}" | awk '{print $1}')" = 811f323c706e11c132af63499e397f7940176383d514f61fc044cf3cd020c615
test "$(sha256sum "${PREP_SCRIPT}" | awk '{print $1}')" = 062c6dd4876d82503d7f92c7bd7feb5625b063127428c71e968cebe57b90e483
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

gpu_state=$(nvidia-smi -i "${PHYSICAL_GPU}" --query-gpu=memory.total,memory.used,utilization.gpu --format=csv,noheader,nounits)
IFS=',' read -r gpu_total gpu_used gpu_utilization <<< "${gpu_state}"
gpu_total=${gpu_total// /}
gpu_used=${gpu_used// /}
gpu_utilization=${gpu_utilization// /}
test "${gpu_total}" -ge 79000
test "${gpu_used}" = 0
test "${gpu_utilization}" = 0
gpu_uuid=$(nvidia-smi -i "${PHYSICAL_GPU}" --query-gpu=uuid --format=csv,noheader,nounits)
! nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader,nounits | grep -Fxq "${gpu_uuid}"

mkdir -p "$(dirname "${PIPELINE_LOG}")"
exec > >(tee "${PIPELINE_LOG}") 2>&1

finish() {
  status=$?
  trap - EXIT
  echo "eval_pipeline_exit_status=${status} at $(date -Is)"
  exit "${status}"
}
trap finish EXIT

echo "RUN_ID=${RUN_ID}"
echo "PROTOCOL_ID=${PROTOCOL_ID}"
echo "CHECKPOINT=${CHECKPOINT}"
echo "PHYSICAL_GPU=${PHYSICAL_GPU}"
echo "eval_pipeline_started_at=$(date -Is)"
echo "host=$(hostname)"

"${COGKIT_PYTHON}" "${INFERENCE_AUDIT}" \
  --run-root "${RUN_ROOT}" \
  --run-id "${RUN_ID}" \
  --output "${INFERENCE_REAUDIT}"
echo "inference_reaudit=passed at $(date -Is)"

"${COGKIT_PYTHON}" "${PREP_SCRIPT}" \
  --inference-root "${INFERENCE_ROOT}" \
  --dataset-root "${DATASET_ROOT}" \
  --index-manifest "${INDEX_MANIFEST}" \
  --output-root "${EVAL_READY}"
test -f "${EVAL_READY}/prepare_audit.json"
test "$(find "${EVAL_READY}/prediction" -maxdepth 1 -type f -name '*.png' | wc -l)" = 280
test "$(find "${EVAL_READY}/gt" -maxdepth 1 -name '*.png' | wc -l)" = 280
test "$(find "${EVAL_READY}/mask" -maxdepth 1 -name '*.png' | wc -l)" = 280
test -z "$(comm -3 <(find "${EVAL_READY}/prediction" -maxdepth 1 -name '*.png' -printf '%f\n' | sort) <(find "${EVAL_READY}/gt" -maxdepth 1 -name '*.png' -printf '%f\n' | sort))"
test -z "$(comm -3 <(find "${EVAL_READY}/prediction" -maxdepth 1 -name '*.png' -printf '%f\n' | sort) <(find "${EVAL_READY}/mask" -maxdepth 1 -name '*.png' -printf '%f\n' | sort))"
"${COGKIT_PYTHON}" "${EVAL_AUDIT}" \
  --run-root "${RUN_ROOT}" \
  --stage prepared \
  --output "${PREPARED_AUDIT}"
echo "eval_prepare_audit=passed at $(date -Is)"

mkdir -p "$(dirname "${EVAL_OUTPUT}")" "$(dirname "${EVAL_LOG}")"
echo "launching all-eight eval on physical GPU ${PHYSICAL_GPU} at $(date -Is)"
cd "${EVAL_CWD}"
source "${MODEL_ENV}"
CUDA_VISIBLE_DEVICES="${PHYSICAL_GPU}" "${EVAL_PYTHON}" "${EVALUATOR}" \
  "${EVAL_READY}/prediction" \
  "${EVAL_READY}/gt" \
  --mask_dir "${EVAL_READY}/mask" \
  --output_dir "${EVAL_OUTPUT}" \
  --metrics psnr,ssim,lpips,fid,cmmd,as,cfd,remove \
  --crop_border 0 \
  --spatial_protocol full_frame \
  --fid_batch_size 50 \
  --cmmd_batch_size 32 \
  --remove_crop \
  --device cuda:0 \
  >"${EVAL_LOG}" 2>&1

test -f "${EVAL_OUTPUT}/summary.json"
test -f "${EVAL_OUTPUT}/pairwise_metrics.csv"
test "$(wc -l < "${EVAL_OUTPUT}/pairwise_metrics.csv")" = 281
"${COGKIT_PYTHON}" "${EVAL_AUDIT}" \
  --run-root "${RUN_ROOT}" \
  --stage evaluation \
  --output "${EVALUATION_AUDIT}"
echo "evaluation_artifact_audit=passed at $(date -Is)"
echo "eval_pipeline_completed_at=$(date -Is)"
