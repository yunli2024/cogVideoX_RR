#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 {1800|2200|2400}" >&2
  exit 2
fi

STEP=$1
PROTOCOL_ID=orp-04315389ca758503
case "${STEP}" in
  1800)
    RUN_ID=run-20260807-214150-d62ba25b
    GPU_ID=4
    LORA_SHA256=97ad2c59de2d09eca5daa567b67afad6a19b93e62cef773a5c59e18adda29df5
    ;;
  2200)
    RUN_ID=run-20260807-214150-97d08014
    GPU_ID=5
    LORA_SHA256=db46680166757682b2d33edc7102bb0d33d8e17228ab0e4e6877453c301a0cb1
    ;;
  2400)
    RUN_ID=run-20260807-214151-8daf3121
    GPU_ID=6
    LORA_SHA256=5c18018f3e3ac160f08ca9a83c97850f73bc6961702cadec09bdad490133a596
    ;;
  *)
    echo "unsupported checkpoint: ${STEP}" >&2
    exit 2
    ;;
esac

CWD=/mnt/cpfs/jiachengliu/code/CogKit
PYTHON=/mnt/cpfs/jiachengliu/envs/cogkit-py310-cu124/bin/python
SCRIPT_ROOT=${CWD}/quickstart/scripts/rolloutremover_cogvideox
INFER=${SCRIPT_ROOT}/infer_explicit_scheduler_v2.py
CORE=${CWD}/src/cogkit/finetune/diffusion/models/cogvideo/cogvideox_rollout_t2v/inference.py
SAMPLE_IDS_FILE=${SCRIPT_ROOT}/dpm_selection5_ids_v1.txt
BASE=/mnt/cpfs/jiachengliu/pretrained_models/CogVideoX-5b
LORA_ROOT=/mnt/cpfs/jiachengliu/dataset/CORNE/ckpt/cogkit_rolloutremover_cogvideox5b_480x720_c4r5g1p2_seed42_v1_lora_exports
LORA=${LORA_ROOT}/checkpoint-${STEP}
CONTRACT=/mnt/cpfs/jiachengliu/dataset/CORNE/ckpt/cogkit_rolloutremover_cogvideox5b_480x720_c4r5g1p2_seed42_v1/rollout_training_contract.json
MANIFEST=/mnt/cpfs/jiachengliu/dataset/CORNE/cache/cogkit_rr_rord280_selection12_v1/manifests/selection12.jsonl
CACHE=/mnt/cpfs/jiachengliu/dataset/CORNE/cache/cogkit_rr_rord280_selection12_v1/latents_v2
OUTPUT=/mnt/cpfs/jiachengliu/dataset/CORNE/inference/cogkit_rr_rord280_selection5_dpm_v1/${RUN_ID}/checkpoint-${STEP}
LOG=/mnt/cpfs/jiachengliu/dataset/CORNE/logs/cogkit_rr_rord280_selection5_dpm_v1/${RUN_ID}.log

echo "RUN_ID=${RUN_ID}"
echo "PROTOCOL_ID=${PROTOCOL_ID}"
echo "CHECKPOINT=${STEP}"
echo "GPU_ID=${GPU_ID}"
date -Is

test -x "${PYTHON}"
test -f "${INFER}"
test -f "${CORE}"
test -f "${SAMPLE_IDS_FILE}"
test -f "${LORA}/adapter_model.safetensors"
test -f "${CONTRACT}"
test -f "${MANIFEST}"
test ! -e "${OUTPUT}"
test ! -e "${LOG}"
test "$(sha256sum "${INFER}" | awk '{print $1}')" = a19c6c096fcc0e61428a9b16151e7afc320cb578e2c4b6ef34811835654e2cba
test "$(sha256sum "${CORE}" | awk '{print $1}')" = 3ecf83947db43c60d20ff7c871baaac683f1d1ac81643bb58e63822ad2750b85
test "$(sha256sum "${SAMPLE_IDS_FILE}" | awk '{print $1}')" = aa2f0d07bf1dbe118df4eb4956326c67afc14867f7a6a04ca50ee297576d73bc
test "$(sha256sum "${LORA}/adapter_model.safetensors" | awk '{print $1}')" = "${LORA_SHA256}"
test "$(sha256sum "${CONTRACT}" | awk '{print $1}')" = 4064edfc0436a0003b5f2fbbf4a5da1c9044b50ed56021190f12daa8cf6dc9f1
test "$(sha256sum "${MANIFEST}" | awk '{print $1}')" = b590e2bd809ffd5d7d44965f303b3c92c2245d054eb9cf23d587eddcbb64f895
test "$(sha256sum "${BASE}/scheduler/scheduler_config.json" | awk '{print $1}')" = 247ecd6635dae7bf889a7ec69ba951c44d746cb4edef13b9c3b1b16bfeeedba5

gpu_state=$(nvidia-smi -i "${GPU_ID}" --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits)
gpu_memory=${gpu_state%%,*}
gpu_utilization=${gpu_state##*,}
gpu_memory=${gpu_memory// /}
gpu_utilization=${gpu_utilization// /}
test "${gpu_memory}" = 0
test "${gpu_utilization}" = 0

mkdir -p "$(dirname "${OUTPUT}")" "$(dirname "${LOG}")"
cd "${CWD}"
export CUDA_VISIBLE_DEVICES="${GPU_ID}"

exec "${PYTHON}" "${INFER}" \
  --model_path "${BASE}" \
  --lora_path "${LORA}" \
  --training_contract "${CONTRACT}" \
  --manifest "${MANIFEST}" \
  --cache_dir "${CACHE}" \
  --output_dir "${OUTPUT}" \
  --sample_ids_file "${SAMPLE_IDS_FILE}" \
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
  --save_trajectory \
  --save_raw_latents \
  >"${LOG}" 2>&1
