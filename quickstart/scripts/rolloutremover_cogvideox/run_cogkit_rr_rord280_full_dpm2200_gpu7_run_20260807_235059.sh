#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 {0|1|2|3}" >&2
  exit 2
fi

SHARD=$1
RUN_ID=run-20260807-235059-17ff7603
PROTOCOL_ID=orp-8003d3911e53dc05
CHECKPOINT=2200
GPU_ID=7
case "${SHARD}" in
  0) SHARD_SHA256=2a127c0e70da8ff7a4cf7c949a29e18c109afbb2f3425199159f16ccb3111538 ;;
  1) SHARD_SHA256=49b48c01fab6cc333aa8cff7cb5a620613eb75c22275973f29bfcb4c35af1f0a ;;
  2) SHARD_SHA256=30e74450ed42712766a609dfddf1c1850e848d363ac4c30bd489dde2894ba1b5 ;;
  3) SHARD_SHA256=5c2a2b6d491d86705716e2c4dd5fcc02d4ee8d5beda4b32d370f77ebf3956ceb ;;
  *)
    echo "unsupported shard: ${SHARD}" >&2
    exit 2
    ;;
esac

CWD=/mnt/cpfs/jiachengliu/code/CogKit
PYTHON=/mnt/cpfs/jiachengliu/envs/cogkit-py310-cu124/bin/python
SCRIPT_ROOT=${CWD}/quickstart/scripts/rolloutremover_cogvideox
INFER=${SCRIPT_ROOT}/infer_explicit_scheduler_v2.py
CORE=${CWD}/src/cogkit/finetune/diffusion/models/cogvideo/cogvideox_rollout_t2v/inference.py
SHARD_IDS=${SCRIPT_ROOT}/rord280_full_shard${SHARD}_ids_v1.txt
BASE=/mnt/cpfs/jiachengliu/pretrained_models/CogVideoX-5b
LORA=/mnt/cpfs/jiachengliu/dataset/CORNE/ckpt/cogkit_rolloutremover_cogvideox5b_480x720_c4r5g1p2_seed42_v1_lora_exports/checkpoint-2200
CONTRACT=/mnt/cpfs/jiachengliu/dataset/CORNE/ckpt/cogkit_rolloutremover_cogvideox5b_480x720_c4r5g1p2_seed42_v1/rollout_training_contract.json
MANIFEST=/mnt/cpfs/jiachengliu/dataset/CORNE/cache/cogkit_rr_rord280_full_v1/manifests/rord280_full.jsonl
CACHE=/mnt/cpfs/jiachengliu/dataset/CORNE/cache/cogkit_rr_rord280_full_v1/latents_mode_480x720
RUN_ROOT=/mnt/cpfs/jiachengliu/dataset/CORNE/inference/cogkit_rr_rord280_full_dpm_v1/${RUN_ID}
OUTPUT=${RUN_ROOT}/inference/shard${SHARD}
LOG=${RUN_ROOT}/logs/shard${SHARD}.log
MIN_FREE_MIB=45000

echo "RUN_ID=${RUN_ID}"
echo "PROTOCOL_ID=${PROTOCOL_ID}"
echo "CHECKPOINT=${CHECKPOINT}"
echo "SHARD=${SHARD}"
echo "GPU_ID=${GPU_ID}"
date -Is

test -x "${PYTHON}"
test -f "${INFER}"
test -f "${CORE}"
test -f "${SHARD_IDS}"
test -f "${LORA}/adapter_model.safetensors"
test -f "${CONTRACT}"
test -f "${MANIFEST}"
test -d "${CACHE}"
test ! -e "${OUTPUT}"
test ! -e "${LOG}"
test "$(sha256sum "${INFER}" | awk '{print $1}')" = a19c6c096fcc0e61428a9b16151e7afc320cb578e2c4b6ef34811835654e2cba
test "$(sha256sum "${CORE}" | awk '{print $1}')" = 3ecf83947db43c60d20ff7c871baaac683f1d1ac81643bb58e63822ad2750b85
test "$(sha256sum "${SHARD_IDS}" | awk '{print $1}')" = "${SHARD_SHA256}"
test "$(sha256sum "${LORA}/adapter_model.safetensors" | awk '{print $1}')" = db46680166757682b2d33edc7102bb0d33d8e17228ab0e4e6877453c301a0cb1
test "$(sha256sum "${CONTRACT}" | awk '{print $1}')" = 4064edfc0436a0003b5f2fbbf4a5da1c9044b50ed56021190f12daa8cf6dc9f1
test "$(sha256sum "${MANIFEST}" | awk '{print $1}')" = fda189ab9ce7e2ae85c02d58987181b089a3cf55caf3dcf81d33d164bff0306a
test "$(sha256sum "${BASE}/scheduler/scheduler_config.json" | awk '{print $1}')" = 247ecd6635dae7bf889a7ec69ba951c44d746cb4edef13b9c3b1b16bfeeedba5
test "$(wc -l < "${SHARD_IDS}")" = 70
test "$(find "${CACHE}" -maxdepth 1 -type f -name '*.pt' | wc -l)" = 280
test "$(find "${CACHE}" -maxdepth 1 -type f -name '*.json' | wc -l)" = 280

gpu_state=$(nvidia-smi -i "${GPU_ID}" --query-gpu=memory.total,memory.used,memory.free --format=csv,noheader,nounits)
IFS=',' read -r gpu_total gpu_used gpu_free <<< "${gpu_state}"
gpu_total=${gpu_total// /}
gpu_used=${gpu_used// /}
gpu_free=${gpu_free// /}
test "${gpu_total}" -ge 79000
test "${gpu_free}" -ge "${MIN_FREE_MIB}"
echo "gpu_total_mib=${gpu_total} gpu_used_mib=${gpu_used} gpu_free_mib=${gpu_free}"

mkdir -p "$(dirname "${OUTPUT}")" "$(dirname "${LOG}")"
cd "${CWD}"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="${GPU_ID}"

exec "${PYTHON}" "${INFER}" \
  --model_path "${BASE}" \
  --lora_path "${LORA}" \
  --training_contract "${CONTRACT}" \
  --manifest "${MANIFEST}" \
  --cache_dir "${CACHE}" \
  --output_dir "${OUTPUT}" \
  --sample_ids_file "${SHARD_IDS}" \
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
  >"${LOG}" 2>&1
