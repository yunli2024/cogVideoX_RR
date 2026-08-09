#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 4 ]]; then
  echo "usage: $0 <run-id> <protocol-id> <checkpoint-step> <shard:0-7>" >&2
  exit 2
fi

RUN_ID="$1"
PROTOCOL_ID="$2"
CHECKPOINT="$3"
SHARD="$4"
[[ "${RUN_ID}" =~ ^run-[0-9]{8}-[0-9]{6}-[a-f0-9]{8}$ ]] || { echo "unsafe run id" >&2; exit 2; }
[[ "${PROTOCOL_ID}" =~ ^orp-[a-f0-9]{16}$ ]] || { echo "unsafe protocol id" >&2; exit 2; }
[[ "${CHECKPOINT}" =~ ^[1-9][0-9]*$ ]] || { echo "invalid checkpoint" >&2; exit 2; }
[[ "${SHARD}" =~ ^[0-7]$ ]] || { echo "invalid shard" >&2; exit 2; }

case "${SHARD}" in
  0) SHARD_SHA256=5785d03146405e44661f6dac4191727ebeaf7154163a4503b0e445e6a84c78ca ;;
  1) SHARD_SHA256=40a4b2356c9d8df35d1aea53fda14595d9cee40f76544b3febe64b09434bed42 ;;
  2) SHARD_SHA256=43c1035a6cc3eeb38d5173440a36bb8b689136ed5b7b3f4c443ea3c10316671a ;;
  3) SHARD_SHA256=0c470924366b1d029b24b0f14a2033442accb18e499569ebe9f3430e97504922 ;;
  4) SHARD_SHA256=b7d6a51eeeb0bc2ff8123b81e47de2d1a56b808199e560808a98b40a6e99eee2 ;;
  5) SHARD_SHA256=220e7ffda0aaa331cbaa32079437a280184e563dacef91f37d7d756df8fe6584 ;;
  6) SHARD_SHA256=8309957c21ba7a7200a674904631c7c43d5bbd0db4c8d4107873de2d92cb61b3 ;;
  7) SHARD_SHA256=9525a046febce6adcbff70ded028362e4b9ed559d48d8fa9b72d663ef306cf1c ;;
esac

CWD=/mnt/cpfs/jiachengliu/code/CogKit-endpoint-inference
PYTHON=/mnt/cpfs/jiachengliu/envs/cogkit-py310-cu124/bin/python
SCRIPT_ROOT=${CWD}/quickstart/scripts/rolloutremover_cogvideox
INFER=${SCRIPT_ROOT}/infer_endpoint.py
CORE=${CWD}/src/cogkit/finetune/diffusion/models/cogvideo/cogvideox_endpoint_t2v/inference.py
SHARD_IDS=${SCRIPT_ROOT}/rord280_full_8way_shard${SHARD}_ids_v1.txt
BASE=/mnt/cpfs/jiachengliu/pretrained_models/CogVideoX-5b
LORA=/mnt/cpfs/jiachengliu/dataset/CORNE/ckpt/cogkit_endpointremover_cogvideox5b_480x720_seed42_v1_lora_exports/checkpoint-${CHECKPOINT}
CONTRACT=/mnt/cpfs/jiachengliu/dataset/CORNE/ckpt/cogkit_endpointremover_cogvideox5b_480x720_seed42_v1/endpoint_training_contract.json
MANIFEST=/mnt/cpfs/jiachengliu/dataset/CORNE/cache/cogkit_rr_rord280_full_v1/manifests/rord280_full.jsonl
CACHE=/mnt/cpfs/jiachengliu/dataset/CORNE/cache/cogkit_rr_rord280_full_v1/latents_mode_480x720
RUN_ROOT=/mnt/cpfs/jiachengliu/dataset/CORNE/inference/cogkit_endpoint_rord280_8gpu_v1/${RUN_ID}
OUTPUT=${RUN_ROOT}/inference/shard${SHARD}
LOG=${RUN_ROOT}/logs/shard${SHARD}.log
MIN_FREE_MIB=45000

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
test "$(sha256sum "${SHARD_IDS}" | awk '{print $1}')" = "${SHARD_SHA256}"
test "$(sha256sum "${LORA}/adapter_model.safetensors" | awk '{print $1}')" = 324bc5fd791d2a874c3bd482643a8fd2b2f32a7c0a29e691e77cd880edcac135
test "$(sha256sum "${CONTRACT}" | awk '{print $1}')" = c676fe9b741245b70d26d95d46a24c6363ec49be36a9c928e8d488f220c709ee
test "$(sha256sum "${MANIFEST}" | awk '{print $1}')" = fda189ab9ce7e2ae85c02d58987181b089a3cf55caf3dcf81d33d164bff0306a
test "$(sha256sum "${BASE}/scheduler/scheduler_config.json" | awk '{print $1}')" = 247ecd6635dae7bf889a7ec69ba951c44d746cb4edef13b9c3b1b16bfeeedba5
test "$(wc -l < "${SHARD_IDS}")" = 35
test "$(find "${CACHE}" -maxdepth 1 -type f -name '*.pt' | wc -l)" = 280
test "$(find "${CACHE}" -maxdepth 1 -type f -name '*.json' | wc -l)" = 280

gpu_free=$(nvidia-smi -i "${SHARD}" --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')
test "${gpu_free}" -ge "${MIN_FREE_MIB}"

mkdir -p "$(dirname "${OUTPUT}")" "$(dirname "${LOG}")"
cd "${CWD}"
export PYTHONPATH="${CWD}/src${PYTHONPATH:+:${PYTHONPATH}}"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="${SHARD}"

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
  --no-save_states \
  >"${LOG}" 2>&1
