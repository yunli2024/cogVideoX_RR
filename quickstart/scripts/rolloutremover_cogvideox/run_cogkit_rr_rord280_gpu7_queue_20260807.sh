#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=7
cd /mnt/cpfs/jiachengliu/code/CogKit

PYTHON=/mnt/cpfs/jiachengliu/envs/cogkit-py310-cu124/bin/python
BASE=/mnt/cpfs/jiachengliu/pretrained_models/CogVideoX-5b
CKPT_ROOT=/mnt/cpfs/jiachengliu/dataset/CORNE/ckpt/cogkit_rolloutremover_cogvideox5b_480x720_c4r5g1p2_seed42_v1_lora_exports
CONTRACT=/mnt/cpfs/jiachengliu/dataset/CORNE/ckpt/cogkit_rolloutremover_cogvideox5b_480x720_c4r5g1p2_seed42_v1/rollout_training_contract.json
MANIFEST=/mnt/cpfs/jiachengliu/dataset/CORNE/cache/cogkit_rr_rord280_selection12_v1/manifests/selection12.jsonl
CACHE=/mnt/cpfs/jiachengliu/dataset/CORNE/cache/cogkit_rr_rord280_selection12_v1/latents_v2
OUTPUT_ROOT=/mnt/cpfs/jiachengliu/dataset/CORNE/inference/cogkit_rr_rord280_selection12_v1
LOG_ROOT=/mnt/cpfs/jiachengliu/dataset/CORNE/logs/cogkit_rr_rord280_selection12_v1
SAMPLE_IDS=(00000022 00000050 00000090 00000091 00000111 00000120 00000137 00000200 00000209 00000211 00000215 00000235)

mkdir -p "$LOG_ROOT"

run_one() {
  local step="$1"
  local run_id="$2"
  "$PYTHON" quickstart/scripts/rolloutremover_cogvideox/infer.py \
    --model_path "$BASE" \
    --lora_path "$CKPT_ROOT/checkpoint-$step" \
    --training_contract "$CONTRACT" \
    --manifest "$MANIFEST" \
    --cache_dir "$CACHE" \
    --output_dir "$OUTPUT_ROOT/$run_id/checkpoint-$step" \
    --sample_ids "${SAMPLE_IDS[@]}" \
    --mask_condition_kind mask_check \
    --height 480 --width 720 \
    --num_inference_steps 50 --guidance_scale 6 --no-use_dynamic_cfg \
    --seed 42 --dtype bf16 --device cuda --save_trajectory \
    >"$LOG_ROOT/$run_id.log" 2>&1
}

run_one 2000 run-20260807-134508-8d06e8ea
run_one 2400 run-20260807-134508-d4bf5429
