#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=7
cd /mnt/cpfs/jiachengliu/code/CogKit

PYTHON=/mnt/cpfs/jiachengliu/envs/cogkit-py310-cu124/bin/python
OUTPUT=/mnt/cpfs/jiachengliu/dataset/CORNE/inference/cogkit_rr_rord280_selection12_v1/run-20260807-134939-29d949cd/checkpoint-2000-smoke1
LOG=/mnt/cpfs/jiachengliu/dataset/CORNE/logs/cogkit_rr_rord280_selection12_v1/run-20260807-134939-29d949cd.log

mkdir -p "$(dirname "$LOG")"

exec "$PYTHON" quickstart/scripts/rolloutremover_cogvideox/infer.py \
  --model_path /mnt/cpfs/jiachengliu/pretrained_models/CogVideoX-5b \
  --lora_path /mnt/cpfs/jiachengliu/dataset/CORNE/ckpt/cogkit_rolloutremover_cogvideox5b_480x720_c4r5g1p2_seed42_v1_lora_exports/checkpoint-2000 \
  --training_contract /mnt/cpfs/jiachengliu/dataset/CORNE/ckpt/cogkit_rolloutremover_cogvideox5b_480x720_c4r5g1p2_seed42_v1/rollout_training_contract.json \
  --manifest /mnt/cpfs/jiachengliu/dataset/CORNE/cache/cogkit_rr_rord280_selection12_v1/manifests/selection12.jsonl \
  --cache_dir /mnt/cpfs/jiachengliu/dataset/CORNE/cache/cogkit_rr_rord280_selection12_v1/latents_v2 \
  --output_dir "$OUTPUT" \
  --sample_ids 00000022 \
  --mask_condition_kind mask_check \
  --height 480 \
  --width 720 \
  --num_inference_steps 50 \
  --guidance_scale 6 \
  --no-use_dynamic_cfg \
  --seed 42 \
  --dtype bf16 \
  --device cuda \
  --save_trajectory \
  >"$LOG" 2>&1
