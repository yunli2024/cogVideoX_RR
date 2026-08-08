#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=7
export PYTHONPATH=/mnt/cpfs/jiachengliu/code/cogvideox_rollout_v2

cd /mnt/cpfs/jiachengliu/code/cogvideox_rollout_v2

PYTHON=/mnt/cpfs/jiachengliu/envs/cogkit-py310-cu124/bin/python
ROOT=/mnt/cpfs/jiachengliu/dataset/CORNE/cache/cogkit_rr_rord280_selection12_v1
LOG_ROOT=/mnt/cpfs/jiachengliu/dataset/CORNE/logs/cogkit_rr_rord280_selection12_v1

exec "$PYTHON" cache_cogvideox_keyframes.py \
  --pretrained_model_name_or_path /mnt/cpfs/jiachengliu/pretrained_models/CogVideoX-5b \
  --dataset_root /mnt/cpfs/jiachengliu/dataset \
  --manifest "$ROOT/manifests/selection12.jsonl" \
  --output_dir "$ROOT/latents_v2" \
  --preset wan14b_maskpred_c4r5_g1p2_14f \
  --height 480 \
  --width 720 \
  --seed 42 \
  --posterior_mode mode \
  --device cuda \
  --dtype bfloat16 \
  >"$LOG_ROOT/cache_v2.log" 2>&1
