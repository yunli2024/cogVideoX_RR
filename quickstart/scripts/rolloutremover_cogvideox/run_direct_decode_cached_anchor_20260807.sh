#!/usr/bin/env bash
set -euo pipefail

task_cwd=/mnt/cpfs/jiachengliu/code/CogKit
task_python=/mnt/cpfs/jiachengliu/envs/cogkit-py310-cu124/bin/python
task_script=/mnt/cpfs/jiachengliu/code/CogKit/quickstart/scripts/rolloutremover_cogvideox/direct_decode_cached_anchor.py
task_model=/mnt/cpfs/jiachengliu/pretrained_models/CogVideoX-5b
task_cache=/mnt/cpfs/jiachengliu/dataset/CORNE/cache/cogkit_rr_rord280_selection12_v1/latents_v2/00000022.pt
task_metadata=/mnt/cpfs/jiachengliu/dataset/CORNE/cache/cogkit_rr_rord280_selection12_v1/latents_v2/00000022.json
task_output=/mnt/cpfs/jiachengliu/dataset/CORNE/diagnostics/cog_f7_direct_decode/run-20260807-202155-b17169aa
task_log=/mnt/cpfs/jiachengliu/dataset/CORNE/diagnostics/cog_f7_direct_decode/run-20260807-202155-b17169aa.log

cd "$task_cwd"
mkdir -p /mnt/cpfs/jiachengliu/dataset/CORNE/diagnostics/cog_f7_direct_decode
test ! -e "$task_output"
test ! -e "$task_log"
export CUDA_VISIBLE_DEVICES=0

"$task_python" "$task_script" \
  --model_path "$task_model" \
  --cache_path "$task_cache" \
  --metadata_path "$task_metadata" \
  --output_dir "$task_output" \
  --sample_id 00000022 \
  --global_seed 42 \
  --device cuda:0 \
  >"$task_log" 2>&1
