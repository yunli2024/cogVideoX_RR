#!/usr/bin/env bash
set -euo pipefail

cd /mnt/cpfs/jiachengliu/code/CogKit

PYTHON=/mnt/cpfs/jiachengliu/envs/cogkit-py310-cu124/bin/python
EXPORTER=quickstart/scripts/rolloutremover_cogvideox/export_dcp_lora.py
CHECKPOINT_ROOT=/mnt/cpfs/jiachengliu/dataset/CORNE/ckpt/cogkit_rolloutremover_cogvideox5b_480x720_c4r5g1p2_seed42_v1
OUTPUT_ROOT=/mnt/cpfs/jiachengliu/dataset/CORNE/ckpt/cogkit_rolloutremover_cogvideox5b_480x720_c4r5g1p2_seed42_v1_lora_exports
CONFIG=quickstart/scripts/rolloutremover_cogvideox/config_train_480x720_c4r5g1p2_seed42_v1.yaml
CONTRACT="$CHECKPOINT_ROOT/rollout_training_contract.json"
LOG_ROOT=/mnt/cpfs/jiachengliu/dataset/CORNE/logs/cogkit_rr_lora_export_rord280_screen_v1

mkdir -p "$LOG_ROOT"

for step in 1800 2000 2200 2400; do
  "$PYTHON" "$EXPORTER" \
    --checkpoint-dir "$CHECKPOINT_ROOT/checkpoint-$step" \
    --output-dir "$OUTPUT_ROOT/checkpoint-$step" \
    --code-root /mnt/cpfs/jiachengliu/code/CogKit \
    --expected-shards 7 \
    --expected-tensor-count 336 \
    --expected-numel 132120576 \
    --provenance-file "$CONFIG" \
    --provenance-file "$CONTRACT" \
    >"$LOG_ROOT/checkpoint-$step.log" 2>&1
done
