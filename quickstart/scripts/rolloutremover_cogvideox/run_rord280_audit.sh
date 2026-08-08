#!/usr/bin/env bash
set -euo pipefail

cd /mnt/cpfs/jiachengliu/code/CogKit

PYTHON=/mnt/cpfs/jiachengliu/envs/cogkit-py310-cu124/bin/python
SCRIPT=quickstart/scripts/rolloutremover_cogvideox/build_rord280_selection.py
TRAIN_MANIFEST=/mnt/cpfs/jiachengliu/dataset/CORNE/cache/cogvideox_rollout_5b_14f_mode_480x720_v1/manifests/manifest_64505.jsonl
DATASET_ROOT=/mnt/cpfs/jiachengliu/dataset/eval_dataset/RORD-280-indexed-v1
INDEX_MANIFEST=/mnt/cpfs/jiachengliu/dataset/eval_dataset/RORD-280-indexed-v1/index_manifest.json
OUTPUT_ROOT=/mnt/cpfs/jiachengliu/dataset/CORNE/cache/cogkit_rr_rord280_selection12_v1
LOG_ROOT=/mnt/cpfs/jiachengliu/dataset/CORNE/logs/cogkit_rr_rord280_selection12_v1

mkdir -p "$LOG_ROOT"

exec "$PYTHON" "$SCRIPT" \
  --training-manifest "$TRAIN_MANIFEST" \
  --dataset-root "$DATASET_ROOT" \
  --index-manifest "$INDEX_MANIFEST" \
  --output-root "$OUTPUT_ROOT" \
  --workers 24 \
  >"$LOG_ROOT/overlap_selection.log" 2>&1
