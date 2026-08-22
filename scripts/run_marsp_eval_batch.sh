#!/usr/bin/env bash
set -e

VIDEO_LIST="${1:-configs/marsp_eval_videos.txt}"
CHECKPOINT="${2:-checkpoints/best_model.pt}"
WINDOW_SIZE="${3:-5}"
THRESHOLD="${4:-0.5}"

while IFS= read -r VIDEO_PATH || [ -n "$VIDEO_PATH" ]; do
  [ -z "$VIDEO_PATH" ] && continue

  VIDEO_FILE="$(basename "$VIDEO_PATH")"
  VIDEO_NAME="${VIDEO_FILE%.mp4}"

  echo "========================================"
  echo "Running MARSP pipeline for: $VIDEO_NAME"
  echo "Input: $VIDEO_PATH"
  echo "========================================"

  python scripts/run_marsp_pipeline.py \
    --video-name "$VIDEO_NAME" \
    --input "$VIDEO_PATH" \
    --checkpoint "$CHECKPOINT" \
    --window-size "$WINDOW_SIZE" \
    --threshold "$THRESHOLD"

done < "$VIDEO_LIST"

echo "Batch MARSP evaluation complete."