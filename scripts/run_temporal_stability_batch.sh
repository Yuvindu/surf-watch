#!/usr/bin/env bash
set -e

mkdir -p outputs/temporal_stability

for summary in outputs/marsp/*_pipeline_summary.json; do
  [ -e "$summary" ] || continue

  video_name="$(basename "$summary" _pipeline_summary.json)"

  raw_mask="outputs/video_inference/${video_name}_stabilised_mask.mp4"
  processed_mask="outputs/temporal_aggregation/${video_name}_stabilised_agg_mask.mp4"
  output_json="outputs/temporal_stability/${video_name}_stability.json"

  if [ ! -f "$raw_mask" ] || [ ! -f "$processed_mask" ]; then
    echo "Skipping ${video_name} (missing mask files)"
    continue
  fi

  echo "Running temporal stability evaluation for ${video_name}"

  python scripts/evaluate_temporal_stability.py \
    --raw-mask-video "$raw_mask" \
    --processed-mask-video "$processed_mask" \
    --output-json "$output_json"
done

echo "Temporal stability batch evaluation complete."