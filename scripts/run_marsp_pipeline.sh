#!/usr/bin/env bash
set -e

# Usage:
# bash scripts/run_marsp_pipeline.sh RipVIS-051 ../RipVIS/train/videos/RipVIS-051.mp4

VIDEO_NAME="${1:-RipVIS-051}"
INPUT_VIDEO="${2:-../RipVIS/train/videos/RipVIS-051.mp4}"
CHECKPOINT="${3:-checkpoints/best_model.pt}"

STAB_VIDEO="outputs/motion_compensation/${VIDEO_NAME}_stabilised.mp4"
STAB_COMPARE="outputs/motion_compensation/${VIDEO_NAME}_comparison.mp4"
STAB_RESULTS="outputs/motion_compensation/${VIDEO_NAME}_results.json"

ORIG_OVERLAY="outputs/video_inference/${VIDEO_NAME}_original_overlay.mp4"
ORIG_MASK="outputs/video_inference/${VIDEO_NAME}_original_mask.mp4"
ORIG_PROB="outputs/video_inference/${VIDEO_NAME}_original_prob.mp4"

STAB_OVERLAY="outputs/video_inference/${VIDEO_NAME}_stabilised_overlay.mp4"
STAB_MASK="outputs/video_inference/${VIDEO_NAME}_stabilised_mask.mp4"
STAB_PROB="outputs/video_inference/${VIDEO_NAME}_stabilised_prob.mp4"

ORIG_AGG_PROB="outputs/temporal_aggregation/${VIDEO_NAME}_original_agg_prob.mp4"
ORIG_AGG_MASK="outputs/temporal_aggregation/${VIDEO_NAME}_original_agg_mask.mp4"
ORIG_AGG_COMPARE="outputs/temporal_aggregation/${VIDEO_NAME}_original_agg_compare.mp4"
ORIG_AGG_RESULTS="outputs/temporal_aggregation/${VIDEO_NAME}_original_agg_results.json"

STAB_AGG_PROB="outputs/temporal_aggregation/${VIDEO_NAME}_stabilised_agg_prob.mp4"
STAB_AGG_MASK="outputs/temporal_aggregation/${VIDEO_NAME}_stabilised_agg_mask.mp4"
STAB_AGG_COMPARE="outputs/temporal_aggregation/${VIDEO_NAME}_stabilised_agg_compare.mp4"
STAB_AGG_RESULTS="outputs/temporal_aggregation/${VIDEO_NAME}_stabilised_agg_results.json"

echo "=== Motion compensation ==="
python scripts/run_motion_compensation.py \
  --input "${INPUT_VIDEO}" \
  --output "${STAB_VIDEO}" \
  --comparison-output "${STAB_COMPARE}" \
  --results "${STAB_RESULTS}"

echo "=== Segmentation: original video ==="
python scripts/run_video_segmentation.py \
  --input "${INPUT_VIDEO}" \
  --checkpoint "${CHECKPOINT}" \
  --output-overlay "${ORIG_OVERLAY}" \
  --output-mask "${ORIG_MASK}" \
  --output-prob "${ORIG_PROB}"

echo "=== Segmentation: stabilised video ==="
python scripts/run_video_segmentation.py \
  --input "${STAB_VIDEO}" \
  --checkpoint "${CHECKPOINT}" \
  --output-overlay "${STAB_OVERLAY}" \
  --output-mask "${STAB_MASK}" \
  --output-prob "${STAB_PROB}"

echo "=== Temporal aggregation: original video ==="
python scripts/run_temporal_aggregation.py \
  --input-prob-video "${ORIG_PROB}" \
  --output-prob-video "${ORIG_AGG_PROB}" \
  --output-mask-video "${ORIG_AGG_MASK}" \
  --output-comparison-video "${ORIG_AGG_COMPARE}" \
  --results "${ORIG_AGG_RESULTS}" \
  --window-size 5 \
  --threshold 0.5

echo "=== Temporal aggregation: stabilised video ==="
python scripts/run_temporal_aggregation.py \
  --input-prob-video "${STAB_PROB}" \
  --output-prob-video "${STAB_AGG_PROB}" \
  --output-mask-video "${STAB_AGG_MASK}" \
  --output-comparison-video "${STAB_AGG_COMPARE}" \
  --results "${STAB_AGG_RESULTS}" \
  --window-size 5 \
  --threshold 0.5

echo "=== Done ==="
echo "Motion compensation outputs: outputs/motion_compensation/"
echo "Segmentation outputs: outputs/video_inference/"
echo "Temporal aggregation outputs: outputs/temporal_aggregation/"