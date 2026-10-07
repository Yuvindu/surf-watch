# Multi-Model Probability Fusion

## Purpose

SCRUM-86 evaluates whether the complementary SegFormer-B0 and U-Net ResNet34
outputs improve semantic segmentation when fused before thresholding. The
baseline ensemble uses a weighted arithmetic mean of source-aligned rip-current
probability maps:

`fused_probability = sum(model_probability * model_weight) / sum(model_weight)`

The binary mask is created only after fusion. Component masks are not averaged
or combined, because doing so would discard probability information.

## Validation Rules

- At least two unique registered component models are required.
- Omitted weights select the reproducible equal-weight baseline.
- If any explicit weight is supplied, every component must have exactly one.
- Weights must be finite, non-negative, and have a positive total.
- Every component probability map must be finite, bounded by 0 and 1, and
  aligned to the same source-frame dimensions.
- The ensemble threshold must be between 0 and 1.

Invalid configuration or component output fails the ensemble clearly. A
component model can still retain its own evaluation result when another model
or the ensemble fails.

## Smoke Test

Use a small labelled subset to validate software and checkpoint compatibility:

```bash
python scripts/evaluate_held_out_models.py \
  --run-name scrum86-real-checkpoint-smoke \
  --ripvis-root ../RipVIS \
  --processed-root data/processed \
  --model-checkpoint segformer=checkpoints/best_model.pt \
  --model-checkpoint unet-resnet34=checkpoints/unet_resnet34_best_model.pt \
  --ensemble \
  --model-weight segformer=1 \
  --model-weight unet-resnet34=1 \
  --threshold 0.5 \
  --max-frames 8 \
  --device auto
```

Omit both `--model-weight` arguments to select the same equal-weight baseline.
A partial explicit weight list is rejected rather than silently assigning a
default to the missing model.

## Formal Evaluation

The formal experiment uses all 4,349 labelled frames from all 36 official
RipVIS validation videos:

```bash
python scripts/evaluate_held_out_models.py \
  --run-name segformer-unet-equal-fusion-val-full \
  --ripvis-root ../RipVIS \
  --processed-root data/processed \
  --model-checkpoint segformer=checkpoints/best_model.pt \
  --model-checkpoint unet-resnet34=checkpoints/unet_resnet34_best_model.pt \
  --ensemble \
  --model-weight segformer=1 \
  --model-weight unet-resnet34=1 \
  --threshold 0.5 \
  --device cuda
```

Run formal evidence from a clean, committed Git revision on suitable GPU
hardware. Preserve the generated JSON and CSV files before stopping or deleting
the cloud instance.

## Outputs and Provenance

The existing held-out evaluator adds `ensemble` rows to the same artifacts as
the component models:

- dataset-level JSON summary;
- frame-level CSV metrics;
- video-level CSV metrics.

The JSON records the fusion method, raw and normalized weights, threshold,
component checkpoint paths and SHA-256 hashes, data selection, annotation hash,
Git revision, source-file hashes, runtime, failures, and adapter metadata.
Frame, video, and dataset summaries include foreground and class-mean metrics,
empty-prediction counts, mean foreground probability, and mean binary
confidence. Confidence is recorded for later calibration work but is not
treated as calibrated probability evidence in this experiment.

## Interpretation Boundary

The validation split was used for checkpoint selection, so this is a matched
local validation experiment rather than an untouched public-test benchmark.
The equal-weight run tests a predeclared baseline. Any later weight or threshold
tuning must be documented separately and must not be reported as independent
held-out performance on the same validation data.

## Completed Baseline

The formal equal-weight evaluation completed on 23 September 2026. It achieved
foreground IoU 0.4701, compared with 0.5027 for SegFormer and 0.4201 for U-Net.
The ensemble was the unique per-video winner on 8 of 36 videos but did not beat
SegFormer overall. See
[`experiments/ensemble_fusion_01.md`](experiments/ensemble_fusion_01.md) for the
full metrics, provenance, artifact hashes, and interpretation.
