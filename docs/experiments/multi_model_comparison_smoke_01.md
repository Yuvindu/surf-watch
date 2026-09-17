# Multi-Model Comparison Smoke Test 01

## Objective

Validate that SCRUM-77 can run SegFormer-B0 and U-Net ResNet34 against the same
video, preserve separate baseline and MARSP artifacts, and produce one
machine-readable comparison summary.

This is an integration smoke test. The short unlabelled test excerpt and local
CPU runtime are not sufficient for a model-quality conclusion.

## Configuration

- Date: 2026-09-17
- Run name: `scrum77-smoke-ripvis002`
- Input: first 24 frames of held-out test video `RipVIS-002`
- Resolution and frame rate: 1280 x 720 at 30 FPS
- Models: `segformer`, `unet-resnet34`
- Window size: 5
- Threshold: 0.5
- Device: local CPU

## Results

| Model | Status | Frames | Baseline time (s) | MARSP time (s) |
|---|---:|---:|---:|---:|
| SegFormer-B0 | Success | 24 | 8.32 | 20.93 |
| U-Net ResNet34 | Success | 24 | 5.90 | 16.12 |

Both adapters generated complete model-scoped artifact sets and were included
in the aggregate JSON and CSV. SegFormer produced non-empty predictions, while
U-Net's baseline mask was empty throughout this short excerpt. MARSP introduced
a small non-empty U-Net region after stabilised inference and aggregation.
This difference is retained as useful failure-case evidence rather than being
treated as a quality improvement.

On this excerpt, SegFormer MARSP decreased consecutive IoU by `0.0079` and
increased small-blob count from `0.0833` to `0.7917`. U-Net's empty baseline
made its baseline temporal IoU `1.0`, demonstrating why temporal stability
metrics must be interpreted alongside semantic accuracy and mask occupancy.

## Acceptance evidence

- Both models processed the same 24 input frames without manual renaming.
- Each model produced separate `baseline/`, `marsp/`, and `comparison/`
  directories.
- Aggregate JSON and CSV summaries recorded model, checkpoint, input, runtime,
  frame count, metrics, and artifact paths.
- Automated tests cover partial failure reporting without aborting subsequent
  model runs.
- SegFormer completed from its local checkpoint without a model download.

## Follow-up

Run the workflow across a stratified held-out set and combine temporal metrics
with annotation-based IoU, Dice, precision, recall, and mask occupancy. Use
those results to choose ensemble candidates and avoid interpreting empty-mask
stability as good performance.
