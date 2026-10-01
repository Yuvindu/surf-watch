# MARSP Ablation Evaluation

## Purpose

SCRUM-87 measures which model and MARSP components contribute to semantic
quality, temporal stability, fragmentation, and runtime. Every configuration
uses the same ordered RipVIS validation samples, checkpoints, threshold policy,
and metric implementation.

The experiment reuses component probability maps within each video. This keeps
the configurations paired and avoids treating repeated model inference as an
algorithmic difference.

## Predeclared Matrix

| ID | Model path | Temporal or post-processing | Reference | Purpose |
| --- | --- | --- | --- | --- |
| A0 | SegFormer | None | A0 | Established frame-level reference |
| A1 | U-Net ResNet34 | None | A0 | Isolate architecture choice |
| A2 | Equal-weight ensemble | None | A0 | Isolate probability fusion |
| A3 | Selected best path | Fixed temporal smoothing | Selected path raw result | Measure smoothing alone |
| A4 | Selected best path | Motion-adaptive weighting | Selected path raw result | Measure motion feedback |
| A5 | Selected best path | Small-component cleanup | Selected path raw result | Measure precision-fragmentation trade-off |
| A6 | Selected best path | Motion-adaptive weighting and cleanup | Selected path raw result | Evaluate the combined source-coordinate temporal path |

SegFormer is the default selected path for A3-A6 because the formal SCRUM-86
evaluation found it strongest overall. The CLI can select U-Net or the ensemble
for a separately named follow-up run.

## Coordinate-Alignment Boundary

Quantitative semantic metrics are computed in source-frame coordinates. Motion
is estimated between consecutive ordered sampled images and is converted into
the same motion scores used by MARSP. Those scores control temporal window size
and neighbour influence, but the prediction maps are not geometrically warped.

RipVIS sampling cadence varies by video. The predeclared maximum gap is six
source frames. A larger gap starts a new segment: temporal aggregation and
consecutive-mask stability metrics never cross that boundary. Sparse frames
still contribute to semantic metrics, and their excluded temporal pairs are
counted in the artifacts. The five-frame window counts labelled samples rather
than a fixed duration, so temporal metrics should be interpreted with the
recorded frame gaps, not as time-normalised stability.

This is intentional. The available ground-truth masks align with the original
sampled frames. Scoring stabilised predictions directly against those masks
would compare different coordinate systems. Geometric stabilisation must only
be included in labelled scoring after a verified transform path maps each
prediction or mask into the same coordinate system.

## Metrics

Semantic results are reported per frame, video, and dataset:

- foreground IoU, Dice, precision, and recall;
- background IoU, mean IoU, and mean Dice;
- foreground fraction and empty-prediction counts;
- mean foreground probability and binary confidence.

Temporal and fragmentation results are reported per video and macro-averaged:

- consecutive-frame IoU and Dice;
- pixel change rate;
- mask-area standard deviation and mean absolute area change;
- connected-component count and small-component count.

Runtime is separated into inference, motion estimation, and post-processing.
The estimated independent runtime for each configuration includes only the
stages that configuration requires, even though inference is reused during the
actual experiment run.
Existing run directories are not overwritten; choose a new run name for each
repeat or parameter change.

## Smoke Test

Use a small selection before a formal GPU run:

```bash
python scripts/evaluate_marsp_ablation.py \
  --run-name scrum87-smoke \
  --ripvis-root ../RipVIS \
  --processed-root data/processed \
  --model-checkpoint segformer=checkpoints/best_model.pt \
  --model-checkpoint unet-resnet34=checkpoints/unet_resnet34_best_model.pt \
  --selected-model-path segformer \
  --window-size 5 \
  --threshold 0.5 \
  --min-component-area 500 \
  --max-frames 8 \
  --device auto
```

Omitting `--model-weight` selects equal weighting. Explicit weights must be
equal because A2 is predeclared as the equal-weight ensemble.

## Formal Run

Run all 4,349 labelled validation frames from a clean committed revision on a
GPU host:

```bash
python scripts/evaluate_marsp_ablation.py \
  --run-name marsp-ablation-val-full \
  --ripvis-root /workspace/RipVIS \
  --processed-root /workspace/surfwatch/data/processed \
  --model-checkpoint segformer=/workspace/surfwatch/checkpoints/best_model.pt \
  --model-checkpoint unet-resnet34=/workspace/surfwatch/checkpoints/unet_resnet34_best_model.pt \
  --model-weight segformer=1 \
  --model-weight unet-resnet34=1 \
  --selected-model-path segformer \
  --window-size 5 \
  --threshold 0.5 \
  --min-component-area 500 \
  --device cuda
```

Do not tune the window, threshold, or cleanup area after inspecting the formal
run and then report the same run as independent evidence. Parameter sensitivity
must use a separately named experiment with its selection boundary documented.

## Artifacts

Each run writes:

- `marsp_ablation_summary.json`, containing the matrix, aggregate results,
  paired deltas, motion evidence, runtime, checkpoints, parameters, data
  selection, file hashes, and Git state;
- `marsp_ablation_frame_metrics.csv`, containing one row per configuration and
  labelled frame;
- `marsp_ablation_video_metrics.csv`, containing semantic, temporal, and runtime
  summaries for each configuration and video.

Generated outputs remain excluded from Git. Copy the complete run directory to
persistent or local storage before stopping a cloud instance.
