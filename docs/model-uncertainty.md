# Confidence and Model-Uncertainty Evaluation

SCRUM-88 measures descriptive probability reliability for the fixed SegFormer-B0,
U-Net ResNet34, and equal-weight ensemble checkpoints. It uses the same ordered,
source-aligned RipVIS validation frames for every model path. No temperature,
threshold, or ensemble weight is fitted by this evaluator.

## Metrics

For every labelled pixel, `p` is the predicted rip-current probability and `y`
is the binary ground truth. The evaluator reports:

- Brier score: mean `(p - y)^2` (lower is better).
- Binary negative log-likelihood: mean `-y log(p) - (1-y) log(1-p)`, with
  probabilities clipped to `[1e-7, 1-1e-7]` for numerical stability.
- Foreground expected calibration error (ECE): equal-width bins over the
  *foreground probability* `p` in `[0,1]`; for each non-empty bin, compare
  mean `p` with observed foreground frequency and weight the absolute gap by
  the bin's pixel count. `p=1` belongs to the last bin. The default is 15 bins.
- Mean binary predictive entropy in natural-log units. High entropy means the
  model's probability is near 0.5, not necessarily that the model is wrong.
- SegFormer-versus-U-Net disagreement: mean absolute probability difference
  and fraction of pixels whose masks differ at the predeclared threshold 0.5.

All aggregate scores pool pixels, rather than averaging frame or video scores.
Reliability-bin counts, mean probabilities, and empirical foreground frequencies
are retained so plots can be reproduced without saving full probability maps.
Empty bins have null means; an empty cohort has null scores rather than zero.

Whole-image metrics can look strong because beach frames contain extensive
background. Therefore every model also receives a **candidate-region** view:
pixels that are positive in the ground truth **or** have probability at least
the threshold in either component model. The region is identical for all three
model paths. It uses ground truth and both model outputs, so it is a diagnostic
cohort, **not** a deployable region proposal or an unbiased deployment metric.
Report both views and their pixel counts together.

## Smoke Run

Use the two verified checkpoints and the existing semantic masks:

```bash
python scripts/evaluate_model_uncertainty.py \
  --run-name scrum88-smoke \
  --ripvis-root ../RipVIS \
  --processed-root data/processed \
  --model-checkpoint segformer=checkpoints/best_model.pt \
  --model-checkpoint unet-resnet34=checkpoints/unet_resnet34_best_model.pt \
  --max-frames 1 \
  --device cpu
```

A one-frame local smoke run completed on 7 October 2026 with matching
checkpoint hashes and produced all four expected artifacts. Its numerical
scores are only a software check, not research evidence.

## Full Validation Run

Run from a clean committed revision with both checkpoints and the complete
RipVIS validation split available. Use a new run name for any repeat; existing
run directories are never overwritten.

```bash
python scripts/evaluate_model_uncertainty.py \
  --run-name scrum88-val-full \
  --ripvis-root ../RipVIS \
  --processed-root data/processed \
  --model-checkpoint segformer=checkpoints/best_model.pt \
  --model-checkpoint unet-resnet34=checkpoints/unet_resnet34_best_model.pt \
  --bins 15 \
  --threshold 0.5 \
  --device cuda
```

`--video` can restrict the smoke run to a named video; `--max-frames` limits
the ordered sample list. The evaluator loads each component model once, runs
both on each source frame, and fuses their aligned probability maps without
repeating inference. Results are written to
`outputs/uncertainty_evaluation/<run-name>/`:

- `uncertainty_summary.json`: aggregate and per-video scores, reliability
  bins, top disagreement frames, runtime, parameters, sample-list hash,
  checkpoint and annotation hashes, adapter metadata, and code provenance.
- `uncertainty_frame_metrics.csv`: per-frame and per-model scores in both
  cohorts, with paired model-disagreement metrics.
- `uncertainty_video_metrics.csv`: per-video and per-model summaries.
- `uncertainty_reliability_bins.csv`: aggregate bin data for reliability plots.

The official RipVIS validation split already influenced checkpoint selection.
These results must therefore be called **descriptive validation diagnostics**,
not independent estimates of calibration on unseen beaches. Fitting a
temperature or selecting weights from these outputs would be a different,
predeclared experiment with a video-disjoint assessment boundary. The public
test split has no released quantitative masks; do not claim quantitative test
performance from it.

The complete 7 October run and artifact hashes are recorded in the
[formal experiment record](experiments/model_uncertainty_01.md). Generated
artifacts remain excluded from Git and must be retained separately.
