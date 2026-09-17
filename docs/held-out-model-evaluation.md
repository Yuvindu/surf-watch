# Held-Out Model Evaluation

## Purpose

`scripts/evaluate_held_out_models.py` compares registered segmentation models on
the same labelled RipVIS validation frames. It complements the video-level
baseline-versus-MARSP workflow by measuring semantic accuracy against ground
truth before temporal processing or ensemble fusion is introduced.

The public RipVIS test split has no public ground-truth masks. Quantitative
metrics therefore use the official video-separated validation split; test
videos remain available for qualitative inference.

## Run the matched evaluation

From the repository root with the project environment active:

```bash
python scripts/evaluate_held_out_models.py \
  --run-name segformer-unet-val \
  --ripvis-root ../RipVIS \
  --processed-root data/processed \
  --model-checkpoint segformer=checkpoints/best_model.pt \
  --model-checkpoint unet-resnet34=checkpoints/unet_resnet34_best_model.pt \
  --threshold 0.5 \
  --device auto
```

For a quick smoke test, restrict the run with `--max-frames 8`. Use one or more
`--video RipVIS-001` arguments to evaluate complete selected validation videos.
Do not use a truncated run as formal experimental evidence.

## Outputs

Each run writes under `outputs/held_out_evaluation/<run-name>/`:

- `held_out_evaluation_summary.json`: dataset, checkpoint, code revision,
  exact evaluator/metric file hashes, working-tree state, parameters, runtime,
  and dataset-level results
- `held_out_frame_metrics.csv`: one row per frame and model
- `held_out_video_metrics.csv`: confusion-derived and macro metrics grouped by
  source video and model

Foreground IoU, Dice, precision, and recall are reported separately from
class-mean scores. Frames with no ground-truth and no predicted foreground have
undefined foreground overlap and are excluded from macro foreground averages.
Empty-ground-truth and empty-prediction counts are reported explicitly so an
all-background model cannot appear successful through background accuracy.
Predictions are resized by each adapter to the source-frame dimensions and are
scored against the source-resolution semantic mask. The adapter's internal
model input size is retained in its metadata.

## Reproducibility rules

- Run every model with the same frame selection and threshold.
- Keep the official video-level split intact.
- Use the best checkpoint selected during training, not a later epoch chosen
  after viewing these results.
- Preserve the generated JSON and CSV files with each formal experiment.
- Treat `--max-frames` as a software check only unless the subset protocol was
  specified before evaluation.

The next experiment layer can use the same aligned frame selection for
equal-weight and validation-tuned probability fusion.

The first complete matched evaluation is recorded in
`docs/experiments/held_out_model_evaluation_01.md`.
