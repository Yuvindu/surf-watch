# Multi-Model Comparison Workflow

## Purpose

`scripts/run_multi_model_comparison.py` runs the existing baseline-versus-MARSP
comparison once for every requested segmentation adapter. All models receive
the same input video, temporal window, and mask threshold so their outputs can
be compared without manual file renaming or directory preparation.

The workflow currently supports the registered `segformer` and
`unet-resnet34` adapters. Each adapter requires a compatible local checkpoint.

## Run a comparison

From the repository root with the project virtual environment active:

```bash
python scripts/run_multi_model_comparison.py \
  --run-name held-out-ripvis-002 \
  --video-name RipVIS-002 \
  --input ../RipVIS/test/videos/RipVIS-002.mp4 \
  --model-checkpoint segformer=checkpoints/best_model.pt \
  --model-checkpoint unet-resnet34=checkpoints/unet_resnet34_best_model.pt \
  --window-size 5 \
  --threshold 0.5
```

Repeat `--model-checkpoint MODEL=CHECKPOINT` once per adapter. Model names must
be registered in `src/models/model_registry.py`, and a model may only appear
once in a run. `--run-name` and `--video-name` accept letters, numbers, dots,
underscores, and hyphens.

Use `--output-root <directory>` to change the default `outputs/multi_model`
root. Use `--reuse-existing` only when the individual comparison runner can
validate that existing artifacts match the selected input, model, checkpoint,
and parameters.

## Output structure

```text
outputs/multi_model/<run-name>/
|-- multi_model_comparison_summary.json
|-- multi_model_comparison_summary.csv
|-- segformer/
|   |-- baseline/
|   |-- marsp/
|   |   |-- motion_compensation/
|   |   |-- video_inference/
|   |   |-- temporal_aggregation/
|   |   `-- summary/
|   `-- comparison/
`-- unet-resnet34/
    |-- baseline/
    |-- marsp/
    |   |-- motion_compensation/
    |   |-- video_inference/
    |   |-- temporal_aggregation/
    |   `-- summary/
    `-- comparison/
```

The aggregate JSON and CSV record the input identity, shared parameters,
requested/successful/failed models, checkpoint identity, runtimes, frame
counts, baseline and MARSP metrics, deltas, and artifact paths.

## Failure behaviour

Models run as independent subprocesses. A missing checkpoint, non-zero runner
exit, missing metrics file, or invalid metrics JSON marks only that model as
failed. Remaining adapters still run, successful artifacts remain intact, and
the summaries contain the failure reason. The command exits with status `1`
when any model fails so automated experiments can detect a partial result.

## Reproducibility notes

- Use one run name per experiment configuration.
- Record the command and aggregate summary with each formal experiment.
- SegFormer inference constructs its architecture locally before loading the
  checkpoint, so inference does not require a Hugging Face network request.
- Local smoke-test runtimes are operational evidence only; use controlled
  hardware and held-out inputs for scientific timing comparisons.

See `docs/experiments/multi_model_comparison_smoke_01.md` for the first
end-to-end validation of this workflow.
