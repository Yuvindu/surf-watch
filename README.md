# SurfWatch

SurfWatch is a semantic video segmentation project focused on detecting rip-current regions in beach footage. The current project direction is to build a leakage-safe machine learning pipeline around the RipVIS dataset and evaluate models on fully unseen videos rather than mixed frame samples.

## Project Goal

The goal of SurfWatch is to support rip-current detection from video in a way that is practical for real-world beach conditions and honest about model limitations. At this stage, the project is centered on:

- preparing a reproducible dataset pipeline
- converting RipVIS instance masks into a single semantic `rip` class
- training and validating baseline segmentation models
- extending the baseline into a motion-aware MARSP pipeline
- evaluating only on held-out videos
- documenting decisions, risks, and assumptions clearly

## Dataset And Evaluation Strategy

SurfWatch uses the official RipVIS train/validation/test split. The split is defined at the video level, which means all frames, annotations, and derived masks from a source video must remain in the same partition.

This choice matters because frame-level random splitting can create:

- frame leakage between train and evaluation sets
- temporal leakage from near-duplicate consecutive frames
- misleadingly optimistic metrics
- imbalances across viewpoints, durations, and rip-current types

The current planned split sizes are:

- Train: 112 videos
- Validation: 36 videos
- Test: 36 videos

Further detail is documented in [docs/split-strategy.md](/Users/rashmikecaldera/Developer/curtin/CSP/surfwatch/docs/split-strategy.md).

## Current Project Decisions

The repo currently records a few key decisions that shape the project:

- SurfWatch is the project name
- delivery follows 1-week Scrum sprints
- the official RipVIS split is preserved at the video level for all experiments
- RipVIS instance annotations are converted into binary semantic segmentation masks for SurfWatch
- local quantitative evaluation uses the validation split because the public test split does not include local labels
- baseline training uses SegFormer for binary rip-current segmentation
- U-Net ResNet34 is the second trained segmentation model; its best checkpoint is selected by validation IoU
- MARSP work extends the baseline with motion compensation and temporal aggregation

See [docs/decision-log.md](/Users/rashmikecaldera/Developer/curtin/CSP/surfwatch/docs/decision-log.md) for the running decision history.

## Risks And Constraints

The main risks identified so far are:

- insufficient model performance on unseen beach conditions
- scope creep across ML and app work
- demo instability from environment or model issues
- data leakage or biased evaluation from incorrect splitting

Current mitigations include conservative scoping, validation on held-out data, stable demo preparation, and strict preservation of source-video partitions throughout preprocessing and evaluation.

See [docs/risk-register.md](/Users/rashmikecaldera/Developer/curtin/CSP/surfwatch/docs/risk-register.md) for the detailed register.

## Literature Context

The project is informed by background review material on:

- RipVIS as the core dataset and benchmark context
- rip-current detection work relevant to mobile or deployable use cases
- motion-aware processing considerations for video-based prediction stability

Supporting review documents are stored here:

- [docs/lit-review/ripvis-lit-review.pdf](/Users/rashmikecaldera/Developer/curtin/CSP/surfwatch/docs/lit-review/ripvis-lit-review.pdf)
- [docs/lit-review/RipFinder_mobile_lit-Review.pdf](/Users/rashmikecaldera/Developer/curtin/CSP/surfwatch/docs/lit-review/RipFinder_mobile_lit-Review.pdf)

## Repository Structure

Key project areas currently include:

- `configs/`: training configuration
- `scripts/`: dataset conversion, training, inference, motion compensation, temporal aggregation, and MARSP pipeline runners
- `src/data/`: dataset loading
- `src/models/`: baseline model definition
- `src/preprocessing/`: motion compensation logic
- `src/postprocessing/`: temporal aggregation logic
- `src/training/`: training loops, metrics, losses, and utilities
- `docs/`: decision log, risk register, split strategy, and experiment notes

## Setup

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

## Dataset Layout

The default local setup expects the RipVIS dataset as a sibling directory to this repository:

```text
../RipVIS/
```

If using a different location, set environment variables before running scripts:

```bash
export RIPVIS_ROOT=/path/to/RipVIS
export PROCESSED_ROOT=/path/to/surfwatch/data/processed
```

Further detail for cloud use is documented in [docs/cloud-training-setup.md](/Users/rashmikecaldera/Developer/curtin/CSP/surfwatch/docs/cloud-training-setup.md).

## Preparing Semantic Segmentation Masks

Convert RipVIS instance annotations into SurfWatch semantic masks:

```bash
python scripts/convert_ripvis_to_semantic.py --split train
python scripts/convert_ripvis_to_semantic.py --split val
```

You can then verify the dataloader:

```bash
python scripts/test_dataloader.py
```

## Baseline Training

Train the SegFormer baseline with an isolated artifact directory:

```bash
python scripts/train_baseline.py \
  --ripvis-root ../RipVIS \
  --processed-root data/processed \
  --run-dir outputs/training/segformer_reproduction_02 \
  --image-size 512 \
  --batch-size 4 \
  --epochs 5 \
  --learning-rate 1e-4 \
  --weight-decay 1e-4 \
  --seed 42
```

This produces:

- an immutable checkpoint for every completed epoch
- explicit best and last checkpoint aliases
- validation prediction comparisons
- an incrementally written training manifest containing configuration,
  environment, dataset annotation hashes, metric history, smoke-inference
  evidence, and checkpoint hashes

The command refuses to use a non-empty run directory, preventing a later run
from silently replacing an earlier checkpoint. The original Phase 1 epoch-4
checkpoint is no longer available, so its metrics remain historical evidence;
the controlled replacement procedure is documented in
[docs/segformer-reproduction-training.md](docs/segformer-reproduction-training.md).

The replacement run completed on 23 September 2026 and selected epoch 4 at
validation IoU 0.7565. Its checkpoint and complete run artifacts were verified
after local transfer, and the checkpoint is installed at the runtime path
`checkpoints/best_model.pt`. The corrected source-resolution comparison achieved
foreground IoU 0.5027 for SegFormer and 0.4201 for U-Net; see
[docs/experiments/segformer_reproduction_02.md](docs/experiments/segformer_reproduction_02.md).

## U-Net ResNet34 Training

Train the second segmentation model with the matched RipVIS protocol:

```bash
python scripts/train_unet_resnet34.py \
  --ripvis-root ../RipVIS \
  --processed-root data/processed \
  --image-size 512 \
  --batch-size 4 \
  --epochs 20 \
  --encoder-weights imagenet \
  --seed 42
```

The completed RTX 4090 run selected epoch 10 by validation IoU. Its best validation metrics were IoU `0.7407`, Dice `0.8311`, precision `0.9034`, and recall `0.7828`. The compatible runtime checkpoint is stored at `checkpoints/unet_resnet34_best_model.pt`.

The training pipeline records its full configuration, metric history, runtime, environment, and adapter smoke check in `unet_resnet34_training_manifest.json`. See [docs/unet-resnet34-training.md](/Users/rashmikecaldera/Developer/curtin/CSP/surfwatch/docs/unet-resnet34-training.md) for reproduction instructions and [docs/experiments/unet_resnet34_cloud_run_01.md](/Users/rashmikecaldera/Developer/curtin/CSP/surfwatch/docs/experiments/unet_resnet34_cloud_run_01.md) for the completed experiment.

## Running Motion Compensation

Run the feature-based partial affine motion compensation prototype on a sample video:

```bash
python scripts/run_motion_compensation.py \
  --input ../RipVIS/train/videos/RipVIS-051.mp4 \
  --output outputs/motion_compensation/RipVIS-051_stabilised.mp4 \
  --comparison-output outputs/motion_compensation/RipVIS-051_comparison.mp4 \
  --results outputs/motion_compensation/RipVIS-051_results.json
```

## Running Video Segmentation Inference

Run frame-level segmentation inference on a video and save overlay, binary mask, and probability outputs:

```bash
python scripts/run_video_segmentation.py \
  --input ../RipVIS/train/videos/RipVIS-051.mp4 \
  --model segformer \
  --checkpoint checkpoints/best_model.pt \
  --output-overlay outputs/video_inference/RipVIS-051_original_overlay.mp4 \
  --output-mask outputs/video_inference/RipVIS-051_original_mask.mp4 \
  --output-prob outputs/video_inference/RipVIS-051_original_prob.mp4 \
  --threshold 0.5
```

The model-agnostic interface also supports U-Net with a ResNet34 encoder:

```bash
python scripts/run_video_segmentation.py \
  --input ../RipVIS/train/videos/RipVIS-051.mp4 \
  --model unet-resnet34 \
  --checkpoint checkpoints/unet_resnet34_best_model.pt \
  --output-overlay outputs/video_inference/RipVIS-051_unet_overlay.mp4 \
  --output-mask outputs/video_inference/RipVIS-051_unet_mask.mp4 \
  --output-prob outputs/video_inference/RipVIS-051_unet_prob.mp4 \
  --threshold 0.5
```

The trained checkpoint at `checkpoints/unet_resnet34_best_model.pt` makes U-Net available through the Analyse page and the shared baseline/MARSP comparison flow. If that ignored local artifact is absent on another machine, the model registry reports U-Net as unavailable until the checkpoint is restored.

## Held-Out Model Evaluation

Compare registered models against the same labelled RipVIS validation frames:

```bash
python scripts/evaluate_held_out_models.py \
  --run-name segformer-unet-val \
  --ripvis-root ../RipVIS \
  --processed-root data/processed \
  --model-checkpoint segformer=checkpoints/best_model.pt \
  --model-checkpoint unet-resnet34=checkpoints/unet_resnet34_best_model.pt
```

The evaluator writes dataset-level JSON plus frame- and video-level CSV files,
including model, checkpoint, dataset, parameter, and code-revision provenance.
See [docs/held-out-model-evaluation.md](/Users/rashmikecaldera/Developer/curtin/CSP/surfwatch/docs/held-out-model-evaluation.md)
for metric definitions and experiment rules.

## Running Temporal Aggregation

Run temporal aggregation over a probability video:

```bash
python scripts/run_temporal_aggregation.py \
  --input-prob-video outputs/video_inference/RipVIS-051_original_prob.mp4 \
  --output-prob-video outputs/temporal_aggregation/RipVIS-051_original_agg_prob.mp4 \
  --output-mask-video outputs/temporal_aggregation/RipVIS-051_original_agg_mask.mp4 \
  --output-comparison-video outputs/temporal_aggregation/RipVIS-051_original_agg_compare.mp4 \
  --results outputs/temporal_aggregation/RipVIS-051_original_agg_results.json \
  --window-size 5 \
  --threshold 0.5 \
  --motion-results outputs/motion_compensation/RipVIS-051_results.json
```

## Running The Integrated MARSP Pipeline

Run the full motion-aware SurfWatch pipeline end to end:

```bash
python scripts/run_marsp_pipeline.py \
  --video-name RipVIS-051 \
  --input ../RipVIS/train/videos/RipVIS-051.mp4 \
  --model segformer \
  --checkpoint checkpoints/best_model.pt \
  --window-size 5 \
  --threshold 0.5
```

This pipeline currently performs:

1. motion compensation
2. frame-level segmentation inference on original and stabilised video
3. motion-adaptive temporal aggregation
4. generation of stage-wise outputs and a pipeline summary JSON


Outputs are written under:

- `outputs/motion_compensation/`
- `outputs/video_inference/`
- `outputs/temporal_aggregation/`
- `outputs/marsp/`

## Running Baseline vs MARSP Comparison

Run the baseline-vs-MARSP comparison workflow on a single video:

```bash
python scripts/run_baseline_vs_marsp_compare.py \
  --video-name RipVIS-051 \
  --input ../RipVIS/train/videos/RipVIS-051.mp4 \
  --checkpoint checkpoints/best_model.pt \
  --window-size 5 \
  --threshold 0.5
```

This workflow currently performs:

1. baseline SegFormer inference on the original video
2. full MARSP processing on the same video
3. rendering of baseline and MARSP overlay outputs on the original video frames
4. generation of a side-by-side comparison video
5. computation of temporal stability metrics for baseline and MARSP outputs
6. generation of a single JSON comparison summary with side-by-side metric values

Outputs are written under:

- `outputs/comparisons/`

Typical artifacts include:

- `<video_name>_baseline_overlay.mp4`
- `<video_name>_baseline_mask.mp4`
- `<video_name>_marsp_overlay.mp4`
- `<video_name>_baseline_vs_marsp.mp4`
- `<video_name>_baseline_vs_marsp_metrics.json`

## Running Multi-Model Comparisons

Run the same baseline and MARSP comparison for multiple registered model
adapters with one command:

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

Each model runs independently. A failed adapter is recorded in the final
summary without removing artifacts generated by the other adapters. Outputs
are grouped by run, model, and workflow under `outputs/multi_model/`, with
aggregate JSON and CSV summaries at the run root.

See [docs/multi-model-comparison.md](docs/multi-model-comparison.md) for the
CLI contract, output structure, failure behaviour, and a complete example.

## Running The Web Demo

The React frontend runs against a local Python backend adapter that wraps the baseline-vs-MARSP comparison workflow.

Start the backend from the repository root in one terminal:

```bash
source .venv/bin/activate
python -m backend.server
```

The backend listens at:

```text
http://127.0.0.1:8000
```

Start the frontend in a second terminal:

```bash
cd frontend
npm install
npm run dev
```

Open the app at:

```text
http://localhost:5173/
```

Use `http://localhost:5173/analyse` to go straight to the upload screen. Do not open `http://127.0.0.1:8000` in the browser for the app; that is the backend API server and `GET /` returns `404` by design.

If the frontend shows a blank page after dependency changes, clear Vite's optimized dependency cache and restart:

```bash
cd frontend
rm -rf node_modules/.vite
npm run dev -- --force
```

See [docs/frontend-backend-comparison-flow.md](/Users/rashmikecaldera/Developer/curtin/CSP/surfwatch/docs/frontend-backend-comparison-flow.md) for the full startup steps, API contract, artifact locations, and troubleshooting notes.

## Status


SurfWatch now has:

- a working semantic segmentation dataset pipeline
- trained SegFormer-B0 and U-Net ResNet34 models
- a model-agnostic segmentation interface with adapters for both models
- motion compensation and temporal aggregation prototypes
- an integrated MARSP pipeline runner for end-to-end experimentation

The current stage is a repeatable multi-model comparison workflow for
SegFormer-B0 and U-Net ResNet34, with each model tested both with and without
MARSP on the same input. The resulting model-scoped artifacts and aggregate
provenance summaries prepare the project for broader held-out evaluation and
ensemble fusion experiments. DeepLabV3+ remains an optional later extension.
