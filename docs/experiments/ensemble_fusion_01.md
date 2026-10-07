# Equal-Weight Ensemble Fusion 01

## Status

Completed on 23 September 2026. All component and ensemble evaluations
succeeded, and the JSON and CSV artifacts were copied from RunPod and
hash-verified locally.

## Objective

Evaluate whether equal-weight averaging of the source-aligned SegFormer-B0 and
U-Net ResNet34 probability maps improves binary rip-current segmentation before
thresholding.

## Configuration

- Dataset: official RipVIS validation split
- Coverage: 4,349 labelled frames from all 36 validation videos
- Evaluation resolution: source-frame resolution
- SegFormer weight: 0.5 normalized
- U-Net ResNet34 weight: 0.5 normalized
- Binary mask threshold: 0.5
- Hardware: NVIDIA GeForce RTX 4090, 24 GB VRAM
- PyTorch: 2.4.1 with CUDA 12.4 runtime
- Git revision: `1fd019f9ba0bc54e427559a68a48914556e705a2`
- Working tree: clean
- Validation annotation SHA-256: `98e3193fa751475ce44efe4340cfae8964a6ea3072dc25574c7ecd07eb5a7cf1`

Checkpoint identities:

| Model | Checkpoint SHA-256 |
|---|---|
| SegFormer-B0 | `3fca0f0fd315f306e07d33b21a1641b223a1e5c16a41abe7c121679cd89f7371` |
| U-Net ResNet34 | `b3a0be0a536fb6aa0e01b157d5e9f521e041a100629749f7c070d99bc4d8917b` |

## Command

```bash
python scripts/evaluate_held_out_models.py \
  --run-name segformer-unet-equal-fusion-val-full-20260923 \
  --ripvis-root /workspace/RipVIS \
  --processed-root /workspace/surfwatch/data/processed \
  --model-checkpoint \
    segformer=outputs/training/segformer_reproduction_02/checkpoints/segformer_best_model.pt \
  --model-checkpoint \
    unet-resnet34=/workspace/surfwatch/checkpoints/unet_resnet34_best_model.pt \
  --ensemble \
  --model-weight segformer=1 \
  --model-weight unet-resnet34=1 \
  --threshold 0.5 \
  --device cuda
```

## Dataset-Level Results

Micro metrics are derived from the confusion matrix accumulated over every
source-resolution validation pixel.

| Model | Foreground IoU | Foreground Dice | Precision | Recall | Mean IoU | Mean Dice | Empty predictions | Runtime (s) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| SegFormer-B0 | **0.5027** | **0.6690** | 0.7698 | **0.5916** | **0.7341** | **0.8258** | 1,305 | 316.60 |
| U-Net ResNet34 | 0.4201 | 0.5916 | 0.8633 | 0.4500 | 0.6919 | 0.7866 | 1,695 | 308.19 |
| Equal-weight ensemble | 0.4701 | 0.6395 | **0.8912** | 0.4987 | 0.7186 | 0.8114 | 1,882 | 345.54 |

The ensemble improves foreground IoU by 0.0500 over U-Net but remains 0.0326
below SegFormer. It produces the highest precision but lower recall than
SegFormer and the most empty predictions. Averaging at threshold 0.5 therefore
acts conservatively: agreement between models suppresses false positives but
also removes valid rip-current pixels detected strongly by only one model.

Mean foreground probabilities were 0.0538 for SegFormer, 0.0385 for U-Net, and
0.0461 for the ensemble. Mean binary confidence was 0.9830, 0.9891, and 0.9809
respectively. These descriptive confidence values are not calibration metrics.

## Per-Video Comparison

Using per-video micro foreground IoU:

- SegFormer-B0 is the unique winner on 18 videos.
- The equal-weight ensemble is the unique winner on 8 videos.
- U-Net ResNet34 is the unique winner on 5 videos.
- Five videos are tied.

The ensemble's largest improvements over the better component occur on
RipVIS-014, RipVIS-072, and RipVIS-090. Its eight per-video wins confirm that
the component outputs contain complementary information, even though the fixed
50/50 configuration is not the best dataset-wide model.

## Artifacts

The formal artifacts are stored locally under:

`outputs/held_out_evaluation/segformer-unet-equal-fusion-val-full-20260923/`

| Artifact | SHA-256 |
|---|---|
| JSON summary | `47767cd3cdb87abc75003c84f0ccee26f5633622957b8b525faae56a39b3973b` |
| Frame metrics CSV | `ba9c9db11c3ddedb952a35b3ceb5c29a409dd98a7ee35d100d15e70926babd33` |
| Video metrics CSV | `b2b5b4771f0b3b58e3332f27223018ec591de03008505231f998496b516cb273` |

Generated evaluation artifacts are intentionally excluded from Git. The
experiment record, implementation, command, checkpoint identities, and artifact
hashes are versioned.

## Limitations

- Validation data was used for checkpoint selection and is not an untouched
  public-test benchmark.
- The experiment evaluates only the predeclared 50/50 weights and threshold
  0.5; it does not establish whether calibrated or tuned fusion can outperform
  SegFormer.
- Selecting weights or a threshold from these results would be validation-set
  tuning and requires a separately documented evaluation boundary.
- This experiment measures frame-level semantic quality, not MARSP temporal
  stability.

## Decision

Retain SegFormer-B0 as the strongest standalone semantic model. Keep the 50/50
ensemble as the reproducible fusion baseline and evidence of per-video
complementarity. Any weighted-fusion or threshold-calibration follow-up must be
treated as a separate experiment rather than retroactively changing this run.
