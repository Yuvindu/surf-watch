# Held-Out Model Evaluation 01

## Status

Completed on 17 September 2026.

## Objective

Compare the fixed best SegFormer-B0 and U-Net ResNet34 checkpoints on exactly
the same labelled RipVIS validation frames before implementing probability
fusion. This experiment measures semantic segmentation accuracy independently
of MARSP temporal processing.

## Configuration

- Dataset: official RipVIS validation split
- Coverage: 4,349 labelled frames from all 36 validation videos
- Evaluation resolution: source-frame resolution
- Adapter input size: 512 x 512 for both models
- Binary mask threshold: 0.5
- Hardware: NVIDIA GeForce RTX 4090, 24 GB VRAM
- PyTorch: 2.4.1 with CUDA 12.4 runtime
- Git base revision: `38956c27512cd241f099bbb7e1c8c389ffdc0e1e`
- Working tree: dirty because the evaluator was retained uncommitted for review
- Evaluator SHA-256: `eee713880ee1de146f1c8291efd61dbfb69771e992911ea587764f7959b19488`
- Metric module SHA-256: `bd9ad7dd4d56fb7bdab35492b6c76074643bef0d07c3388b4707f00f61cddee2`
- RipVIS validation annotation SHA-256: `98e3193fa751475ce44efe4340cfae8964a6ea3072dc25574c7ecd07eb5a7cf1`

Checkpoint identities:

| Model | Selected epoch | Checkpoint SHA-256 |
|---|---:|---|
| SegFormer-B0 | 1 | `fceaf655fa33e9f8fb13094ca7fd6dcbe83dbc78cef0d7d9a3c4af3c8fc844b7` |
| U-Net ResNet34 | 10 | `b3a0be0a536fb6aa0e01b157d5e9f521e041a100629749f7c070d99bc4d8917b` |

## Command

```bash
python scripts/evaluate_held_out_models.py \
  --run-name segformer-unet-val-full-2026-09-17 \
  --ripvis-root /workspace/RipVIS \
  --processed-root /workspace/surfwatch/data/processed \
  --model-checkpoint segformer=/workspace/surfwatch/checkpoints/best_model.pt \
  --model-checkpoint unet-resnet34=/workspace/surfwatch/checkpoints/unet_resnet34_best_model.pt \
  --threshold 0.5 \
  --device cuda
```

## Dataset-Level Results

Micro metrics are calculated from the confusion matrix accumulated over every
source-resolution validation pixel.

| Model | Foreground IoU | Foreground Dice | Precision | Recall | Mean IoU | Mean Dice | Empty predictions | Runtime (s) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| SegFormer-B0 | 0.1007 | 0.1830 | 0.2717 | 0.1380 | 0.5151 | 0.5732 | 1,756 | 201.56 |
| U-Net ResNet34 | 0.4201 | 0.5916 | 0.8633 | 0.4500 | 0.6919 | 0.7866 | 1,695 | 224.92 |

The validation set contains 1,488 frames with empty ground-truth foreground.
Across all frames, the mean ground-truth foreground fraction is 0.0571.
SegFormer predicts a mean foreground fraction of 0.0375 and U-Net predicts
0.0368, indicating that both checkpoints tend to under-segment the rip class.

## Per-Video Comparison

Using per-video micro foreground IoU:

- U-Net ResNet34 performs better on 22 of 36 videos.
- SegFormer-B0 performs better on 7 of 36 videos.
- The models tie on 7 videos, including cases with no measurable foreground
  overlap.

U-Net therefore provides the strongest single-model result, but SegFormer wins
on a non-trivial subset. The largest SegFormer advantages occur on RipVIS-067,
RipVIS-128, RipVIS-137, and RipVIS-039. This scene-dependent reversal provides
evidence that probability-level fusion is worth evaluating rather than simply
discarding SegFormer.

## Interpretation

U-Net is the stronger model overall and is especially precise, but its recall
of 0.4500 and lower predicted foreground fraction show that it still misses a
substantial portion of labelled rip regions. SegFormer's lower precision and
recall make it unsuitable as the preferred standalone model under this
protocol, while its per-video wins suggest potentially complementary outputs.

These source-resolution foreground metrics must not be compared directly with
the validation values stored in the training checkpoints. Training reported
class-mean metrics on masks resized to 512 x 512; this experiment reports both
foreground-specific and class-mean metrics after predictions are mapped back to
the original frame resolution.

## Limitations

- The official validation split was also used for checkpoint selection, so this
  is held-out from training but is not an untouched public test benchmark.
- Public RipVIS test labels are unavailable for local quantitative scoring.
- A fixed 0.5 threshold was used for both models; no threshold tuning or
  calibration was performed.
- The experiment evaluates semantic accuracy only and does not measure MARSP
  temporal stability.
- Empty-ground-truth frames are excluded from undefined macro foreground
  overlap averages and are reported separately.

## Artifacts

The complete JSON summary, 8,698 frame/model rows, and 72 video/model rows are
stored locally under:

`outputs/held_out_evaluation/segformer-unet-val-full-2026-09-17/`

The same artifacts remain on the persistent RunPod volume under:

`/workspace/surfwatch/outputs/held_out_evaluation/segformer-unet-val-full-2026-09-17/`

## Decision

Proceed with SCRUM-86 using equal-weight probability fusion as the first
ensemble baseline. Compare the ensemble against both component models on this
same validation selection before considering tuned weights or a third model.
