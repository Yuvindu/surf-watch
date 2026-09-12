# U-Net ResNet34 Training

## Purpose

This workflow trains the second SurfWatch segmentation model using the same RipVIS video-level train and validation splits and binary mask semantics as the SegFormer baseline. U-Net inputs use ImageNet normalization during both training and inference.

## Full GPU Run

Set the dataset paths and run from the repository root:

```bash
export RIPVIS_ROOT=/workspace/RipVIS
export PROCESSED_ROOT=/workspace/surfwatch/data/processed

python scripts/train_unet_resnet34.py \
  --ripvis-root /workspace/RipVIS \
  --processed-root /workspace/surfwatch/data/processed \
  --image-size 512 \
  --batch-size 4 \
  --num-workers 4 \
  --epochs 20 \
  --learning-rate 0.0001 \
  --weight-decay 0.0001 \
  --encoder-weights imagenet \
  --threshold 0.5 \
  --seed 42 \
  --checkpoint-dir outputs/unet_resnet34/checkpoints \
  --output-dir outputs/unet_resnet34/predictions
```

Adjust batch size to available GPU memory. Keep the seed, split, preprocessing, and metric definitions fixed when comparing U-Net with SegFormer.

## Completed Run

The first full run completed on 2 September 2026 using one NVIDIA RTX 4090 with PyTorch 2.8.0 and CUDA 12.8.

| Property | Value |
|---|---:|
| Training frames | 14,616 |
| Validation frames | 4,349 |
| Epochs | 20 |
| Runtime | 3,325.4 seconds (55m 25s) |
| Best epoch | 10 |
| Validation loss | 0.1366 |
| Validation IoU | 0.7407 |
| Validation Dice | 0.8311 |
| Validation precision | 0.9034 |
| Validation recall | 0.7828 |

Validation IoU peaked at epoch 10. Training loss continued to fall afterward while validation loss generally increased, so the widening loss gap indicates overfitting. Best-checkpoint selection retained epoch 10 rather than the final epoch. The complete epoch history and interpretation are recorded in [experiments/unet_resnet34_cloud_run_01.md](experiments/unet_resnet34_cloud_run_01.md).

## Pipeline Smoke Run

Use a tiny subset to verify the training and checkpoint path before allocating a full GPU run:

```bash
python scripts/train_unet_resnet34.py \
  --ripvis-root ../RipVIS \
  --processed-root data/processed \
  --image-size 64 \
  --batch-size 2 \
  --num-workers 0 \
  --epochs 1 \
  --train-subset-size 2 \
  --val-subset-size 2 \
  --encoder-weights none \
  --checkpoint-dir /tmp/surfwatch-unet-smoke/checkpoints \
  --output-dir /tmp/surfwatch-unet-smoke/predictions
```

The smoke run verifies execution and artifact compatibility only. Its metrics are not research results.

## Artifacts And Provenance

The archived full run contains:

- `outputs/unet_resnet34/checkpoints/unet_resnet34_best_model.pt`: epoch-10 checkpoint selected by validation IoU.
- `outputs/unet_resnet34/checkpoints/unet_resnet34_last_model.pt`: final-epoch recovery checkpoint.
- `outputs/unet_resnet34/checkpoints/unet_resnet34_training_manifest.json`: configuration, environment, dataset counts, epoch history, best metrics, runtime, checkpoint path, and adapter smoke-inference evidence.
- `outputs/unet_resnet34/predictions/`: four labelled validation prediction samples.
- `logs/unet_resnet34_training.log`: complete console metric history.

The best checkpoint is also installed at `checkpoints/unet_resnet34_best_model.pt`, which is the default path used by the model registry, backend, CLI, and frontend. Generated checkpoints and outputs are intentionally ignored by Git. The best checkpoint SHA-256 is `b3a0be0a536fb6aa0e01b157d5e9f521e041a100629749f7c070d99bc4d8917b`.

Both checkpoints contain `model_state_dict`, optimiser state, epoch, metrics, model identifier, and training configuration. The best checkpoint therefore loads directly through the existing `unet-resnet34` inference adapter.

Post-training smoke inference on RunPod and local CPU inference both loaded the best checkpoint without adapter changes and returned `720 x 1280` probability maps and binary masks for a representative validation frame.

## Evaluation Boundary

Quantitative local reporting uses the labelled RipVIS validation split. The public test split remains reserved for qualitative held-out inference because its annotations are unavailable locally. Full baseline-versus-MARSP validation with the trained U-Net belongs to SCRUM-76.

Qualitative validation samples show that the model captures the main rip-current region but may under-segment boundaries and produce small isolated false-positive blobs. This is consistent with precision exceeding recall. Threshold calibration, small-blob cleanup, and broader held-out video evaluation remain follow-up work rather than changes to this training result.
