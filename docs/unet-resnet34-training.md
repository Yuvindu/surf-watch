# U-Net ResNet34 Training

## Purpose

This workflow trains the second SurfWatch segmentation model using the same RipVIS video-level train and validation splits and binary mask semantics as the SegFormer baseline. U-Net inputs use ImageNet normalization during both training and inference.

## Full GPU Run

Set the dataset paths and run from the repository root:

```bash
export RIPVIS_ROOT=/workspace/RipVIS
export PROCESSED_ROOT=/workspace/surfwatch/data/processed

python scripts/train_unet_resnet34.py \
  --image-size 512 \
  --batch-size 4 \
  --epochs 20 \
  --learning-rate 0.0001 \
  --weight-decay 0.0001 \
  --encoder-weights imagenet \
  --seed 42
```

Adjust batch size to available GPU memory. Keep the seed, split, preprocessing, and metric definitions fixed when comparing U-Net with SegFormer.

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

The full run produces:

- `checkpoints/unet_resnet34_best_model.pt`: checkpoint selected by validation IoU.
- `checkpoints/unet_resnet34_last_model.pt`: final-epoch recovery checkpoint.
- `checkpoints/unet_resnet34_training_manifest.json`: configuration, environment, dataset counts, epoch history, best metrics, runtime, checkpoint path, and adapter smoke-inference evidence.
- `outputs/predictions/unet_resnet34/`: labelled validation prediction samples.

Both checkpoints contain `model_state_dict`, optimiser state, epoch, metrics, model identifier, and training configuration. The best checkpoint therefore loads directly through the existing `unet-resnet34` inference adapter.

## Evaluation Boundary

Quantitative local reporting uses the labelled RipVIS validation split. The public test split remains reserved for qualitative held-out inference because its annotations are unavailable locally. Full baseline-versus-MARSP validation with the trained U-Net belongs to SCRUM-76.
