# Baseline Cloud Run 01

## Objective
Run the first full cloud-based baseline training experiment for SurfWatch using SegFormer on the RipVIS semantic segmentation dataset.

## Environment
- Platform: RunPod
- GPU: NVIDIA A40 (48 GB VRAM)
- Training device: CUDA
- Workspace layout:
  - `/workspace/surfwatch`
  - `/workspace/RipVIS`

## Model
- Architecture: SegFormer baseline
- Pretrained checkpoint: `nvidia/segformer-b0-finetuned-ade-512-512`
- Task: Binary semantic segmentation
  - `0 = background`
  - `1 = rip current`

## Dataset
- Source dataset: RipVIS
- Train split: full train split
- Validation split: full validation split
- Input size: `512 x 512`

## Configuration
- Batch size: `4`
- Epochs: `5`
- Learning rate: `1e-4`
- Weight decay: `1e-4`
- Num workers: `4`
- Train subset size: `None`
- Validation subset size: `None`
- Output directory: `outputs/predictions/baseline_cloud_run_01`

## Results
### Epoch 1
- train_loss: `0.0683`
- val_loss: `0.0917`
- val_iou: `0.7324`
- val_dice: `0.8242`
- val_precision: `0.8756`
- val_recall: `0.7866`

### Epoch 2
- train_loss: `0.0363`
- val_loss: `0.1010`
- val_iou: `0.7444`
- val_dice: `0.8350`
- val_precision: `0.8463`
- val_recall: `0.8244`

### Epoch 3
- train_loss: `0.0309`
- val_loss: `0.1098`
- val_iou: `0.7224`
- val_dice: `0.8152`
- val_precision: `0.8877`
- val_recall: `0.7678`

### Epoch 4
- train_loss: `0.0281`
- val_loss: `0.1036`
- val_iou: `0.7551`
- val_dice: `0.8437`
- val_precision: `0.8574`
- val_recall: `0.8312`

### Epoch 5
- train_loss: `0.0260`
- val_loss: `0.1121`
- val_iou: `0.7456`
- val_dice: `0.8354`
- val_precision: `0.8900`
- val_recall: `0.7957`

## Best checkpoint
The best validation result was obtained at **epoch 4**.

- best val_iou: `0.7551`
- best val_dice: `0.8437`
- best val_precision: `0.8574`
- best val_recall: `0.8312`

Checkpoint:
- `checkpoints/best_model.pt`

## Interpretation
The full cloud baseline run produced stable validation performance across 5 epochs and established a strong frame-level segmentation baseline for SurfWatch. The model showed slightly higher precision than recall in several epochs, suggesting somewhat conservative rip-current predictions, which aligns with earlier qualitative inspection showing mild under-segmentation in some cases.

## Outputs
- Best checkpoint: `checkpoints/best_model.pt`
- Last checkpoint: `checkpoints/last_model.pt`
- Comparison images: `outputs/predictions/baseline_cloud_run_01/`

## Next steps
- Review saved comparison outputs
- Use this run as the baseline reference for future motion-aware improvements
- Compare later MARSP-style experiments against this checkpoint and metric set