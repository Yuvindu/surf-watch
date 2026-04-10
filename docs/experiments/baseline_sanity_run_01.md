# Baseline Sanity Run 01

## Objective
Validate that the SurfWatch baseline training pipeline works end to end using the RipVIS semantic segmentation dataset and SegFormer baseline model.

## Setup
- Model: SegFormer baseline
- Task: Binary semantic segmentation (`background` vs `rip current`)
- Dataset:
  - Train split: RipVIS train with converted semantic masks
  - Validation split: RipVIS val with converted semantic masks
- Input size: `512 x 512`
- Batch size: `2`
- Epochs: `1`
- Train subset size: `200`
- Validation subset size: `50`

## Purpose
This was a sanity-check run intended to confirm:
- the dataloader works correctly
- the model can train without crashing
- validation metrics can be computed
- checkpoints can be saved
- prediction outputs can be exported

This run was **not intended to serve as a final benchmark**.

## Results
- Train loss: `0.3239`
- Validation loss: `0.1929`
- Validation IoU: `0.6989`
- Validation Dice: `0.7999`

## Outputs produced
- Checkpoint saved: `checkpoints/best_model.pt`
- Sample predictions saved under: `outputs/predictions/`

## Qualitative observation
The initial prediction captured the approximate location of the rip-current region but under-segmented the full ground-truth area and simplified the target shape into smaller connected components. This suggests that the pipeline is functioning correctly and that the model is learning meaningful spatial cues, although segmentation quality is still coarse and requires further training.

## Notes
- The SegFormer decoder classification head was reinitialized for 2 classes, which is expected because the pretrained checkpoint was originally trained for a different label space.
- The public RipVIS test split remains excluded from local quantitative evaluation because public ground-truth annotations are not available.
- This sanity run is suitable as evidence that the baseline pipeline is operational, but not as a final performance report.

## Next steps
- Inspect more validation prediction/ground-truth pairs
- Run a larger controlled baseline experiment
- Compare qualitative improvements across more epochs or a larger subset