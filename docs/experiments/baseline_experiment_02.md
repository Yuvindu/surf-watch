# Baseline Experiment 02

## Objective
Run a larger controlled baseline experiment for SurfWatch after the initial sanity run, in order to observe learning behavior across multiple epochs and obtain a more meaningful validation signal.

## Setup
- Model: SegFormer baseline
- Task: Binary semantic segmentation (`background` vs `rip current`)
- Dataset:
  - Train split: RipVIS train with converted semantic masks
  - Validation split: RipVIS val with converted semantic masks
- Input size: `512 x 512`
- Batch size: `2`
- Epochs: `3`
- Train subset size: `1000`
- Validation subset size: `200`

## Purpose
This run was intended to move beyond simple pipeline validation and evaluate whether the baseline model improves meaningfully over multiple epochs on a larger subset of the dataset.

## Results
### Epoch 1
- Train loss: `0.1623`
- Validation loss: `0.1563`
- Validation IoU: `0.6875`
- Validation Dice: `0.7908`

### Epoch 2
- Train loss: `0.0684`
- Validation loss: `0.0938`
- Validation IoU: `0.7571`
- Validation Dice: `0.8466`

### Epoch 3
- Train loss: `0.0535`
- Validation loss: `0.1545`
- Validation IoU: `0.7045`
- Validation Dice: `0.8045`

## Best epoch
The best validation performance was observed at **Epoch 2**:
- **Best Validation IoU:** `0.7571`
- **Best Validation Dice:** `0.8466`

## Interpretation
The baseline model improved clearly from epoch 1 to epoch 2, indicating that the training pipeline and model setup are able to learn meaningful segmentation behavior from the RipVIS semantic masks.

Validation performance then declined at epoch 3 while training loss continued to decrease. This suggests early overfitting on the current subset configuration. As a result, epoch 2 is the most representative checkpoint from this experiment.

## Qualitative observation
Qualitative inspection showed that the model captured the approximate location of rip-current regions, but predictions remained smoother and more compact than the ground-truth masks. In some cases, the predicted region was broken into smaller connected components rather than matching the full connected shape in the annotation.

## Outputs produced
- Best checkpoint saved under: `checkpoints/`
- Sample predictions saved under: `outputs/predictions/`

## Notes
- The pretrained SegFormer classifier head was reinitialized for 2 classes, which is expected because the original checkpoint was trained on a different label space.
- The public RipVIS test split remains excluded from local quantitative evaluation because public ground-truth annotations are not available.
- This experiment provides a stronger baseline reference than the initial sanity run, but it is still based on a subset rather than full split training.

## Conclusion
This run confirms that the SurfWatch baseline segmentation pipeline is functioning correctly and that SegFormer can learn meaningful rip-current segmentation behavior on RipVIS. The results also indicate that checkpoint selection based on validation performance is important, with epoch 2 outperforming epoch 3 despite lower training loss at the later stage.

## Next steps
- Inspect additional validation prediction vs ground-truth pairs
- Decide whether to scale to a larger subset or full train/validation run
- Consider adding stronger logging and experiment tracking
- Use this result as the baseline reference before introducing motion-aware improvements