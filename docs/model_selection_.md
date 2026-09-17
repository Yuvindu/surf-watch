# SurfWatch Model Selection Rationale

## Current objective

SurfWatch performs **binary semantic segmentation** of beach imagery and video frames. Each pixel is classified as either rip current or background. The segmentation model supplies frame-level probability maps to the model-agnostic MARSP temporal processing layer, which applies motion compensation, temporal aggregation, thresholding, and post-processing.

The Phase 2 research question is broader than selecting a single high-scoring model: it asks whether MARSP improves temporal stability and detection quality consistently across different segmentation architectures, and whether complementary models can later be combined through ensemble fusion.

## Evolution from the Phase 1 plan

The Phase 1 proposal considered SegFormer-B2 as the primary model and DeepLabV3+ as a conventional convolutional baseline. That document described intended work rather than completed experiments.

The implemented Phase 2 system currently uses:

- **SegFormer-B0**, retained from the original SurfWatch pipeline.
- **U-Net with a ResNet34 encoder**, added through the model-agnostic segmentation interface.

DeepLabV3+ has not been trained or evaluated in SurfWatch. It remains a possible third architecture if the initial two-model experiments show that additional architectural diversity is needed.

## Implemented models

### SegFormer-B0

SegFormer-B0 is a compact transformer-based semantic segmentation model. It remains the reference model because it is already integrated into the pipeline and has an established SurfWatch checkpoint and evaluation record.

The best recorded full validation run achieved:

| Metric | Value |
|---|---:|
| IoU | 0.7551 |
| Dice | 0.8437 |
| Precision | 0.8574 |
| Recall | 0.8312 |

The detailed run record is available in [`experiments/baseline_cloud_run_01.md`](experiments/baseline_cloud_run_01.md).

### U-Net ResNet34

U-Net ResNet34 provides a convolutional encoder-decoder architecture with skip connections. It was selected as the second model because it differs structurally from SegFormer while remaining practical to train and deploy on the available dataset and hardware.

The best checkpoint from the first full cloud training run was produced at epoch 10 and achieved:

| Metric | Value |
|---|---:|
| IoU | 0.7407 |
| Dice | 0.8311 |
| Precision | 0.9034 |
| Recall | 0.7828 |

The detailed run record is available in [`experiments/unet_resnet34_cloud_run_01.md`](experiments/unet_resnet34_cloud_run_01.md).

## Initial comparison

| Property | SegFormer-B0 | U-Net ResNet34 |
|---|---|---|
| Architecture family | Transformer-based encoder | Convolutional encoder-decoder |
| Validation IoU | 0.7551 | 0.7407 |
| Validation Dice | 0.8437 | 0.8311 |
| Validation precision | 0.8574 | 0.9034 |
| Validation recall | 0.8312 | 0.7828 |
| Current role | Reference model | Second model and ensemble candidate |

SegFormer currently has the stronger IoU, Dice, and recall, while U-Net has higher precision. This suggests potentially useful complementary behaviour, but the frame-level results alone do not establish whether an ensemble will improve video-level performance. Both models must first be evaluated under the same baseline and MARSP configurations on the same held-out videos.

## DeepLabV3+ status

DeepLabV3+ is no longer the immediate next training target. Training another model before completing the controlled SegFormer-versus-U-Net evaluation would increase compute and experimental scope without answering the current model-agnostic MARSP question.

DeepLabV3+ may be added later if:

- the two-model ensemble lacks sufficient diversity;
- a third architecture is required for a broader ablation study; or
- the research evaluation specifically benefits from an atrous-convolution baseline.

## Planned evaluation sequence

1. Run SegFormer-B0 and U-Net ResNet34 on the same held-out video set.
2. Evaluate each model with and without MARSP using identical thresholds, temporal parameters, and post-processing settings.
3. Compare segmentation quality, temporal stability, runtime, and per-video failure cases.
4. Implement multi-model experiment orchestration and provenance reporting.
5. Evaluate ensemble fusion using the two trained models.
6. Decide whether DeepLabV3+ would add enough research value to justify training it.

This sequence keeps model selection evidence-driven and separates the effect of the segmentation architecture from the effect of MARSP temporal processing.

## Evaluation boundary

Quantitative model selection must use the documented validation or held-out video split and must not tune against the public test set. All reports should preserve checkpoint identity, dataset split, preprocessing, threshold, MARSP parameters, and code revision so results can be reproduced.

## References

- Xie, E. et al. (2021). *SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers*. https://arxiv.org/abs/2105.15203
- Ronneberger, O., Fischer, P., and Brox, T. (2015). *U-Net: Convolutional Networks for Biomedical Image Segmentation*. https://arxiv.org/abs/1505.04597
- Chen, L.-C. et al. (2018). *Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation*. https://arxiv.org/abs/1802.02611
