# U-Net ResNet34 Cloud Run 01

## Objective

Train and evaluate the second SurfWatch segmentation architecture using the same RipVIS split, binary mask semantics, preprocessing resolution, and validation metrics as the SegFormer baseline.

## Provenance

- Date: 2 September 2026
- Jira: SCRUM-85
- Git commit: `197bac8b99a584e28ef2a771a149f84ddd03840a`
- Branch: `codex/SCRUM-85-unet-training`
- Platform: RunPod
- GPU: NVIDIA RTX 4090, 24 GB VRAM
- Python: 3.11.10
- PyTorch: 2.8.0
- CUDA: 12.8
- `segmentation-models-pytorch`: 0.5.0

## Model And Dataset

- Architecture: U-Net with ResNet34 encoder
- Encoder initialisation: ImageNet
- Output: one-channel binary segmentation logits
- Training frames: 14,616
- Validation frames: 4,349
- Input size: `512 x 512`
- Split boundary: official RipVIS video-level train and validation partitions

## Configuration

- Batch size: `4`
- Epochs: `20`
- Learning rate: `1e-4`
- Weight decay: `1e-4`
- Dataloader workers: `4`
- Threshold: `0.5`
- Seed: `42`
- Checkpoint criterion: highest validation IoU
- Training runtime: `3,325.4` seconds (`55m 25s`)

## Epoch History

| Epoch | Train loss | Val loss | Val IoU | Val Dice | Precision | Recall |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.1004 | 0.1116 | 0.7237 | 0.8167 | 0.8687 | 0.7790 |
| 2 | 0.0350 | 0.1289 | 0.7116 | 0.8051 | 0.9215 | 0.7428 |
| 3 | 0.0278 | 0.1343 | 0.7250 | 0.8173 | 0.9177 | 0.7589 |
| 4 | 0.0250 | 0.1531 | 0.7017 | 0.7962 | 0.8884 | 0.7424 |
| 5 | 0.0230 | 0.1422 | 0.7150 | 0.8083 | 0.9181 | 0.7477 |
| 6 | 0.0201 | 0.1531 | 0.6839 | 0.7785 | 0.9400 | 0.7088 |
| 7 | 0.0189 | 0.1262 | 0.7290 | 0.8215 | 0.8605 | 0.7910 |
| 8 | 0.0169 | 0.1463 | 0.7292 | 0.8213 | 0.8926 | 0.7739 |
| 9 | 0.0160 | 0.1646 | 0.7269 | 0.8191 | 0.9058 | 0.7655 |
| **10** | **0.0151** | **0.1366** | **0.7407** | **0.8311** | **0.9034** | **0.7828** |
| 11 | 0.0135 | 0.1842 | 0.7114 | 0.8050 | 0.9169 | 0.7440 |
| 12 | 0.0127 | 0.1692 | 0.7166 | 0.8099 | 0.8991 | 0.7560 |
| 13 | 0.0121 | 0.2009 | 0.7151 | 0.8087 | 0.8889 | 0.7584 |
| 14 | 0.0118 | 0.1984 | 0.7047 | 0.7989 | 0.9081 | 0.7393 |
| 15 | 0.0111 | 0.1882 | 0.7124 | 0.8058 | 0.9229 | 0.7432 |
| 16 | 0.0104 | 0.1924 | 0.7050 | 0.7992 | 0.9004 | 0.7421 |
| 17 | 0.0097 | 0.2213 | 0.7081 | 0.8020 | 0.9094 | 0.7427 |
| 18 | 0.0098 | 0.2040 | 0.7092 | 0.8030 | 0.9140 | 0.7424 |
| 19 | 0.0088 | 0.2598 | 0.6945 | 0.7891 | 0.9200 | 0.7246 |
| 20 | 0.0085 | 0.1957 | 0.6854 | 0.7801 | 0.9229 | 0.7140 |

## Selected Checkpoint

Epoch 10 produced the highest validation IoU.

- Validation IoU: `0.7407`
- Validation Dice: `0.8311`
- Validation precision: `0.9034`
- Validation recall: `0.7828`
- Validation loss: `0.1366`
- Runtime checkpoint: `checkpoints/unet_resnet34_best_model.pt`
- SHA-256: `b3a0be0a536fb6aa0e01b157d5e9f521e041a100629749f7c070d99bc4d8917b`

## Verification

- All 28 repository tests passed in the RunPod CUDA environment.
- The best checkpoint loaded through `UnetResNet34SegmentationAdapter` without code changes.
- RunPod post-training smoke inference returned a `720 x 1280` probability map and binary mask.
- Local CPU adapter inference reproduced the expected output dimensions.
- Four labelled validation comparisons were saved under `outputs/unet_resnet34/predictions/`.

## Interpretation And Limitations

U-Net ResNet34 reached performance close to the established SegFormer baseline while using the same evaluation boundary. Precision remained higher than recall, indicating conservative predictions and some under-segmentation. The saved comparisons show that the primary rip-current area is generally captured, with missed boundary regions and occasional small isolated false-positive blobs.

Training loss continued to improve after epoch 10, but validation loss increased and validation IoU declined to `0.6854` by epoch 20. This divergence is evidence of overfitting and justifies retaining the validation-IoU-selected checkpoint rather than the last checkpoint.

These are validation results, not labelled public-test results. The public RipVIS test annotations are unavailable locally, so the test split remains limited to qualitative held-out inference. Threshold calibration, small-blob cleanup, broader held-out video evaluation, and U-Net baseline-versus-MARSP evaluation are follow-up work.

## Artifacts

- Best and last checkpoints: `outputs/unet_resnet34/checkpoints/`
- Training manifest: `outputs/unet_resnet34/checkpoints/unet_resnet34_training_manifest.json`
- Prediction comparisons: `outputs/unet_resnet34/predictions/`
- Training log: `logs/unet_resnet34_training.log`

Generated model artifacts are intentionally excluded from Git. The experiment record, reproduction code, configuration, metric history, and checksum are versioned so the archived checkpoint can be verified after transfer.
