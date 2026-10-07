# SegFormer Reproduction 02

## Status

Completed on 23 September 2026. The full training run, checkpoints, manifest,
and replacement held-out evaluation were copied from RunPod and verified
locally before the pod was stopped.

## Objective

Reproduce the deleted Phase 1 SegFormer-B0 checkpoint under the documented
five-epoch protocol, preserve stronger provenance, and replace the invalidated
SegFormer row in the first matched U-Net comparison.

## Training Configuration

- Dataset: official RipVIS train and validation splits
- Coverage: 14,616 training frames and 4,349 validation frames
- Model: `nvidia/segformer-b0-finetuned-ade-512-512`
- Input size: 512 x 512
- Batch size: 4
- Epochs: 5
- Optimizer: AdamW
- Learning rate: `1e-4`
- Weight decay: `1e-4`
- Seed: 42
- Hardware: NVIDIA GeForce RTX 5090, 32 GB VRAM
- PyTorch: 2.8.0 with CUDA 12.8 runtime
- Git revision: `4714edd44efe0030613f99428175b4893f096802`
- Working tree: clean
- Train annotation SHA-256: `d925b30bf2718169dc221994d855397739b38615555bd7a0685841a069969fec`
- Validation annotation SHA-256: `98e3193fa751475ce44efe4340cfae8964a6ea3072dc25574c7ecd07eb5a7cf1`

## Training Results

| Epoch | Train loss | Validation loss | IoU | Dice | Precision | Recall |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.0692 | 0.0905 | 0.7498 | 0.8390 | 0.8825 | 0.8054 |
| 2 | 0.0384 | 0.0945 | 0.7442 | 0.8349 | 0.8426 | 0.8275 |
| 3 | 0.0320 | 0.1175 | 0.7258 | 0.8180 | 0.9140 | 0.7611 |
| 4 | 0.0290 | 0.1053 | **0.7565** | **0.8449** | 0.8577 | **0.8331** |
| 5 | 0.0268 | 0.1116 | 0.7379 | 0.8288 | 0.8925 | 0.7846 |

Epoch 4 was selected by validation mean IoU. Its IoU of 0.7565 closely
reproduces and slightly exceeds the historical Phase 1 best value of 0.7551.
The complete run took 814.95 seconds.

The selected checkpoint is 44,845,613 bytes and has SHA-256:

`3fca0f0fd315f306e07d33b21a1641b223a1e5c16a41abe7c121679cd89f7371`

## Replacement Held-Out Evaluation

The reproduced checkpoint and the fixed U-Net ResNet34 checkpoint were scored
at source-frame resolution on the same 4,349 labelled frames from all 36
RipVIS validation videos. Both used a binary threshold of 0.5.

| Model | Foreground IoU | Foreground Dice | Precision | Recall | Mean IoU | Empty predictions | Runtime (s) |
|---|---:|---:|---:|---:|---:|---:|---:|
| SegFormer-B0 | **0.5027** | **0.6690** | 0.7698 | **0.5916** | **0.7341** | 1,305 | 190.60 |
| U-Net ResNet34 | 0.4201 | 0.5916 | **0.8633** | 0.4500 | 0.6919 | 1,695 | 202.53 |

SegFormer leads on foreground IoU for 19 of 36 videos, U-Net leads on 11, and
the models tie on 6. SegFormer has higher overlap and recall overall, while
U-Net remains more precise. This is evidence for testing probability-level
fusion because the models retain different error profiles.

Training-time validation metrics and this held-out evaluation are not directly
interchangeable. Training reports class-mean metrics on resized 512 x 512
masks; the matched evaluator reports foreground-specific and class-mean
metrics after mapping predictions to source-frame resolution.

## Artifact Verification

The training manifest successfully verified every epoch, best, and last
checkpoint after local transfer. The replacement evaluation artifacts also
matched their RunPod hashes:

| Artifact | SHA-256 |
|---|---|
| JSON summary | `3bed3c0bda77d7b10621045edebcd06f6956c86d077bd5232729f4cb4cb75487` |
| Frame metrics CSV | `ec54f73103ae4d779e1037abdfc94f2238e4de9351065a26e37c48b36399d09a` |
| Video metrics CSV | `8c2db2f5d826eb64409f38199b0c1d9e3b48ca36c6d34732e579dc68d6edfa0a` |

Generated artifacts are intentionally excluded from Git. They are stored
locally under:

- `outputs/training/segformer_reproduction_02/`
- `outputs/held_out_evaluation/matched-segformer-unet-val-full-20260923/`

The selected checkpoint is installed at the application's existing SegFormer
runtime path, `checkpoints/best_model.pt`. The unrelated epoch-1 artifact was
retained as `checkpoints/segformer_invalid_epoch1_model.pt` for provenance and
must not be used for performance reporting.

## Decision

Accept the reproduced epoch-4 checkpoint as the current SegFormer research
checkpoint. Supersede the invalid SegFormer performance row in
`held_out_model_evaluation_01.md` with this replacement run, and begin SCRUM-86
with equal-weight probability fusion before considering tuned weights or a
third architecture.
