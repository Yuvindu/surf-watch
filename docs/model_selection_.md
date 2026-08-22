# SurfWatch — Model Selection Rationale

## Objective

Semantic segmentation of surf/ocean scenes to identify scene classes such as water, waves, sky, beach, rocks, and surfers. Selection criteria: segmentation accuracy, training complexity, inference speed, and framework compatibility.

---

## Primary Model: SegFormer-B2

**Architecture:** Hierarchical transformer encoder (Mix Transformer — MiT-B2) + lightweight MLP decoder \[[1](#ref1)\]

**Why SegFormer-B2:**

- **Accuracy:** ~46.5% mIoU on ADE20K (single-scale) \[[1](#ref1)\], which covers outdoor/nature scene classes (sky, water, sand, vegetation) directly relevant to surf scenes.
- **Multi-scale context:** The hierarchical encoder captures both fine-grained wave texture and large-scale sky/water regions without positional encoding interpolation artifacts — a known failure mode for other vision transformers \[[1](#ref1)\].
- **Speed/accuracy balance:** B2 sits at the practical midpoint of the SegFormer family. B0/B1 sacrifice too much accuracy; B4/B5 add latency without proportional gain for MVP purposes \[[1](#ref1)\].
- **Pretrained weights:** `nvidia/mit-b2` (ImageNet-1k → ADE20K) available via HuggingFace `transformers`; fine-tuning on a surf-specific dataset requires minimal additional infrastructure \[[2](#ref2)\].
- **Framework support:** First-class HuggingFace support (`SegformerForSemanticSegmentation`) with PyTorch backend, enabling straightforward training loops and export \[[2](#ref2)\].

**Pretrained checkpoint:** `nvidia/segformer-b2-finetuned-ade-512-512` \[[2](#ref2)\]

---

## Baseline Comparison Model: DeepLabV3+ (ResNet-50 backbone)

**Architecture:** Encoder-decoder with atrous spatial pyramid pooling (ASPP) + ResNet-50 backbone \[[3](#ref3)\]

**Why DeepLabV3+ as baseline:**

- **Established benchmark:** DeepLabV3+ is the canonical CNN-based semantic segmentation baseline; comparing against it quantifies the benefit of the transformer approach \[[3](#ref3)\].
- **Strong edge preservation:** Atrous convolutions and the decoder module preserve water/sky/beach boundary sharpness, making it a meaningful (not trivially weak) comparison \[[3](#ref3)\].
- **Availability:** Available directly from `torchvision.models.segmentation.deeplabv3_resnet50` with COCO-pretrained weights — zero additional dependencies \[[4](#ref4)\].
- **Training complexity:** Lower than SegFormer; easier to reproduce and diagnose during MVP iteration \[[1](#ref1)\]\[[3](#ref3)\].

**Pretrained checkpoint:** `torchvision` — `deeplabv3_resnet50(weights=DeepLabV3_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1)` \[[4](#ref4)\]

---

## Comparison Summary

| Criterion              | SegFormer-B2 (Primary)          | DeepLabV3+-R50 (Baseline)        |
|------------------------|----------------------------------|-----------------------------------|
| Architecture           | Transformer encoder, MLP decoder \[[1](#ref1)\] | CNN encoder-decoder + ASPP \[[3](#ref3)\] |
| ADE20K mIoU (SS)       | ~46.5% \[[1](#ref1)\]          | ~44.1% (ResNet-50 backbone) \[[5](#ref5)\] |
| Inference speed (GPU)  | ~25 FPS @ 512×512 (approx.) \[[1](#ref1)\] | ~35 FPS @ 512×512 \[[5](#ref5)\] |
| Parameters             | 27.4 M \[[1](#ref1)\]          | 42.0 M \[[4](#ref4)\]           |
| Training complexity    | Medium (higher GPU memory req.) \[[1](#ref1)\] | Low (mature, stable training) \[[3](#ref3)\] |
| Pretrained source      | HuggingFace (`nvidia/mit-b2`) \[[2](#ref2)\] | torchvision (COCO) \[[4](#ref4)\] |
| Surf scene suitability | High (ADE20K covers water/sky) \[[1](#ref1)\] | Good (atrous conv for multi-scale) \[[3](#ref3)\] |

> **Note:** The ADE20K mIoU of ~44.1% for DeepLabV3+ is reported with a ResNet-50 backbone; published results using ResNet-101 are marginally higher (~44.17%) \[[5](#ref5)\]. The ResNet-50 variant used here is chosen for training simplicity and lower parameter count.

---

## Alignment with SurfWatch Objective

SurfWatch requires reliable pixel-level scene understanding across variable lighting, wave states, and camera angles. SegFormer-B2's transformer encoder better generalises across these distribution shifts compared to fixed-receptive-field CNNs \[[1](#ref1)\], which is the primary reason it is selected as the MVP model over DeepLabV3+.

DeepLabV3+-ResNet50 serves as the baseline to demonstrate this improvement quantitatively on the SurfWatch evaluation set \[[3](#ref3)\].

> **Dataset note:** Both models will be fine-tuned on a surf-specific dataset. The target dataset for fine-tuning should be defined and documented before training begins to fully satisfy dataset alignment criteria.

---

## Not Selected

**FCN (Fully Convolutional Network):** Excluded \[[6](#ref6)\]. Lacks contextual aggregation (no ASPP or attention), produces spatially inconsistent predictions, and is outperformed by 15–30 mIoU points on all relevant benchmarks \[[1](#ref1)\]\[[3](#ref3)\]\[[6](#ref6)\]. Not a meaningful comparison partner for a 2025 system.

---

## References

<a id="ref1"></a>**[1]** Xie, E., Wang, W., Yu, Z., Anandkumar, A., Alvarez, J. M., & Luo, P. (2021). *SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers.* NeurIPS 2021.
<https://arxiv.org/abs/2105.15203>

<a id="ref2"></a>**[2]** NVIDIA. *SegFormer-B2 fine-tuned on ADE20K (512×512).* HuggingFace Model Hub.
<https://huggingface.co/nvidia/segformer-b2-finetuned-ade-512-512>

<a id="ref3"></a>**[3]** Chen, L.-C., Zhu, Y., Papandreou, G., Schroff, F., & Adam, H. (2018). *Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation (DeepLabV3+).* ECCV 2018.
<https://arxiv.org/abs/1802.02611>

<a id="ref4"></a>**[4]** PyTorch / torchvision. *deeplabv3_resnet50 — Torchvision documentation.*
<https://pytorch.org/vision/main/models/generated/torchvision.models.segmentation.deeplabv3_resnet50.html>

<a id="ref5"></a>**[5]** Databricks. *Setting a Baseline for Image Segmentation Speedups.* (2024).
<https://www.databricks.com/blog/behind-the-scenes>

<a id="ref6"></a>**[6]** Long, J., Shelhamer, E., & Darrell, T. (2015). *Fully Convolutional Networks for Semantic Segmentation.* CVPR 2015.
<https://arxiv.org/abs/1411.4038>
