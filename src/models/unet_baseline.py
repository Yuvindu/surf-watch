import torch.nn as nn
import segmentation_models_pytorch as smp


class SegmentationOutput:
    """Lightweight wrapper that mimics the `.logits` attribute returned by
    HuggingFace's SegformerForSemanticSegmentation, so the training engine
    (src/training/engine.py) can call any model as `model(pixel_values=...).logits`
    without needing model-specific branches.
    """

    def __init__(self, logits):
        self.logits = logits


class UNetForSemanticSegmentation(nn.Module):
    """U-Net (segmentation_models_pytorch) wrapped to match the SegFormer
    baseline's calling convention, so it drops into the existing training
    engine, loss functions, and metrics with no changes required there.
    """

    def __init__(
        self,
        encoder_name: str = "resnet34",
        num_classes: int = 2,
        encoder_weights: str = "imagenet",
    ):
        super().__init__()
        self.unet = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=3,
            classes=num_classes,
            activation=None,  # raw logits, matching SegFormer's un-activated output
        )

    def forward(self, pixel_values):
        logits = self.unet(pixel_values)
        return SegmentationOutput(logits)


def build_unet_model(
    encoder_name: str = "resnet34",
    num_classes: int = 2,
    encoder_weights: str = "imagenet",
) -> nn.Module:
    return UNetForSemanticSegmentation(
        encoder_name=encoder_name,
        num_classes=num_classes,
        encoder_weights=encoder_weights,
    )
