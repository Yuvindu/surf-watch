import torch.nn as nn


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_unet_resnet34_model(encoder_weights=None) -> nn.Module:
    try:
        import segmentation_models_pytorch as smp
    except ImportError as exc:
        raise RuntimeError(
            "U-Net support requires segmentation-models-pytorch. "
            "Install the project requirements before selecting unet-resnet34."
        ) from exc

    return smp.Unet(
        encoder_name="resnet34",
        encoder_weights=encoder_weights,
        in_channels=3,
        classes=1,
        activation=None,
    )
