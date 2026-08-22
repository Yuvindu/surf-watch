from configs.baseline_config import BaselineConfig
from src.models.model_registry import (
    SUPPORTED_SEGMENTATION_MODELS,
    get_segmentation_model_option,
)
from src.models.segformer_adapter import SegFormerSegmentationAdapter
from src.models.segmentation_interface import SegmentationModelAdapter
from src.models.unet_resnet34_adapter import UnetResNet34SegmentationAdapter


_ADAPTER_BUILDERS = {
    "segformer": SegFormerSegmentationAdapter,
    "unet-resnet34": UnetResNet34SegmentationAdapter,
}


def build_segmentation_adapter(
    model_name: str,
    checkpoint_path: str,
    device,
    config: BaselineConfig,
    threshold: float,
) -> SegmentationModelAdapter:
    option = get_segmentation_model_option(model_name)
    builder = _ADAPTER_BUILDERS.get(option.id)
    if builder is None:
        raise RuntimeError(f"No adapter builder is registered for '{option.id}'.")

    return builder(
        checkpoint_path=checkpoint_path,
        device=device,
        config=config,
        threshold=threshold,
    )
