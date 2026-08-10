from configs.baseline_config import BaselineConfig
from src.models.segformer_adapter import SegFormerSegmentationAdapter
from src.models.segmentation_interface import SegmentationModelAdapter


SUPPORTED_SEGMENTATION_MODELS = ("segformer",)


def build_segmentation_adapter(
    model_name: str,
    checkpoint_path: str,
    device,
    config: BaselineConfig,
    threshold: float,
) -> SegmentationModelAdapter:
    normalized_name = model_name.strip().lower()

    if normalized_name == "segformer":
        return SegFormerSegmentationAdapter(
            checkpoint_path=checkpoint_path,
            device=device,
            config=config,
            threshold=threshold,
        )

    supported = ", ".join(SUPPORTED_SEGMENTATION_MODELS)
    raise ValueError(f"Unsupported segmentation model '{model_name}'. Supported models: {supported}")
