from dataclasses import dataclass


@dataclass(frozen=True)
class SegmentationModelOption:
    id: str
    label: str
    description: str
    default_checkpoint: str


SEGMENTATION_MODEL_OPTIONS = (
    SegmentationModelOption(
        id="segformer",
        label="SegFormer",
        description="Transformer-based binary rip-current segmentation baseline.",
        default_checkpoint="checkpoints/best_model.pt",
    ),
    SegmentationModelOption(
        id="unet-resnet34",
        label="U-Net (ResNet34)",
        description="CNN encoder-decoder using a ResNet34 backbone.",
        default_checkpoint="checkpoints/unet_resnet34_best_model.pt",
    ),
)

SUPPORTED_SEGMENTATION_MODELS = tuple(option.id for option in SEGMENTATION_MODEL_OPTIONS)
DEFAULT_SEGMENTATION_MODEL = SEGMENTATION_MODEL_OPTIONS[0].id


def normalize_model_name(model_name: str) -> str:
    return model_name.strip().lower()


def get_segmentation_model_option(model_name: str) -> SegmentationModelOption:
    normalized_name = normalize_model_name(model_name)
    for option in SEGMENTATION_MODEL_OPTIONS:
        if option.id == normalized_name:
            return option

    supported = ", ".join(SUPPORTED_SEGMENTATION_MODELS)
    raise ValueError(
        f"Unsupported segmentation model '{model_name}'. Supported models: {supported}"
    )


def segmentation_model_options_payload() -> list[dict[str, str]]:
    return [
        {
            "id": option.id,
            "label": option.label,
            "description": option.description,
        }
        for option in SEGMENTATION_MODEL_OPTIONS
    ]
