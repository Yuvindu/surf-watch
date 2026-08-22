from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class SegmentationModelOption:
    id: str
    label: str
    description: str


SEGMENTATION_MODEL_OPTIONS = (
    SegmentationModelOption(
        id="segformer",
        label="SegFormer",
        description="Transformer-based binary rip-current segmentation baseline.",
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
    return [asdict(option) for option in SEGMENTATION_MODEL_OPTIONS]
