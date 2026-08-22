from dataclasses import dataclass
from typing import Any, Protocol, Tuple

import numpy as np


@dataclass(frozen=True)
class SegmentationModelMetadata:
    name: str
    input_size: Tuple[int, int]
    threshold: float
    device: str
    output_format: str = "probability_map_and_binary_mask"


@dataclass(frozen=True)
class SegmentationResult:
    probability_map: np.ndarray
    binary_mask: np.ndarray
    metadata: SegmentationModelMetadata


class SegmentationModelAdapter(Protocol):
    def metadata(self) -> SegmentationModelMetadata:
        """Return model identity and output contract metadata."""

    def preprocess_frame(self, frame: np.ndarray) -> Any:
        """Convert a BGR uint8 video frame into model-ready input."""

    def predict_probability_map(self, frame: np.ndarray) -> np.ndarray:
        """Return a float32 rip-current probability map aligned to the source frame."""

    def probability_to_mask(self, probability_map: np.ndarray) -> np.ndarray:
        """Convert a probability map into a uint8 binary mask using the adapter threshold."""

    def predict(self, frame: np.ndarray) -> SegmentationResult:
        """Run preprocessing, inference, and thresholding for one BGR frame."""
