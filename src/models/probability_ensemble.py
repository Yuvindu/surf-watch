from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

from src.models.segmentation_interface import (
    SegmentationModelAdapter,
    SegmentationModelMetadata,
    SegmentationResult,
)


def resolve_model_weights(
    model_names: Sequence[str],
    weight_specs: Sequence[tuple[str, float]] | None = None,
) -> dict[str, float]:
    names = [str(name).strip() for name in model_names]
    if len(names) < 2:
        raise ValueError("ensemble fusion requires at least two component models")
    if any(not name for name in names):
        raise ValueError("component model names cannot be empty")
    duplicates = sorted(name for name in set(names) if names.count(name) > 1)
    if duplicates:
        raise ValueError("component models must be unique: " + ", ".join(duplicates))

    if not weight_specs:
        return {name: 1.0 for name in names}

    weights: dict[str, float] = {}
    for model_name, raw_weight in weight_specs:
        name = str(model_name).strip()
        if name in weights:
            raise ValueError(f"duplicate model weight: {name}")
        weight = float(raw_weight)
        if not np.isfinite(weight):
            raise ValueError(f"model weight must be finite: {name}")
        if weight < 0.0:
            raise ValueError(f"model weight cannot be negative: {name}")
        weights[name] = weight

    expected = set(names)
    supplied = set(weights)
    missing = sorted(expected - supplied)
    unknown = sorted(supplied - expected)
    if unknown:
        raise ValueError("weights supplied for unknown models: " + ", ".join(unknown))
    if missing:
        raise ValueError("missing model weights: " + ", ".join(missing))
    if sum(weights.values()) <= 0.0:
        raise ValueError("at least one model weight must be greater than zero")
    return {name: weights[name] for name in names}


def normalize_model_weights(weights: Mapping[str, float]) -> dict[str, float]:
    resolved = resolve_model_weights(list(weights), list(weights.items()))
    total = sum(resolved.values())
    return {name: weight / total for name, weight in resolved.items()}


def fuse_probability_maps(
    probability_maps: Mapping[str, np.ndarray],
    weights: Mapping[str, float],
) -> np.ndarray:
    if set(probability_maps) != set(weights):
        missing = sorted(set(probability_maps) - set(weights))
        unknown = sorted(set(weights) - set(probability_maps))
        details = []
        if missing:
            details.append("missing weights for " + ", ".join(missing))
        if unknown:
            details.append("weights without probability maps for " + ", ".join(unknown))
        raise ValueError("probability maps and weights must match: " + "; ".join(details))

    normalized = normalize_model_weights(weights)
    expected_shape: tuple[int, int] | None = None
    fused: np.ndarray | None = None
    for model_name in weights:
        probability_map = probability_maps[model_name]
        if not isinstance(probability_map, np.ndarray):
            raise ValueError(f"probability map must be a numpy array: {model_name}")
        if probability_map.ndim != 2:
            raise ValueError(f"probability map must be 2D: {model_name}")
        if not np.all(np.isfinite(probability_map)):
            raise ValueError(f"probability map contains non-finite values: {model_name}")
        if probability_map.size and (
            float(probability_map.min()) < 0.0
            or float(probability_map.max()) > 1.0
        ):
            raise ValueError(f"probability map must be within [0, 1]: {model_name}")
        if expected_shape is None:
            expected_shape = probability_map.shape
            fused = np.zeros(expected_shape, dtype=np.float64)
        elif probability_map.shape != expected_shape:
            raise ValueError(
                "component probability maps must have matching source-aligned shapes"
            )
        fused += probability_map.astype(np.float64, copy=False) * normalized[model_name]

    if fused is None:
        raise ValueError("at least one probability map is required")
    return fused.astype(np.float32)


class ProbabilityEnsembleAdapter:
    def __init__(
        self,
        components: Mapping[str, SegmentationModelAdapter],
        weights: Mapping[str, float],
        threshold: float = 0.5,
    ) -> None:
        if set(components) != set(weights):
            raise ValueError("ensemble components and model weights must match")
        self.components = dict(components)
        self.weights = resolve_model_weights(
            list(self.components), list(weights.items())
        )
        self.normalized_weights = normalize_model_weights(self.weights)
        self.threshold = self._validate_threshold(threshold)

    def metadata(self) -> SegmentationModelMetadata:
        component_metadata = [adapter.metadata() for adapter in self.components.values()]
        input_sizes = {metadata.input_size for metadata in component_metadata}
        devices = {metadata.device for metadata in component_metadata}
        return SegmentationModelMetadata(
            name="Probability Ensemble (" + " + ".join(self.components) + ")",
            input_size=(
                next(iter(input_sizes)) if len(input_sizes) == 1 else (0, 0)
            ),
            threshold=self.threshold,
            device=next(iter(devices)) if len(devices) == 1 else "mixed",
        )

    def preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        return frame

    def predict_probability_map(self, frame: np.ndarray) -> np.ndarray:
        probability_maps = {
            model_name: adapter.predict_probability_map(frame)
            for model_name, adapter in self.components.items()
        }
        return fuse_probability_maps(probability_maps, self.weights)

    def probability_to_mask(self, probability_map: np.ndarray) -> np.ndarray:
        if probability_map.ndim != 2:
            raise ValueError("probability_map must be a 2D array")
        return (probability_map >= self.threshold).astype(np.uint8)

    def predict(self, frame: np.ndarray) -> SegmentationResult:
        probability_map = self.predict_probability_map(frame)
        return SegmentationResult(
            probability_map=probability_map,
            binary_mask=self.probability_to_mask(probability_map),
            metadata=self.metadata(),
        )

    @staticmethod
    def _validate_threshold(threshold: float) -> float:
        value = float(threshold)
        if not np.isfinite(value) or value < 0.0 or value > 1.0:
            raise ValueError("threshold must be between 0.0 and 1.0")
        return value
