from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal, Sequence

import cv2
import numpy as np

from src.evaluation.segmentation_metrics import mean_defined
from src.postprocessing.temporal_aggregation import (
    TemporalAggregationConfig,
    aggregate_probability_maps,
    aggregate_probability_maps_motion_adaptive,
    remove_small_components_from_masks,
    threshold_probability_maps,
)


TemporalMode = Literal["none", "fixed", "motion_adaptive"]
SUPPORTED_MODEL_PATHS = ("segformer", "unet-resnet34", "ensemble")


@dataclass(frozen=True)
class AblationConfiguration:
    experiment_id: str
    label: str
    model_path: str
    temporal_mode: TemporalMode
    window_size: int
    threshold: float
    min_component_area: int
    reference_id: str
    purpose: str

    def to_dict(self) -> dict:
        return asdict(self)


def build_ablation_matrix(
    *,
    selected_model_path: str = "segformer",
    window_size: int = 5,
    threshold: float = 0.5,
    min_component_area: int = 500,
) -> list[AblationConfiguration]:
    if selected_model_path not in SUPPORTED_MODEL_PATHS:
        raise ValueError(
            "selected_model_path must be one of: "
            + ", ".join(SUPPORTED_MODEL_PATHS)
        )
    if window_size < 1 or window_size % 2 == 0:
        raise ValueError("window_size must be a positive odd integer")
    if not np.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be between 0.0 and 1.0")
    if min_component_area < 1:
        raise ValueError("min_component_area must be greater than zero")

    selected_reference = {
        "segformer": "A0",
        "unet-resnet34": "A1",
        "ensemble": "A2",
    }[selected_model_path]
    common = {
        "window_size": window_size,
        "threshold": float(threshold),
    }
    return [
        AblationConfiguration(
            "A0",
            "SegFormer frame-level baseline",
            "segformer",
            "none",
            min_component_area=0,
            reference_id="A0",
            purpose="Reference frame-level baseline",
            **common,
        ),
        AblationConfiguration(
            "A1",
            "U-Net ResNet34 frame-level baseline",
            "unet-resnet34",
            "none",
            min_component_area=0,
            reference_id="A0",
            purpose="Alternative frame-level architecture",
            **common,
        ),
        AblationConfiguration(
            "A2",
            "Equal-weight ensemble",
            "ensemble",
            "none",
            min_component_area=0,
            reference_id="A0",
            purpose="Isolate multi-model probability fusion",
            **common,
        ),
        AblationConfiguration(
            "A3",
            "Fixed temporal smoothing",
            selected_model_path,
            "fixed",
            min_component_area=0,
            reference_id=selected_reference,
            purpose="Measure temporal smoothing alone",
            **common,
        ),
        AblationConfiguration(
            "A4",
            "Motion-adaptive weighting",
            selected_model_path,
            "motion_adaptive",
            min_component_area=0,
            reference_id=selected_reference,
            purpose="Measure the contribution of motion feedback",
            **common,
        ),
        AblationConfiguration(
            "A5",
            "Small-component cleanup",
            selected_model_path,
            "none",
            min_component_area=min_component_area,
            reference_id=selected_reference,
            purpose="Measure the precision-fragmentation trade-off",
            **common,
        ),
        AblationConfiguration(
            "A6",
            "Motion-adaptive aggregation and cleanup",
            selected_model_path,
            "motion_adaptive",
            min_component_area=min_component_area,
            reference_id=selected_reference,
            purpose="Evaluate motion-adaptive aggregation with cleanup",
            **common,
        ),
    ]


def apply_ablation_configuration(
    probability_maps: Sequence[np.ndarray],
    configuration: AblationConfiguration,
    *,
    motion_scores: Sequence[float] | None = None,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    maps = [np.asarray(probability_map, dtype=np.float32) for probability_map in probability_maps]
    if not maps:
        raise ValueError("at least one probability map is required")
    expected_shape = maps[0].shape
    for probability_map in maps:
        if probability_map.ndim != 2 or probability_map.shape != expected_shape:
            raise ValueError("probability maps must be aligned 2D arrays")
        if not np.all(np.isfinite(probability_map)):
            raise ValueError("probability maps must contain only finite values")
        if probability_map.size and (
            float(probability_map.min()) < 0.0
            or float(probability_map.max()) > 1.0
        ):
            raise ValueError("probability maps must be within [0, 1]")

    temporal_config = TemporalAggregationConfig(
        window_size=configuration.window_size,
        threshold=configuration.threshold,
    )
    if configuration.temporal_mode == "none":
        processed_maps = [probability_map.copy() for probability_map in maps]
    elif configuration.temporal_mode == "fixed":
        processed_maps = aggregate_probability_maps(maps, temporal_config)
    elif configuration.temporal_mode == "motion_adaptive":
        if motion_scores is None:
            raise ValueError("motion_scores are required for motion-adaptive ablations")
        scores = [float(score) for score in motion_scores]
        if len(scores) != len(maps):
            raise ValueError("motion_scores and probability_maps must have the same length")
        if any(not np.isfinite(score) or not 0.0 <= score <= 1.0 for score in scores):
            raise ValueError("motion_scores must be finite values within [0, 1]")
        processed_maps = aggregate_probability_maps_motion_adaptive(
            maps,
            scores,
            temporal_config,
        )
    else:
        raise ValueError(f"unsupported temporal mode: {configuration.temporal_mode}")

    masks = threshold_probability_maps(processed_maps, configuration.threshold)
    if configuration.min_component_area > 0:
        masks = remove_small_components_from_masks(
            masks,
            configuration.min_component_area,
        )
    return processed_maps, masks


def _mask_overlap(first: np.ndarray, second: np.ndarray) -> tuple[float, float]:
    first_bool = first.astype(bool, copy=False)
    second_bool = second.astype(bool, copy=False)
    intersection = int(np.count_nonzero(first_bool & second_bool))
    union = int(np.count_nonzero(first_bool | second_bool))
    total = int(np.count_nonzero(first_bool)) + int(np.count_nonzero(second_bool))
    iou = 1.0 if union == 0 else intersection / union
    dice = 1.0 if total == 0 else (2.0 * intersection) / total
    return float(iou), float(dice)


def _component_counts(mask: np.ndarray, small_component_area: int) -> tuple[int, int]:
    component_count, _, stats, _ = cv2.connectedComponentsWithStats(
        (mask > 0).astype(np.uint8),
        connectivity=8,
    )
    areas = [
        int(stats[component_id, cv2.CC_STAT_AREA])
        for component_id in range(1, component_count)
    ]
    return len(areas), sum(area <= small_component_area for area in areas)


def summarize_temporal_masks(
    masks: Sequence[np.ndarray],
    *,
    small_component_area: int = 500,
    pair_breaks: set[int] | None = None,
) -> dict:
    if not masks:
        raise ValueError("at least one mask is required")
    if small_component_area < 1:
        raise ValueError("small_component_area must be greater than zero")
    pair_breaks = pair_breaks or set()
    if any(index < 1 or index >= len(masks) for index in pair_breaks):
        raise ValueError("pair_breaks must identify a valid later frame")

    binary_masks = [(np.asarray(mask) > 0).astype(np.uint8) for mask in masks]
    expected_shape = binary_masks[0].shape
    if any(mask.ndim != 2 or mask.shape != expected_shape for mask in binary_masks):
        raise ValueError("masks must be aligned 2D arrays")

    areas = [int(mask.sum()) for mask in binary_masks]
    component_counts = []
    small_component_counts = []
    for mask in binary_masks:
        component_count, small_count = _component_counts(mask, small_component_area)
        component_counts.append(component_count)
        small_component_counts.append(small_count)

    consecutive_ious = []
    consecutive_dices = []
    pixel_change_rates = []
    absolute_area_changes = []
    for index in range(1, len(binary_masks)):
        if index in pair_breaks:
            continue
        previous, current = binary_masks[index - 1], binary_masks[index]
        previous_area, current_area = areas[index - 1], areas[index]
        iou, dice = _mask_overlap(previous, current)
        consecutive_ious.append(iou)
        consecutive_dices.append(dice)
        pixel_change_rates.append(float(np.mean(previous != current)))
        absolute_area_changes.append(abs(current_area - previous_area))

    return {
        "frame_count": len(binary_masks),
        "evaluated_pair_count": len(consecutive_ious),
        "skipped_pair_count": len(pair_breaks),
        "mean_consecutive_iou": mean_defined(consecutive_ious),
        "mean_consecutive_dice": mean_defined(consecutive_dices),
        "mean_pixel_change_rate": mean_defined(pixel_change_rates),
        "std_mask_area_px": float(np.std(areas)),
        "mean_abs_area_change_px": mean_defined(absolute_area_changes),
        "mean_component_count": float(np.mean(component_counts)),
        "mean_small_component_count": float(np.mean(small_component_counts)),
        "empty_prediction_frames": sum(area == 0 for area in areas),
    }


def summarize_temporal_video_records(records: Sequence[dict]) -> dict:
    if not records:
        raise ValueError("at least one temporal video record is required")
    metric_names = (
        "mean_consecutive_iou",
        "mean_consecutive_dice",
        "mean_pixel_change_rate",
        "std_mask_area_px",
        "mean_abs_area_change_px",
        "mean_component_count",
        "mean_small_component_count",
    )
    return {
        "video_count": len(records),
        "frame_count": sum(int(record["frame_count"]) for record in records),
        "evaluated_pair_count": sum(int(record["evaluated_pair_count"]) for record in records),
        "skipped_pair_count": sum(int(record["skipped_pair_count"]) for record in records),
        "empty_prediction_frames": sum(
            int(record["empty_prediction_frames"]) for record in records
        ),
        "macro_per_video": {
            name: mean_defined(record[name] for record in records)
            for name in metric_names
        },
    }
