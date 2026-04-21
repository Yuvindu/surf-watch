from dataclasses import dataclass
from pathlib import Path
from typing import List

import cv2
import numpy as np


@dataclass
class TemporalAggregationConfig:
    window_size: int = 5
    threshold: float = 0.5

    def __post_init__(self):
        if self.window_size < 1 or self.window_size % 2 == 0:
            raise ValueError("window_size must be a positive odd integer")


def aggregate_probability_maps(
    probability_maps: List[np.ndarray],
    config: TemporalAggregationConfig,
) -> List[np.ndarray]:
    """
    Apply centered sliding-window averaging over a sequence of probability maps.

    Each probability map is expected to be a 2D float array in [0, 1].
    """
    radius = config.window_size // 2
    aggregated = []

    for t in range(len(probability_maps)):
        start = max(0, t - radius)
        end = min(len(probability_maps), t + radius + 1)

        window = probability_maps[start:end]
        mean_map = np.mean(window, axis=0).astype(np.float32)
        aggregated.append(mean_map)

    return aggregated


def threshold_probability_maps(
    probability_maps: List[np.ndarray],
    threshold: float,
) -> List[np.ndarray]:
    """
    Convert probability maps to binary masks using a fixed threshold.
    """
    masks = []
    for prob in probability_maps:
        mask = (prob >= threshold).astype(np.uint8)
        masks.append(mask)
    return masks


def save_mask_video(
    masks: List[np.ndarray],
    output_path: str,
    fps: float,
) -> None:
    """
    Save binary masks as a grayscale video.
    """
    if not masks:
        raise ValueError("No masks provided")

    height, width = masks[0].shape
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    writer = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
        isColor=False,
    )

    for mask in masks:
        frame = (mask * 255).astype(np.uint8)
        writer.write(frame)

    writer.release()


def save_probability_video(
    probability_maps: List[np.ndarray],
    output_path: str,
    fps: float,
) -> None:
    """
    Save probability maps as grayscale videos for inspection.
    """
    if not probability_maps:
        raise ValueError("No probability maps provided")

    height, width = probability_maps[0].shape
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    writer = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
        isColor=False,
    )

    for prob in probability_maps:
        frame = np.clip(prob * 255.0, 0, 255).astype(np.uint8)
        writer.write(frame)

    writer.release()


def save_comparison_video(
    raw_masks: List[np.ndarray],
    aggregated_masks: List[np.ndarray],
    output_path: str,
    fps: float,
) -> None:
    """
    Save side-by-side comparison of raw vs aggregated binary masks.
    """
    if not raw_masks or not aggregated_masks:
        raise ValueError("Mask lists must not be empty")

    if len(raw_masks) != len(aggregated_masks):
        raise ValueError("raw_masks and aggregated_masks must have the same length")

    height, width = raw_masks[0].shape
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    writer = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width * 2, height),
    )

    for raw, agg in zip(raw_masks, aggregated_masks):
        raw_rgb = np.stack([(raw * 255).astype(np.uint8)] * 3, axis=-1)
        agg_rgb = np.stack([(agg * 255).astype(np.uint8)] * 3, axis=-1)

        cv2.putText(raw_rgb, "Raw", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.putText(agg_rgb, "Aggregated", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        combined = np.hstack([raw_rgb, agg_rgb])
        writer.write(combined)

    writer.release()