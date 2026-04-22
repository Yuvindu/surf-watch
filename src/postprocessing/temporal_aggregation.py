from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np


@dataclass
class TemporalAggregationConfig:
    window_size: int = 5
    threshold: float = 0.5
    low_motion_translation_px: float = 1.0
    high_motion_translation_px: float = 4.0
    low_motion_rotation_deg: float = 0.05
    high_motion_rotation_deg: float = 0.25
    min_neighbor_weight: float = 0.1

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


def _normalize_motion(value: float, low: float, high: float) -> float:
    if high <= low:
        return 0.0
    if value <= low:
        return 0.0
    if value >= high:
        return 1.0
    return float((value - low) / (high - low))


def build_motion_scores_from_metadata(
    frame_motion_metadata: Optional[List[dict]],
    num_frames: int,
    config: TemporalAggregationConfig,
) -> Optional[List[float]]:
    """
    Build one motion score per frame in [0, 1], where higher means less reliable
    temporal aggregation due to stronger motion or failed alignment.

    The motion metadata is expected to come from the motion compensation stage and
    to describe transforms between consecutive frames. Therefore metadata length is
    usually num_frames - 1, and score[0] is set to 0.0 because frame 0 has no
    previous-frame transform.
    """
    if not frame_motion_metadata:
        return None

    scores = [0.0] * num_frames

    usable = min(len(frame_motion_metadata), max(0, num_frames - 1))
    for i in range(usable):
        item = frame_motion_metadata[i]

        if item.get("fallback", False) or not item.get("success", False):
            scores[i + 1] = 1.0
            continue

        translation_mag = float(item.get("translation_magnitude_px", 0.0))
        rotation_deg = abs(float(item.get("rotation_deg", 0.0)))

        translation_score = _normalize_motion(
            translation_mag,
            config.low_motion_translation_px,
            config.high_motion_translation_px,
        )
        rotation_score = _normalize_motion(
            rotation_deg,
            config.low_motion_rotation_deg,
            config.high_motion_rotation_deg,
        )

        scores[i + 1] = float((translation_score + rotation_score) / 2.0)

    return scores


def _adaptive_radius(base_radius: int, motion_score: float) -> int:
    if motion_score >= 0.75:
        return 0
    if motion_score >= 0.40:
        return min(1, base_radius)
    return base_radius


def _neighbor_weight(
    distance: int,
    target_motion_score: float,
    neighbor_motion_score: float,
    config: TemporalAggregationConfig,
) -> float:
    if distance == 0:
        return 1.0

    if target_motion_score >= 0.75 or neighbor_motion_score >= 0.75:
        return 0.0

    temporal_decay = 1.0 / (1.0 + distance)
    motion_penalty = 1.0 - max(target_motion_score, neighbor_motion_score)
    weight = temporal_decay * (0.25 + 0.75 * motion_penalty)
    return max(config.min_neighbor_weight, float(weight))


def aggregate_probability_maps_motion_adaptive(
    probability_maps: List[np.ndarray],
    motion_scores: List[float],
    config: TemporalAggregationConfig,
) -> List[np.ndarray]:
    """
    Apply motion-adaptive temporal aggregation.

    High-motion frames use a smaller effective temporal window, and neighboring
    frames are weighted down when motion magnitude is high or alignment is poor.
    """
    if len(probability_maps) != len(motion_scores):
        raise ValueError("probability_maps and motion_scores must have the same length")

    base_radius = config.window_size // 2
    aggregated = []

    for t in range(len(probability_maps)):
        radius = _adaptive_radius(base_radius, motion_scores[t])
        start = max(0, t - radius)
        end = min(len(probability_maps), t + radius + 1)

        weighted_sum = np.zeros_like(probability_maps[t], dtype=np.float32)
        total_weight = 0.0

        for j in range(start, end):
            distance = abs(j - t)
            weight = _neighbor_weight(distance, motion_scores[t], motion_scores[j], config)
            if weight <= 0.0:
                continue

            weighted_sum += probability_maps[j] * weight
            total_weight += weight

        if total_weight <= 0.0:
            aggregated.append(probability_maps[t].astype(np.float32))
        else:
            aggregated.append((weighted_sum / total_weight).astype(np.float32))

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