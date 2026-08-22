import argparse
import json
from pathlib import Path
from typing import List

import cv2
import numpy as np


def load_mask_video(path: str) -> tuple[list[np.ndarray], float]:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open mask video: {path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    masks: list[np.ndarray] = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame.ndim == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame

        mask = (gray >= 127).astype(np.uint8)
        masks.append(mask)

    cap.release()
    return masks, fps


def iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = np.logical_and(a == 1, b == 1).sum()
    union = np.logical_or(a == 1, b == 1).sum()
    if union == 0:
        return 1.0
    return float(inter / union)


def dice(a: np.ndarray, b: np.ndarray) -> float:
    inter = np.logical_and(a == 1, b == 1).sum()
    total = (a == 1).sum() + (b == 1).sum()
    if total == 0:
        return 1.0
    return float((2.0 * inter) / total)


def pixel_change_rate(a: np.ndarray, b: np.ndarray) -> float:
    changed = (a != b).sum()
    total = a.size
    return float(changed / total)


def small_component_count(mask: np.ndarray, max_area: int) -> int:
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    count = 0
    for label in range(1, num_labels):
        area = stats[label, cv2.CC_STAT_AREA]
        if area <= max_area:
            count += 1
    return count


def summarize_sequence(masks: List[np.ndarray], small_blob_area: int) -> dict:
    if len(masks) < 2:
        raise ValueError("Need at least 2 frames to compute temporal stability metrics")

    consecutive_ious = []
    consecutive_dices = []
    change_rates = []
    areas = []
    area_diffs = []
    small_blob_counts = []

    prev_area = int(masks[0].sum())
    areas.append(prev_area)
    small_blob_counts.append(small_component_count(masks[0], small_blob_area))

    for i in range(1, len(masks)):
        prev_mask = masks[i - 1]
        curr_mask = masks[i]

        consecutive_ious.append(iou(prev_mask, curr_mask))
        consecutive_dices.append(dice(prev_mask, curr_mask))
        change_rates.append(pixel_change_rate(prev_mask, curr_mask))

        curr_area = int(curr_mask.sum())
        areas.append(curr_area)
        area_diffs.append(abs(curr_area - prev_area))
        prev_area = curr_area

        small_blob_counts.append(small_component_count(curr_mask, small_blob_area))

    return {
        "total_frames": len(masks),
        "mean_consecutive_iou": float(np.mean(consecutive_ious)),
        "std_consecutive_iou": float(np.std(consecutive_ious)),
        "mean_consecutive_dice": float(np.mean(consecutive_dices)),
        "std_consecutive_dice": float(np.std(consecutive_dices)),
        "mean_pixel_change_rate": float(np.mean(change_rates)),
        "std_pixel_change_rate": float(np.std(change_rates)),
        "mean_mask_area_px": float(np.mean(areas)),
        "std_mask_area_px": float(np.std(areas)),
        "mean_abs_area_change_px": float(np.mean(area_diffs)) if area_diffs else 0.0,
        "mean_small_blob_count": float(np.mean(small_blob_counts)),
        "std_small_blob_count": float(np.std(small_blob_counts)),
    }


def compare_sequences(raw_summary: dict, processed_summary: dict) -> dict:
    return {
        "delta_mean_consecutive_iou": processed_summary["mean_consecutive_iou"] - raw_summary["mean_consecutive_iou"],
        "delta_mean_consecutive_dice": processed_summary["mean_consecutive_dice"] - raw_summary["mean_consecutive_dice"],
        "delta_mean_pixel_change_rate": processed_summary["mean_pixel_change_rate"] - raw_summary["mean_pixel_change_rate"],
        "delta_std_mask_area_px": processed_summary["std_mask_area_px"] - raw_summary["std_mask_area_px"],
        "delta_mean_abs_area_change_px": processed_summary["mean_abs_area_change_px"] - raw_summary["mean_abs_area_change_px"],
        "delta_mean_small_blob_count": processed_summary["mean_small_blob_count"] - raw_summary["mean_small_blob_count"],
    }


def interpret_comparison(comparison: dict) -> dict:
    return {
        "higher_consecutive_iou_is_better": comparison["delta_mean_consecutive_iou"] > 0,
        "higher_consecutive_dice_is_better": comparison["delta_mean_consecutive_dice"] > 0,
        "lower_pixel_change_rate_is_better": comparison["delta_mean_pixel_change_rate"] < 0,
        "lower_area_variation_is_better": comparison["delta_std_mask_area_px"] < 0,
        "lower_abs_area_change_is_better": comparison["delta_mean_abs_area_change_px"] < 0,
        "lower_small_blob_count_is_better": comparison["delta_mean_small_blob_count"] < 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-mask-video", required=True, help="Path to the raw/before mask video")
    parser.add_argument("--processed-mask-video", required=True, help="Path to the processed/after mask video")
    parser.add_argument("--output-json", required=True, help="Path to save the metric summary JSON")
    parser.add_argument("--small-blob-area", type=int, default=500, help="Max connected-component area to count as a small blob")
    args = parser.parse_args()

    raw_masks, raw_fps = load_mask_video(args.raw_mask_video)
    processed_masks, processed_fps = load_mask_video(args.processed_mask_video)

    if len(raw_masks) != len(processed_masks):
        raise ValueError(
            f"Mask videos must have the same number of frames. "
            f"Got raw={len(raw_masks)} processed={len(processed_masks)}"
        )

    raw_summary = summarize_sequence(raw_masks, args.small_blob_area)
    processed_summary = summarize_sequence(processed_masks, args.small_blob_area)
    comparison = compare_sequences(raw_summary, processed_summary)
    interpretation = interpret_comparison(comparison)

    result = {
        "raw_mask_video": args.raw_mask_video,
        "processed_mask_video": args.processed_mask_video,
        "raw_fps": raw_fps,
        "processed_fps": processed_fps,
        "small_blob_area_threshold": args.small_blob_area,
        "raw_summary": raw_summary,
        "processed_summary": processed_summary,
        "comparison": comparison,
        "interpretation": interpretation,
    }

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
