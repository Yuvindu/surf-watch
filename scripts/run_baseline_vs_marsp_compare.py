# scripts/run_baseline_vs_marsp_compare.py
import argparse
import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.models.adapter_factory import SUPPORTED_SEGMENTATION_MODELS


def run_command(command: list[str]) -> None:
    print("\n[RUN]", " ".join(command))
    result = subprocess.run(command)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {result.returncode}")


def load_video_frames(path: str) -> tuple[list[np.ndarray], float]:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frames: list[np.ndarray] = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)

    cap.release()
    return frames, fps


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

        masks.append((gray >= 127).astype(np.uint8))

    cap.release()
    return masks, fps


def save_overlay_video(
    original_video: str,
    mask_video: str,
    output_path: str,
    label: str,
    alpha: float = 0.35,
) -> None:
    frames, fps = load_video_frames(original_video)
    masks, _ = load_mask_video(mask_video)

    if len(frames) != len(masks):
        raise ValueError(
            f"Frame count mismatch for overlay generation: "
            f"video={len(frames)} masks={len(masks)}"
        )

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    height, width = frames[0].shape[:2]
    writer = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    for frame, mask in zip(frames, masks):
        overlay = frame.copy()

        # Red mask overlay
        red = np.zeros_like(frame)
        red[:, :, 2] = 255

        mask_3c = np.stack([mask * 255] * 3, axis=-1) > 0
        overlay = np.where(
            mask_3c,
            cv2.addWeighted(frame, 1.0 - alpha, red, alpha, 0.0),
            overlay,
        )

        cv2.putText(
            overlay,
            label,
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 255, 255),
            2,
        )

        writer.write(overlay)

    writer.release()


def save_side_by_side_video(
    left_video: str,
    right_video: str,
    output_path: str,
) -> None:
    left_frames, fps_left = load_video_frames(left_video)
    right_frames, fps_right = load_video_frames(right_video)

    if len(left_frames) != len(right_frames):
        raise ValueError(
            f"Video length mismatch: left={len(left_frames)} right={len(right_frames)}"
        )

    if not left_frames:
        raise ValueError("No frames found for comparison video")

    h1, w1 = left_frames[0].shape[:2]
    h2, w2 = right_frames[0].shape[:2]

    if (h1, w1) != (h2, w2):
        raise ValueError("Left and right videos must have matching frame sizes")

    fps = fps_left or fps_right
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    writer = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (w1 * 2, h1),
    )

    for left, right in zip(left_frames, right_frames):
        left = left.copy()
        cv2.putText(
            left,
            "Baseline",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 255, 255),
            2,
        )
        combined = np.hstack([left, right])
        writer.write(combined)

    writer.release()


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
    return float((a != b).sum() / a.size)


def small_component_count(mask: np.ndarray, max_area: int = 500) -> int:
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8),
        connectivity=8,
    )
    count = 0
    for label in range(1, num_labels):
        area = stats[label, cv2.CC_STAT_AREA]
        if area <= max_area:
            count += 1
    return count


def summarize_mask_sequence(masks: list[np.ndarray], small_blob_area: int = 500) -> dict:
    if len(masks) < 2:
        raise ValueError("Need at least two masks to compute temporal stability metrics")

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
        "mean_consecutive_dice": float(np.mean(consecutive_dices)),
        "mean_pixel_change_rate": float(np.mean(change_rates)),
        "std_mask_area_px": float(np.std(areas)),
        "mean_abs_area_change_px": float(np.mean(area_diffs)),
        "mean_small_blob_count": float(np.mean(small_blob_counts)),
    }


def compare_metric_summaries(baseline: dict, marsp: dict) -> dict:
    return {
        "delta_mean_consecutive_iou": marsp["mean_consecutive_iou"] - baseline["mean_consecutive_iou"],
        "delta_mean_consecutive_dice": marsp["mean_consecutive_dice"] - baseline["mean_consecutive_dice"],
        "delta_mean_pixel_change_rate": marsp["mean_pixel_change_rate"] - baseline["mean_pixel_change_rate"],
        "delta_std_mask_area_px": marsp["std_mask_area_px"] - baseline["std_mask_area_px"],
        "delta_mean_abs_area_change_px": marsp["mean_abs_area_change_px"] - baseline["mean_abs_area_change_px"],
        "delta_mean_small_blob_count": marsp["mean_small_blob_count"] - baseline["mean_small_blob_count"],
    }


def build_metric_table(baseline: dict, marsp: dict) -> list[dict]:
    return [
        {
            "metric": "Consecutive IoU",
            "baseline": round(baseline["mean_consecutive_iou"], 4),
            "marsp": round(marsp["mean_consecutive_iou"], 4),
            "delta": round(marsp["mean_consecutive_iou"] - baseline["mean_consecutive_iou"], 4),
            "preferred_direction": "higher",
        },
        {
            "metric": "Consecutive Dice",
            "baseline": round(baseline["mean_consecutive_dice"], 4),
            "marsp": round(marsp["mean_consecutive_dice"], 4),
            "delta": round(marsp["mean_consecutive_dice"] - baseline["mean_consecutive_dice"], 4),
            "preferred_direction": "higher",
        },
        {
            "metric": "Pixel change rate",
            "baseline": round(baseline["mean_pixel_change_rate"], 6),
            "marsp": round(marsp["mean_pixel_change_rate"], 6),
            "delta": round(marsp["mean_pixel_change_rate"] - baseline["mean_pixel_change_rate"], 6),
            "preferred_direction": "lower",
        },
        {
            "metric": "Mask area variation",
            "baseline": round(baseline["std_mask_area_px"], 1),
            "marsp": round(marsp["std_mask_area_px"], 1),
            "delta": round(marsp["std_mask_area_px"] - baseline["std_mask_area_px"], 1),
            "preferred_direction": "lower",
        },
        {
            "metric": "Mean absolute area change",
            "baseline": round(baseline["mean_abs_area_change_px"], 1),
            "marsp": round(marsp["mean_abs_area_change_px"], 1),
            "delta": round(marsp["mean_abs_area_change_px"] - baseline["mean_abs_area_change_px"], 1),
            "preferred_direction": "lower",
        },
        {
            "metric": "Mean small blob count",
            "baseline": round(baseline["mean_small_blob_count"], 4),
            "marsp": round(marsp["mean_small_blob_count"], 4),
            "delta": round(marsp["mean_small_blob_count"] - baseline["mean_small_blob_count"], 4),
            "preferred_direction": "lower",
        },
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video-name", required=True)
    parser.add_argument("--input", required=True, help="Path to original input video")
    parser.add_argument("--checkpoint", default="checkpoints/best_model.pt")
    parser.add_argument(
        "--model",
        default="segformer",
        choices=SUPPORTED_SEGMENTATION_MODELS,
        help="Segmentation model adapter to use",
    )
    parser.add_argument("--window-size", type=int, default=5)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--reuse-existing", action="store_true")
    args = parser.parse_args()

    video_name = args.video_name
    input_video = args.input

    compare_dir = Path("outputs/comparisons")
    compare_dir.mkdir(parents=True, exist_ok=True)

    # Baseline outputs
    baseline_overlay = str(compare_dir / f"{video_name}_baseline_overlay.mp4")
    baseline_mask = str(compare_dir / f"{video_name}_baseline_mask.mp4")
    baseline_prob = str(compare_dir / f"{video_name}_baseline_prob.mp4")

    # MARSP outputs from main pipeline
    marsp_overlay = str(compare_dir / f"{video_name}_marsp_overlay.mp4")
    side_by_side = str(compare_dir / f"{video_name}_baseline_vs_marsp.mp4")

    comparison_metrics_json = str(compare_dir / f"{video_name}_baseline_vs_marsp_metrics.json")

    # Reused MARSP pipeline outputs
    marsp_final_mask = f"outputs/temporal_aggregation/{video_name}_stabilised_agg_mask.mp4"
    marsp_summary = f"outputs/marsp/{video_name}_pipeline_summary.json"

    # 1. Run baseline inference
    if not args.reuse_existing or not (
        Path(baseline_overlay).exists()
        and Path(baseline_mask).exists()
        and Path(baseline_prob).exists()
    ):
        print("[STAGE] Running baseline segmentation adapter", flush=True)
        run_command([
            sys.executable,
            "scripts/run_video_segmentation.py",
            "--input", input_video,
            "--model", args.model,
            "--checkpoint", args.checkpoint,
            "--output-overlay", baseline_overlay,
            "--output-mask", baseline_mask,
            "--output-prob", baseline_prob,
            "--threshold", str(args.threshold),
        ])

    # 2. Run MARSP pipeline
    if not args.reuse_existing or not (
        Path(marsp_final_mask).exists()
        and Path(marsp_summary).exists()
    ):
        print("[STAGE] Running MARSP motion-aware pipeline", flush=True)
        run_command([
            sys.executable,
            "scripts/run_marsp_pipeline.py",
            "--video-name", video_name,
            "--input", input_video,
            "--model", args.model,
            "--checkpoint", args.checkpoint,
            "--window-size", str(args.window_size),
            "--threshold", str(args.threshold),
        ])

    # 3. Build MARSP overlay on original frames
    print("[STAGE] Rendering MARSP overlay on original frames", flush=True)
    save_overlay_video(
        original_video=input_video,
        mask_video=marsp_final_mask,
        output_path=marsp_overlay,
        label="MARSP",
    )

    # 4. Build side-by-side baseline vs MARSP overlay video
    print("[STAGE] Rendering baseline vs MARSP comparison video", flush=True)
    save_side_by_side_video(
        left_video=baseline_overlay,
        right_video=marsp_overlay,
        output_path=side_by_side,
    )

    # 5. Compute temporal stability metrics for baseline and MARSP final masks
    print("[STAGE] Computing baseline vs MARSP temporal stability metrics", flush=True)
    baseline_masks, _ = load_mask_video(baseline_mask)
    marsp_masks, _ = load_mask_video(marsp_final_mask)

    baseline_summary = summarize_mask_sequence(baseline_masks)
    marsp_summary_metrics = summarize_mask_sequence(marsp_masks)
    comparison = compare_metric_summaries(baseline_summary, marsp_summary_metrics)
    metric_table = build_metric_table(baseline_summary, marsp_summary_metrics)

    combined = {
        "video_name": video_name,
        "input_video": input_video,
        "segmentation_model": args.model,
        "baseline": baseline_summary,
        "marsp": marsp_summary_metrics,
        "comparison": comparison,
        "metric_table": metric_table,
        "artifacts": {
            "baseline_overlay": baseline_overlay,
            "baseline_mask": baseline_mask,
            "marsp_overlay": marsp_overlay,
            "marsp_final_mask": marsp_final_mask,
            "side_by_side_video": side_by_side,
        },
    }

    Path(comparison_metrics_json).write_text(
        json.dumps(combined, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(combined, indent=2))
    print("\n[OK] Baseline vs MARSP comparison complete")


if __name__ == "__main__":
    main()
