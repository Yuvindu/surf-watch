import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.postprocessing.temporal_aggregation import (
    TemporalAggregationConfig,
    aggregate_probability_maps,
    aggregate_probability_maps_motion_adaptive,
    build_motion_scores_from_metadata,
    save_comparison_video,
    save_mask_video,
    save_probability_video,
    threshold_probability_maps,
)


def load_probability_video(path: str) -> tuple[list[np.ndarray], float]:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open probability video: {path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame.ndim == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame

        prob = gray.astype(np.float32) / 255.0
        frames.append(prob)

    cap.release()
    return frames, fps


def load_motion_metadata(path: Optional[str]) -> Optional[list[dict]]:
    if path is None:
        return None

    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    return payload.get("per_frame_metadata")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-prob-video", required=True, help="Path to raw probability video")
    parser.add_argument("--output-prob-video", required=True, help="Path to aggregated probability video")
    parser.add_argument("--output-mask-video", required=True, help="Path to aggregated binary mask video")
    parser.add_argument("--output-comparison-video", required=True, help="Path to raw vs aggregated comparison video")
    parser.add_argument("--results", default="outputs/temporal_aggregation/results.json")
    parser.add_argument("--window-size", type=int, default=5)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument(
        "--motion-results",
        default=None,
        help="Optional motion compensation results JSON for motion-adaptive aggregation",
    )
    args = parser.parse_args()

    config = TemporalAggregationConfig(
        window_size=args.window_size,
        threshold=args.threshold,
    )

    raw_prob_maps, fps = load_probability_video(args.input_prob_video)
    if not raw_prob_maps:
        raise ValueError("No probability frames were loaded from the input video")

    print(f"Loaded {len(raw_prob_maps)} probability frames at {fps:.2f} fps")
    print(f"Window size: {config.window_size}, threshold: {config.threshold}")

    motion_metadata = load_motion_metadata(args.motion_results)
    motion_scores = build_motion_scores_from_metadata(motion_metadata, len(raw_prob_maps), config)

    if motion_scores is not None:
        aggregated_prob_maps = aggregate_probability_maps_motion_adaptive(
            raw_prob_maps,
            motion_scores,
            config,
        )
        aggregation_type = "motion-adaptive sliding-window probability averaging"
        mean_motion_score = float(np.mean(motion_scores))
        max_motion_score = float(np.max(motion_scores))
    else:
        aggregated_prob_maps = aggregate_probability_maps(raw_prob_maps, config)
        aggregation_type = "fixed sliding-window probability averaging"
        mean_motion_score = None
        max_motion_score = None

    raw_binary_masks = threshold_probability_maps(raw_prob_maps, config.threshold)
    aggregated_binary_masks = threshold_probability_maps(aggregated_prob_maps, config.threshold)

    save_probability_video(aggregated_prob_maps, args.output_prob_video, fps)
    save_mask_video(aggregated_binary_masks, args.output_mask_video, fps)
    save_comparison_video(
        raw_binary_masks,
        aggregated_binary_masks,
        args.output_comparison_video,
        fps,
    )

    Path(args.results).parent.mkdir(parents=True, exist_ok=True)
    results = {
        "input_probability_video": args.input_prob_video,
        "output_probability_video": args.output_prob_video,
        "output_mask_video": args.output_mask_video,
        "output_comparison_video": args.output_comparison_video,
        "window_size": config.window_size,
        "threshold": config.threshold,
        "total_frames": len(raw_prob_maps),
        "fps": fps,
        "frame_height": raw_prob_maps[0].shape[0],
        "frame_width": raw_prob_maps[0].shape[1],
        "aggregation_type": aggregation_type,
        "motion_results": args.motion_results,
        "mean_motion_score": mean_motion_score,
        "max_motion_score": max_motion_score,
    }

    with open(args.results, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()