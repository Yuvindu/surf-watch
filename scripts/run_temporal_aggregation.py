import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.postprocessing.temporal_aggregation import (
    TemporalAggregationConfig,
    aggregate_probability_maps,
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-prob-video", required=True, help="Path to raw probability video")
    parser.add_argument("--output-prob-video", required=True, help="Path to aggregated probability video")
    parser.add_argument("--output-mask-video", required=True, help="Path to aggregated binary mask video")
    parser.add_argument("--output-comparison-video", required=True, help="Path to raw vs aggregated comparison video")
    parser.add_argument("--results", default="outputs/temporal_aggregation/results.json")
    parser.add_argument("--window-size", type=int, default=5)
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    config = TemporalAggregationConfig(
        window_size=args.window_size,
        threshold=args.threshold,
    )

    raw_prob_maps, fps = load_probability_video(args.input_prob_video)
    aggregated_prob_maps = aggregate_probability_maps(raw_prob_maps, config)

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
        "method": "sliding-window probability averaging",
    }

    with open(args.results, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()  