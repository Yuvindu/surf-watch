import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))

from configs.baseline_config import BaselineConfig
from src.models.segformer_adapter import SegFormerSegmentationAdapter
from src.training.utils import get_device


def make_overlay(frame: np.ndarray, mask: np.ndarray, alpha: float = 0.35) -> np.ndarray:
    overlay = frame.copy()
    red = np.zeros_like(frame)
    red[:, :, 2] = 255  # BGR red
    overlay[mask == 1] = cv2.addWeighted(frame, 1 - alpha, red, alpha, 0)[mask == 1]
    return overlay


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to input video")
    parser.add_argument("--checkpoint", required=True, help="Path to trained checkpoint")
    parser.add_argument("--output-overlay", required=True, help="Path to output overlay video")
    parser.add_argument("--output-mask", required=True, help="Path to output mask video")
    parser.add_argument("--output-prob", required=True, help="Path to output probability video")
    parser.add_argument("--threshold", type=float, default=0.5, help="Threshold for converting rip-current probabilities into a binary mask")
    args = parser.parse_args()

    config = BaselineConfig()
    device = get_device()

    print(f"Using device: {device}")
    print(f"Loading checkpoint: {args.checkpoint}")
    print(f"Mask threshold: {args.threshold}")

    Path(args.output_overlay).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_mask).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_prob).parent.mkdir(parents=True, exist_ok=True)

    adapter = SegFormerSegmentationAdapter(
        checkpoint_path=args.checkpoint,
        device=device,
        config=config,
        threshold=args.threshold,
    )
    metadata = adapter.metadata()
    print(f"Segmentation adapter: {metadata.name}")
    print(f"Adapter input size: {metadata.input_size}")

    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {args.input}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    overlay_writer = cv2.VideoWriter(
        args.output_overlay,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    mask_writer = cv2.VideoWriter(
        args.output_mask,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
        isColor=False,
    )
    prob_writer = cv2.VideoWriter(
        args.output_prob,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
        isColor=False,
    )

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        result = adapter.predict(frame)
        prob_map = result.probability_map
        mask = result.binary_mask
        overlay = make_overlay(frame, mask)
        mask_frame = (mask * 255).astype(np.uint8)
        prob_frame = np.clip(prob_map * 255.0, 0, 255).astype(np.uint8)

        overlay_writer.write(overlay)
        mask_writer.write(mask_frame)
        prob_writer.write(prob_frame)

        frame_idx += 1
        if frame_idx % 50 == 0 or frame_idx == total:
            print(f"Processed {frame_idx}/{total} frames")

    cap.release()
    overlay_writer.release()
    mask_writer.release()
    prob_writer.release()

    print(f"Saved overlay video to: {args.output_overlay}")
    print(f"Saved mask video to: {args.output_mask}")
    print(f"Saved probability video to: {args.output_prob}")


if __name__ == "__main__":
    main()
