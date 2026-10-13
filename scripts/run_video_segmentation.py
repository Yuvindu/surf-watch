import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))

from configs.baseline_config import BaselineConfig
from src.models.adapter_factory import (
    SUPPORTED_SEGMENTATION_MODELS,
    build_segmentation_adapter,
)
from src.models.model_registry import get_segmentation_model_option
from src.models.probability_ensemble import ProbabilityEnsembleAdapter
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
    parser.add_argument("--checkpoint", help="Path to trained checkpoint")
    parser.add_argument("--ensemble-checkpoint", action="append", default=[], metavar="MODEL=CHECKPOINT")
    parser.add_argument("--output-overlay", required=True, help="Path to output overlay video")
    parser.add_argument("--output-mask", required=True, help="Path to output mask video")
    parser.add_argument("--output-prob", required=True, help="Path to output probability video")
    parser.add_argument(
        "--model",
        default="segformer",
        choices=(*SUPPORTED_SEGMENTATION_MODELS, "ensemble"),
        help="Segmentation model adapter to use",
    )
    parser.add_argument("--threshold", type=float, default=0.5, help="Threshold for converting rip-current probabilities into a binary mask")
    args = parser.parse_args()

    if args.model == "ensemble":
        if args.checkpoint or len(args.ensemble_checkpoint) < 2:
            parser.error("ensemble requires at least two --ensemble-checkpoint values and no --checkpoint")
        component_paths = {}
        for value in args.ensemble_checkpoint:
            if "=" not in value:
                parser.error("--ensemble-checkpoint must use MODEL=CHECKPOINT")
            model_name, checkpoint_path = value.split("=", 1)
            try:
                model_name = get_segmentation_model_option(model_name).id
            except ValueError as exc:
                parser.error(str(exc))
            if not checkpoint_path or model_name in component_paths:
                parser.error("ensemble checkpoint paths must be non-empty and models unique")
            component_paths[model_name] = checkpoint_path
    elif not args.checkpoint or args.ensemble_checkpoint:
        parser.error("a single model requires --checkpoint and no --ensemble-checkpoint")

    config = BaselineConfig()
    device = get_device()

    print(f"Using device: {device}")
    print(f"Loading checkpoint: {args.checkpoint or component_paths}")
    print(f"Mask threshold: {args.threshold}")

    Path(args.output_overlay).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_mask).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_prob).parent.mkdir(parents=True, exist_ok=True)

    if args.model == "ensemble":
        components = {
            name: build_segmentation_adapter(name, path, device, config, args.threshold)
            for name, path in component_paths.items()
        }
        adapter = ProbabilityEnsembleAdapter(
            components, {name: 1.0 for name in components}, threshold=args.threshold
        )
    else:
        adapter = build_segmentation_adapter(
            model_name=args.model,
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
