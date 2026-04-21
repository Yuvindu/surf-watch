import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.append(str(Path(__file__).resolve().parents[1]))

from configs.baseline_config import BaselineConfig
from src.models.segformer_baseline import build_segformer_model
from src.training.utils import get_device


def load_model(checkpoint_path: str, device: torch.device, config: BaselineConfig):
    model = build_segformer_model(
        model_name=config.pretrained_model_name,
        num_classes=config.num_classes,
    ).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def preprocess_frame(frame: np.ndarray, image_size: tuple[int, int]) -> torch.Tensor:
    resized = cv2.resize(frame, (image_size[1], image_size[0]), interpolation=cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
    return tensor.unsqueeze(0)


@torch.no_grad()
def predict_probability_map(
    model,
    frame: np.ndarray,
    device: torch.device,
    image_size: tuple[int, int],
) -> np.ndarray:
    input_tensor = preprocess_frame(frame, image_size).to(device)

    logits = model(pixel_values=input_tensor).logits
    logits = torch.nn.functional.interpolate(
        logits,
        size=image_size,
        mode="bilinear",
        align_corners=False,
    )

    probs = torch.softmax(logits, dim=1)
    rip_prob = probs[:, 1, :, :].squeeze(0).cpu().numpy().astype(np.float32)
    rip_prob_resized = cv2.resize(
        rip_prob,
        (frame.shape[1], frame.shape[0]),
        interpolation=cv2.INTER_LINEAR,
    )
    return rip_prob_resized


@torch.no_grad()
def predict_mask(
    model,
    frame: np.ndarray,
    device: torch.device,
    image_size: tuple[int, int],
    threshold: float = 0.5,
) -> np.ndarray:
    prob_map = predict_probability_map(model, frame, device, image_size)
    return (prob_map >= threshold).astype(np.uint8)


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
    args = parser.parse_args()

    config = BaselineConfig()
    device = get_device()

    print(f"Using device: {device}")
    print(f"Loading checkpoint: {args.checkpoint}")

    Path(args.output_overlay).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_mask).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_prob).parent.mkdir(parents=True, exist_ok=True)

    model = load_model(args.checkpoint, device, config)

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

        prob_map = predict_probability_map(model, frame, device, config.image_size)
        mask = (prob_map >= 0.5).astype(np.uint8)
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