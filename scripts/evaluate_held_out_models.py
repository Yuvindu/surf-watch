from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import sys
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.append(str(Path(__file__).resolve().parents[1]))

from configs.baseline_config import BaselineConfig
from scripts.run_baseline_vs_marsp_compare import file_identity
from src.evaluation.segmentation_metrics import (
    binary_confusion,
    metrics_from_confusion,
    summarize_records,
)
from src.models.adapter_factory import build_segmentation_adapter
from src.models.model_registry import get_segmentation_model_option


RUN_NAME_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
FRAME_NAME_PATTERN = re.compile(r"^(?P<video>.+)_(?P<frame>\d+)$")
METRIC_NAMES = (
    "foreground_iou",
    "foreground_dice",
    "foreground_precision",
    "foreground_recall",
    "background_iou",
    "mean_iou",
    "mean_dice",
)


@dataclass(frozen=True)
class ModelEvaluationSpec:
    model: str
    checkpoint: Path


def parse_model_checkpoint(value: str) -> ModelEvaluationSpec:
    if "=" not in value:
        raise argparse.ArgumentTypeError(
            "model checkpoint must use MODEL=CHECKPOINT format"
        )
    model_value, checkpoint_value = value.split("=", 1)
    if not checkpoint_value.strip():
        raise argparse.ArgumentTypeError("checkpoint path cannot be empty")
    try:
        model = get_segmentation_model_option(model_value).id
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc
    return ModelEvaluationSpec(
        model=model,
        checkpoint=Path(checkpoint_value.strip()).expanduser().resolve(),
    )


def validate_run_name(value: str) -> str:
    if not RUN_NAME_PATTERN.fullmatch(value):
        raise argparse.ArgumentTypeError(
            "run name must contain only letters, numbers, dots, underscores, and hyphens"
        )
    return value


def validate_unique_models(specs: list[ModelEvaluationSpec]) -> None:
    models = [spec.model for spec in specs]
    duplicates = sorted(model for model in set(models) if models.count(model) > 1)
    if duplicates:
        raise ValueError("Each model can appear only once: " + ", ".join(duplicates))


def parse_frame_identity(filename: str) -> tuple[str, int]:
    match = FRAME_NAME_PATTERN.fullmatch(Path(filename).stem)
    if match is None:
        raise ValueError(
            f"Cannot derive video and frame index from RipVIS filename: {filename}"
        )
    return match.group("video"), int(match.group("frame"))


def discover_samples(
    ripvis_root: Path,
    processed_root: Path,
    split: str,
    selected_videos: set[str] | None = None,
    max_frames: int | None = None,
) -> list[dict]:
    images_dir = ripvis_root / split / "sampled_images" / "sampled_images" / "images"
    masks_dir = processed_root / split / "masks"
    image_paths = sorted(images_dir.glob("*.jpg"))
    if not image_paths:
        raise FileNotFoundError(f"No sampled images found in {images_dir}")

    samples = []
    missing_masks = []
    for image_path in image_paths:
        video_id, frame_index = parse_frame_identity(image_path.name)
        if selected_videos and video_id not in selected_videos:
            continue
        mask_path = masks_dir / image_path.with_suffix(".png").name
        if not mask_path.is_file():
            missing_masks.append(mask_path)
            continue
        samples.append(
            {
                "video_id": video_id,
                "frame_index": frame_index,
                "image_path": image_path,
                "mask_path": mask_path,
            }
        )

    if missing_masks:
        example = ", ".join(path.name for path in missing_masks[:5])
        raise FileNotFoundError(
            f"Missing {len(missing_masks)} ground-truth masks. Examples: {example}"
        )
    if not samples:
        selection = ", ".join(sorted(selected_videos or [])) or "all videos"
        raise ValueError(f"No matched image/mask samples found for {selection}")
    return samples[:max_frames] if max_frames is not None else samples


def git_revision() -> str | None:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True
    )
    if completed.returncode != 0:
        return None
    return completed.stdout.strip() or None


def git_worktree_dirty() -> bool | None:
    completed = subprocess.run(
        ["git", "status", "--porcelain"], capture_output=True, text=True
    )
    if completed.returncode != 0:
        return None
    return bool(completed.stdout.strip())


def code_provenance() -> dict:
    repository_root = Path(__file__).resolve().parents[1]
    return {
        "git_revision": git_revision(),
        "git_worktree_dirty": git_worktree_dirty(),
        "files": {
            "evaluator": file_identity(str(Path(__file__))),
            "metrics": file_identity(
                str(repository_root / "src/evaluation/segmentation_metrics.py")
            ),
        },
    }


def evaluate_model(
    spec: ModelEvaluationSpec,
    samples: list[dict],
    device: torch.device,
    threshold: float,
) -> dict:
    started_at = time.perf_counter()
    if not spec.checkpoint.is_file():
        return {
            "model": spec.model,
            "status": "failed",
            "error": f"Checkpoint not found: {spec.checkpoint}",
            "runtime_seconds": time.perf_counter() - started_at,
            "frame_records": [],
            "video_records": [],
        }

    try:
        adapter = build_segmentation_adapter(
            model_name=spec.model,
            checkpoint_path=str(spec.checkpoint),
            device=device,
            config=BaselineConfig(),
            threshold=threshold,
        )
        records = []
        total = len(samples)
        for sample_number, sample in enumerate(samples, start=1):
            frame = cv2.imread(str(sample["image_path"]), cv2.IMREAD_COLOR)
            target = cv2.imread(str(sample["mask_path"]), cv2.IMREAD_GRAYSCALE)
            if frame is None:
                raise ValueError(f"Could not read image: {sample['image_path']}")
            if target is None:
                raise ValueError(f"Could not read mask: {sample['mask_path']}")
            target = (target > 0).astype(np.uint8)

            result = adapter.predict(frame)
            prediction = result.binary_mask
            if prediction.shape != target.shape:
                raise ValueError(
                    f"Prediction shape {prediction.shape} does not match target "
                    f"shape {target.shape} for {sample['image_path'].name}"
                )

            confusion = binary_confusion(prediction, target)
            pixel_count = int(target.size)
            target_positive = int(target.sum())
            prediction_positive = int(prediction.sum())
            records.append(
                {
                    "model": spec.model,
                    "video_id": sample["video_id"],
                    "frame_index": sample["frame_index"],
                    "filename": sample["image_path"].name,
                    "pixel_count": pixel_count,
                    "ground_truth_positive_pixels": target_positive,
                    "prediction_positive_pixels": prediction_positive,
                    "ground_truth_foreground_fraction": target_positive / pixel_count,
                    "prediction_foreground_fraction": prediction_positive / pixel_count,
                    "confusion": confusion,
                    "metrics": metrics_from_confusion(confusion),
                }
            )
            if sample_number % 100 == 0 or sample_number == total:
                print(f"[{spec.model}] evaluated {sample_number}/{total} frames", flush=True)

        by_video = defaultdict(list)
        for record in records:
            by_video[record["video_id"]].append(record)
        video_records = [
            {
                "model": spec.model,
                "video_id": video_id,
                **summarize_records(records_for_video),
            }
            for video_id, records_for_video in sorted(by_video.items())
        ]
        return {
            "model": spec.model,
            "status": "success",
            "error": None,
            "checkpoint": file_identity(str(spec.checkpoint)),
            "adapter_metadata": asdict(adapter.metadata()),
            "runtime_seconds": time.perf_counter() - started_at,
            "summary": summarize_records(records),
            "frame_records": records,
            "video_records": video_records,
        }
    except Exception as exc:
        return {
            "model": spec.model,
            "status": "failed",
            "error": f"{type(exc).__name__}: {exc}",
            "checkpoint": file_identity(str(spec.checkpoint)),
            "runtime_seconds": time.perf_counter() - started_at,
            "frame_records": [],
            "video_records": [],
        }


def _flatten_summary(prefix: str, summary: dict) -> dict:
    row = {
        "frame_count": summary["frame_count"],
        "empty_ground_truth_frames": summary["empty_ground_truth_frames"],
        "empty_prediction_frames": summary["empty_prediction_frames"],
        "mean_ground_truth_foreground_fraction": summary[
            "mean_ground_truth_foreground_fraction"
        ],
        "mean_prediction_foreground_fraction": summary[
            "mean_prediction_foreground_fraction"
        ],
    }
    for aggregation in ("micro", "macro_per_frame"):
        for metric in METRIC_NAMES:
            row[f"{prefix}{aggregation}_{metric}"] = summary[aggregation][metric]
    return row


def write_artifacts(output_dir: Path, payload: dict) -> tuple[Path, Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "held_out_evaluation_summary.json"
    frame_csv_path = output_dir / "held_out_frame_metrics.csv"
    video_csv_path = output_dir / "held_out_video_metrics.csv"

    json_payload = {
        **payload,
        "results": [
            {
                key: value
                for key, value in result.items()
                if key not in {"frame_records", "video_records"}
            }
            for result in payload["results"]
        ],
    }
    json_path.write_text(json.dumps(json_payload, indent=2), encoding="utf-8")

    frame_rows = []
    for result in payload["results"]:
        for record in result["frame_records"]:
            row = {
                key: record[key]
                for key in (
                    "model",
                    "video_id",
                    "frame_index",
                    "filename",
                    "pixel_count",
                    "ground_truth_positive_pixels",
                    "prediction_positive_pixels",
                    "ground_truth_foreground_fraction",
                    "prediction_foreground_fraction",
                )
            }
            row.update(record["confusion"])
            row.update(record["metrics"])
            frame_rows.append(row)
    _write_csv(frame_csv_path, frame_rows)

    video_rows = []
    for result in payload["results"]:
        for record in result["video_records"]:
            video_rows.append(
                {
                    "model": record["model"],
                    "video_id": record["video_id"],
                    **_flatten_summary("", record),
                }
            )
    _write_csv(video_csv_path, video_rows)
    return json_path, frame_csv_path, video_csv_path


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def print_summary(results: list[dict]) -> None:
    print("\nHeld-out semantic segmentation evaluation")
    print("model             status   frames   fg IoU   fg Dice   runtime(s)")
    print("----------------  -------  -------  -------  --------  ----------")
    for result in results:
        if result["status"] == "success":
            summary = result["summary"]
            iou = summary["micro"]["foreground_iou"]
            dice = summary["micro"]["foreground_dice"]
            iou_text = f"{iou:.4f}" if iou is not None else "n/a"
            dice_text = f"{dice:.4f}" if dice is not None else "n/a"
            print(
                f"{result['model']:<16}  success  {summary['frame_count']:<7}  "
                f"{iou_text:<7}  {dice_text:<8}  {result['runtime_seconds']:.2f}"
            )
        else:
            print(f"{result['model']:<16}  failed   -        -        -         -")
            print(f"  error: {result['error']}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate registered segmentation models on labelled RipVIS frames."
    )
    parser.add_argument("--run-name", required=True, type=validate_run_name)
    parser.add_argument(
        "--model-checkpoint",
        action="append",
        required=True,
        type=parse_model_checkpoint,
        metavar="MODEL=CHECKPOINT",
    )
    parser.add_argument("--ripvis-root", default="../RipVIS")
    parser.add_argument("--processed-root", default="data/processed")
    parser.add_argument("--split", choices=("val",), default="val")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--video", action="append", dest="videos")
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--output-root", default="outputs/held_out_evaluation")
    args = parser.parse_args()

    if not 0.0 <= args.threshold <= 1.0:
        parser.error("--threshold must be between 0.0 and 1.0")
    if args.max_frames is not None and args.max_frames <= 0:
        parser.error("--max-frames must be greater than zero")
    try:
        validate_unique_models(args.model_checkpoint)
    except ValueError as exc:
        parser.error(str(exc))

    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA was requested but is not available")
    use_cuda = args.device == "cuda" or (
        args.device == "auto" and torch.cuda.is_available()
    )
    device = torch.device("cuda" if use_cuda else "cpu")
    ripvis_root = Path(args.ripvis_root).expanduser().resolve()
    processed_root = Path(args.processed_root).expanduser().resolve()
    try:
        samples = discover_samples(
            ripvis_root=ripvis_root,
            processed_root=processed_root,
            split=args.split,
            selected_videos=set(args.videos) if args.videos else None,
            max_frames=args.max_frames,
        )
    except (FileNotFoundError, ValueError) as exc:
        parser.error(str(exc))

    print(
        f"Evaluating {len(args.model_checkpoint)} model(s) on {len(samples)} "
        f"labelled {args.split} frames using {device}."
    )
    results = [
        evaluate_model(spec, samples, device, args.threshold)
        for spec in args.model_checkpoint
    ]
    annotation_path = (
        ripvis_root / args.split / "coco_annotations" / f"{args.split}.json"
    )
    payload = {
        "schema_version": 1,
        "run_name": args.run_name,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "code": code_provenance(),
        "dataset": {
            "name": "RipVIS",
            "split": args.split,
            "ripvis_root": str(ripvis_root),
            "processed_root": str(processed_root),
            "annotation": file_identity(str(annotation_path)),
            "evaluated_frame_count": len(samples),
            "evaluated_video_count": len({sample["video_id"] for sample in samples}),
            "evaluated_video_ids": sorted({sample["video_id"] for sample in samples}),
            "selection": {
                "requested_videos": sorted(args.videos) if args.videos else None,
                "max_frames": args.max_frames,
            },
        },
        "parameters": {
            "threshold": args.threshold,
            "device": str(device),
            "evaluation_resolution": "source_frame",
        },
        "models_requested": [spec.model for spec in args.model_checkpoint],
        "successful_models": [r["model"] for r in results if r["status"] == "success"],
        "failed_models": [r["model"] for r in results if r["status"] == "failed"],
        "results": results,
    }
    output_dir = Path(args.output_root).expanduser().resolve() / args.run_name
    json_path, frame_csv, video_csv = write_artifacts(output_dir, payload)
    print_summary(results)
    print(f"\n[OK] JSON summary: {json_path}")
    print(f"[OK] Frame CSV: {frame_csv}")
    print(f"[OK] Video CSV: {video_csv}")
    return 1 if payload["failed_models"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
