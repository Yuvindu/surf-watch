from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.append(str(Path(__file__).resolve().parents[1]))

from configs.baseline_config import BaselineConfig
from scripts.evaluate_held_out_models import (
    discover_samples,
    git_revision,
    git_worktree_dirty,
    parse_model_checkpoint,
    validate_run_name,
    validate_unique_models,
)
from scripts.run_baseline_vs_marsp_compare import file_identity
from src.evaluation.calibration_metrics import (
    CalibrationAccumulator,
    DisagreementAccumulator,
    NLL_EPSILON,
    foreground_candidate_mask,
)
from src.models.adapter_factory import build_segmentation_adapter
from src.models.probability_ensemble import fuse_probability_maps


COMPONENTS = ("segformer", "unet-resnet34")
MODEL_PATHS = (*COMPONENTS, "ensemble")
COHORTS = ("all_pixels", "candidate_region")


def new_model_accumulators(bin_count: int) -> dict:
    return {
        model: {cohort: CalibrationAccumulator(bin_count) for cohort in COHORTS}
        for model in MODEL_PATHS
    }


def new_disagreement_accumulators() -> dict:
    return {cohort: DisagreementAccumulator() for cohort in COHORTS}


def summarize_model_accumulators(accumulators: dict) -> dict:
    return {
        model: {cohort: accumulator.summary() for cohort, accumulator in cohorts.items()}
        for model, cohorts in accumulators.items()
    }


def summarize_disagreement(accumulators: dict) -> dict:
    return {cohort: accumulator.summary() for cohort, accumulator in accumulators.items()}


def evaluate_samples(
    samples: list[dict],
    adapters: dict,
    *,
    bin_count: int,
    threshold: float,
) -> dict:
    if set(adapters) != set(COMPONENTS):
        raise ValueError("uncertainty evaluation requires SegFormer and U-Net ResNet34")
    dataset_models = new_model_accumulators(bin_count)
    dataset_disagreement = new_disagreement_accumulators()
    video_models = {}
    video_disagreement = {}
    video_frame_counts = {}
    frame_rows = []
    inference_seconds = {model: 0.0 for model in COMPONENTS}

    for sample_number, sample in enumerate(samples, start=1):
        frame = cv2.imread(str(sample["image_path"]), cv2.IMREAD_COLOR)
        target = cv2.imread(str(sample["mask_path"]), cv2.IMREAD_GRAYSCALE)
        if frame is None or target is None:
            raise ValueError(f"Could not read image or mask for {sample['image_path']}")
        target = (target > 0).astype(np.uint8)

        probabilities = {}
        for model in COMPONENTS:
            started_at = time.perf_counter()
            probabilities[model] = adapters[model].predict_probability_map(frame)
            inference_seconds[model] += time.perf_counter() - started_at
        candidate = foreground_candidate_mask(
            target, probabilities[COMPONENTS[0]], probabilities[COMPONENTS[1]], threshold
        )
        probabilities["ensemble"] = fuse_probability_maps(
            {model: probabilities[model] for model in COMPONENTS},
            {model: 1.0 for model in COMPONENTS},
        )

        video_id = sample["video_id"]
        if video_id not in video_models:
            video_models[video_id] = new_model_accumulators(bin_count)
            video_disagreement[video_id] = new_disagreement_accumulators()
            video_frame_counts[video_id] = 0
        video_frame_counts[video_id] += 1

        for model in MODEL_PATHS:
            frame_summaries = {}
            for cohort, selection in (("all_pixels", None), ("candidate_region", candidate)):
                frame_accumulator = CalibrationAccumulator(bin_count)
                frame_accumulator.update(probabilities[model], target, selection)
                dataset_models[model][cohort].merge(frame_accumulator)
                video_models[video_id][model][cohort].merge(frame_accumulator)
                frame_summaries[cohort] = frame_accumulator.summary()
            frame_rows.append(
                {
                    "video_id": video_id,
                    "frame_index": sample["frame_index"],
                    "filename": sample["image_path"].name,
                    "model": model,
                    **flatten_model_summaries(frame_summaries),
                }
            )

        frame_disagreement = {}
        for cohort, selection in (("all_pixels", None), ("candidate_region", candidate)):
            frame_accumulator = DisagreementAccumulator()
            frame_accumulator.update(
                probabilities[COMPONENTS[0]],
                probabilities[COMPONENTS[1]],
                threshold,
                selection,
            )
            dataset_disagreement[cohort].merge(frame_accumulator)
            video_disagreement[video_id][cohort].merge(frame_accumulator)
            frame_disagreement[cohort] = frame_accumulator.summary()
        for row in frame_rows[-len(MODEL_PATHS) :]:
            row.update(flatten_disagreement(frame_disagreement))

        if sample_number % 100 == 0 or sample_number == len(samples):
            print(f"Evaluated {sample_number}/{len(samples)} frames", flush=True)

    video_records = [
        {
            "video_id": video_id,
            "frame_count": video_frame_counts[video_id],
            "models": summarize_model_accumulators(video_models[video_id]),
            "disagreement": summarize_disagreement(video_disagreement[video_id]),
        }
        for video_id in sorted(video_models)
    ]
    high_disagreement_frames = sorted(
        (row for row in frame_rows if row["model"] == "ensemble"),
        key=lambda row: (
            -(row["candidate_region_mask_disagreement_fraction"] or 0.0),
            row["video_id"],
            row["frame_index"],
        ),
    )[:10]
    return {
        "aggregate": {
            "models": summarize_model_accumulators(dataset_models),
            "disagreement": summarize_disagreement(dataset_disagreement),
        },
        "video_records": video_records,
        "frame_rows": frame_rows,
        "high_disagreement_frames": [
            {
                key: row[key]
                for key in (
                    "video_id",
                    "frame_index",
                    "filename",
                    "candidate_region_pixel_count",
                    "candidate_region_mean_absolute_probability_difference",
                    "candidate_region_mask_disagreement_fraction",
                )
            }
            for row in high_disagreement_frames
        ],
        "inference_seconds": inference_seconds,
    }


def flatten_model_summaries(summaries: dict) -> dict:
    fields = (
        "pixel_count",
        "positive_pixel_count",
        "foreground_prevalence",
        "brier_score",
        "negative_log_likelihood",
        "mean_entropy_nats",
        "foreground_ece",
    )
    return {
        f"{cohort}_{field}": summary[field]
        for cohort, summary in summaries.items()
        for field in fields
    }


def flatten_disagreement(summaries: dict) -> dict:
    fields = (
        "mean_absolute_probability_difference",
        "mask_disagreement_fraction",
    )
    return {
        f"{cohort}_{field}": summary[field]
        for cohort, summary in summaries.items()
        for field in fields
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        if not rows:
            return
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_artifacts(output_dir: Path, payload: dict, evaluated: dict) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=False)
    summary_path = output_dir / "uncertainty_summary.json"
    frame_path = output_dir / "uncertainty_frame_metrics.csv"
    video_path = output_dir / "uncertainty_video_metrics.csv"
    reliability_path = output_dir / "uncertainty_reliability_bins.csv"
    summary_path.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    write_csv(frame_path, evaluated["frame_rows"])

    video_rows = []
    for video in evaluated["video_records"]:
        for model, cohorts in video["models"].items():
            video_rows.append(
                {
                    "video_id": video["video_id"],
                    "frame_count": video["frame_count"],
                    "model": model,
                    **flatten_model_summaries(cohorts),
                    **flatten_disagreement(video["disagreement"]),
                }
            )
    write_csv(video_path, video_rows)

    bin_rows = []
    for model, cohorts in evaluated["aggregate"]["models"].items():
        for cohort, summary in cohorts.items():
            for index, bin_record in enumerate(summary["bins"]):
                bin_rows.append(
                    {"model": model, "cohort": cohort, "bin_index": index, **bin_record}
                )
    write_csv(reliability_path, bin_rows)
    return [summary_path, frame_path, video_path, reliability_path]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Measure descriptive calibration and disagreement on matched RipVIS frames."
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
    parser.add_argument("--video", action="append", dest="videos")
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--bins", type=int, default=15)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--output-root", default="outputs/uncertainty_evaluation")
    args = parser.parse_args()

    try:
        validate_unique_models(args.model_checkpoint)
    except ValueError as exc:
        parser.error(str(exc))
    specs = {spec.model: spec for spec in args.model_checkpoint}
    if set(specs) != set(COMPONENTS):
        parser.error("provide exactly segformer and unet-resnet34 checkpoints")
    if args.bins < 2:
        parser.error("--bins must be at least 2")
    if not np.isfinite(args.threshold) or not 0.0 <= args.threshold <= 1.0:
        parser.error("--threshold must be within [0, 1]")
    if args.max_frames is not None and args.max_frames < 1:
        parser.error("--max-frames must be greater than zero")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA was requested but is not available")
    for spec in specs.values():
        if not spec.checkpoint.is_file():
            parser.error(f"checkpoint not found: {spec.checkpoint}")
    use_cuda = args.device == "cuda" or (
        args.device == "auto" and torch.cuda.is_available()
    )
    device = torch.device("cuda" if use_cuda else "cpu")
    ripvis_root = Path(args.ripvis_root).expanduser().resolve()
    processed_root = Path(args.processed_root).expanduser().resolve()
    try:
        samples = discover_samples(
            ripvis_root,
            processed_root,
            "val",
            selected_videos=set(args.videos) if args.videos else None,
            max_frames=args.max_frames,
        )
    except (FileNotFoundError, ValueError) as exc:
        parser.error(str(exc))
    output_dir = Path(args.output_root).expanduser().resolve() / args.run_name
    if output_dir.exists():
        parser.error(f"run directory already exists: {output_dir}")

    started_at = time.perf_counter()
    adapters = {
        model: build_segmentation_adapter(
            model_name=model,
            checkpoint_path=str(specs[model].checkpoint),
            device=device,
            config=BaselineConfig(),
            threshold=args.threshold,
        )
        for model in COMPONENTS
    }
    evaluated = evaluate_samples(
        samples, adapters, bin_count=args.bins, threshold=args.threshold
    )
    sample_ids = "\n".join(
        f"{sample['video_id']}:{sample['frame_index']}:{sample['image_path'].name}"
        for sample in samples
    )
    repository_root = Path(__file__).resolve().parents[1]
    payload = {
        "schema_version": 1,
        "run_name": args.run_name,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "code": {
            "git_revision": git_revision(),
            "git_worktree_dirty": git_worktree_dirty(),
            "files": {
                "evaluator": file_identity(str(Path(__file__))),
                "metrics": file_identity(
                    str(repository_root / "src/evaluation/calibration_metrics.py")
                ),
                "ensemble": file_identity(
                    str(repository_root / "src/models/probability_ensemble.py")
                ),
            },
        },
        "dataset": {
            "name": "RipVIS",
            "split": "val",
            "ripvis_root": str(ripvis_root),
            "processed_root": str(processed_root),
            "annotation": file_identity(
                str(ripvis_root / "val" / "coco_annotations" / "val.json")
            ),
            "evaluated_frame_count": len(samples),
            "evaluated_video_ids": sorted({sample["video_id"] for sample in samples}),
            "ordered_sample_sha256": hashlib.sha256(sample_ids.encode("utf-8")).hexdigest(),
            "selection": {
                "videos": sorted(args.videos) if args.videos else None,
                "max_frames": args.max_frames,
            },
        },
        "models": {
            model: {
                "checkpoint": file_identity(str(specs[model].checkpoint)),
                "adapter_metadata": asdict(adapters[model].metadata()),
            }
            for model in COMPONENTS
        },
        "ensemble": {
            "method": "weighted_probability_mean",
            "weights": {model: 0.5 for model in COMPONENTS},
        },
        "parameters": {
            "device": str(device),
            "threshold": args.threshold,
            "bin_count": args.bins,
            "nll_epsilon": NLL_EPSILON,
            "evaluation_resolution": "source_frame",
            "candidate_region": (
                "ground-truth foreground OR either component probability >= threshold"
            ),
            "calibration_transform": None,
        },
        "aggregate": evaluated["aggregate"],
        "video_records": evaluated["video_records"],
        "high_disagreement_frames": evaluated["high_disagreement_frames"],
        "runtime_seconds": {
            "component_inference": evaluated["inference_seconds"],
            "total": time.perf_counter() - started_at,
        },
        "interpretation": (
            "Descriptive validation diagnostics only; checkpoint selection used this split."
        ),
    }
    paths = write_artifacts(output_dir, payload, evaluated)
    for path in paths:
        print(f"[OK] {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
