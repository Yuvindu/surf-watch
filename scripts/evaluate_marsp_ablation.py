from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from collections import defaultdict
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.append(str(Path(__file__).resolve().parents[1]))

from configs.baseline_config import BaselineConfig
from scripts.evaluate_held_out_models import (
    METRIC_NAMES,
    ModelEvaluationSpec,
    file_identity,
    discover_samples,
    parse_model_checkpoint,
    parse_model_weight,
    validate_run_name,
    validate_unique_models,
)
from src.evaluation.marsp_ablation import (
    AblationConfiguration,
    apply_ablation_configuration,
    build_ablation_matrix,
    summarize_temporal_masks,
    summarize_temporal_video_records,
)
from src.evaluation.segmentation_metrics import (
    binary_confusion,
    metrics_from_confusion,
    summarize_records,
)
from src.models.adapter_factory import build_segmentation_adapter
from src.models.probability_ensemble import (
    fuse_probability_maps,
    normalize_model_weights,
    resolve_model_weights,
)
from src.postprocessing.temporal_aggregation import (
    TemporalAggregationConfig,
    build_motion_scores_from_metadata,
)
from src.preprocessing.motion_compensation import (
    MotionCompensationConfig,
    affine_to_params,
    estimate_partial_affine_transform,
)


REQUIRED_MODELS = ("segformer", "unet-resnet34")
TEMPORAL_METRICS = (
    "mean_consecutive_iou",
    "mean_consecutive_dice",
    "mean_pixel_change_rate",
    "std_mask_area_px",
    "mean_abs_area_change_px",
    "mean_component_count",
    "mean_small_component_count",
)


def git_revision() -> str | None:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        return None
    return completed.stdout.strip() or None


def git_worktree_dirty() -> bool | None:
    completed = subprocess.run(
        ["git", "status", "--porcelain"],
        capture_output=True,
        text=True,
    )
    return bool(completed.stdout.strip()) if completed.returncode == 0 else None


def code_provenance() -> dict:
    root = Path(__file__).resolve().parents[1]
    paths = {
        "runner": Path(__file__),
        "ablation_core": root / "src/evaluation/marsp_ablation.py",
        "metrics": root / "src/evaluation/segmentation_metrics.py",
        "temporal_aggregation": root / "src/postprocessing/temporal_aggregation.py",
        "motion_compensation": root / "src/preprocessing/motion_compensation.py",
        "ensemble": root / "src/models/probability_ensemble.py",
    }
    return {
        "git_revision": git_revision(),
        "git_worktree_dirty": git_worktree_dirty(),
        "files": {name: file_identity(str(path)) for name, path in paths.items()},
    }


def group_samples_by_video(samples: list[dict]) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for sample in samples:
        grouped[sample["video_id"]].append(sample)

    ordered = {}
    for video_id, video_samples in sorted(grouped.items()):
        video_samples = sorted(video_samples, key=lambda item: item["frame_index"])
        frame_indices = [int(item["frame_index"]) for item in video_samples]
        if len(frame_indices) != len(set(frame_indices)):
            raise ValueError(f"duplicate frame indices found for {video_id}")
        ordered[video_id] = video_samples
    return ordered


def sampled_segments(samples: list[dict], max_frame_gap: int) -> list[slice]:
    if max_frame_gap < 1:
        raise ValueError("max_frame_gap must be greater than zero")
    if not samples:
        raise ValueError("at least one sample is required")
    starts = [0]
    for index in range(1, len(samples)):
        gap = int(samples[index]["frame_index"]) - int(samples[index - 1]["frame_index"])
        if gap < 1:
            raise ValueError("sample frame indices must increase")
        if gap > max_frame_gap:
            starts.append(index)
    return [slice(start, end) for start, end in zip(starts, starts[1:] + [len(samples)])]


def validate_equal_weights(weights: dict[str, float]) -> None:
    normalized = normalize_model_weights(weights)
    if any(abs(weight - 0.5) > 1e-9 for weight in normalized.values()):
        raise ValueError("A2 is predeclared as an equal-weight ensemble; model weights must match")


def load_video_samples(samples: list[dict]) -> tuple[list[np.ndarray], list[np.ndarray]]:
    frames = []
    targets = []
    for sample in samples:
        frame = cv2.imread(str(sample["image_path"]), cv2.IMREAD_COLOR)
        target = cv2.imread(str(sample["mask_path"]), cv2.IMREAD_GRAYSCALE)
        if frame is None:
            raise ValueError(f"Could not read image: {sample['image_path']}")
        if target is None:
            raise ValueError(f"Could not read mask: {sample['mask_path']}")
        frames.append(frame)
        targets.append((target > 0).astype(np.uint8))
    return frames, targets


def estimate_sampled_motion_metadata(
    frames: list[np.ndarray],
    frame_indices: list[int],
    config: MotionCompensationConfig | None = None,
) -> tuple[list[dict], float]:
    if len(frames) != len(frame_indices):
        raise ValueError("frames and frame_indices must have the same length")
    if not frames:
        raise ValueError("at least one frame is required")

    started_at = time.perf_counter()
    config = config or MotionCompensationConfig()
    orb = cv2.ORB_create(nfeatures=config.orb_n_features)
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    metadata = []
    previous_gray = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
    for position, frame in enumerate(frames[1:], start=1):
        current_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        transform, info = estimate_partial_affine_transform(
            orb,
            matcher,
            previous_gray,
            current_gray,
            config,
        )
        if transform is None:
            dx = dy = rotation_radians = 0.0
        else:
            dx, dy, rotation_radians = affine_to_params(transform)
        metadata.append(
            {
                "from_frame_index": int(frame_indices[position - 1]),
                "frame_index": int(frame_indices[position]),
                "frame_gap": int(frame_indices[position] - frame_indices[position - 1]),
                "success": bool(info["success"]),
                "fallback": bool(info["fallback"]),
                "good_matches": int(info["good_matches"]),
                "inliers": int(info["inliers"]),
                "translation_x_px": float(dx),
                "translation_y_px": float(dy),
                "translation_magnitude_px": float(np.hypot(dx, dy)),
                "rotation_deg": float(np.degrees(rotation_radians)),
            }
        )
        previous_gray = current_gray
    return metadata, time.perf_counter() - started_at


def summarize_motion(
    metadata: list[dict], motion_scores: list[float], skipped_gap_count: int
) -> dict:
    return {
        "method": "ORB partial-affine motion estimated between ordered sampled frames",
        "coordinate_policy": (
            "Motion metadata controls temporal weights; predictions remain in source-frame "
            "coordinates for comparison with source-aligned ground truth."
        ),
        "pair_count": len(metadata),
        "skipped_gap_count": skipped_gap_count,
        "successful_pairs": sum(item["success"] for item in metadata),
        "fallback_pairs": sum(item["fallback"] for item in metadata),
        "mean_frame_gap": (
            float(np.mean([item["frame_gap"] for item in metadata]))
            if metadata
            else None
        ),
        "mean_motion_score": float(np.mean(motion_scores)),
        "max_motion_score": float(np.max(motion_scores)),
    }


def validate_probability_map(
    probability_map: np.ndarray,
    target: np.ndarray,
    label: str,
) -> np.ndarray:
    probability_map = np.asarray(probability_map, dtype=np.float32)
    if probability_map.shape != target.shape:
        raise ValueError(
            f"{label} probability shape {probability_map.shape} does not match "
            f"target shape {target.shape}"
        )
    if not np.all(np.isfinite(probability_map)):
        raise ValueError(f"{label} probability map contains non-finite values")
    if probability_map.size and (
        float(probability_map.min()) < 0.0
        or float(probability_map.max()) > 1.0
    ):
        raise ValueError(f"{label} probability map is outside [0, 1]")
    return probability_map


def infer_video_probability_maps(
    adapters: dict,
    frames: list[np.ndarray],
    targets: list[np.ndarray],
    weights: dict[str, float],
) -> tuple[dict[str, list[np.ndarray]], dict[str, float]]:
    maps_by_path = {model_name: [] for model_name in adapters}
    runtime = {model_name: 0.0 for model_name in adapters}
    for model_name, adapter in adapters.items():
        started_at = time.perf_counter()
        for frame, target in zip(frames, targets):
            probability_map = adapter.predict_probability_map(frame)
            maps_by_path[model_name].append(
                validate_probability_map(probability_map, target, model_name)
            )
        runtime[model_name] = time.perf_counter() - started_at

    fusion_started_at = time.perf_counter()
    maps_by_path["ensemble"] = [
        fuse_probability_maps(
            {model_name: maps_by_path[model_name][index] for model_name in adapters},
            weights,
        )
        for index in range(len(frames))
    ]
    fusion_seconds = time.perf_counter() - fusion_started_at
    runtime["ensemble_fusion"] = fusion_seconds
    runtime["ensemble"] = sum(runtime[name] for name in adapters) + fusion_seconds
    return maps_by_path, runtime


def build_frame_record(
    configuration: AblationConfiguration,
    sample: dict,
    probability_map: np.ndarray,
    prediction: np.ndarray,
    target: np.ndarray,
) -> dict:
    confusion = binary_confusion(prediction, target)
    pixel_count = int(target.size)
    target_positive = int(target.sum())
    prediction_positive = int(prediction.sum())
    return {
        "experiment_id": configuration.experiment_id,
        "label": configuration.label,
        "model_path": configuration.model_path,
        "video_id": sample["video_id"],
        "frame_index": int(sample["frame_index"]),
        "filename": sample["image_path"].name,
        "pixel_count": pixel_count,
        "ground_truth_positive_pixels": target_positive,
        "prediction_positive_pixels": prediction_positive,
        "ground_truth_foreground_fraction": target_positive / pixel_count,
        "prediction_foreground_fraction": prediction_positive / pixel_count,
        "mean_foreground_probability": float(np.mean(probability_map)),
        "mean_prediction_confidence": float(
            np.mean(np.maximum(probability_map, 1.0 - probability_map))
        ),
        "confusion": confusion,
        "metrics": metrics_from_confusion(confusion),
    }


def semantic_metric_deltas(candidate: dict, reference: dict) -> dict:
    deltas = {}
    for aggregation in ("micro", "macro_per_frame"):
        deltas[aggregation] = {}
        for metric in METRIC_NAMES:
            candidate_value = candidate[aggregation][metric]
            reference_value = reference[aggregation][metric]
            deltas[aggregation][metric] = (
                None
                if candidate_value is None or reference_value is None
                else float(candidate_value - reference_value)
            )
    return deltas


def temporal_metric_deltas(candidate: dict, reference: dict) -> dict:
    return {
        metric: (
            None
            if candidate["macro_per_video"][metric] is None
            or reference["macro_per_video"][metric] is None
            else float(
                candidate["macro_per_video"][metric]
                - reference["macro_per_video"][metric]
            )
        )
        for metric in TEMPORAL_METRICS
    }


def temporal_video_metric_deltas(candidate: dict, reference: dict) -> dict:
    return {
        metric: (
            None
            if candidate[metric] is None or reference[metric] is None
            else float(candidate[metric] - reference[metric])
        )
        for metric in TEMPORAL_METRICS
    }


def _flatten_semantic_summary(summary: dict) -> dict:
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
        "mean_foreground_probability": summary.get("mean_foreground_probability"),
        "mean_prediction_confidence": summary.get("mean_prediction_confidence"),
    }
    for aggregation in ("micro", "macro_per_frame"):
        for metric in METRIC_NAMES:
            row[f"{aggregation}_{metric}"] = summary[aggregation][metric]
    return row


def _flatten_temporal_summary(summary: dict) -> dict:
    row = {
        "temporal_video_count": summary["video_count"],
        "temporal_frame_count": summary["frame_count"],
        "temporal_empty_prediction_frames": summary["empty_prediction_frames"],
    }
    for metric in TEMPORAL_METRICS:
        row[f"temporal_macro_per_video_{metric}"] = summary["macro_per_video"][metric]
    return row


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_artifacts(output_dir: Path, payload: dict) -> tuple[Path, Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "marsp_ablation_summary.json"
    frame_path = output_dir / "marsp_ablation_frame_metrics.csv"
    video_path = output_dir / "marsp_ablation_video_metrics.csv"

    summary_payload = {
        **payload,
        "results": [
            {
                key: value
                for key, value in result.items()
                if key not in {"frame_records"}
            }
            for result in payload["results"]
        ],
    }
    summary_path.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")

    frame_rows = []
    for result in payload["results"]:
        for record in result["frame_records"]:
            row = {
                key: record[key]
                for key in (
                    "experiment_id",
                    "label",
                    "model_path",
                    "video_id",
                    "frame_index",
                    "filename",
                    "pixel_count",
                    "ground_truth_positive_pixels",
                    "prediction_positive_pixels",
                    "ground_truth_foreground_fraction",
                    "prediction_foreground_fraction",
                    "mean_foreground_probability",
                    "mean_prediction_confidence",
                )
            }
            row.update(record["confusion"])
            row.update(record["metrics"])
            frame_rows.append(row)
    _write_csv(frame_path, frame_rows)

    video_rows = []
    for result in payload["results"]:
        for record in result["video_records"]:
            semantic_delta = record["delta_from_reference"]["semantic"]
            temporal_delta = record["delta_from_reference"]["temporal"]
            video_rows.append(
                {
                    "experiment_id": result["configuration"]["experiment_id"],
                    "label": result["configuration"]["label"],
                    "model_path": result["configuration"]["model_path"],
                    "video_id": record["video_id"],
                    **_flatten_semantic_summary(record["semantic"]),
                    **{
                        f"temporal_{key}": value
                        for key, value in record["temporal"].items()
                    },
                    **{
                        f"runtime_{key}": value
                        for key, value in record["runtime_seconds"].items()
                    },
                    **{
                        f"delta_{aggregation}_{metric}": semantic_delta[aggregation][metric]
                        for aggregation in ("micro", "macro_per_frame")
                        for metric in METRIC_NAMES
                    },
                    **{
                        f"delta_temporal_{metric}": temporal_delta[metric]
                        for metric in TEMPORAL_METRICS
                    },
                }
            )
    _write_csv(video_path, video_rows)
    return summary_path, frame_path, video_path


def print_summary(results: list[dict]) -> None:
    def format_metric(value: float | None) -> str:
        return "n/a" if value is None else f"{value:.4f}"

    print("\nMARSP ablation evaluation")
    print("ID  path             fg IoU   fg Dice   precision  recall   total(s)")
    print("--  ---------------  -------  --------  ---------  -------  --------")
    for result in results:
        metrics = result["semantic_summary"]["micro"]
        print(
            f"{result['configuration']['experiment_id']:<3} "
            f"{result['configuration']['model_path']:<16} "
            f"{format_metric(metrics['foreground_iou']):<8} "
            f"{format_metric(metrics['foreground_dice']):<9} "
            f"{format_metric(metrics['foreground_precision']):<10} "
            f"{format_metric(metrics['foreground_recall']):<8} "
            f"{result['runtime_seconds']['estimated_total']:.2f}"
        )


def validate_model_specs(specs: list[ModelEvaluationSpec]) -> dict[str, ModelEvaluationSpec]:
    validate_unique_models(specs)
    by_model = {spec.model: spec for spec in specs}
    missing = sorted(set(REQUIRED_MODELS) - set(by_model))
    extra = sorted(set(by_model) - set(REQUIRED_MODELS))
    if missing or extra:
        details = []
        if missing:
            details.append("missing " + ", ".join(missing))
        if extra:
            details.append("unsupported for A0-A6: " + ", ".join(extra))
        raise ValueError("model checkpoints must define SegFormer and U-Net: " + "; ".join(details))
    return by_model


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the controlled A0-A6 MARSP ablation matrix."
    )
    parser.add_argument("--run-name", required=True, type=validate_run_name)
    parser.add_argument(
        "--model-checkpoint",
        action="append",
        required=True,
        type=parse_model_checkpoint,
        metavar="MODEL=CHECKPOINT",
    )
    parser.add_argument(
        "--model-weight",
        action="append",
        type=parse_model_weight,
        metavar="MODEL=WEIGHT",
        help="Optional ensemble weights; defaults to equal weighting.",
    )
    parser.add_argument(
        "--selected-model-path",
        choices=("segformer", "unet-resnet34", "ensemble"),
        default="segformer",
        help="Model path used for A3-A6. SegFormer is the current formal winner.",
    )
    parser.add_argument("--ripvis-root", default="../RipVIS")
    parser.add_argument("--processed-root", default="data/processed")
    parser.add_argument("--split", choices=("val",), default="val")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--window-size", type=int, default=5)
    parser.add_argument("--min-component-area", type=int, default=500)
    parser.add_argument("--small-component-area", type=int, default=500)
    parser.add_argument("--video", action="append", dest="videos")
    parser.add_argument("--max-frames", type=int)
    parser.add_argument(
        "--max-frame-gap",
        type=int,
        default=6,
        help="Do not aggregate or compare masks across larger source-frame gaps.",
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--output-root", default="outputs/marsp_ablation")
    args = parser.parse_args()

    try:
        specs_by_model = validate_model_specs(args.model_checkpoint)
        matrix = build_ablation_matrix(
            selected_model_path=args.selected_model_path,
            window_size=args.window_size,
            threshold=args.threshold,
            min_component_area=args.min_component_area,
        )
        weights = resolve_model_weights(
            REQUIRED_MODELS,
            [
                (weight_spec.model, weight_spec.weight)
                for weight_spec in (args.model_weight or [])
            ]
            or None,
        )
        validate_equal_weights(weights)
    except ValueError as exc:
        parser.error(str(exc))
    if args.small_component_area < 1:
        parser.error("--small-component-area must be greater than zero")
    if args.max_frames is not None and args.max_frames <= 0:
        parser.error("--max-frames must be greater than zero")
    if args.max_frame_gap < 1:
        parser.error("--max-frame-gap must be greater than zero")
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
        samples_by_video = group_samples_by_video(samples)
    except (FileNotFoundError, ValueError) as exc:
        parser.error(str(exc))

    for spec in specs_by_model.values():
        if not spec.checkpoint.is_file():
            parser.error(f"checkpoint not found: {spec.checkpoint}")
    output_dir = Path(args.output_root).expanduser().resolve() / args.run_name
    if output_dir.exists():
        parser.error(f"run output already exists; choose another run name: {output_dir}")

    adapters = {
        model_name: build_segmentation_adapter(
            model_name=model_name,
            checkpoint_path=str(spec.checkpoint),
            device=device,
            config=BaselineConfig(),
            threshold=args.threshold,
        )
        for model_name, spec in specs_by_model.items()
    }
    records_by_config = {config.experiment_id: [] for config in matrix}
    video_records_by_config = {config.experiment_id: [] for config in matrix}
    runtime_by_config = {
        config.experiment_id: {
            "inference": 0.0,
            "motion_estimation": 0.0,
            "postprocessing": 0.0,
            "estimated_total": 0.0,
        }
        for config in matrix
    }
    motion_records = []

    print(
        f"Running A0-A6 on {len(samples)} {args.split} frames from "
        f"{len(samples_by_video)} videos using {device}."
    )
    for video_number, (video_id, video_samples) in enumerate(
        samples_by_video.items(),
        start=1,
    ):
        frames, targets = load_video_samples(video_samples)
        maps_by_path, path_runtime = infer_video_probability_maps(
            adapters,
            frames,
            targets,
            weights,
        )
        frame_indices = [int(sample["frame_index"]) for sample in video_samples]
        segments = sampled_segments(video_samples, args.max_frame_gap)
        pair_breaks = {segment.start for segment in segments[1:]}
        motion_metadata = []
        motion_scores = []
        motion_seconds = 0.0
        temporal_config = TemporalAggregationConfig(
            window_size=args.window_size,
            threshold=args.threshold,
        )
        for segment in segments:
            segment_metadata, segment_seconds = estimate_sampled_motion_metadata(
                frames[segment], frame_indices[segment]
            )
            motion_metadata.extend(segment_metadata)
            motion_seconds += segment_seconds
            segment_scores = build_motion_scores_from_metadata(
                segment_metadata, len(frames[segment]), temporal_config
            )
            motion_scores.extend(segment_scores or [0.0] * len(frames[segment]))
        motion_summary = summarize_motion(motion_metadata, motion_scores, len(pair_breaks))
        motion_records.append({"video_id": video_id, **motion_summary})

        for configuration in matrix:
            processing_started_at = time.perf_counter()
            processed_maps = []
            masks = []
            for segment in segments:
                segment_maps, segment_masks = apply_ablation_configuration(
                    maps_by_path[configuration.model_path][segment],
                    configuration,
                    motion_scores=motion_scores[segment],
                )
                processed_maps.extend(segment_maps)
                masks.extend(segment_masks)
            processing_seconds = time.perf_counter() - processing_started_at
            frame_records = [
                build_frame_record(
                    configuration,
                    sample,
                    probability_map,
                    mask,
                    target,
                )
                for sample, probability_map, mask, target in zip(
                    video_samples,
                    processed_maps,
                    masks,
                    targets,
                )
            ]
            semantic_summary = summarize_records(frame_records)
            temporal_summary = summarize_temporal_masks(
                masks,
                small_component_area=args.small_component_area,
                pair_breaks=pair_breaks,
            )
            inference_seconds = path_runtime[configuration.model_path]
            adaptive_motion_seconds = (
                motion_seconds
                if configuration.temporal_mode == "motion_adaptive"
                else 0.0
            )
            runtime = {
                "inference": inference_seconds,
                "motion_estimation": adaptive_motion_seconds,
                "postprocessing": processing_seconds,
                "estimated_total": (
                    inference_seconds + adaptive_motion_seconds + processing_seconds
                ),
            }
            records_by_config[configuration.experiment_id].extend(frame_records)
            video_records_by_config[configuration.experiment_id].append(
                {
                    "video_id": video_id,
                    "semantic": semantic_summary,
                    "temporal": temporal_summary,
                    "runtime_seconds": runtime,
                }
            )
            for key, value in runtime.items():
                runtime_by_config[configuration.experiment_id][key] += value

        print(
            f"[{video_number}/{len(samples_by_video)}] {video_id}: "
            f"{len(video_samples)} frames, motion fallbacks="
            f"{motion_summary['fallback_pairs']}",
            flush=True,
        )

    results = []
    summaries_by_id = {}
    for configuration in matrix:
        experiment_id = configuration.experiment_id
        semantic_summary = summarize_records(records_by_config[experiment_id])
        temporal_summary = summarize_temporal_video_records(
            [
                {"video_id": record["video_id"], **record["temporal"]}
                for record in video_records_by_config[experiment_id]
            ]
        )
        summaries_by_id[experiment_id] = {
            "semantic": semantic_summary,
            "temporal": temporal_summary,
        }
        results.append(
            {
                "configuration": configuration.to_dict(),
                "runtime_seconds": runtime_by_config[experiment_id],
                "semantic_summary": semantic_summary,
                "temporal_summary": temporal_summary,
                "frame_records": records_by_config[experiment_id],
                "video_records": video_records_by_config[experiment_id],
            }
        )

    for result in results:
        reference_id = result["configuration"]["reference_id"]
        reference_videos = {
            record["video_id"]: record
            for record in video_records_by_config[reference_id]
        }
        for video_record in result["video_records"]:
            reference_record = reference_videos[video_record["video_id"]]
            video_record["delta_from_reference"] = {
                "reference_id": reference_id,
                "semantic": semantic_metric_deltas(
                    video_record["semantic"],
                    reference_record["semantic"],
                ),
                "temporal": temporal_video_metric_deltas(
                    video_record["temporal"],
                    reference_record["temporal"],
                ),
            }
        result["delta_from_reference"] = {
            "reference_id": reference_id,
            "semantic": semantic_metric_deltas(
                result["semantic_summary"],
                summaries_by_id[reference_id]["semantic"],
            ),
            "temporal": temporal_metric_deltas(
                result["temporal_summary"],
                summaries_by_id[reference_id]["temporal"],
            ),
        }

    annotation_path = ripvis_root / args.split / "coco_annotations" / f"{args.split}.json"
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
            "evaluated_video_count": len(samples_by_video),
            "evaluated_video_ids": list(samples_by_video),
            "selection": {
                "requested_videos": sorted(args.videos) if args.videos else None,
                "max_frames": args.max_frames,
            },
            "sequence_policy": (
                "Ordered labelled samples are divided where the source-frame gap exceeds "
                f"{args.max_frame_gap}; smoothing and consecutive-mask metrics never cross "
                "a segment boundary. The window counts labelled samples, not elapsed time."
            ),
        },
        "models": {
            model_name: {
                "checkpoint": file_identity(str(spec.checkpoint)),
                "adapter_metadata": asdict(adapters[model_name].metadata()),
            }
            for model_name, spec in specs_by_model.items()
        },
        "ensemble": {
            "method": "weighted_probability_mean",
            "weights": weights,
            "normalized_weights": normalize_model_weights(weights),
        },
        "parameters": {
            "device": str(device),
            "selected_model_path": args.selected_model_path,
            "threshold": args.threshold,
            "window_size": args.window_size,
            "min_component_area": args.min_component_area,
            "small_component_area": args.small_component_area,
            "max_frame_gap": args.max_frame_gap,
            "evaluation_resolution": "source_frame",
        },
        "coordinate_alignment": {
            "semantic_evaluation": "source-frame coordinates",
            "geometric_stabilisation_scored": False,
            "reason": (
                "Stabilised predictions cannot be compared with unwarped source masks. "
                "Motion estimates are used only for adaptive temporal weighting."
            ),
        },
        "motion_evidence": motion_records,
        "matrix": [configuration.to_dict() for configuration in matrix],
        "results": results,
    }
    summary_path, frame_path, video_path = write_artifacts(output_dir, payload)
    print_summary(results)
    print(f"\n[OK] JSON summary: {summary_path}")
    print(f"[OK] Frame CSV: {frame_path}")
    print(f"[OK] Video CSV: {video_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
