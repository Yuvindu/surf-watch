from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))

from scripts.run_baseline_vs_marsp_compare import file_identity, open_comparison_video_writer
from src.models.model_registry import get_segmentation_model_option


RUN_NAME_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
METRIC_KEYS = (
    "mean_consecutive_iou",
    "mean_consecutive_dice",
    "mean_pixel_change_rate",
    "std_mask_area_px",
    "mean_abs_area_change_px",
    "mean_small_blob_count",
)


@dataclass(frozen=True)
class ModelRunSpec:
    model: str
    checkpoint: Path | None


def parse_model_checkpoint(value: str) -> ModelRunSpec:
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

    return ModelRunSpec(
        model=model,
        checkpoint=Path(checkpoint_value.strip()).expanduser().resolve(),
    )


def validate_run_name(value: str) -> str:
    if not RUN_NAME_PATTERN.fullmatch(value):
        raise argparse.ArgumentTypeError(
            "run name must contain only letters, numbers, dots, underscores, and hyphens"
        )
    return value


def validate_unique_models(specs: Iterable[ModelRunSpec]) -> None:
    models = [spec.model for spec in specs]
    duplicates = sorted(model for model in set(models) if models.count(model) > 1)
    if duplicates:
        raise ValueError("Each model can appear only once: " + ", ".join(duplicates))


def build_model_command(
    *,
    spec: ModelRunSpec,
    video_name: str,
    input_video: Path,
    model_output_root: Path,
    window_size: int,
    threshold: float,
    reuse_existing: bool,
    ensemble_specs: tuple[ModelRunSpec, ...] = (),
) -> list[str]:
    command = [
        sys.executable,
        "scripts/run_baseline_vs_marsp_compare.py",
        "--video-name",
        video_name,
        "--input",
        str(input_video),
        "--model",
        spec.model,
        "--window-size",
        str(window_size),
        "--threshold",
        str(threshold),
        "--output-root",
        str(model_output_root),
    ]
    if spec.model == "ensemble":
        for component in ensemble_specs:
            command.extend(["--ensemble-checkpoint", f"{component.model}={component.checkpoint}"])
    else:
        command.extend(["--checkpoint", str(spec.checkpoint)])
    if reuse_existing:
        command.append("--reuse-existing")
    return command


def failed_result(
    *,
    spec: ModelRunSpec,
    model_output_root: Path,
    runtime_seconds: float,
    error: str,
) -> dict:
    return {
        "model": spec.model,
        "status": "failed",
        "error": error,
        "checkpoint": str(spec.checkpoint) if spec.checkpoint else None,
        "model_output_root": str(model_output_root),
        "runner_runtime_seconds": runtime_seconds,
    }


def run_model(
    *,
    spec: ModelRunSpec,
    video_name: str,
    input_video: Path,
    run_root: Path,
    window_size: int,
    threshold: float,
    reuse_existing: bool,
    ensemble_specs: tuple[ModelRunSpec, ...] = (),
) -> dict:
    model_output_root = run_root / spec.model
    model_output_root.mkdir(parents=True, exist_ok=True)
    started_at = time.perf_counter()

    checkpoints = ensemble_specs if spec.model == "ensemble" else (spec,)
    missing = next((item for item in checkpoints if item.checkpoint is None or not item.checkpoint.is_file()), None)
    if missing is not None or not checkpoints:
        return failed_result(
            spec=spec,
            model_output_root=model_output_root,
            runtime_seconds=time.perf_counter() - started_at,
            error=f"Checkpoint not found: {missing.checkpoint if missing else 'ensemble components'}",
        )

    command = build_model_command(
        spec=spec,
        video_name=video_name,
        input_video=input_video,
        model_output_root=model_output_root,
        window_size=window_size,
        threshold=threshold,
        reuse_existing=reuse_existing,
        ensemble_specs=ensemble_specs,
    )
    print(f"\n[MODEL] {spec.model}", flush=True)
    print("[RUN]", " ".join(command), flush=True)
    completed = subprocess.run(command)
    runtime_seconds = time.perf_counter() - started_at

    if completed.returncode != 0:
        return failed_result(
            spec=spec,
            model_output_root=model_output_root,
            runtime_seconds=runtime_seconds,
            error=f"Comparison command exited with code {completed.returncode}",
        )

    metrics_path = (
        model_output_root
        / "comparison"
        / f"{video_name}_baseline_vs_marsp_metrics.json"
    )
    if not metrics_path.is_file():
        return failed_result(
            spec=spec,
            model_output_root=model_output_root,
            runtime_seconds=runtime_seconds,
            error=f"Comparison metrics were not generated: {metrics_path}",
        )

    try:
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return failed_result(
            spec=spec,
            model_output_root=model_output_root,
            runtime_seconds=runtime_seconds,
            error=f"Comparison metrics are invalid JSON: {exc}",
        )

    checkpoint_identity = metrics.get("provenance", {}).get("checkpoint")
    if checkpoint_identity is None and spec.checkpoint:
        checkpoint_identity = file_identity(str(spec.checkpoint))

    return {
        "model": spec.model,
        "status": "success",
        "error": None,
        "checkpoint": checkpoint_identity,
        "component_checkpoints": metrics.get("provenance", {}).get("component_checkpoints"),
        "model_output_root": str(model_output_root),
        "runner_runtime_seconds": runtime_seconds,
        "frame_count": metrics.get("baseline", {}).get("total_frames"),
        "workflow_timing_seconds": metrics.get("timing_seconds", {}),
        "provenance": metrics.get("provenance", {}),
        "baseline": metrics.get("baseline", {}),
        "marsp": metrics.get("marsp", {}),
        "comparison": metrics.get("comparison", {}),
        "metric_table": metrics.get("metric_table", []),
        "artifacts": metrics.get("artifacts", {}),
        "metrics_path": str(metrics_path),
    }


def csv_row(result: dict) -> dict:
    row = {
        "model": result["model"],
        "status": result["status"],
        "error": result.get("error") or "",
        "frame_count": result.get("frame_count", ""),
        "runner_runtime_seconds": result.get("runner_runtime_seconds", ""),
        "baseline_runtime_seconds": result.get(
            "workflow_timing_seconds", {}
        ).get("baseline", ""),
        "marsp_runtime_seconds": result.get("workflow_timing_seconds", {}).get(
            "marsp", ""
        ),
        "metrics_path": result.get("metrics_path", ""),
    }
    for metric in METRIC_KEYS:
        row[f"baseline_{metric}"] = result.get("baseline", {}).get(metric, "")
        row[f"marsp_{metric}"] = result.get("marsp", {}).get(metric, "")
        row[f"delta_{metric}"] = result.get("comparison", {}).get(
            f"delta_{metric}", ""
        )
    return row


def write_summary_artifacts(run_root: Path, summary: dict) -> tuple[Path, Path]:
    run_root.mkdir(parents=True, exist_ok=True)
    json_path = run_root / "multi_model_comparison_summary.json"
    csv_path = run_root / "multi_model_comparison_summary.csv"

    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    rows = [csv_row(result) for result in summary["results"]]
    fieldnames = list(csv_row({"model": "", "status": ""}).keys())
    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return json_path, csv_path


def print_result_table(results: list[dict]) -> None:
    print("\nMulti-model comparison summary")
    print("model             status   frames   baseline(s)   marsp(s)")
    print("----------------  -------  -------  ------------  --------")
    for result in results:
        timing = result.get("workflow_timing_seconds", {})
        baseline = timing.get("baseline")
        marsp = timing.get("marsp")
        baseline_text = f"{baseline:.2f}" if isinstance(baseline, (int, float)) else "-"
        marsp_text = f"{marsp:.2f}" if isinstance(marsp, (int, float)) else "-"
        print(
            f"{result['model']:<16}  "
            f"{result['status']:<7}  "
            f"{str(result.get('frame_count', '-')):<7}  "
            f"{baseline_text:<12}  "
            f"{marsp_text}"
        )
        if result.get("error"):
            print(f"  error: {result['error']}")


def save_three_way_video(results: list[dict], output_path: Path) -> None:
    expected = ("segformer", "unet-resnet34", "ensemble")
    paths = [
        next(result for result in results if result["model"] == name)["artifacts"]["marsp_overlay"]
        for name in expected
    ]
    captures = [cv2.VideoCapture(path) for path in paths]
    writer = None
    try:
        if not all(capture.isOpened() for capture in captures):
            raise ValueError("Could not open every MARSP overlay for the three-way video")
        sizes = [(
            int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        ) for capture in captures]
        if len(set(sizes)) != 1 or min(sizes[0]) <= 0:
            raise ValueError("Three-way videos must have matching frame sizes")
        rates = [capture.get(cv2.CAP_PROP_FPS) for capture in captures]
        if max(rates) - min(rates) > 0.01:
            raise ValueError("Three-way videos must have matching frame rates")
        width, height = sizes[0]
        panel_width = min(width, 640)
        panel_width -= panel_width % 2
        panel_height = max(2, round(height * panel_width / width))
        panel_height -= panel_height % 2
        output_path.parent.mkdir(parents=True, exist_ok=True)
        writer = open_comparison_video_writer(
            str(output_path), rates[0] or 30.0, (panel_width * 3, panel_height)
        )
        labels = ("SegFormer", "U-Net ResNet34", "Equal-weight ensemble")
        count = 0
        while True:
            reads = [capture.read() for capture in captures]
            if not any(ok for ok, _ in reads):
                break
            if not all(ok for ok, _ in reads):
                raise ValueError("Three-way videos have different frame counts")
            panels = []
            for (_, frame), label in zip(reads, labels):
                panel = cv2.resize(frame, (panel_width, panel_height))
                cv2.rectangle(panel, (0, 0), (panel_width, 42), (0, 0, 0), -1)
                cv2.putText(panel, label, (12, 29), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                panels.append(panel)
            writer.write(np.hstack(panels))
            count += 1
        if count < 2:
            raise ValueError("Three-way comparison needs at least two frames")
    finally:
        for capture in captures:
            capture.release()
        if writer is not None:
            writer.release()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", required=True, type=validate_run_name)
    parser.add_argument("--video-name", required=True, type=validate_run_name)
    parser.add_argument("--input", required=True, help="Path to the shared input video")
    parser.add_argument(
        "--model-checkpoint",
        action="append",
        required=True,
        type=parse_model_checkpoint,
        metavar="MODEL=CHECKPOINT",
        help="Registered model and checkpoint pair; repeat for each model",
    )
    parser.add_argument("--window-size", type=int, default=5)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument(
        "--output-root",
        default="outputs/multi_model",
        help="Root directory for all multi-model runs",
    )
    parser.add_argument("--reuse-existing", action="store_true")
    parser.add_argument("--include-ensemble", action="store_true", help="Run equal-weight probability fusion as a third comparison")
    args = parser.parse_args()

    if args.window_size <= 0:
        parser.error("--window-size must be greater than zero")
    if not 0.0 <= args.threshold <= 1.0:
        parser.error("--threshold must be between 0.0 and 1.0")

    try:
        validate_unique_models(args.model_checkpoint)
    except ValueError as exc:
        parser.error(str(exc))
    if args.include_ensemble and len(args.model_checkpoint) != 2:
        parser.error("--include-ensemble requires both registered component models")

    input_video = Path(args.input).expanduser().resolve()
    if not input_video.is_file():
        parser.error(f"Input video not found: {input_video}")

    run_root = Path(args.output_root).expanduser().resolve() / args.run_name
    started_at = time.perf_counter()
    results = [
        run_model(
            spec=spec,
            video_name=args.video_name,
            input_video=input_video,
            run_root=run_root,
            window_size=args.window_size,
            threshold=args.threshold,
            reuse_existing=args.reuse_existing,
        )
        for spec in args.model_checkpoint
    ]
    if args.include_ensemble:
        results.append(run_model(
            spec=ModelRunSpec("ensemble", None),
            video_name=args.video_name,
            input_video=input_video,
            run_root=run_root,
            window_size=args.window_size,
            threshold=args.threshold,
            reuse_existing=args.reuse_existing,
            ensemble_specs=tuple(args.model_checkpoint),
        ))

    three_way_video = None
    if all(any(result["model"] == model and result["status"] == "success" for result in results)
           for model in ("segformer", "unet-resnet34", "ensemble")):
        print("[STAGE] Rendering three-way comparison video", flush=True)
        path = run_root / "three_way_marsp.mp4"
        save_three_way_video(results, path)
        three_way_video = str(path)

    summary = {
        "schema_version": 1,
        "run_name": args.run_name,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "input": file_identity(str(input_video)),
        "parameters": {
            "video_name": args.video_name,
            "window_size": args.window_size,
            "threshold": args.threshold,
        },
        "models_requested": [spec.model for spec in args.model_checkpoint] + (["ensemble"] if args.include_ensemble else []),
        "ensemble": {"method": "weighted_probability_mean", "weights": {spec.model: 0.5 for spec in args.model_checkpoint}} if args.include_ensemble else None,
        "artifacts": {"three_way_video": three_way_video} if three_way_video else {},
        "successful_models": [
            result["model"] for result in results if result["status"] == "success"
        ],
        "failed_models": [
            result["model"] for result in results if result["status"] == "failed"
        ],
        "total_runtime_seconds": time.perf_counter() - started_at,
        "results": results,
    }
    json_path, csv_path = write_summary_artifacts(run_root, summary)
    print_result_table(results)
    print(f"\n[OK] JSON summary: {json_path}")
    print(f"[OK] CSV summary: {csv_path}")

    return 1 if summary["failed_models"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
