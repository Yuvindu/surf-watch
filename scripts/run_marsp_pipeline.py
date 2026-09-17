import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Optional

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.models.adapter_factory import SUPPORTED_SEGMENTATION_MODELS


def resolve_output_directories(output_root: Optional[str]) -> dict:
    if output_root is None:
        return {
            "motion": Path("outputs/motion_compensation"),
            "inference": Path("outputs/video_inference"),
            "aggregation": Path("outputs/temporal_aggregation"),
            "summary": Path("outputs/marsp"),
        }

    root = Path(output_root).expanduser().resolve()
    return {
        "motion": root / "motion_compensation",
        "inference": root / "video_inference",
        "aggregation": root / "temporal_aggregation",
        "summary": root / "summary",
    }


def run_command(command: list[str]) -> None:
    print("\n[RUN]", " ".join(command))
    result = subprocess.run(command)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {result.returncode}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video-name", required=True, help="Logical name for this run, e.g. RipVIS-051")
    parser.add_argument("--input", required=True, help="Path to input video")
    parser.add_argument(
        "--checkpoint",
        default="checkpoints/best_model.pt",
        help="Path to trained segmentation model checkpoint",
    )
    parser.add_argument(
        "--model",
        default="segformer",
        choices=SUPPORTED_SEGMENTATION_MODELS,
        help="Segmentation model adapter to use",
    )
    parser.add_argument("--window-size", type=int, default=5)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument(
        "--output-root",
        help="Optional root for structured MARSP outputs",
    )
    args = parser.parse_args()

    video_name = args.video_name
    input_video = args.input
    checkpoint = args.checkpoint

    output_directories = resolve_output_directories(args.output_root)
    motion_dir = output_directories["motion"]
    infer_dir = output_directories["inference"]
    agg_dir = output_directories["aggregation"]
    marsp_dir = output_directories["summary"]

    motion_dir.mkdir(parents=True, exist_ok=True)
    infer_dir.mkdir(parents=True, exist_ok=True)
    agg_dir.mkdir(parents=True, exist_ok=True)
    marsp_dir.mkdir(parents=True, exist_ok=True)

    # Motion compensation outputs
    stabilised_video = str(motion_dir / f"{video_name}_stabilised.mp4")
    motion_compare = str(motion_dir / f"{video_name}_comparison.mp4")
    motion_results = str(motion_dir / f"{video_name}_results.json")

    # Original inference outputs
    orig_overlay = str(infer_dir / f"{video_name}_original_overlay.mp4")
    orig_mask = str(infer_dir / f"{video_name}_original_mask.mp4")
    orig_prob = str(infer_dir / f"{video_name}_original_prob.mp4")

    # Stabilised inference outputs
    stab_overlay = str(infer_dir / f"{video_name}_stabilised_overlay.mp4")
    stab_mask = str(infer_dir / f"{video_name}_stabilised_mask.mp4")
    stab_prob = str(infer_dir / f"{video_name}_stabilised_prob.mp4")

    # Original aggregation outputs
    orig_agg_prob = str(agg_dir / f"{video_name}_original_agg_prob.mp4")
    orig_agg_mask = str(agg_dir / f"{video_name}_original_agg_mask.mp4")
    orig_agg_compare = str(agg_dir / f"{video_name}_original_agg_compare.mp4")
    orig_agg_results = str(agg_dir / f"{video_name}_original_agg_results.json")

    # Stabilised aggregation outputs
    stab_agg_prob = str(agg_dir / f"{video_name}_stabilised_agg_prob.mp4")
    stab_agg_mask = str(agg_dir / f"{video_name}_stabilised_agg_mask.mp4")
    stab_agg_compare = str(agg_dir / f"{video_name}_stabilised_agg_compare.mp4")
    stab_agg_results = str(agg_dir / f"{video_name}_stabilised_agg_results.json")

    # 1. Motion compensation
    run_command([
        sys.executable,
        "scripts/run_motion_compensation.py",
        "--input", input_video,
        "--output", stabilised_video,
        "--comparison-output", motion_compare,
        "--results", motion_results,
    ])

    # 2. Segmentation on original video
    run_command([
        sys.executable,
        "scripts/run_video_segmentation.py",
        "--input", input_video,
        "--model", args.model,
        "--checkpoint", checkpoint,
        "--output-overlay", orig_overlay,
        "--output-mask", orig_mask,
        "--output-prob", orig_prob,
        "--threshold", str(args.threshold),
    ])

    # 3. Segmentation on stabilised video
    run_command([
        sys.executable,
        "scripts/run_video_segmentation.py",
        "--input", stabilised_video,
        "--model", args.model,
        "--checkpoint", checkpoint,
        "--output-overlay", stab_overlay,
        "--output-mask", stab_mask,
        "--output-prob", stab_prob,
        "--threshold", str(args.threshold),
    ])

    # 4. Temporal aggregation on original probability video
    run_command([
        sys.executable,
        "scripts/run_temporal_aggregation.py",
        "--input-prob-video", orig_prob,
        "--output-prob-video", orig_agg_prob,
        "--output-mask-video", orig_agg_mask,
        "--output-comparison-video", orig_agg_compare,
        "--results", orig_agg_results,
        "--window-size", str(args.window_size),
        "--threshold", str(args.threshold),
        "--motion-results", motion_results,
    ])

    # 5. Temporal aggregation on stabilised probability video
    run_command([
        sys.executable,
        "scripts/run_temporal_aggregation.py",
        "--input-prob-video", stab_prob,
        "--output-prob-video", stab_agg_prob,
        "--output-mask-video", stab_agg_mask,
        "--output-comparison-video", stab_agg_compare,
        "--results", stab_agg_results,
        "--window-size", str(args.window_size),
        "--threshold", str(args.threshold),
        "--motion-results", motion_results,
    ])

    summary = {
        "video_name": video_name,
        "input_video": input_video,
        "segmentation_model": args.model,
        "checkpoint": checkpoint,
        "window_size": args.window_size,
        "threshold": args.threshold,
        "motion_adaptive_temporal_aggregation": True,
        "motion_compensation": {
            "stabilised_video": stabilised_video,
            "comparison_video": motion_compare,
            "results": motion_results,
        },
        "segmentation": {
            "original": {
                "overlay": orig_overlay,
                "mask": orig_mask,
                "prob": orig_prob,
            },
            "stabilised": {
                "overlay": stab_overlay,
                "mask": stab_mask,
                "prob": stab_prob,
            },
        },
        "temporal_aggregation": {
            "original": {
                "agg_prob": orig_agg_prob,
                "agg_mask": orig_agg_mask,
                "compare": orig_agg_compare,
                "results": orig_agg_results,
            },
            "stabilised": {
                "agg_prob": stab_agg_prob,
                "agg_mask": stab_agg_mask,
                "compare": stab_agg_compare,
                "results": stab_agg_results,
            },
        },
    }

    summary_path = marsp_dir / f"{video_name}_pipeline_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n[OK] MARSP pipeline complete")
    print(f"[OK] Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
