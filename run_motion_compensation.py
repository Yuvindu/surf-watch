"""
run_motion_compensation.py
───────────────────────────────────────────────────────────────
scripts/run_motion_compensation.py

Runner script for the SurfWatch motion compensation module.
Calls src/preprocessing/motion_compensation.py and saves
results to outputs/motion_compensation/

Usage:
    python scripts/run_motion_compensation.py --input video.mp4
    python scripts/run_motion_compensation.py --input video.mp4 --smooth 30
    python scripts/run_motion_compensation.py --input video.mp4 --no-comparison
"""

import sys
import json
import argparse
from pathlib import Path

# Allow importing from src/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.preprocessing.motion_compensation import (
    MotionCompensator,
    MotionCompensationConfig,
)

OUTPUT_DIR = Path("outputs/motion_compensation")


def main():
    parser = argparse.ArgumentParser(
        description="SurfWatch — Run Motion Compensation (SCRUM-54)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/run_motion_compensation.py --input RipVIS-016.mp4
  python scripts/run_motion_compensation.py --input clip.mp4 --smooth 30 --features 1500
  python scripts/run_motion_compensation.py --input clip.mp4 --no-comparison
        """
    )
    parser.add_argument("--input",         required=True,
                        help="Path to input video")
    parser.add_argument("--output",        default=None,
                        help="Output filename (default: auto-generated)")
    parser.add_argument("--smooth",        type=int,   default=20,
                        help="Smoothing radius in frames (default: 20)")
    parser.add_argument("--features",      type=int,   default=1000,
                        help="Max ORB features per frame (default: 1000)")
    parser.add_argument("--min-matches",   type=int,   default=6,
                        help="Min matches before fallback (default: 6)")
    parser.add_argument("--crop",          type=float, default=0.03,
                        help="Border crop ratio (default: 0.03)")
    parser.add_argument("--no-comparison", action="store_true",
                        help="Output stabilised only (no side-by-side)")
    args = parser.parse_args()

    # ── Set up output path ────────────────────────────────────────────────────
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    input_stem = Path(args.input).stem
    if args.output:
        output_path = str(OUTPUT_DIR / args.output)
    else:
        output_path = str(OUTPUT_DIR / f"{input_stem}_stabilised.mp4")

    # ── Build config ──────────────────────────────────────────────────────────
    config = MotionCompensationConfig(
        smoothing_radius = args.smooth,
        max_features     = args.features,
        min_match_count  = args.min_matches,
        crop_ratio       = args.crop,
        show_comparison  = not args.no_comparison,
    )

    # ── Run pipeline ──────────────────────────────────────────────────────────
    mc     = MotionCompensator(args.input, config=config)
    result = mc.run(output_path)
    result.print_summary()

    # ── Save report ───────────────────────────────────────────────────────────
    report_path = OUTPUT_DIR / f"{input_stem}_report.json"
    with open(report_path, "w") as f:
        json.dump(result.to_dict(), f, indent=2)
    print(f"[OK] Report saved  → {report_path}")
    print(f"[OK] Video saved   → {output_path}")


if __name__ == "__main__":
    main()
