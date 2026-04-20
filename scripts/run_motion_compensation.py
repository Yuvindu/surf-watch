import argparse
import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.preprocessing.motion_compensation import (
    MotionCompensationConfig,
    stabilise_video,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to input video")
    parser.add_argument("--output", required=True, help="Path to stabilised output video")
    parser.add_argument(
        "--comparison-output",
        default=None,
        help="Optional side-by-side original vs stabilised output video",
    )
    parser.add_argument(
        "--results",
        default="outputs/motion_compensation/results.json",
        help="Path to save metrics/results JSON",
    )
    args = parser.parse_args()

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.results).parent.mkdir(parents=True, exist_ok=True)
    if args.comparison_output is not None:
        Path(args.comparison_output).parent.mkdir(parents=True, exist_ok=True)

    config = MotionCompensationConfig()
    results = stabilise_video(
        input_path=args.input,
        output_path=args.output,
        comparison_output_path=args.comparison_output,
        config=config,
    )

    with open(args.results, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()