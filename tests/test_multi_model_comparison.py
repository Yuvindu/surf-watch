import argparse
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.run_marsp_pipeline import resolve_output_directories
from scripts.run_multi_model_comparison import (
    ModelRunSpec,
    build_model_command,
    parse_model_checkpoint,
    run_model,
    validate_run_name,
    validate_unique_models,
    write_summary_artifacts,
)


class MultiModelComparisonTests(unittest.TestCase):
    def test_model_checkpoint_parser_normalizes_registered_model(self) -> None:
        spec = parse_model_checkpoint(" SegFormer =/tmp/model.pt")

        self.assertEqual(spec.model, "segformer")
        self.assertEqual(spec.checkpoint, Path("/tmp/model.pt").resolve())

    def test_unknown_model_is_rejected(self) -> None:
        with self.assertRaisesRegex(
            argparse.ArgumentTypeError,
            "Unsupported segmentation model",
        ):
            parse_model_checkpoint("unknown=/tmp/model.pt")

    def test_run_name_rejects_paths(self) -> None:
        with self.assertRaisesRegex(argparse.ArgumentTypeError, "only letters"):
            validate_run_name("../comparison")

    def test_marsp_structured_output_directories_share_the_requested_root(
        self,
    ) -> None:
        directories = resolve_output_directories("/tmp/model/marsp")

        self.assertEqual(
            directories["motion"],
            Path("/tmp/model/marsp/motion_compensation").resolve(),
        )
        self.assertEqual(
            directories["summary"],
            Path("/tmp/model/marsp/summary").resolve(),
        )

    def test_marsp_default_directories_remain_backward_compatible(self) -> None:
        directories = resolve_output_directories(None)

        self.assertEqual(directories["motion"], Path("outputs/motion_compensation"))
        self.assertEqual(directories["summary"], Path("outputs/marsp"))

    def test_duplicate_model_specs_are_rejected(self) -> None:
        specs = [
            ModelRunSpec("segformer", Path("/tmp/one.pt")),
            ModelRunSpec("segformer", Path("/tmp/two.pt")),
        ]

        with self.assertRaisesRegex(ValueError, "only once"):
            validate_unique_models(specs)

    def test_model_command_uses_model_scoped_output_root(self) -> None:
        command = build_model_command(
            spec=ModelRunSpec("unet-resnet34", Path("/tmp/unet.pt")),
            video_name="RipVIS-072",
            input_video=Path("/tmp/input.mp4"),
            model_output_root=Path("/tmp/run/unet-resnet34"),
            window_size=5,
            threshold=0.5,
            reuse_existing=True,
        )

        self.assertEqual(
            command[command.index("--output-root") + 1],
            "/tmp/run/unet-resnet34",
        )
        self.assertEqual(
            command[command.index("--model") + 1],
            "unet-resnet34",
        )
        self.assertIn("--reuse-existing", command)

    @patch(
        "scripts.run_multi_model_comparison.subprocess.run",
        return_value=subprocess.CompletedProcess([], 4),
    )
    def test_model_failure_is_returned_without_raising(self, _) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "model.pt"
            input_video = root / "input.mp4"
            checkpoint.write_bytes(b"checkpoint")
            input_video.write_bytes(b"video")

            result = run_model(
                spec=ModelRunSpec("segformer", checkpoint),
                video_name="case",
                input_video=input_video,
                run_root=root / "run",
                window_size=5,
                threshold=0.5,
                reuse_existing=False,
            )

        self.assertEqual(result["status"], "failed")
        self.assertIn("code 4", result["error"])

    def test_summary_writes_json_and_csv_for_success_and_failure(self) -> None:
        summary = {
            "results": [
                {
                    "model": "segformer",
                    "status": "success",
                    "frame_count": 24,
                    "runner_runtime_seconds": 1.0,
                    "workflow_timing_seconds": {
                        "baseline": 0.2,
                        "marsp": 0.7,
                    },
                    "baseline": {"mean_consecutive_iou": 0.8},
                    "marsp": {"mean_consecutive_iou": 0.9},
                    "comparison": {"delta_mean_consecutive_iou": 0.1},
                },
                {
                    "model": "unet-resnet34",
                    "status": "failed",
                    "error": "checkpoint missing",
                },
            ]
        }
        with tempfile.TemporaryDirectory() as directory:
            json_path, csv_path = write_summary_artifacts(Path(directory), summary)
            csv_text = csv_path.read_text(encoding="utf-8")

            self.assertTrue(json_path.is_file())
            self.assertIn("segformer,success", csv_text)
            self.assertIn("unet-resnet34,failed,checkpoint missing", csv_text)


if __name__ == "__main__":
    unittest.main()
