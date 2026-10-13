import unittest
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Optional
from unittest.mock import MagicMock, patch

from backend.server import (
    JOBS,
    build_multi_case_response,
    build_multi_model_command,
    build_comparison_command,
    default_checkpoint_for_model,
    parse_comparison_mode,
    parse_model_field,
    parse_models_field,
    run_multi_model_comparison,
    segmentation_models_response,
)


class StubForm:
    def __init__(self, model: Optional[str], mode: Optional[str] = None, models=None) -> None:
        self.model = model
        self.mode = mode
        self.models = models or []

    def getfirst(self, name: str):
        return {"model": self.model, "mode": self.mode}.get(name)

    def getlist(self, name: str):
        return self.models if name == "models" else []


class BackendModelSelectionTests(unittest.TestCase):
    def test_missing_model_uses_default(self) -> None:
        self.assertEqual(parse_model_field(StubForm(None)), "segformer")

    def test_registered_model_is_normalized(self) -> None:
        self.assertEqual(parse_model_field(StubForm("SegFormer")), "segformer")

    def test_unknown_model_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unsupported segmentation model"):
            parse_model_field(StubForm("unknown"))

    def test_selected_model_is_passed_to_comparison_cli(self) -> None:
        command = build_comparison_command(
            video_name="case-1",
            input_path=Path("input.mp4"),
            checkpoint=Path("checkpoint.pt"),
            model_name="segformer",
            window_size=7,
            threshold=0.6,
        )

        model_flag = command.index("--model")
        self.assertEqual(command[model_flag + 1], "segformer")

    def test_unet_uses_its_own_default_checkpoint(self) -> None:
        checkpoint = default_checkpoint_for_model("unet-resnet34")

        self.assertEqual(checkpoint.name, "unet_resnet34_best_model.pt")

    def test_comparison_mode_and_model_set_validation(self) -> None:
        self.assertEqual(parse_comparison_mode(StubForm(None)), "single")
        self.assertEqual(parse_comparison_mode(StubForm(None, "multi")), "multi")
        with self.assertRaisesRegex(ValueError, "mode must"):
            parse_comparison_mode(StubForm(None, "other"))
        with self.assertRaisesRegex(ValueError, "two different"):
            parse_models_field(StubForm(None, models=["segformer", "segformer"]))
        with patch("backend.server.default_checkpoint_for_model", return_value=Path(__file__)):
            self.assertEqual(
                parse_models_field(StubForm(None, models=["segformer", "unet-resnet34"])),
                ["segformer", "unet-resnet34"],
            )

    @patch("backend.server.default_checkpoint_for_model")
    def test_multi_model_command_uses_separate_registered_checkpoints(self, checkpoint) -> None:
        checkpoint.side_effect = lambda model: Path(f"/checkpoints/{model}.pt")
        command = build_multi_model_command(
            "case-1", Path("input.mp4"), ["segformer", "unet-resnet34"], 7, 0.6
        )
        pairs = [command[index + 1] for index, value in enumerate(command) if value == "--model-checkpoint"]
        self.assertEqual(pairs, [
            "segformer=/checkpoints/segformer.pt",
            "unet-resnet34=/checkpoints/unet-resnet34.pt",
        ])
        self.assertEqual(command[command.index("--window-size") + 1], "7")

    def test_multi_response_preserves_success_failure_and_provenance(self) -> None:
        summary = {
            "input": {"path": "/tmp/video.mp4", "size_bytes": 10, "sha256": "inputhash"},
            "total_runtime_seconds": 12.0,
            "results": [
                {
                    "model": "segformer", "status": "success", "frame_count": 24,
                    "checkpoint": {"sha256": "checkpoint"},
                    "provenance": {"git_revision": "abc123"},
                    "artifacts": {"side_by_side_video": "outputs/multi_model/case/segformer/compare.mp4"},
                    "metrics_path": "outputs/multi_model/case/segformer/metrics.json",
                },
                {"model": "unet-resnet34", "status": "failed", "error": "checkpoint missing"},
            ],
        }
        response = build_multi_case_response(
            "http://localhost:8000", "case", "video.mp4", Path("outputs/frontend_uploads/video.mp4"),
            "2026-10-11T00:00:00Z", ["segformer", "unet-resnet34"], 5, 0.5, summary,
        )
        self.assertEqual(response["mode"], "multi")
        self.assertEqual(response["modelResults"][0]["provenance"]["git_revision"], "abc123")
        self.assertIn("/artifacts/outputs/multi_model/case/segformer/compare.mp4", response["overlayUrl"])
        self.assertEqual(response["modelResults"][1]["error"], "checkpoint missing")

    @patch("backend.server.subprocess.Popen")
    def test_partial_multi_model_failure_keeps_successful_result(self, popen) -> None:
        process = MagicMock()
        process.stdout = [
            "[MODEL] segformer\n",
            "[STAGE] Running baseline segmentation adapter\n",
            "[MODEL] unet-resnet34\n",
        ]
        process.wait.return_value = 1
        popen.return_value = process
        with TemporaryDirectory() as directory, patch("backend.server.MULTI_MODEL_DIR", Path(directory)):
            run_root = Path(directory) / "case"
            run_root.mkdir()
            (run_root / "multi_model_comparison_summary.json").write_text(json.dumps({
                "successful_models": ["segformer"],
                "failed_models": ["unet-resnet34"],
                "results": [
                    {"model": "segformer", "status": "success"},
                    {"model": "unet-resnet34", "status": "failed", "error": "checkpoint missing"},
                ],
            }), encoding="utf-8")
            JOBS["case"] = {"jobId": "case"}
            try:
                summary = run_multi_model_comparison(
                    "case", Path("video.mp4"), ["segformer", "unet-resnet34"], 5, 0.5, "case"
                )
                self.assertEqual(summary["successful_models"], ["segformer"])
                self.assertEqual(JOBS["case"]["currentModel"], "unet-resnet34")
                self.assertEqual(JOBS["case"]["modelIndex"], 2)
            finally:
                JOBS.pop("case", None)

    @patch("backend.server.Path.is_file", return_value=False)
    def test_api_marks_models_without_checkpoints_unavailable(self, _) -> None:
        response = segmentation_models_response()

        self.assertTrue(response["models"])
        self.assertTrue(all(not model["available"] for model in response["models"]))


if __name__ == "__main__":
    unittest.main()
