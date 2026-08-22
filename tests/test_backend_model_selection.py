import unittest
from pathlib import Path
from typing import Optional
from unittest.mock import patch

from backend.server import (
    build_comparison_command,
    default_checkpoint_for_model,
    parse_model_field,
    segmentation_models_response,
)


class StubForm:
    def __init__(self, model: Optional[str]) -> None:
        self.model = model

    def getfirst(self, name: str):
        return self.model if name == "model" else None


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

    @patch("backend.server.Path.is_file", return_value=False)
    def test_api_marks_models_without_checkpoints_unavailable(self, _) -> None:
        response = segmentation_models_response()

        self.assertTrue(response["models"])
        self.assertTrue(all(not model["available"] for model in response["models"]))


if __name__ == "__main__":
    unittest.main()
