import argparse
import tempfile
import unittest
from pathlib import Path

import numpy as np

from scripts.evaluate_held_out_models import (
    ModelEvaluationSpec,
    evaluate_ensemble,
    parse_model_weight,
)
from src.models.probability_ensemble import (
    ProbabilityEnsembleAdapter,
    fuse_probability_maps,
    normalize_model_weights,
    resolve_model_weights,
)
from src.models.segmentation_interface import SegmentationModelMetadata


class StaticAdapter:
    def __init__(self, probability_map: np.ndarray) -> None:
        self.probability_map = probability_map

    def predict_probability_map(self, frame: np.ndarray) -> np.ndarray:
        return self.probability_map.copy()

    def metadata(self) -> SegmentationModelMetadata:
        return SegmentationModelMetadata(
            name="static",
            input_size=(2, 2),
            threshold=0.5,
            device="cpu",
        )


class ProbabilityEnsembleTests(unittest.TestCase):
    def test_omitted_weights_resolve_to_equal_weight_baseline(self) -> None:
        self.assertEqual(
            resolve_model_weights(["first", "second"]),
            {"first": 1.0, "second": 1.0},
        )

    def test_equal_weight_fusion_averages_probability_maps(self) -> None:
        first = np.array([[0.0, 0.2], [0.8, 1.0]], dtype=np.float32)
        second = np.array([[1.0, 0.6], [0.4, 0.0]], dtype=np.float32)

        fused = fuse_probability_maps(
            {"first": first, "second": second},
            {"first": 1.0, "second": 1.0},
        )

        np.testing.assert_allclose(fused, (first + second) / 2.0)
        self.assertEqual(fused.dtype, np.float32)

    def test_explicit_weights_are_normalized_before_fusion(self) -> None:
        first = np.ones((2, 2), dtype=np.float32)
        second = np.zeros((2, 2), dtype=np.float32)

        fused = fuse_probability_maps(
            {"first": first, "second": second},
            {"first": 3.0, "second": 1.0},
        )

        np.testing.assert_allclose(fused, np.full((2, 2), 0.75, np.float32))
        self.assertEqual(
            normalize_model_weights({"first": 3.0, "second": 1.0}),
            {"first": 0.75, "second": 0.25},
        )

    def test_adapter_thresholds_only_after_fusion(self) -> None:
        adapter = ProbabilityEnsembleAdapter(
            {
                "first": StaticAdapter(np.full((2, 2), 0.8, np.float32)),
                "second": StaticAdapter(np.full((2, 2), 0.4, np.float32)),
            },
            {"first": 1.0, "second": 1.0},
            threshold=0.6,
        )

        result = adapter.predict(np.zeros((2, 2, 3), dtype=np.uint8))

        np.testing.assert_allclose(result.probability_map, 0.6)
        np.testing.assert_array_equal(result.binary_mask, np.ones((2, 2), np.uint8))

    def test_partial_explicit_weights_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "missing model weights: second"):
            resolve_model_weights(["first", "second"], [("first", 1.0)])

    def test_duplicate_weights_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "duplicate model weight"):
            resolve_model_weights(
                ["first", "second"],
                [("first", 1.0), ("first", 2.0), ("second", 1.0)],
            )

    def test_negative_and_all_zero_weights_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "cannot be negative"):
            resolve_model_weights(
                ["first", "second"], [("first", -1.0), ("second", 1.0)]
            )
        with self.assertRaisesRegex(ValueError, "greater than zero"):
            resolve_model_weights(
                ["first", "second"], [("first", 0.0), ("second", 0.0)]
            )

    def test_unknown_weight_and_unaligned_maps_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "unknown models: third"):
            resolve_model_weights(
                ["first", "second"], [("first", 1.0), ("third", 1.0)]
            )
        with self.assertRaisesRegex(ValueError, "source-aligned shapes"):
            fuse_probability_maps(
                {
                    "first": np.zeros((2, 2), np.float32),
                    "second": np.zeros((3, 2), np.float32),
                },
                {"first": 1.0, "second": 1.0},
            )

    def test_nonfinite_and_out_of_range_probabilities_are_rejected(self) -> None:
        for invalid_map, message in (
            (np.array([[np.nan]], np.float32), "non-finite"),
            (np.array([[1.1]], np.float32), "within \\[0, 1\\]"),
        ):
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    fuse_probability_maps(
                        {
                            "first": invalid_map,
                            "second": np.zeros((1, 1), np.float32),
                        },
                        {"first": 1.0, "second": 1.0},
                    )

    def test_cli_weight_validation_rejects_negative_value(self) -> None:
        with self.assertRaisesRegex(argparse.ArgumentTypeError, "cannot be negative"):
            parse_model_weight("segformer=-0.5")

    def test_missing_component_checkpoint_fails_ensemble_cleanly(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result = evaluate_ensemble(
                [
                    ModelEvaluationSpec("segformer", root / "missing-segformer.pt"),
                    ModelEvaluationSpec("unet-resnet34", root / "missing-unet.pt"),
                ],
                {"segformer": 1.0, "unet-resnet34": 1.0},
                samples=[],
                device="cpu",
                threshold=0.5,
            )

        self.assertEqual(result["status"], "failed")
        self.assertIn("Missing component checkpoint", result["error"])
        self.assertEqual(result["frame_records"], [])


if __name__ == "__main__":
    unittest.main()
