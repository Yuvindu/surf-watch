import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from scripts.evaluate_model_uncertainty import evaluate_samples, write_artifacts
from src.evaluation.calibration_metrics import (
    CalibrationAccumulator,
    DisagreementAccumulator,
    foreground_candidate_mask,
)


class CalibrationMetricTests(unittest.TestCase):
    def test_perfect_extreme_probabilities_are_finite_and_reliable(self) -> None:
        target = np.array([[0, 1]], dtype=np.uint8)
        probabilities = target.astype(np.float32)
        accumulator = CalibrationAccumulator(bin_count=2)
        accumulator.update(probabilities, target)
        summary = accumulator.summary()

        self.assertEqual(summary["pixel_count"], 2)
        self.assertEqual([item["pixel_count"] for item in summary["bins"]], [1, 1])
        self.assertAlmostEqual(summary["brier_score"], 0.0)
        self.assertAlmostEqual(summary["foreground_ece"], 0.0)
        self.assertAlmostEqual(summary["mean_entropy_nats"], 0.0)
        self.assertTrue(np.isfinite(summary["negative_log_likelihood"]))

    def test_overconfident_errors_have_finite_nll(self) -> None:
        target = np.array([[0, 1]], dtype=np.uint8)
        probabilities = np.array([[1.0, 0.0]], dtype=np.float32)
        accumulator = CalibrationAccumulator(bin_count=2)
        accumulator.update(probabilities, target)
        summary = accumulator.summary()

        self.assertAlmostEqual(summary["brier_score"], 1.0)
        self.assertAlmostEqual(summary["foreground_ece"], 1.0)
        self.assertGreater(summary["negative_log_likelihood"], 10.0)

    def test_bin_boundaries_and_pixel_weighted_merge(self) -> None:
        first = CalibrationAccumulator(bin_count=2)
        first.update(np.array([[0.0, 0.5]], np.float32), np.array([[0, 1]], np.uint8))
        second = CalibrationAccumulator(bin_count=2)
        second.update(np.array([[1.0]], np.float32), np.array([[1]], np.uint8))
        first.merge(second)
        summary = first.summary()

        self.assertEqual([item["pixel_count"] for item in summary["bins"]], [1, 2])
        self.assertAlmostEqual(summary["brier_score"], 0.25 / 3)
        self.assertAlmostEqual(summary["foreground_ece"], 0.5 / 3)

    def test_empty_candidate_region_has_defined_empty_summary(self) -> None:
        target = np.zeros((2, 2), dtype=np.uint8)
        first = np.zeros((2, 2), dtype=np.float32)
        second = np.full((2, 2), 0.1, dtype=np.float32)
        candidate = foreground_candidate_mask(target, first, second, 0.5)
        self.assertFalse(candidate.any())

        accumulator = CalibrationAccumulator()
        accumulator.update(first, target, candidate)
        summary = accumulator.summary()
        self.assertEqual(summary["pixel_count"], 0)
        self.assertIsNone(summary["foreground_ece"])
        self.assertIsNone(summary["brier_score"])

    def test_candidate_region_includes_truth_and_either_prediction(self) -> None:
        target = np.array([[1, 0, 0]], dtype=np.uint8)
        first = np.array([[0.1, 0.8, 0.2]], dtype=np.float32)
        second = np.array([[0.2, 0.1, 0.7]], dtype=np.float32)
        np.testing.assert_array_equal(
            foreground_candidate_mask(target, first, second, 0.5),
            np.array([[True, True, True]]),
        )

    def test_invalid_inputs_are_rejected(self) -> None:
        target = np.zeros((2, 2), dtype=np.uint8)
        accumulator = CalibrationAccumulator()
        for bad in (
            np.zeros((1, 2), dtype=np.float32),
            np.full((2, 2), np.nan, dtype=np.float32),
            np.full((2, 2), 1.1, dtype=np.float32),
        ):
            with self.assertRaises(ValueError):
                accumulator.update(bad, target)
        with self.assertRaisesRegex(ValueError, "binary"):
            accumulator.update(np.zeros((2, 2), np.float32), np.full((2, 2), 2))
        with self.assertRaisesRegex(ValueError, "boolean mask"):
            accumulator.update(np.zeros((2, 2), np.float32), target, np.ones((2, 2)))
        with self.assertRaises(ValueError):
            CalibrationAccumulator(bin_count=1)

    def test_disagreement_counts_probability_and_mask_differences(self) -> None:
        first = np.array([[0.2, 0.8]], dtype=np.float32)
        second = np.array([[0.4, 0.3]], dtype=np.float32)
        accumulator = DisagreementAccumulator()
        accumulator.update(first, second, 0.5)
        summary = accumulator.summary()

        self.assertEqual(summary["pixel_count"], 2)
        self.assertAlmostEqual(summary["mean_absolute_probability_difference"], 0.35)
        self.assertAlmostEqual(summary["mask_disagreement_fraction"], 0.5)


class FakeAdapter:
    def __init__(self, probability_map: np.ndarray) -> None:
        self.probability_map = probability_map
        self.calls = 0

    def predict_probability_map(self, frame: np.ndarray) -> np.ndarray:
        self.calls += 1
        return self.probability_map.copy()


class UncertaintyEvaluatorTests(unittest.TestCase):
    def test_evaluator_reuses_component_predictions_and_writes_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            samples = []
            for video_id in ("RipVIS-001", "RipVIS-007"):
                image_path = root / f"{video_id}_00001.jpg"
                mask_path = root / f"{video_id}_00001.png"
                cv2.imwrite(str(image_path), np.zeros((2, 2, 3), dtype=np.uint8))
                cv2.imwrite(str(mask_path), np.array([[255, 0], [0, 0]], dtype=np.uint8))
                samples.append(
                    {
                        "video_id": video_id,
                        "frame_index": 1,
                        "image_path": image_path,
                        "mask_path": mask_path,
                    }
                )
            adapters = {
                "segformer": FakeAdapter(np.array([[0.9, 0.1], [0.1, 0.1]], np.float32)),
                "unet-resnet34": FakeAdapter(np.array([[0.7, 0.7], [0.1, 0.1]], np.float32)),
            }
            evaluated = evaluate_samples(samples, adapters, bin_count=2, threshold=0.5)
            self.assertEqual(adapters["segformer"].calls, 2)
            self.assertEqual(adapters["unet-resnet34"].calls, 2)
            self.assertEqual(len(evaluated["frame_rows"]), 6)
            self.assertEqual(len(evaluated["video_records"]), 2)
            self.assertEqual(
                [record["video_id"] for record in evaluated["video_records"]],
                ["RipVIS-001", "RipVIS-007"],
            )
            self.assertEqual(
                evaluated["aggregate"]["models"]["ensemble"]["all_pixels"]["pixel_count"],
                8,
            )

            payload = {"aggregate": evaluated["aggregate"]}
            paths = write_artifacts(root / "run", payload, evaluated)
            self.assertEqual(len(paths), 4)
            self.assertEqual(json.loads(paths[0].read_text())["aggregate"], payload["aggregate"])
            self.assertEqual(len(paths[1].read_text().splitlines()), 7)
            self.assertEqual(len(paths[2].read_text().splitlines()), 7)
            self.assertEqual(len(paths[3].read_text().splitlines()), 13)
            with self.assertRaises(FileExistsError):
                write_artifacts(root / "run", payload, evaluated)


if __name__ == "__main__":
    unittest.main()
