import argparse
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from scripts.evaluate_held_out_models import (
    discover_samples,
    parse_frame_identity,
    parse_model_checkpoint,
    validate_unique_models,
)
from src.evaluation.segmentation_metrics import (
    binary_confusion,
    metrics_from_confusion,
    summarize_records,
)


class HeldOutEvaluationTests(unittest.TestCase):
    def test_parse_frame_identity_preserves_nr_video_id(self) -> None:
        self.assertEqual(
            parse_frame_identity("RipVIS-NR-020_01194.jpg"),
            ("RipVIS-NR-020", 1194),
        )

    def test_model_checkpoint_uses_registered_model(self) -> None:
        spec = parse_model_checkpoint(" UNeT-ResNet34 = /tmp/unet.pt")
        self.assertEqual(spec.model, "unet-resnet34")
        self.assertEqual(spec.checkpoint, Path("/tmp/unet.pt").resolve())

    def test_duplicate_models_are_rejected(self) -> None:
        spec = parse_model_checkpoint("segformer=/tmp/model.pt")
        with self.assertRaisesRegex(ValueError, "only once"):
            validate_unique_models([spec, spec])

    def test_unknown_model_is_rejected(self) -> None:
        with self.assertRaisesRegex(argparse.ArgumentTypeError, "Unsupported"):
            parse_model_checkpoint("other=/tmp/model.pt")

    def test_discover_samples_filters_complete_video_ids(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image_dir = root / "RipVIS/val/sampled_images/sampled_images/images"
            mask_dir = root / "processed/val/masks"
            image_dir.mkdir(parents=True)
            mask_dir.mkdir(parents=True)
            for filename in ("RipVIS-001_00001", "RipVIS-NR-020_00002"):
                cv2.imwrite(
                    str(image_dir / f"{filename}.jpg"),
                    np.zeros((2, 2, 3), np.uint8),
                )
                cv2.imwrite(
                    str(mask_dir / f"{filename}.png"),
                    np.zeros((2, 2), np.uint8),
                )

            samples = discover_samples(
                root / "RipVIS",
                root / "processed",
                "val",
                selected_videos={"RipVIS-NR-020"},
            )

        self.assertEqual(len(samples), 1)
        self.assertEqual(samples[0]["video_id"], "RipVIS-NR-020")

    def test_binary_metrics_report_foreground_scores(self) -> None:
        prediction = np.array([[1, 1], [0, 0]], dtype=np.uint8)
        target = np.array([[1, 0], [1, 0]], dtype=np.uint8)
        confusion = binary_confusion(prediction, target)
        metrics = metrics_from_confusion(confusion)

        self.assertEqual(
            confusion,
            {
                "true_negative": 1,
                "false_positive": 1,
                "false_negative": 1,
                "true_positive": 1,
            },
        )
        self.assertAlmostEqual(metrics["foreground_iou"], 1 / 3)
        self.assertAlmostEqual(metrics["foreground_dice"], 0.5)
        self.assertAlmostEqual(metrics["mean_iou"], 1 / 3)

    def test_empty_foreground_is_undefined_and_excluded_from_macro(self) -> None:
        empty = np.zeros((2, 2), dtype=np.uint8)
        empty_confusion = binary_confusion(empty, empty)
        empty_metrics = metrics_from_confusion(empty_confusion)
        self.assertIsNone(empty_metrics["foreground_iou"])

        positive = np.ones((2, 2), dtype=np.uint8)
        positive_confusion = binary_confusion(positive, positive)
        records = [
            {
                "confusion": empty_confusion,
                "metrics": empty_metrics,
                "ground_truth_positive_pixels": 0,
                "prediction_positive_pixels": 0,
                "ground_truth_foreground_fraction": 0.0,
                "prediction_foreground_fraction": 0.0,
            },
            {
                "confusion": positive_confusion,
                "metrics": metrics_from_confusion(positive_confusion),
                "ground_truth_positive_pixels": 4,
                "prediction_positive_pixels": 4,
                "ground_truth_foreground_fraction": 1.0,
                "prediction_foreground_fraction": 1.0,
            },
        ]
        summary = summarize_records(records)

        self.assertEqual(summary["empty_ground_truth_frames"], 1)
        self.assertAlmostEqual(summary["macro_per_frame"]["foreground_iou"], 1.0)
        self.assertAlmostEqual(summary["micro"]["foreground_iou"], 1.0)


if __name__ == "__main__":
    unittest.main()
