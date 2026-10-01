import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from src.evaluation.marsp_ablation import (
    apply_ablation_configuration,
    build_ablation_matrix,
    summarize_temporal_masks,
)
from src.postprocessing.temporal_aggregation import remove_small_components
from src.preprocessing.motion_compensation import (
    MotionCompensationConfig,
    estimate_partial_affine_transform,
)
from scripts.evaluate_marsp_ablation import (
    group_samples_by_video,
    sampled_segments,
    semantic_metric_deltas,
    temporal_video_metric_deltas,
    validate_equal_weights,
    validate_model_specs,
)
from scripts.evaluate_held_out_models import ModelEvaluationSpec


class MarspAblationTests(unittest.TestCase):
    def test_matrix_declares_a0_to_a6_and_selected_reference(self) -> None:
        matrix = build_ablation_matrix(selected_model_path="ensemble")

        self.assertEqual([item.experiment_id for item in matrix], [f"A{i}" for i in range(7)])
        self.assertEqual(matrix[3].model_path, "ensemble")
        self.assertEqual(matrix[3].reference_id, "A2")
        self.assertEqual(matrix[1].reference_id, "A0")
        self.assertEqual(matrix[2].reference_id, "A0")
        self.assertEqual(matrix[6].temporal_mode, "motion_adaptive")
        self.assertEqual(matrix[6].min_component_area, 500)

    def test_matrix_rejects_invalid_parameters(self) -> None:
        with self.assertRaisesRegex(ValueError, "positive odd"):
            build_ablation_matrix(window_size=4)
        with self.assertRaisesRegex(ValueError, "between"):
            build_ablation_matrix(threshold=1.2)
        with self.assertRaisesRegex(ValueError, "one of"):
            build_ablation_matrix(selected_model_path="other")

    def test_fixed_temporal_smoothing_reduces_single_frame_outlier(self) -> None:
        configuration = build_ablation_matrix(window_size=3)[3]
        maps = [
            np.zeros((2, 2), dtype=np.float32),
            np.ones((2, 2), dtype=np.float32),
            np.zeros((2, 2), dtype=np.float32),
        ]

        processed, masks = apply_ablation_configuration(maps, configuration)

        self.assertAlmostEqual(float(processed[1][0, 0]), 1 / 3)
        self.assertEqual(int(masks[1].sum()), 0)

    def test_motion_adaptive_mode_requires_aligned_scores(self) -> None:
        configuration = build_ablation_matrix()[4]
        maps = [np.zeros((2, 2), dtype=np.float32)] * 2

        with self.assertRaisesRegex(ValueError, "required"):
            apply_ablation_configuration(maps, configuration)
        with self.assertRaisesRegex(ValueError, "same length"):
            apply_ablation_configuration(maps, configuration, motion_scores=[0.0])

    def test_high_motion_falls_back_to_current_frame(self) -> None:
        configuration = build_ablation_matrix(window_size=3)[4]
        maps = [
            np.zeros((2, 2), dtype=np.float32),
            np.ones((2, 2), dtype=np.float32),
            np.zeros((2, 2), dtype=np.float32),
        ]

        processed, _ = apply_ablation_configuration(
            maps,
            configuration,
            motion_scores=[0.0, 1.0, 0.0],
        )

        np.testing.assert_array_equal(processed[1], maps[1])

    def test_small_component_cleanup_preserves_large_region(self) -> None:
        mask = np.zeros((8, 8), dtype=np.uint8)
        mask[0, 0] = 1
        mask[3:6, 3:6] = 1

        cleaned = remove_small_components(mask, min_component_area=4)

        self.assertEqual(cleaned[0, 0], 0)
        self.assertEqual(int(cleaned.sum()), 9)

    def test_temporal_summary_reports_stability_and_fragmentation(self) -> None:
        first = np.zeros((4, 4), dtype=np.uint8)
        second = np.zeros((4, 4), dtype=np.uint8)
        first[1:3, 1:3] = 1
        second[1:3, 1:3] = 1

        summary = summarize_temporal_masks([first, second], small_component_area=3)

        self.assertEqual(summary["frame_count"], 2)
        self.assertEqual(summary["mean_consecutive_iou"], 1.0)
        self.assertEqual(summary["mean_pixel_change_rate"], 0.0)
        self.assertEqual(summary["mean_small_component_count"], 0.0)

    def test_temporal_summary_excludes_sparse_gap_pairs(self) -> None:
        first = np.zeros((2, 2), dtype=np.uint8)
        second = np.ones((2, 2), dtype=np.uint8)

        summary = summarize_temporal_masks([first, second], pair_breaks={1})

        self.assertEqual(summary["evaluated_pair_count"], 0)
        self.assertEqual(summary["skipped_pair_count"], 1)
        self.assertIsNone(summary["mean_consecutive_iou"])

    def test_sparse_frame_gap_splits_temporal_segments(self) -> None:
        samples = [{"frame_index": index} for index in (1, 7, 13, 43, 49)]

        segments = sampled_segments(samples, max_frame_gap=6)

        self.assertEqual([(part.start, part.stop) for part in segments], [(0, 3), (3, 5)])
        with self.assertRaisesRegex(ValueError, "greater than zero"):
            sampled_segments(samples, max_frame_gap=0)

    def test_a2_rejects_unequal_ensemble_weights(self) -> None:
        validate_equal_weights({"segformer": 1.0, "unet-resnet34": 1.0})
        with self.assertRaisesRegex(ValueError, "equal-weight"):
            validate_equal_weights({"segformer": 3.0, "unet-resnet34": 1.0})

    def test_group_samples_orders_frames_and_rejects_duplicates(self) -> None:
        grouped = group_samples_by_video(
            [
                {"video_id": "v", "frame_index": 3},
                {"video_id": "v", "frame_index": 1},
            ]
        )
        self.assertEqual([item["frame_index"] for item in grouped["v"]], [1, 3])

        with self.assertRaisesRegex(ValueError, "duplicate"):
            group_samples_by_video(
                [
                    {"video_id": "v", "frame_index": 1},
                    {"video_id": "v", "frame_index": 1},
                ]
            )

    def test_model_specs_require_both_formal_component_models(self) -> None:
        specs = [ModelEvaluationSpec("segformer", Path("/tmp/segformer.pt"))]
        with self.assertRaisesRegex(ValueError, "missing unet-resnet34"):
            validate_model_specs(specs)

    def test_semantic_deltas_preserve_undefined_metrics(self) -> None:
        candidate = {
            "micro": {name: 0.5 for name in (
                "foreground_iou", "foreground_dice", "foreground_precision",
                "foreground_recall", "background_iou", "mean_iou", "mean_dice"
            )},
            "macro_per_frame": {name: 0.5 for name in (
                "foreground_iou", "foreground_dice", "foreground_precision",
                "foreground_recall", "background_iou", "mean_iou", "mean_dice"
            )},
        }
        reference = {
            "micro": dict(candidate["micro"]),
            "macro_per_frame": dict(candidate["macro_per_frame"]),
        }
        reference["micro"]["foreground_recall"] = None

        delta = semantic_metric_deltas(candidate, reference)

        self.assertEqual(delta["micro"]["foreground_iou"], 0.0)
        self.assertIsNone(delta["micro"]["foreground_recall"])

    def test_temporal_video_deltas_preserve_direction(self) -> None:
        candidate = {
            "mean_consecutive_iou": 0.8,
            "mean_consecutive_dice": 0.9,
            "mean_pixel_change_rate": 0.1,
            "std_mask_area_px": 2.0,
            "mean_abs_area_change_px": 1.0,
            "mean_component_count": 1.0,
            "mean_small_component_count": 0.0,
        }
        reference = {name: 0.5 for name in candidate}

        delta = temporal_video_metric_deltas(candidate, reference)

        self.assertAlmostEqual(delta["mean_consecutive_iou"], 0.3)
        self.assertAlmostEqual(delta["mean_pixel_change_rate"], -0.4)

    def test_motion_estimation_handles_incomplete_knn_pairs(self) -> None:
        class FakeOrb:
            def detectAndCompute(self, _image, _mask):
                return [object()], np.ones((1, 32), dtype=np.uint8)

        class FakeMatcher:
            def knnMatch(self, _first, _second, k):
                self.requested_neighbors = k
                return [[SimpleNamespace(distance=1.0, queryIdx=0, trainIdx=0)]]

        transform, info = estimate_partial_affine_transform(
            FakeOrb(),
            FakeMatcher(),
            np.zeros((2, 2), dtype=np.uint8),
            np.zeros((2, 2), dtype=np.uint8),
            MotionCompensationConfig(min_good_matches=1),
        )

        self.assertIsNone(transform)
        self.assertTrue(info["fallback"])


if __name__ == "__main__":
    unittest.main()
