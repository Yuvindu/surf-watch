import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.run_baseline_vs_marsp_compare import (
    build_provenance,
    file_identity,
    reusable_run_matches,
)


class ComparisonProvenanceTests(unittest.TestCase):
    def test_file_identity_records_resolved_path_size_and_hash(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            artifact = Path(directory) / "checkpoint.pt"
            artifact.write_bytes(b"surfwatch")

            identity = file_identity(str(artifact))

        self.assertEqual(identity["path"], str(artifact.resolve()))
        self.assertEqual(identity["size_bytes"], 9)
        self.assertEqual(
            identity["sha256"],
            "c7800e9290eb8ac0375c2e45e750c392230e1ac2aaadb389e16c5f765b202b92",
        )

    @patch(
        "scripts.run_baseline_vs_marsp_compare.git_revision",
        return_value="abc123",
    )
    def test_provenance_identifies_model_input_checkpoint_and_parameters(
        self,
        _,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "model.pt"
            video = Path(directory) / "input.mp4"
            checkpoint.write_bytes(b"checkpoint")
            video.write_bytes(b"video")

            provenance = build_provenance(
                model="unet-resnet34",
                checkpoint=str(checkpoint),
                input_video=str(video),
                window_size=5,
                threshold=0.5,
            )

        self.assertEqual(provenance["segmentation_model"], "unet-resnet34")
        self.assertEqual(provenance["git_revision"], "abc123")
        self.assertEqual(provenance["parameters"], {"window_size": 5, "threshold": 0.5})
        self.assertEqual(provenance["checkpoint"]["path"], str(checkpoint.resolve()))
        self.assertEqual(provenance["input"]["path"], str(video.resolve()))

    def test_reuse_requires_matching_provenance(self) -> None:
        provenance = {
            "segmentation_model": "unet-resnet34",
            "checkpoint": {"sha256": "checkpoint-hash"},
            "input": {"sha256": "input-hash"},
            "parameters": {"window_size": 5, "threshold": 0.5},
        }
        with tempfile.TemporaryDirectory() as directory:
            metrics_path = Path(directory) / "metrics.json"
            metrics_path.write_text(
                json.dumps({"provenance": provenance}),
                encoding="utf-8",
            )

            self.assertTrue(reusable_run_matches(metrics_path, provenance))

            changed = {**provenance, "segmentation_model": "segformer"}
            self.assertFalse(reusable_run_matches(metrics_path, changed))

    def test_legacy_metrics_are_not_reused(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            metrics_path = Path(directory) / "metrics.json"
            metrics_path.write_text(
                json.dumps({"segmentation_model": "unet-resnet34"}),
                encoding="utf-8",
            )

            self.assertFalse(reusable_run_matches(metrics_path, {}))


if __name__ == "__main__":
    unittest.main()
