import argparse
import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from configs.segformer_training_config import SegformerTrainingConfig
from scripts.train_baseline import (
    config_from_args,
    prepare_run_directory,
    run_training,
    sha256_file,
)


class SegformerTrainingConfigTests(unittest.TestCase):
    def test_phase_one_reproduction_defaults(self):
        config = SegformerTrainingConfig()

        self.assertEqual(config.image_size, (512, 512))
        self.assertEqual(config.batch_size, 4)
        self.assertEqual(config.num_epochs, 5)
        self.assertEqual(config.learning_rate, 1e-4)
        self.assertEqual(config.weight_decay, 1e-4)
        self.assertEqual(config.seed, 42)

    def test_config_from_args_uses_run_scoped_output(self):
        args = argparse.Namespace(
            ripvis_root="/data/RipVIS",
            processed_root="/data/processed",
            image_size=256,
            batch_size=2,
            num_workers=0,
            epochs=3,
            learning_rate=2e-4,
            weight_decay=1e-5,
            train_subset_size=20,
            val_subset_size=10,
            run_dir="outputs/training/test-run",
            pretrained_model_name="segformer-test",
            threshold=0.4,
            seed=7,
            require_clean_git=True,
        )

        config = config_from_args(args)

        self.assertEqual(config.image_size, (256, 256))
        self.assertEqual(config.run_dir, "outputs/training/test-run")
        self.assertEqual(config.seed, 7)
        self.assertTrue(config.require_clean_git)

    def test_invalid_values_are_rejected(self):
        with self.assertRaises(ValueError):
            SegformerTrainingConfig(num_epochs=0).validate()
        with self.assertRaises(ValueError):
            SegformerTrainingConfig(run_dir=" ").validate()


class TrainingArtifactSafetyTests(unittest.TestCase):
    def test_empty_run_directory_is_allowed(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir) / "new-run"
            prepare_run_directory(run_dir)
            self.assertTrue(run_dir.is_dir())

    def test_non_empty_run_directory_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir) / "existing-run"
            run_dir.mkdir()
            (run_dir / "checkpoint.pt").write_bytes(b"existing")

            with self.assertRaises(FileExistsError):
                prepare_run_directory(run_dir)

    def test_sha256_file_records_artifact_identity(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "artifact.bin"
            path.write_bytes(b"surfwatch")

            self.assertEqual(
                sha256_file(path), hashlib.sha256(b"surfwatch").hexdigest()
            )

    def test_formal_run_rejects_dirty_git_before_creating_artifacts(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir) / "formal-run"
            config = SegformerTrainingConfig(
                run_dir=str(run_dir), require_clean_git=True
            )
            with patch(
                "scripts.train_baseline._environment_metadata",
                return_value={"git": {"revision": "abc123", "dirty": True}},
            ):
                with self.assertRaises(RuntimeError):
                    run_training(config)

            self.assertFalse(run_dir.exists())


if __name__ == "__main__":
    unittest.main()
