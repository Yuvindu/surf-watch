import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from scripts.verify_training_artifacts import verify_manifest


class TrainingArtifactVerificationTests(unittest.TestCase):
    def _write_run(self, root: Path, content: bytes = b"checkpoint") -> Path:
        checkpoint_dir = root / "checkpoints"
        checkpoint_dir.mkdir()
        checkpoint = checkpoint_dir / "best.pt"
        checkpoint.write_bytes(content)
        manifest = {
            "status": "complete",
            "checkpoints": [
                {
                    "path": "/workspace/original/checkpoints/best.pt",
                    "bytes": len(content),
                    "sha256": hashlib.sha256(content).hexdigest(),
                }
            ],
        }
        manifest_path = root / "manifest.json"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        return manifest_path

    def test_verifies_relocated_run_directory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest = self._write_run(Path(tmpdir))
            self.assertEqual(verify_manifest(manifest), [])

    def test_detects_checkpoint_corruption(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            manifest = self._write_run(root)
            (root / "checkpoints" / "best.pt").write_bytes(b"corruption")

            errors = verify_manifest(manifest)

            self.assertEqual(len(errors), 1)
            self.assertIn("SHA-256 mismatch", errors[0])

    def test_rejects_incomplete_manifest(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            manifest = self._write_run(root)
            payload = json.loads(manifest.read_text(encoding="utf-8"))
            payload["status"] = "running"
            manifest.write_text(json.dumps(payload), encoding="utf-8")

            errors = verify_manifest(manifest)

            self.assertTrue(any("not 'complete'" in error for error in errors))


if __name__ == "__main__":
    unittest.main()
