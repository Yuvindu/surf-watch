import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from src.data.ripvis_dataset import RipVISSemanticDataset


class RipVISDatasetNormalizationTests(unittest.TestCase):
    def test_optional_normalization_matches_inference_preprocessing(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            images = root / "RipVIS/train/sampled_images/sampled_images/images"
            masks = root / "processed/train/masks"
            images.mkdir(parents=True)
            masks.mkdir(parents=True)
            Image.fromarray(np.full((2, 2, 3), 255, dtype=np.uint8)).save(images / "frame.jpg")
            Image.fromarray(np.full((2, 2), 255, dtype=np.uint8)).save(masks / "frame.png")

            dataset = RipVISSemanticDataset(
                ripvis_root=str(root / "RipVIS"),
                processed_root=str(root / "processed"),
                split="train",
                image_size=(2, 2),
                normalization=((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            )
            image, mask = dataset[0]

        self.assertTrue(np.allclose(image.numpy(), 1.0))
        self.assertEqual(mask.unique().tolist(), [1])


if __name__ == "__main__":
    unittest.main()
