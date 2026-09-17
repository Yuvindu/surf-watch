import importlib.util
import unittest
from unittest.mock import Mock, patch

import numpy as np
import torch

from src.models.unet_resnet34 import build_unet_resnet34_model
from src.models.unet_resnet34_adapter import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    UnetResNet34SegmentationAdapter,
)


class ConstantUnetModel(torch.nn.Module):
    def forward(self, input_tensor):
        batch, _, height, width = input_tensor.shape
        logits = torch.full((batch, 1, height, width), -4.0)
        logits[:, :, :, width // 2 :] = 4.0
        return logits


class UnetResNet34AdapterTests(unittest.TestCase):
    def setUp(self) -> None:
        self.adapter = UnetResNet34SegmentationAdapter.__new__(
            UnetResNet34SegmentationAdapter
        )
        self.adapter.device = torch.device("cpu")
        self.adapter.threshold = 0.5
        self.adapter.image_size = (4, 6)
        self.adapter.model = ConstantUnetModel().eval()

    def test_prediction_matches_segmentation_contract(self) -> None:
        frame = np.zeros((8, 10, 3), dtype=np.uint8)

        result = self.adapter.predict(frame)

        self.assertEqual(result.probability_map.shape, (8, 10))
        self.assertEqual(result.probability_map.dtype, np.float32)
        self.assertGreaterEqual(float(result.probability_map.min()), 0.0)
        self.assertLessEqual(float(result.probability_map.max()), 1.0)
        self.assertEqual(result.binary_mask.dtype, np.uint8)
        self.assertEqual(set(np.unique(result.binary_mask)), {0, 1})
        self.assertEqual(result.metadata.name, "U-Net (ResNet34)")

    def test_preprocessing_converts_bgr_to_normalized_rgb(self) -> None:
        frame = np.zeros((4, 6, 3), dtype=np.uint8)
        frame[:, :, 0] = 255

        tensor = self.adapter.preprocess_frame(frame)

        expected = torch.tensor(
            [
                (0.0 - IMAGENET_MEAN[0]) / IMAGENET_STD[0],
                (0.0 - IMAGENET_MEAN[1]) / IMAGENET_STD[1],
                (1.0 - IMAGENET_MEAN[2]) / IMAGENET_STD[2],
            ]
        )
        torch.testing.assert_close(tensor[0, :, 0, 0], expected)

    def test_unexpected_logits_shape_is_rejected(self) -> None:
        self.adapter.model = Mock(
            return_value=torch.zeros((1, 2, 4, 6), dtype=torch.float32)
        )

        with self.assertRaisesRegex(ValueError, "shape \(1, 1, height, width\)"):
            self.adapter.predict_probability_map(
                np.zeros((4, 6, 3), dtype=np.uint8)
            )

    @patch("src.models.unet_resnet34_adapter.torch.load")
    @patch("src.models.unet_resnet34_adapter.build_unet_resnet34_model")
    def test_checkpoint_requires_model_state_dict(self, build_model, load) -> None:
        build_model.return_value = ConstantUnetModel()
        load.return_value = {"epoch": 1}

        with self.assertRaisesRegex(ValueError, "model_state_dict"):
            UnetResNet34SegmentationAdapter(
                checkpoint_path="checkpoint.pt",
                device=torch.device("cpu"),
                config=Mock(image_size=(4, 6)),
            )

    def test_threshold_outside_probability_range_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "between 0.0 and 1.0"):
            self.adapter._validate_threshold(1.1)


@unittest.skipUnless(
    importlib.util.find_spec("segmentation_models_pytorch"),
    "segmentation-models-pytorch is not installed",
)
class UnetResNet34LibrarySmokeTests(unittest.TestCase):
    def test_model_produces_one_channel_logits(self) -> None:
        model = build_unet_resnet34_model(encoder_weights=None).eval()

        with torch.no_grad():
            logits = model(torch.zeros((1, 3, 64, 64), dtype=torch.float32))

        self.assertEqual(tuple(logits.shape), (1, 1, 64, 64))


if __name__ == "__main__":
    unittest.main()
