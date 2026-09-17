import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import torch

from src.models.segformer_adapter import SegFormerSegmentationAdapter


class ConstantLogitModel:
    def __call__(self, pixel_values: torch.Tensor) -> SimpleNamespace:
        _, _, height, width = pixel_values.shape
        logits = torch.zeros((1, 2, height, width), dtype=torch.float32)
        logits[:, 1, :, width // 2 :] = 4.0
        return SimpleNamespace(logits=logits)


class SegmentationAdapterContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.adapter = SegFormerSegmentationAdapter.__new__(SegFormerSegmentationAdapter)
        self.adapter.device = torch.device("cpu")
        self.adapter.threshold = 0.5
        self.adapter.image_size = (4, 6)
        self.adapter.model = ConstantLogitModel()

    def test_prediction_matches_source_frame_contract(self) -> None:
        frame = np.zeros((8, 10, 3), dtype=np.uint8)
        result = self.adapter.predict(frame)

        self.assertEqual(result.probability_map.shape, frame.shape[:2])
        self.assertEqual(result.probability_map.dtype, np.float32)
        self.assertGreaterEqual(float(result.probability_map.min()), 0.0)
        self.assertLessEqual(float(result.probability_map.max()), 1.0)
        self.assertEqual(result.binary_mask.shape, frame.shape[:2])
        self.assertEqual(result.binary_mask.dtype, np.uint8)
        self.assertTrue(set(np.unique(result.binary_mask)).issubset({0, 1}))
        self.assertEqual(result.metadata.output_format, "probability_map_and_binary_mask")

    def test_invalid_frame_dtype_is_rejected(self) -> None:
        frame = np.zeros((8, 10, 3), dtype=np.float32)
        with self.assertRaisesRegex(ValueError, "dtype uint8"):
            self.adapter.predict(frame)

    def test_non_2d_probability_map_is_rejected(self) -> None:
        probability_map = np.zeros((2, 2, 1), dtype=np.float32)
        with self.assertRaisesRegex(ValueError, "2D array"):
            self.adapter.probability_to_mask(probability_map)

    @patch("src.models.segformer_adapter.torch.load")
    @patch("src.models.segformer_adapter.build_segformer_model")
    def test_checkpoint_inference_does_not_load_pretrained_weights(
        self,
        build_model,
        load_checkpoint,
    ) -> None:
        model = Mock()
        model.to.return_value = model
        build_model.return_value = model
        load_checkpoint.return_value = {"model_state_dict": {}}
        config = Mock(
            image_size=(512, 512),
            pretrained_model_name="segformer-b0",
            num_classes=2,
        )

        SegFormerSegmentationAdapter(
            checkpoint_path="checkpoint.pt",
            device=torch.device("cpu"),
            config=config,
        )

        build_model.assert_called_once_with(
            model_name="segformer-b0",
            num_classes=2,
            load_pretrained_weights=False,
        )
        model.load_state_dict.assert_called_once_with({})


if __name__ == "__main__":
    unittest.main()
