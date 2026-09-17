import unittest
from unittest.mock import Mock, patch

from configs.baseline_config import BaselineConfig
from src.models.adapter_factory import _ADAPTER_BUILDERS, build_segmentation_adapter
from src.models.model_registry import SUPPORTED_SEGMENTATION_MODELS


class AdapterFactoryTests(unittest.TestCase):
    def test_every_public_model_has_an_adapter_builder(self) -> None:
        self.assertEqual(set(_ADAPTER_BUILDERS), set(SUPPORTED_SEGMENTATION_MODELS))

    def test_registered_model_is_built_with_shared_configuration(self) -> None:
        builder = Mock(return_value=Mock())
        config = BaselineConfig()

        with patch("src.models.adapter_factory._ADAPTER_BUILDERS", {"segformer": builder}):
            adapter = build_segmentation_adapter(
                model_name="SegFormer",
                checkpoint_path="checkpoint.pt",
                device="cpu",
                config=config,
                threshold=0.65,
            )

        self.assertIs(adapter, builder.return_value)
        builder.assert_called_once_with(
            checkpoint_path="checkpoint.pt",
            device="cpu",
            config=config,
            threshold=0.65,
        )

    def test_unregistered_model_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unsupported segmentation model"):
            build_segmentation_adapter(
                model_name="unet",
                checkpoint_path="checkpoint.pt",
                device="cpu",
                config=BaselineConfig(),
                threshold=0.5,
            )


if __name__ == "__main__":
    unittest.main()
