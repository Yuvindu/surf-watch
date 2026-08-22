import unittest

from src.models.model_registry import (
    DEFAULT_SEGMENTATION_MODEL,
    SUPPORTED_SEGMENTATION_MODELS,
    get_segmentation_model_option,
    segmentation_model_options_payload,
)


class ModelRegistryTests(unittest.TestCase):
    def test_default_model_is_registered(self) -> None:
        self.assertIn(DEFAULT_SEGMENTATION_MODEL, SUPPORTED_SEGMENTATION_MODELS)

    def test_lookup_is_case_insensitive(self) -> None:
        option = get_segmentation_model_option(" SegFormer ")
        self.assertEqual(option.id, "segformer")

    def test_unknown_model_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unsupported segmentation model"):
            get_segmentation_model_option("unknown")

    def test_payload_contains_frontend_metadata(self) -> None:
        payload = segmentation_model_options_payload()
        self.assertEqual(payload[0]["id"], DEFAULT_SEGMENTATION_MODEL)
        self.assertEqual(
            {option["id"] for option in payload},
            {"segformer", "unet-resnet34"},
        )
        self.assertTrue(payload[0]["label"])
        self.assertTrue(payload[0]["description"])


if __name__ == "__main__":
    unittest.main()
