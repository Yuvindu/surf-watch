import unittest

from configs.unet_resnet34_config import UnetResNet34TrainingConfig
from scripts.train_unet_resnet34 import build_parser, config_from_args


class UnetResNet34TrainingConfigTests(unittest.TestCase):
    def test_parser_builds_reproducible_config(self):
        args = build_parser().parse_args(
            [
                "--ripvis-root",
                "/data/RipVIS",
                "--processed-root",
                "/data/processed",
                "--image-size",
                "256",
                "--batch-size",
                "2",
                "--epochs",
                "3",
                "--encoder-weights",
                "none",
                "--train-subset-size",
                "8",
                "--val-subset-size",
                "4",
            ]
        )

        config = config_from_args(args)

        self.assertEqual(config.image_size, (256, 256))
        self.assertEqual(config.batch_size, 2)
        self.assertEqual(config.num_epochs, 3)
        self.assertIsNone(config.encoder_weights)
        self.assertEqual(config.train_subset_size, 8)
        self.assertEqual(config.best_checkpoint_name, "unet_resnet34_best_model.pt")

    def test_invalid_threshold_is_rejected(self):
        config = UnetResNet34TrainingConfig(threshold=1.1)

        with self.assertRaisesRegex(ValueError, "threshold"):
            config.validate()


if __name__ == "__main__":
    unittest.main()
