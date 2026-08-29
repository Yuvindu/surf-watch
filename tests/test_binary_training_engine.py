import tempfile
import unittest
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

from src.training.engine import (
    forward_binary_logits,
    predict_binary,
    prepare_binary_targets,
    save_best_model,
    train_one_epoch,
    validate,
)
from src.training.losses import get_binary_loss_fn


class BinaryTrainingEngineTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        images = torch.rand(4, 3, 8, 8)
        masks = (images[:, 0] > 0.5).long()
        self.loader = DataLoader(TensorDataset(images, masks), batch_size=2)
        self.model = torch.nn.Conv2d(3, 1, kernel_size=1)
        self.loss_fn = get_binary_loss_fn()
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-2)
        self.device = torch.device("cpu")

    def test_binary_training_and_validation_contract(self):
        loss = train_one_epoch(
            self.model,
            self.loader,
            self.optimizer,
            self.loss_fn,
            self.device,
            forward_logits=forward_binary_logits,
            prepare_targets=prepare_binary_targets,
        )
        metrics = validate(
            self.model,
            self.loader,
            self.loss_fn,
            self.device,
            forward_logits=forward_binary_logits,
            prepare_targets=prepare_binary_targets,
            predict_from_logits=predict_binary,
        )

        self.assertGreater(loss, 0.0)
        self.assertEqual(
            set(metrics),
            {"val_loss", "val_iou", "val_dice", "val_precision", "val_recall"},
        )
        self.assertTrue(all(0.0 <= metrics[name] <= 1.0 for name in metrics if name != "val_loss"))

    def test_checkpoint_includes_training_provenance(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            save_best_model(
                self.model,
                self.optimizer,
                epoch=1,
                metrics={"val_iou": 0.5},
                checkpoint_dir=tmpdir,
                filename="unet.pt",
                extra_state={"model_id": "unet-resnet34", "config": {"seed": 42}},
            )

            checkpoint = torch.load(Path(tmpdir) / "unet.pt", map_location="cpu")

        self.assertIn("model_state_dict", checkpoint)
        self.assertEqual(checkpoint["model_id"], "unet-resnet34")
        self.assertEqual(checkpoint["config"]["seed"], 42)


class _SegformerOutput:
    def __init__(self, logits):
        self.logits = logits


class _SegformerStyleModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.head = torch.nn.Conv2d(3, 2, kernel_size=1)

    def forward(self, pixel_values):
        return _SegformerOutput(self.head(pixel_values))


class ExistingTrainingCompatibilityTests(unittest.TestCase):
    def test_default_engine_path_keeps_segformer_calling_convention(self):
        images = torch.rand(2, 3, 8, 8)
        masks = torch.zeros(2, 8, 8, dtype=torch.long)
        loader = DataLoader(TensorDataset(images, masks), batch_size=2)
        model = _SegformerStyleModel()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
        loss_fn = torch.nn.CrossEntropyLoss()

        train_loss = train_one_epoch(
            model, loader, optimizer, loss_fn, torch.device("cpu")
        )
        metrics = validate(model, loader, loss_fn, torch.device("cpu"))

        self.assertGreater(train_loss, 0.0)
        self.assertIn("val_iou", metrics)


if __name__ == "__main__":
    unittest.main()
