import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

sys.path.append(str(Path(__file__).resolve().parents[1]))

from configs.baseline_config import BaselineConfig
from src.data.ripvis_dataset import RipVISSemanticDataset
from src.models.segformer_baseline import build_segformer_model
from src.training.engine import train_one_epoch, validate, save_sample_predictions, save_best_model
from src.training.losses import get_loss_fn
from src.training.utils import get_device, set_seed


def main():
    config = BaselineConfig()
    set_seed(config.seed)
    device = get_device()

    train_dataset = RipVISSemanticDataset(
        ripvis_root=config.ripvis_root,
        processed_root=config.processed_root,
        split=config.train_split,
        image_size=config.image_size,
        return_filename=True,
    )

    val_dataset = RipVISSemanticDataset(
        ripvis_root=config.ripvis_root,
        processed_root=config.processed_root,
        split=config.val_split,
        image_size=config.image_size,
        return_filename=True,
    )

    if config.train_subset_size is not None:
        train_dataset = Subset(
            train_dataset,
            range(min(config.train_subset_size, len(train_dataset))),
        )

    if config.val_subset_size is not None:
        val_dataset = Subset(
            val_dataset,
            range(min(config.val_subset_size, len(val_dataset))),
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
    )

    model = build_segformer_model(
        model_name=config.pretrained_model_name,
        num_classes=config.num_classes,
    ).to(device)

    loss_fn = get_loss_fn()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    best_val_iou = -1.0

    for epoch in range(config.num_epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device)
        val_metrics = validate(model, val_loader, loss_fn, device)

        print(
            f"Epoch [{epoch + 1}/{config.num_epochs}] "
            f"train_loss={train_loss:.4f} "
            f"val_loss={val_metrics['val_loss']:.4f} "
            f"val_iou={val_metrics['val_iou']:.4f} "
            f"val_dice={val_metrics['val_dice']:.4f} "
            f"val_precision={val_metrics['val_precision']:.4f} "
            f"val_recall={val_metrics['val_recall']:.4f}"
        )

        if val_metrics["val_iou"] > best_val_iou:
            best_val_iou = val_metrics["val_iou"]
            save_best_model(
                model=model,
                optimizer=optimizer,
                epoch=epoch + 1,
                metrics=val_metrics,
                checkpoint_dir=config.checkpoint_dir,
            )

    save_sample_predictions(
        model=model,
        loader=val_loader,
        device=device,
        output_dir=config.output_dir,
        max_samples=4,
    )


if __name__ == "__main__":
    main()