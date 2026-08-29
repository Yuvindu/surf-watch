import argparse
import importlib.metadata
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import cv2
import torch
from torch.utils.data import DataLoader, Subset

sys.path.append(str(Path(__file__).resolve().parents[1]))

from configs.baseline_config import BaselineConfig
from configs.unet_resnet34_config import UnetResNet34TrainingConfig
from src.data.ripvis_dataset import RipVISSemanticDataset
from src.models.unet_resnet34 import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    build_unet_resnet34_model,
)
from src.models.unet_resnet34_adapter import UnetResNet34SegmentationAdapter
from src.training.engine import (
    forward_binary_logits,
    predict_binary,
    prepare_binary_targets,
    save_best_model,
    save_sample_predictions,
    train_one_epoch,
    validate,
)
from src.training.losses import get_binary_loss_fn
from src.training.utils import get_device, set_seed


def build_parser() -> argparse.ArgumentParser:
    defaults = UnetResNet34TrainingConfig()
    parser = argparse.ArgumentParser(description="Train U-Net ResNet34 on RipVIS")
    parser.add_argument("--ripvis-root", default=defaults.ripvis_root)
    parser.add_argument("--processed-root", default=defaults.processed_root)
    parser.add_argument("--image-size", type=int, default=defaults.image_size[0])
    parser.add_argument("--batch-size", type=int, default=defaults.batch_size)
    parser.add_argument("--num-workers", type=int, default=defaults.num_workers)
    parser.add_argument("--epochs", type=int, default=defaults.num_epochs)
    parser.add_argument("--learning-rate", type=float, default=defaults.learning_rate)
    parser.add_argument("--weight-decay", type=float, default=defaults.weight_decay)
    parser.add_argument("--train-subset-size", type=int)
    parser.add_argument("--val-subset-size", type=int)
    parser.add_argument("--checkpoint-dir", default=defaults.checkpoint_dir)
    parser.add_argument("--output-dir", default=defaults.output_dir)
    parser.add_argument(
        "--encoder-weights",
        choices=("imagenet", "none"),
        default=defaults.encoder_weights,
    )
    parser.add_argument("--threshold", type=float, default=defaults.threshold)
    parser.add_argument("--seed", type=int, default=defaults.seed)
    return parser


def config_from_args(args: argparse.Namespace) -> UnetResNet34TrainingConfig:
    config = UnetResNet34TrainingConfig(
        ripvis_root=args.ripvis_root,
        processed_root=args.processed_root,
        image_size=(args.image_size, args.image_size),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        num_epochs=args.epochs,
        train_subset_size=args.train_subset_size,
        val_subset_size=args.val_subset_size,
        checkpoint_dir=args.checkpoint_dir,
        output_dir=args.output_dir,
        encoder_weights=None if args.encoder_weights == "none" else args.encoder_weights,
        threshold=args.threshold,
        seed=args.seed,
    )
    config.validate()
    return config


def _build_dataset(config, split):
    return RipVISSemanticDataset(
        ripvis_root=config.ripvis_root,
        processed_root=config.processed_root,
        split=split,
        image_size=config.image_size,
        return_filename=True,
        normalization=(IMAGENET_MEAN, IMAGENET_STD),
    )


def _subset(dataset, size):
    if size is None:
        return dataset
    return Subset(dataset, range(min(size, len(dataset))))


def _build_loader(dataset, config, device, shuffle):
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=shuffle,
        num_workers=config.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=config.num_workers > 0,
    )


def _denormalize(images):
    mean = torch.tensor(IMAGENET_MEAN, dtype=images.dtype).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, dtype=images.dtype).view(1, 3, 1, 1)
    return (images * std + mean).clamp(0.0, 1.0)


def _sample_path(dataset):
    base_dataset = dataset.dataset if isinstance(dataset, Subset) else dataset
    index = dataset.indices[0] if isinstance(dataset, Subset) else 0
    return base_dataset.samples[index][0]


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _package_versions() -> dict:
    packages = (
        "torch",
        "torchvision",
        "segmentation-models-pytorch",
        "opencv-python",
        "numpy",
    )
    return {name: importlib.metadata.version(name) for name in packages}


def _git_revision() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return result.stdout.strip()


def run_training(config: UnetResNet34TrainingConfig) -> dict:
    set_seed(config.seed)
    device = get_device()
    train_dataset = _subset(_build_dataset(config, config.train_split), config.train_subset_size)
    val_dataset = _subset(_build_dataset(config, config.val_split), config.val_subset_size)
    train_loader = _build_loader(train_dataset, config, device, shuffle=True)
    val_loader = _build_loader(val_dataset, config, device, shuffle=False)

    model = build_unet_resnet34_model(config.encoder_weights).to(device)
    loss_fn = get_binary_loss_fn()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    checkpoint_dir = Path(config.checkpoint_dir)
    history = []
    best_iou = -1.0
    started = time.perf_counter()

    print(
        f"Training {config.model_id} on {device}: "
        f"{len(train_dataset)} train / {len(val_dataset)} validation frames"
    )
    for epoch in range(config.num_epochs):
        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            loss_fn,
            device,
            forward_logits=forward_binary_logits,
            prepare_targets=prepare_binary_targets,
        )
        metrics = validate(
            model,
            val_loader,
            loss_fn,
            device,
            forward_logits=forward_binary_logits,
            prepare_targets=prepare_binary_targets,
            predict_from_logits=predict_binary,
            threshold=config.threshold,
        )
        epoch_result = {"epoch": epoch + 1, "train_loss": train_loss, **metrics}
        history.append(epoch_result)
        print(
            f"Epoch [{epoch + 1}/{config.num_epochs}] "
            f"train_loss={train_loss:.4f} val_loss={metrics['val_loss']:.4f} "
            f"val_iou={metrics['val_iou']:.4f} val_dice={metrics['val_dice']:.4f} "
            f"val_precision={metrics['val_precision']:.4f} "
            f"val_recall={metrics['val_recall']:.4f}"
        )

        checkpoint_metadata = {
            "model_id": config.model_id,
            "config": config.to_dict(),
        }
        if metrics["val_iou"] > best_iou:
            best_iou = metrics["val_iou"]
            save_best_model(
                model,
                optimizer,
                epoch + 1,
                metrics,
                str(checkpoint_dir),
                filename=config.best_checkpoint_name,
                extra_state=checkpoint_metadata,
            )
        save_best_model(
            model,
            optimizer,
            epoch + 1,
            metrics,
            str(checkpoint_dir),
            filename=config.last_checkpoint_name,
            extra_state=checkpoint_metadata,
        )

    training_seconds = time.perf_counter() - started
    best_checkpoint = checkpoint_dir / config.best_checkpoint_name
    checkpoint = torch.load(best_checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    save_sample_predictions(
        model,
        val_loader,
        device,
        config.output_dir,
        forward_logits=forward_binary_logits,
        predict_from_logits=predict_binary,
        threshold=config.threshold,
        denormalize_images=_denormalize,
    )

    adapter = UnetResNet34SegmentationAdapter(
        checkpoint_path=str(best_checkpoint),
        device=device,
        config=BaselineConfig(image_size=config.image_size),
        threshold=config.threshold,
    )
    sample_path = _sample_path(val_dataset)
    frame = cv2.imread(str(sample_path))
    if frame is None:
        raise RuntimeError(f"Could not read smoke-inference frame: {sample_path}")
    smoke_result = adapter.predict(frame)

    manifest = {
        "model_id": config.model_id,
        "config": config.to_dict(),
        "environment": {
            "python": platform.python_version(),
            "device": str(device),
            "dependencies": _package_versions(),
            "git_revision": _git_revision(),
        },
        "dataset": {
            "train_frames": len(train_dataset),
            "validation_frames": len(val_dataset),
        },
        "training_seconds": training_seconds,
        "best_epoch": checkpoint["epoch"],
        "best_metrics": checkpoint["metrics"],
        "checkpoint": str(best_checkpoint),
        "history": history,
        "smoke_inference": {
            "input": str(sample_path),
            "probability_map_shape": list(smoke_result.probability_map.shape),
            "mask_shape": list(smoke_result.binary_mask.shape),
        },
    }
    _write_json(checkpoint_dir / "unet_resnet34_training_manifest.json", manifest)
    return manifest


def main() -> None:
    config = config_from_args(build_parser().parse_args())
    manifest = run_training(config)
    print(f"Best checkpoint: {manifest['checkpoint']}")
    print(f"Best validation IoU: {manifest['best_metrics']['val_iou']:.4f}")


if __name__ == "__main__":
    main()
