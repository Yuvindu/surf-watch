import argparse
import hashlib
import importlib.metadata
import json
import platform
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import torch
from torch.utils.data import DataLoader, Subset

sys.path.append(str(Path(__file__).resolve().parents[1]))

from configs.baseline_config import BaselineConfig
from configs.segformer_training_config import SegformerTrainingConfig
from src.data.ripvis_dataset import RipVISSemanticDataset
from src.models.segformer_adapter import SegFormerSegmentationAdapter
from src.models.segformer_baseline import build_segformer_model
from src.training.engine import (
    save_best_model,
    save_sample_predictions,
    train_one_epoch,
    validate,
)
from src.training.losses import get_loss_fn
from src.training.utils import get_device, set_seed


def build_parser() -> argparse.ArgumentParser:
    defaults = SegformerTrainingConfig()
    parser = argparse.ArgumentParser(
        description="Train a reproducible SegFormer-B0 model on RipVIS"
    )
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
    parser.add_argument("--run-dir", default=defaults.run_dir)
    parser.add_argument(
        "--pretrained-model-name", default=defaults.pretrained_model_name
    )
    parser.add_argument("--threshold", type=float, default=defaults.threshold)
    parser.add_argument("--seed", type=int, default=defaults.seed)
    parser.add_argument(
        "--require-clean-git",
        action="store_true",
        help="Refuse to start unless the Git worktree is clean",
    )
    return parser


def config_from_args(args: argparse.Namespace) -> SegformerTrainingConfig:
    config = SegformerTrainingConfig(
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
        run_dir=args.run_dir,
        pretrained_model_name=args.pretrained_model_name,
        threshold=args.threshold,
        seed=args.seed,
        require_clean_git=args.require_clean_git,
    )
    config.validate()
    return config


def prepare_run_directory(run_dir: Path) -> None:
    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(
            f"Run directory is not empty: {run_dir}. Choose a new --run-dir so "
            "existing checkpoints cannot be overwritten."
        )
    run_dir.mkdir(parents=True, exist_ok=True)


def _build_dataset(config: SegformerTrainingConfig, split: str):
    return RipVISSemanticDataset(
        ripvis_root=config.ripvis_root,
        processed_root=config.processed_root,
        split=split,
        image_size=config.image_size,
        return_filename=True,
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


def _sample_path(dataset) -> Path:
    base_dataset = dataset.dataset if isinstance(dataset, Subset) else dataset
    index = dataset.indices[0] if isinstance(dataset, Subset) else 0
    return Path(base_dataset.samples[index][0])


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _package_versions() -> dict:
    versions = {}
    for name in ("torch", "torchvision", "transformers", "opencv-python", "numpy"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "not-installed"
    return versions


def _git_metadata() -> dict:
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return {"revision": "unknown", "dirty": None}
    return {"revision": revision, "dirty": bool(status)}


def _environment_metadata(device: torch.device) -> dict:
    gpu_name = None
    if device.type == "cuda":
        gpu_name = torch.cuda.get_device_name(device)
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "device": str(device),
        "gpu": gpu_name,
        "cuda_runtime": torch.version.cuda,
        "dependencies": _package_versions(),
        "git": _git_metadata(),
    }


def _annotation_metadata(config: SegformerTrainingConfig) -> dict:
    result = {}
    root = Path(config.ripvis_root)
    for split in (config.train_split, config.val_split):
        path = root / split / "coco_annotations" / f"{split}.json"
        result[split] = {
            "path": str(path),
            "sha256": sha256_file(path) if path.is_file() else None,
        }
    return result


def _checkpoint_metadata(
    config: SegformerTrainingConfig,
    environment: dict,
    created_at: str,
) -> dict:
    return {
        "model_id": config.model_id,
        "config": config.to_dict(),
        "run_created_at": created_at,
        "training_code_revision": environment["git"]["revision"],
    }


def run_training(config: SegformerTrainingConfig) -> dict:
    config.validate()
    set_seed(config.seed)
    device = get_device()
    environment = _environment_metadata(device)
    if config.require_clean_git and environment["git"]["dirty"] is not False:
        raise RuntimeError(
            "Formal training requires a clean Git worktree with a known revision"
        )

    run_dir = Path(config.run_dir)
    prepare_run_directory(run_dir)
    checkpoint_dir = run_dir / "checkpoints"
    prediction_dir = run_dir / "predictions"
    manifest_path = run_dir / config.manifest_name
    checkpoint_dir.mkdir(parents=True)

    created_at = datetime.now(timezone.utc).isoformat()
    train_dataset = _subset(
        _build_dataset(config, config.train_split), config.train_subset_size
    )
    val_dataset = _subset(
        _build_dataset(config, config.val_split), config.val_subset_size
    )
    train_loader = _build_loader(train_dataset, config, device, shuffle=True)
    val_loader = _build_loader(val_dataset, config, device, shuffle=False)

    model = build_segformer_model(
        config.pretrained_model_name,
        config.num_classes,
    ).to(device)
    pretrained_revision = getattr(model.config, "_commit_hash", None)
    loss_fn = get_loss_fn()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    checkpoint_metadata = _checkpoint_metadata(config, environment, created_at)
    history = []
    best_iou = -1.0
    best_epoch = None
    best_checkpoint = checkpoint_dir / config.best_checkpoint_name
    last_checkpoint = checkpoint_dir / config.last_checkpoint_name
    started = time.perf_counter()

    manifest = {
        "status": "running",
        "created_at": created_at,
        "model_id": config.model_id,
        "model": {
            "pretrained_model_name": config.pretrained_model_name,
            "pretrained_revision": pretrained_revision,
        },
        "config": config.to_dict(),
        "environment": environment,
        "dataset": {
            "train_frames": len(train_dataset),
            "validation_frames": len(val_dataset),
            "annotations": _annotation_metadata(config),
        },
        "checkpoint_selection": "highest validation mean IoU",
        "history": history,
    }
    _write_json(manifest_path, manifest)

    print(
        f"Training {config.model_id} on {device}: "
        f"{len(train_dataset)} train / {len(val_dataset)} validation frames"
    )
    for epoch in range(config.num_epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device)
        metrics = validate(model, val_loader, loss_fn, device)
        epoch_number = epoch + 1
        epoch_result = {"epoch": epoch_number, "train_loss": train_loss, **metrics}
        history.append(epoch_result)
        print(
            f"Epoch [{epoch_number}/{config.num_epochs}] "
            f"train_loss={train_loss:.4f} val_loss={metrics['val_loss']:.4f} "
            f"val_iou={metrics['val_iou']:.4f} val_dice={metrics['val_dice']:.4f} "
            f"val_precision={metrics['val_precision']:.4f} "
            f"val_recall={metrics['val_recall']:.4f}"
        )

        epoch_checkpoint = checkpoint_dir / f"segformer_epoch_{epoch_number:03d}.pt"
        save_best_model(
            model,
            optimizer,
            epoch_number,
            metrics,
            str(checkpoint_dir),
            filename=epoch_checkpoint.name,
            extra_state=checkpoint_metadata,
        )
        shutil.copy2(epoch_checkpoint, last_checkpoint)
        if metrics["val_iou"] > best_iou:
            best_iou = metrics["val_iou"]
            best_epoch = epoch_number
            shutil.copy2(epoch_checkpoint, best_checkpoint)

        manifest.update(
            {
                "history": history,
                "best_epoch": best_epoch,
                "best_metrics": history[best_epoch - 1],
                "completed_epochs": epoch_number,
            }
        )
        _write_json(manifest_path, manifest)

    training_seconds = time.perf_counter() - started
    checkpoint = torch.load(best_checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    save_sample_predictions(
        model,
        val_loader,
        device,
        str(prediction_dir),
    )

    adapter = SegFormerSegmentationAdapter(
        checkpoint_path=str(best_checkpoint),
        device=device,
        config=BaselineConfig(
            image_size=config.image_size,
            pretrained_model_name=config.pretrained_model_name,
        ),
        threshold=config.threshold,
    )
    sample_path = _sample_path(val_dataset)
    frame = cv2.imread(str(sample_path))
    if frame is None:
        raise RuntimeError(f"Could not read smoke-inference frame: {sample_path}")
    smoke_result = adapter.predict(frame)

    checkpoint_files = sorted(checkpoint_dir.glob("*.pt"))
    manifest.update(
        {
            "status": "complete",
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "training_seconds": training_seconds,
            "best_epoch": checkpoint["epoch"],
            "best_metrics": checkpoint["metrics"],
            "best_checkpoint": str(best_checkpoint),
            "last_checkpoint": str(last_checkpoint),
            "checkpoints": [
                {
                    "path": str(path),
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
                for path in checkpoint_files
            ],
            "smoke_inference": {
                "input": str(sample_path),
                "probability_map_shape": list(smoke_result.probability_map.shape),
                "mask_shape": list(smoke_result.binary_mask.shape),
            },
        }
    )
    _write_json(manifest_path, manifest)
    return manifest


def main() -> None:
    config = config_from_args(build_parser().parse_args())
    manifest = run_training(config)
    print(f"Best checkpoint: {manifest['best_checkpoint']}")
    print(f"Best validation IoU: {manifest['best_metrics']['val_iou']:.4f}")
    print(f"Manifest: {Path(config.run_dir) / config.manifest_name}")


if __name__ == "__main__":
    main()
