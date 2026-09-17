import os
from dataclasses import asdict, dataclass
from typing import Optional, Tuple


@dataclass
class UnetResNet34TrainingConfig:
    ripvis_root: str = os.getenv("RIPVIS_ROOT", "/workspace/RipVIS")
    processed_root: str = os.getenv(
        "PROCESSED_ROOT", "/workspace/surfwatch/data/processed"
    )
    train_split: str = "train"
    val_split: str = "val"
    image_size: Tuple[int, int] = (512, 512)
    batch_size: int = 4
    num_workers: int = 4
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    num_epochs: int = 20
    train_subset_size: Optional[int] = None
    val_subset_size: Optional[int] = None
    checkpoint_dir: str = "checkpoints"
    output_dir: str = "outputs/predictions/unet_resnet34"
    encoder_weights: Optional[str] = "imagenet"
    threshold: float = 0.5
    seed: int = 42
    model_id: str = "unet-resnet34"
    best_checkpoint_name: str = "unet_resnet34_best_model.pt"
    last_checkpoint_name: str = "unet_resnet34_last_model.pt"

    def validate(self) -> None:
        if len(self.image_size) != 2 or min(self.image_size) <= 0:
            raise ValueError("image_size must contain two positive integers")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.num_workers < 0:
            raise ValueError("num_workers cannot be negative")
        if self.num_epochs <= 0:
            raise ValueError("num_epochs must be positive")
        if self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError("optimizer values must be non-negative and learning rate positive")
        if not 0.0 <= self.threshold <= 1.0:
            raise ValueError("threshold must be between 0.0 and 1.0")
        for name in ("train_subset_size", "val_subset_size"):
            value = getattr(self, name)
            if value is not None and value <= 0:
                raise ValueError(f"{name} must be positive when provided")

    def to_dict(self) -> dict:
        return asdict(self)
