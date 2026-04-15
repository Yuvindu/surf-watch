import os
from dataclasses import dataclass


@dataclass
class BaselineConfig:
    ripvis_root: str = os.getenv("RIPVIS_ROOT", "/workspace/RipVIS")
    processed_root: str = os.getenv("PROCESSED_ROOT", "/workspace/surfwatch/data/processed")

    train_split: str = "train"
    val_split: str = "val"

    image_size: tuple = (512, 512)
    batch_size: int = 4
    num_workers: int = 4

    num_classes: int = 2
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    num_epochs: int = 5

    train_subset_size: int = None
    val_subset_size: int = None

    checkpoint_dir: str = "checkpoints"
    output_dir: str = "outputs/predictions/baseline_cloud_run_01"

    pretrained_model_name: str = "nvidia/segformer-b0-finetuned-ade-512-512"
    save_best_only: bool = True
    seed: int = 42