import os
from dataclasses import dataclass


@dataclass
class UNetConfig:
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
    output_dir: str = "outputs/predictions/unet_cloud_run_01"

    encoder_name: str = "resnet34"
    encoder_weights: str = "imagenet"
    # U-Net's pretrained encoder expects ImageNet-normalized inputs, unlike
    # the SegFormer baseline, which handles its own normalization internally.
    normalize_inputs: bool = True

    save_best_only: bool = True
    seed: int = 42
