from dataclasses import dataclass


@dataclass
class BaselineConfig:
    ripvis_root: str = "../RipVIS"
    processed_root: str = "data/processed"

    train_split: str = "train"
    val_split: str = "val"

    image_size: tuple = (512, 512)
    batch_size: int = 2
    num_workers: int = 0

    num_classes: int = 2
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    num_epochs: int = 3

    train_subset_size: int = 1000
    val_subset_size: int = 200

    checkpoint_dir: str = "checkpoints"
    output_dir: str = "outputs/predictions/baseline_run_02"

    pretrained_model_name: str = "nvidia/segformer-b0-finetuned-ade-512-512"
    save_best_only: bool = True
    seed: int = 42