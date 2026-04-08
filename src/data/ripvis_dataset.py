from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF


class RipVISSemanticDataset(Dataset):
    def __init__(
        self,
        ripvis_root: str,
        processed_root: str,
        split: str,
        image_size=(512, 512),
        return_filename: bool = False,
    ):
        if split not in {"train", "val"}:
            raise ValueError("split must be 'train' or 'val'")

        self.split = split
        self.return_filename = return_filename
        self.image_size = image_size

        self.images_dir = (
            Path(ripvis_root) / split / "sampled_images" / "sampled_images" / "images"
        )
        self.masks_dir = Path(processed_root) / split / "masks"

        self.image_paths = sorted(self.images_dir.glob("*.jpg"))

        if not self.image_paths:
            raise FileNotFoundError(f"No images found in {self.images_dir}")

        self.samples = []
        missing_masks = []

        for image_path in self.image_paths:
            mask_name = image_path.with_suffix(".png").name
            mask_path = self.masks_dir / mask_name

            if mask_path.exists():
                self.samples.append((image_path, mask_path))
            else:
                missing_masks.append(mask_path.name)

        if missing_masks:
            raise FileNotFoundError(
                f"Missing {len(missing_masks)} mask files. Example: {missing_masks[:5]}"
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        image_path, mask_path = self.samples[idx]

        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        # Resize image and mask to the same fixed size
        image = TF.resize(
            image,
            self.image_size,
            interpolation=InterpolationMode.BILINEAR,
        )
        mask = TF.resize(
            mask,
            self.image_size,
            interpolation=InterpolationMode.NEAREST,
        )

        image = TF.to_tensor(image)

        mask_np = np.array(mask)
        mask_np = (mask_np > 0).astype(np.uint8)
        mask = torch.from_numpy(mask_np).long()

        if self.return_filename:
            return image, mask, image_path.name

        return image, mask