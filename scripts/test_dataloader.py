import sys
from pathlib import Path

from torch.utils.data import DataLoader

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.data.ripvis_dataset import RipVISSemanticDataset

dataset = RipVISSemanticDataset(
    ripvis_root="../RipVIS",
    processed_root="data/processed",
    split="train",
    image_size=(512, 512),
    return_filename=True,
)

print("Dataset size:", len(dataset))

image, mask, filename = dataset[0]
print("Filename:", filename)
print("Image shape:", image.shape)
print("Mask shape:", mask.shape)
print("Mask unique values:", mask.unique())

loader = DataLoader(dataset, batch_size=4, shuffle=True)
batch = next(iter(loader))
images, masks, filenames = batch

print("Batch image shape:", images.shape)
print("Batch mask shape:", masks.shape)
print("Batch filenames:", filenames[:2])