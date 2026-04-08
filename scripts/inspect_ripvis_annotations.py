import json
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path

# Paths
RIPVIS_ROOT = Path("../RipVIS")
annotation_path = RIPVIS_ROOT / "train" / "coco_annotations" / "train.json"
images_dir = RIPVIS_ROOT / "train" / "sampled_images" / "sampled_images" / "images"

# Load JSON
with open(annotation_path, "r") as f:
    data = json.load(f)

# Lookup tables
images_by_id = {img["id"]: img for img in data["images"]}

anns_by_image = {}
for ann in data["annotations"]:
    image_id = ann["image_id"]
    anns_by_image.setdefault(image_id, []).append(ann)

print("Number of images:", len(images_by_id))
print("Number of annotated images:", len(anns_by_image))

# Pick one annotated image
test_image_id = next(iter(anns_by_image.keys()))
img_info = images_by_id[test_image_id]

print("\nTesting image:")
print(img_info)
print("Number of annotations for this image:", len(anns_by_image[test_image_id]))

# Create empty mask
height = img_info["height"]
width = img_info["width"]
mask = np.zeros((height, width), dtype=np.uint8)

# Draw polygons
for ann in anns_by_image[test_image_id]:
    for seg in ann["segmentation"]:
        polygon = np.array(seg, dtype=np.float32).reshape(-1, 2)
        polygon = np.round(polygon).astype(np.int32)
        cv2.fillPoly(mask, [polygon], 1)

# Save test mask
output_dir = Path("data/processed/debug")
output_dir.mkdir(parents=True, exist_ok=True)

mask_path = output_dir / "test_mask.png"
Image.fromarray((mask * 255).astype(np.uint8)).save(mask_path)

print("Saved mask to:", mask_path)

# Load original image
image_path = images_dir / img_info["file_name"]
print("Image path:", image_path)

image = np.array(Image.open(image_path).convert("RGB"))

# Create overlay
overlay = image.copy()
overlay[mask > 0] = [255, 0, 0]

# Show result
plt.figure(figsize=(14, 6))

plt.subplot(1, 3, 1)
plt.imshow(image)
plt.title("Original Image")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(mask, cmap="gray")
plt.title("Binary Mask")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(overlay)
plt.title("Overlay")
plt.axis("off")

plt.tight_layout()
plt.show()