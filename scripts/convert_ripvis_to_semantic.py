import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

RIPVIS_ROOT = Path("../RipVIS")
DEBUG_DIR = Path("data/processed/debug")

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert RipVIS instance annotations into binary semantic masks."
    )
    parser.add_argument(
        "--split",
        choices=["train", "val", "test"],
        required=True,
        help="Dataset split to process.",
    )
    return parser.parse_args()


def load_annotations(annotation_path: Path) -> dict:
    with open(annotation_path, "r") as f:
        return json.load(f)


def build_lookup_tables(data: dict) -> tuple[dict, dict]:
    images_by_id = {img["id"]: img for img in data["images"]}

    anns_by_image: dict[int, list] = {}
    for ann in data["annotations"]:
        image_id = ann["image_id"]
        anns_by_image.setdefault(image_id, []).append(ann)

    return images_by_id, anns_by_image


def has_annotations(data: dict) -> bool:
    annotations = data.get("annotations")
    return isinstance(annotations, list) and len(annotations) > 0


def create_binary_mask(img_info: dict, image_annotations: list) -> np.ndarray:
    height = img_info["height"]
    width = img_info["width"]
    mask = np.zeros((height, width), dtype=np.uint8)

    for ann in image_annotations:
        for seg in ann.get("segmentation", []):
            polygon = np.array(seg, dtype=np.float32).reshape(-1, 2)
            polygon = np.round(polygon).astype(np.int32)
            cv2.fillPoly(mask, [polygon], 1)

    return mask


def save_mask(mask: np.ndarray, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((mask * 255).astype(np.uint8)).save(output_path)


def save_debug_overlay(image_path: Path, mask: np.ndarray, debug_dir: Path) -> None:
    image = np.array(Image.open(image_path).convert("RGB"))
    overlay = image.copy()
    overlay[mask > 0] = [255, 0, 0]

    debug_dir.mkdir(parents=True, exist_ok=True)
    mask_debug_path = debug_dir / "test_mask.png"
    overlay_debug_path = debug_dir / "test_overlay.png"

    Image.fromarray((mask * 255).astype(np.uint8)).save(mask_debug_path)
    Image.fromarray(overlay).save(overlay_debug_path)

    print(f"Saved debug mask to: {mask_debug_path}")
    print(f"Saved debug overlay to: {overlay_debug_path}")


def main() -> None:
    args = parse_args()
    split = args.split

    annotation_filename = f"{split}.json"
    if split == "test":
        annotation_filename = "test_without_annotations.json"

    annotation_path = RIPVIS_ROOT / split / "coco_annotations" / annotation_filename
    images_dir = RIPVIS_ROOT / split / "sampled_images" / "sampled_images" / "images"
    output_dir = Path("data/processed") / split / "masks"

    data = load_annotations(annotation_path)

    if split == "test" and not has_annotations(data):
        print("Test split uses metadata without ground-truth annotations.")
        print(f"Loaded image metadata from: {annotation_path}")
        print("Semantic mask conversion is skipped for the test split.")
        return

    images_by_id, anns_by_image = build_lookup_tables(data)

    print("Number of images:", len(images_by_id))
    print("Number of annotated images:", len(anns_by_image))

    converted_count = 0
    empty_mask_count = 0

    for image_id, img_info in images_by_id.items():
        image_annotations = anns_by_image.get(image_id, [])
        if not image_annotations:
            empty_mask_count += 1

        mask = create_binary_mask(img_info, image_annotations)

        mask_filename = Path(img_info["file_name"]).with_suffix(".png").name
        mask_output_path = output_dir / mask_filename
        save_mask(mask, mask_output_path)
        converted_count += 1

    print(f"Converted {converted_count} images for split '{split}'.")
    print(f"Created {empty_mask_count} empty masks for non-annotated images.")
    print(f"Masks saved under: {output_dir}")

    debug_image_id = next(iter(anns_by_image.keys()))
    debug_img_info = images_by_id[debug_image_id]
    debug_mask = create_binary_mask(debug_img_info, anns_by_image[debug_image_id])
    debug_image_path = images_dir / debug_img_info["file_name"]
    save_debug_overlay(debug_image_path, debug_mask, DEBUG_DIR)


if __name__ == "__main__":
    main()