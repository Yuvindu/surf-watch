import json

annotation_path = "../RipVIS/train/coco_annotations/train.json"

with open(annotation_path, "r") as f:
    data = json.load(f)

print("Top-level keys:", data.keys())
print("Number of images:", len(data["images"]))
print("Number of annotations:", len(data["annotations"]))
print("Categories:", data["categories"])
print("Sample image:", data["images"][0])
print("Sample annotation:", data["annotations"][0])