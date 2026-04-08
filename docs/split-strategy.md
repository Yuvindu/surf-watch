# Train / Validation / Test Split Strategy

## Objective
Define a leakage-safe split strategy for SurfWatch semantic video segmentation experiments using the RipVIS dataset. Reference dataset page: https://huggingface.co/datasets/Irikos/RipVIS

## Chosen split strategy
SurfWatch will use the official RipVIS train/validation/test split as provided in the dataset release. The split is predefined at the video level and is distributed as separate train, validation, and test partitions in the dataset structure.

## Rationale
RipVIS is a video instance segmentation benchmark, and the dataset authors explicitly recommend using their manual split structure. They note that automatic splitting can produce misleading results due to:
- data leakage from sampled frames of the same video appearing in multiple splits
- uneven distribution of video durations
- uneven distribution of orientations/viewpoints
- uneven distribution of rip-current types

Using the official split reduces these risks and provides a more realistic evaluation setup.

## Split sizes
According to the RipVIS Hugging Face dataset card (https://huggingface.co/datasets/Irikos/RipVIS):
- Train: 112 videos
- Validation: 36 videos
- Test: 36 videos

## What the official split means in practice
The RipVIS split groups complete videos into one of three partitions: train, validation, or test. This means that all derived data from a video — including sampled frames, annotations, and masks — must remain in the same partition as the source video.

For SurfWatch, this implies that:
- training is performed only on videos assigned to the training split
- validation is used for model selection, tuning, and quantitative evaluation
- test videos are kept completely unseen during model development
- no frames from a test or validation video should be reused in training, even if extracted separately later

The split is therefore not just a frame count division, but a full video-partitioning strategy designed to preserve temporal independence between training and evaluation data.

## Split rule
The split is defined at the **video level**, not at the frame level.
In other words, the video is the atomic unit of splitting, and every frame or annotation derived from that video inherits the same split membership.

This means:
- all frames from a given video remain in the same split
- annotations and masks follow the source video split
- the test split remains untouched during training and tuning
## Annotation availability note
During implementation, SurfWatch confirmed that the RipVIS training and validation splits provide COCO-style instance annotations that can be converted into binary semantic segmentation masks. However, the public test split is distributed as `test_without_annotations.json`, which provides image metadata without public ground-truth annotations for local mask generation.

This means that, for SurfWatch:
- the **train split** is used for model fitting
- the **validation split** is used for checkpoint selection, tuning, and quantitative evaluation
- the **test split** is used only for qualitative inference on unseen videos unless an external benchmark evaluation method is provided

As a result, local metrics such as IoU, Dice, precision, and recall are computed on the validation split rather than the public test split.

## Leakage risks identified
1. Frame leakage  
   - Risk: sampled frames from the same video appear in multiple splits  
   - Mitigation: keep full videos in only one split

2. Temporal leakage  
   - Risk: near-duplicate consecutive frames inflate performance  
   - Mitigation: evaluate using held-out videos only

3. Viewpoint imbalance  
   - Risk: one split may contain more drone, phone, or fixed-camera footage than another  
   - Mitigation: use the official manually curated split

4. Rip-type / scene imbalance  
   - Risk: different rip-current types or beach conditions may be unevenly distributed  
   - Mitigation: rely on the expert-defined split and document limitations

## Notes for SurfWatch
RipVIS is released as an instance segmentation dataset. For SurfWatch, instance masks will be merged into a single semantic “rip” class mask while preserving the original video-level split.

When implementing the data pipeline, we should preserve the original RipVIS split folders or metadata identifiers and ensure that any preprocessing, mask conversion, or frame extraction step keeps outputs attached to the same source-video partition.

For semantic mask conversion specifically:
- `train/coco_annotations/train.json` is used to generate training masks
- `val/coco_annotations/val.json` is used to generate validation masks
- the public test split is not converted into ground-truth semantic masks because public annotations are not provided in the same format