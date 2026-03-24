# Dataset Access and Local Storage

## Dataset

SurfWatch currently uses the **RipVIS** dataset for experimentation.

Reference:
https://huggingface.co/datasets/Irikos/RipVIS

## Purpose

This document describes how RipVIS data is stored locally for SurfWatch and how team members should access it without breaking the project's split strategy.

## Local Storage Location

The dataset is stored locally under the project `data/` directory, which is ignored by git.

Current structure:

```text
surfwatch/
  data/
    ripvis/
      raw/
      subset/
        train/
        val/
        test/
```

## Storage Rules

- keep the original downloaded dataset files under `data/ripvis/raw/`
- place the working subset under `data/ripvis/subset/`
- preserve the official RipVIS split at the video level
- keep train, validation, and test videos in separate folders
- keep any derived frames, masks, or metadata attached to the same source-video partition

## Why This Matters

SurfWatch evaluates models on held-out videos. If frames from the same source video are mixed across `train/`, `val/`, and `test/`, the project can suffer from data leakage and misleading evaluation results.

This storage convention supports the split policy documented in [docs/split-strategy.md](/Users/rashmikecaldera/Developer/curtin/CSP/surfwatch/docs/split-strategy.md).

## Verification Completed

- local dataset directories exist under `data/ripvis/`
- split folders exist for `train`, `val`, and `test`
- the repository is configured to keep `data/` out of version control

## Team Access

Team members can access the dataset by:

1. downloading RipVIS from the official source
2. storing the original files under `data/ripvis/raw/`
3. placing the working split data under `data/ripvis/subset/train/`, `data/ripvis/subset/val/`, and `data/ripvis/subset/test/`
4. preserving the original video-level partition when creating any subset, extracted frames, or converted masks

## Notes

- do not commit dataset files into the repository
- if a team member uses a different local machine path, the same internal folder structure should still be preserved
- any preprocessing pipeline should read split membership from the source video and carry it forward to all outputs
