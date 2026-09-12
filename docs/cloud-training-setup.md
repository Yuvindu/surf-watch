# Cloud Training Setup

## Overview
This document describes the recommended directory layout and environment configuration for running SurfWatch training jobs on a cloud GPU instance such as RunPod.

The goal is to keep the project code, raw RipVIS dataset, processed masks, checkpoints, and prediction outputs organized in a way that is easy to reproduce across training sessions.

## Recommended directory layout
Use the following layout inside the mounted workspace volume:

```text
/workspace/
  surfwatch/
  RipVIS/
```

### Directory roles
- `/workspace/surfwatch/` contains the SurfWatch project repository
- `/workspace/RipVIS/` contains the raw RipVIS dataset used for training and validation

## Required environment variables
The baseline training configuration reads dataset paths from environment variables. Set these before running any training or evaluation scripts.

```bash
export RIPVIS_ROOT=/workspace/RipVIS
export PROCESSED_ROOT=/workspace/surfwatch/data/processed
```

### What these variables mean
- `RIPVIS_ROOT` points to the root of the raw RipVIS dataset
- `PROCESSED_ROOT` points to the SurfWatch-generated processed data directory, including semantic segmentation masks

## Default local fallback behavior
If these environment variables are not set, SurfWatch falls back to the local development defaults:

- `RIPVIS_ROOT=../RipVIS`
- `PROCESSED_ROOT=data/processed`

This allows the same codebase to run both locally and in the cloud without changing source files.

## Expected training inputs
Before starting a cloud training run, ensure the following are available:

### In `/workspace/RipVIS/`
- training split data
- validation split data
- COCO annotation files
- extracted sampled images needed by the training pipeline

### In `/workspace/surfwatch/data/processed/`
- converted semantic segmentation masks for the training split
- converted semantic segmentation masks for the validation split

## Example training workflow
From inside the project directory:

```bash
cd /workspace/surfwatch
source .venv/bin/activate
python scripts/train_baseline.py
```

## Notes
- Keep the raw RipVIS dataset outside the main SurfWatch repository to avoid mixing source data with project code.
- Store checkpoints and prediction outputs inside the SurfWatch project directory so they remain grouped with the corresponding experiment setup.
- Use distinct output folders for each cloud experiment to keep runs easy to compare and review.

## Validated RunPod Configuration

The U-Net ResNet34 cloud run on 2 September 2026 used:

- one NVIDIA RTX 4090 with 24 GB VRAM
- CUDA 12.8 and PyTorch 2.8.0
- 16 vCPUs and 61 GB RAM
- a 50 GB persistent network volume mounted at `/workspace`
- a project virtual environment at `/workspace/surfwatch/.venv`

The run used batch size 4 and four dataloader workers. It sustained high GPU utilisation and completed 20 epochs over 14,616 training and 4,349 validation frames in 55 minutes 25 seconds.

When a dataset upload is interrupted, resume it with checksum verification rather than file-size matching alone. A partial transfer can occasionally leave a same-size corrupt file that passes a size-only check.

```bash
rsync -rc --partial -e "ssh -p PORT -i ~/.ssh/id_ed25519" \
  /local/path/images \
  root@RUNPOD_HOST:/workspace/RipVIS/train/sampled_images/sampled_images/
```

Before terminating a paid pod, copy the best and last checkpoints, manifest, prediction samples, and training log back to local storage. Stopping the pod ends active GPU charges; the network volume remains persistent and separately billed until deleted.
