# SegFormer Baseline Reproduction

## Purpose

The original Phase 1 SegFormer epoch-4 checkpoint was deleted with its RunPod
persistent volume. The recorded Phase 1 metrics remain historical evidence, but
the corresponding weights cannot be recovered from those metrics. This run
reproduces the documented training design and creates a durable replacement
checkpoint for matched evaluation and ensemble experiments.

The reproduction is not expected to produce byte-identical weights. It uses the
same model family, official video-level split, preprocessing, hyperparameters,
and seed, while recording stronger artifact provenance than the original run.

## Fixed configuration

| Setting | Value |
|---|---|
| Model | SegFormer-B0 |
| Pretrained model | `nvidia/segformer-b0-finetuned-ade-512-512` |
| Task | Binary semantic segmentation |
| Train/validation selection | Full official RipVIS splits |
| Input size | 512 x 512 |
| Batch size | 4 |
| Epochs | 5 |
| Optimizer | AdamW |
| Learning rate | `1e-4` |
| Weight decay | `1e-4` |
| Seed | 42 |
| Checkpoint criterion | Highest validation mean IoU |

## RunPod command

From `/workspace/surfwatch` with the project environment active:

```bash
python scripts/train_baseline.py \
  --ripvis-root /workspace/RipVIS \
  --processed-root /workspace/surfwatch/data/processed \
  --run-dir /workspace/surfwatch/outputs/training/segformer_reproduction_02 \
  --image-size 512 \
  --batch-size 4 \
  --num-workers 4 \
  --epochs 5 \
  --learning-rate 1e-4 \
  --weight-decay 1e-4 \
  --seed 42 \
  --require-clean-git
```

The run fails if `--run-dir` already contains files. Choose a new run directory
instead of deleting or overwriting an earlier experiment.

The formal command also requires a clean Git worktree so the manifest points to
the exact committed implementation. Commit and push the reviewed training
changes before creating the RunPod checkout.

## Optional smoke check

Use a separate run directory for a small software check. Its metrics are not
formal experimental evidence.

```bash
python scripts/train_baseline.py \
  --ripvis-root /workspace/RipVIS \
  --processed-root /workspace/surfwatch/data/processed \
  --run-dir /workspace/surfwatch/outputs/training/segformer_reproduction_smoke \
  --train-subset-size 8 \
  --val-subset-size 4 \
  --batch-size 2 \
  --num-workers 0 \
  --epochs 1 \
  --seed 42
```

## Artifacts

The formal run directory contains:

```text
segformer_reproduction_02/
  checkpoints/
    segformer_epoch_001.pt
    ...
    segformer_epoch_005.pt
    segformer_best_model.pt
    segformer_last_model.pt
  predictions/
  segformer_training_manifest.json
```

The manifest is written before training and updated after every epoch. On
completion it records the full configuration, environment and package versions,
Git revision and dirty state, annotation hashes, metric history, best epoch,
runtime, smoke-inference shapes, checkpoint sizes, and SHA-256 hashes.

## Backup before stopping RunPod

Copy the entire formal run directory to the local SurfWatch workspace while the
pod is still accessible. Then compare the copied files against the hashes in
`segformer_training_manifest.json`.

From the local repository, verify the relocated checkpoint files with:

```bash
python scripts/verify_training_artifacts.py \
  outputs/training/segformer_reproduction_02/segformer_training_manifest.json
```

The command exits unsuccessfully if the manifest is incomplete or a checkpoint
is missing, truncated, or has a different SHA-256 hash.

Do not delete the pod or persistent volume until the local manifest, best
checkpoint, last checkpoint, and all epoch checkpoints have been opened or
hashed successfully. Keep a second copy outside the RunPod volume.

## Replacement evaluation

After the artifacts are copied locally, run the matched evaluator with the new
best checkpoint and the fixed U-Net checkpoint:

```bash
python scripts/evaluate_held_out_models.py \
  --run-name segformer-reproduction-unet-val-full \
  --ripvis-root ../RipVIS \
  --processed-root data/processed \
  --model-checkpoint \
    segformer=outputs/training/segformer_reproduction_02/checkpoints/segformer_best_model.pt \
  --model-checkpoint unet-resnet34=checkpoints/unet_resnet34_best_model.pt \
  --threshold 0.5 \
  --device auto
```

The formal reproduction completed on 23 September 2026. Epoch 4 was selected
with validation IoU 0.7565, and all seven checkpoint files passed manifest
verification after local transfer. The selected checkpoint SHA-256 is
`3fca0f0fd315f306e07d33b21a1641b223a1e5c16a41abe7c121679cd89f7371`.
It is installed locally at the registry's runtime path,
`checkpoints/best_model.pt`; the displaced invalid epoch-1 artifact is retained
as `checkpoints/segformer_invalid_epoch1_model.pt`.

The replacement matched evaluation produced foreground IoU 0.5027 for
SegFormer and 0.4201 for U-Net. These results supersede the invalidated
SegFormer rows from `held_out_model_evaluation_01.md`. See
[`experiments/segformer_reproduction_02.md`](experiments/segformer_reproduction_02.md)
for the complete training, provenance, and comparison record.
