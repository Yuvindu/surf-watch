# U-Net ResNet34 MARSP Validation

## Purpose

This experiment validates that the trained U-Net ResNet34 checkpoint can pass through both the baseline video segmentation workflow and the model-agnostic MARSP workflow without SegFormer-specific code paths.

The primary evidence uses a complete video from the official RipVIS validation split. An earlier 24-frame training-video excerpt is retained only as an initial smoke check.

## Primary run configuration

| Field | Value |
|---|---|
| Source video | RipVIS-072 |
| Dataset split | Official RipVIS validation split |
| Video | 62 frames, 1280 x 720, 29.97 fps |
| Model | `unet-resnet34` |
| Checkpoint | `unet_resnet34_best_model.pt` |
| Checkpoint SHA-256 | `b3a0be0a536fb6aa0e01b157d5e9f521e041a100629749f7c070d99bc4d8917b` |
| Input SHA-256 | `0962679e235c04b988534afb362bf47a3db9bcc06812008e271a8cbeadc31817` |
| Window size | 5 |
| Probability threshold | 0.5 |
| Execution device | CPU |
| Code revision at run time | `197bac8b99a584e28ef2a771a149f84ddd03840a` plus the uncommitted SCRUM-76 comparison changes |

The comparison runner records the model, input identity, checkpoint identity, parameters, Git revision, workflow modes, and timings in the generated metrics JSON. Existing artifacts are reused only when their recorded model, checkpoint, input, and parameters match the requested run.

## Reproduction command

```bash
python scripts/run_baseline_vs_marsp_compare.py \
  --video-name RipVIS-072-unet-resnet34-validation \
  --input /path/to/RipVIS/val/videos/RipVIS-072.mp4 \
  --model unet-resnet34 \
  --checkpoint checkpoints/unet_resnet34_best_model.pt \
  --window-size 5 \
  --threshold 0.5
```

Use a model-specific logical video name so validation artifacts cannot overwrite an earlier experiment using another model.

## Compatibility result

The U-Net adapter completed both workflows successfully:

- Baseline inference generated a 62-frame overlay, binary mask video, and probability video.
- MARSP completed motion compensation, inference on original and stabilised frames, temporal aggregation, and final overlay rendering.
- The side-by-side comparison contained 62 frames at 2560 x 720, with visible `Baseline` and `MARSP` labels.
- The U-Net probability output was accepted directly by MARSP; no SegFormer-specific model access or tensor handling was encountered.
- All expected artifacts were present, non-empty, and had the expected frame count and dimensions.

## Temporal stability summary

| Metric | Baseline | MARSP | Delta | Preferred direction |
|---|---:|---:|---:|---|
| Consecutive IoU | 0.9419 | 0.9608 | +0.0189 | Higher |
| Consecutive Dice | 0.9697 | 0.9797 | +0.0100 | Higher |
| Pixel change rate | 0.000334 | 0.000213 | -0.000121 | Lower |
| Mask area variation (px) | 769.1 | 815.3 | +46.3 | Lower |
| Mean absolute area change (px) | 251.7 | 168.6 | -83.2 | Lower |
| Mean small blob count | 0.0000 | 0.0000 | 0.0000 | Lower |

MARSP improved four of the six temporal indicators on this validation video. Small-blob count was unchanged because neither workflow produced small disconnected components. Mask-area variation increased by 46.3 pixels, showing that MARSP reduced rapid frame-to-frame changes without reducing every form of area variation.

## Timing

| Workflow | CPU time (seconds) |
|---|---:|
| Baseline | 10.40 |
| MARSP | 27.43 |
| Complete comparison | 39.12 |

The MARSP timing includes motion compensation, two segmentation passes, two temporal aggregation passes, and pipeline artifact generation. It is therefore expected to be substantially higher than the single-pass baseline timing.

## Visual inspection

The first, middle, and final comparison frames were inspected. Both workflows produced non-blank, spatially plausible rip-current regions with consistent labels. MARSP was more conservative than the baseline around the middle of the video and recovered a similar region by the final frame. This behaviour is consistent with temporal smoothing, but it should be checked against annotations before making a semantic-accuracy claim.

## Initial smoke check

Before the validation run, the same checkpoint was processed through both workflows on the first 24 frames of the RipVIS-051 training video. The smoke check also completed successfully and improved all measured temporal-stability indicators. Its purpose was limited to verifying the interface and output artifacts; it is not used as the primary SCRUM-76 evidence.

## Limitations and follow-up

- This result establishes compatibility on one complete held-out validation video, but broader evaluation across different viewpoints, motion levels, rip conditions, and no-rip videos is still required for the research report.
- The comparison runner reports temporal stability against consecutive predictions. It does not score semantic accuracy against frame annotations.
- RipVIS-072 had a 100% motion-transform success rate, but the recorded motion-adaptation scores were very small. Additional videos with stronger camera movement are needed to evaluate the adaptive weighting behaviour.
- The slight increase in mask-area variation should be examined alongside ground truth and across more videos before changing MARSP parameters.
