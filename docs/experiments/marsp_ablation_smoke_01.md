# MARSP Ablation Smoke Run 01

## Purpose

Verify the SCRUM-87 A0-A6 runner with the real SegFormer and U-Net ResNet34
checkpoints before committing GPU time to the full held-out experiment. This is
a software and artifact smoke test, not research evidence.

## Run

- Date: 1 October 2026
- Run name: `scrum87-local-smoke-3frames`
- Device: CPU
- Data: first three ordered labelled validation frames from `RipVIS-001`
- SegFormer checkpoint: local reproduced best checkpoint
- U-Net checkpoint: local epoch-10 best checkpoint
- Ensemble weights: equal weighting
- Selected A3-A6 path: SegFormer
- Window size: 3
- Threshold: 0.5
- Minimum retained component area: 500 pixels

## Result

All seven configurations completed successfully:

| ID | Foreground IoU | Dice | Precision | Recall |
| --- | ---: | ---: | ---: | ---: |
| A0 | 0.5659 | 0.7227 | 0.7434 | 0.7032 |
| A1 | 0.5370 | 0.6987 | 0.9450 | 0.5543 |
| A2 | 0.5398 | 0.7012 | 0.8170 | 0.6141 |
| A3 | 0.5639 | 0.7212 | 0.7457 | 0.6982 |
| A4 | 0.5645 | 0.7216 | 0.7453 | 0.6995 |
| A5 | 0.5659 | 0.7227 | 0.7434 | 0.7032 |
| A6 | 0.5642 | 0.7214 | 0.7452 | 0.6991 |

The two sampled-frame motion pairs were estimated successfully with no
fallbacks. The JSON summary recorded source-coordinate evaluation, motion
evidence, A0 references for A1 and A2, selected-path references for A3-A6, and
per-video paired deltas. The frame CSV contained 21 data rows and the video CSV
contained seven data rows.

After adding the six-source-frame gap boundary, a second CPU smoke check used
two labelled frames from `RipVIS-015`, separated by 30 source frames. It
completed all seven configurations, retained both frames in semantic scoring,
and recorded one skipped temporal pair with no consecutive-pair metric.

## Verification

- 85 repository tests passed after the gap-boundary review.
- CLI help and argument registration loaded successfully.
- `git diff --check` reported no whitespace errors.
- Generated artifacts were written outside the repository under
  `/private/tmp/surfwatch-ablation-smoke/`.

## Interpretation Boundary

Three frames from one video cannot support a model or MARSP conclusion. The
minor metric differences above only demonstrate that each code path executes
and produces distinct, internally consistent records. Formal findings require
the predeclared five-frame window and all 4,349 validation frames across 36
videos on a clean committed revision.
