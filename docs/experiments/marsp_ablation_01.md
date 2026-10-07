# MARSP Ablation 01: Full RipVIS Validation Run

## Status

Completed on 3 October 2026 at clean revision
`342d37302f649933dc5000bda9951ed982c83b30`. The teammate first passed a
smoke run, then evaluated all seven predeclared configurations on the full
validation split. This record interprets the archived JSON and CSV outputs; it
does not report a new run.

## Objective and Protocol

Measure the contributions of model choice, equal-weight probability fusion,
fixed temporal smoothing, motion-adaptive weighting, and small-component
cleanup under the [predeclared matrix](../marsp-ablation.md). All configurations
use the same 4,349 labelled frames from 36 RipVIS validation videos. A0 is the
SegFormer frame-level reference; A3-A6 use SegFormer as their selected model
path. The evaluation uses source-frame masks and does **not** score geometric
camera stabilisation.

- Threshold: 0.5; temporal window: five labelled samples.
- Cleanup threshold: components smaller than 500 pixels.
- Temporal segments split at source-frame gaps greater than six; 3,951
  consecutive pairs were eligible and 362 were excluded.
- Equal-weight fusion: 0.5 SegFormer-B0 and 0.5 U-Net ResNet34.
- Validation annotation SHA-256:
  `98e3193fa751475ce44efe4340cfae8964a6ea3072dc25574c7ecd07eb5a7cf1`.
- SegFormer checkpoint SHA-256:
  `3fca0f0fd315f306e07d33b21a1641b223a1e5c16a41abe7c121679cd89f7371`.
- U-Net checkpoint SHA-256:
  `b3a0be0a536fb6aa0e01b157d5e9f521e041a100629749f7c070d99bc4d8917b`.

The teammate reported Windows, an RTX 5050 with 8 GB VRAM, 16 GB system RAM,
and Python 3.12. Semantic masks were regenerated with
`convert_ripvis_to_semantic.py`. Only the evaluator's dependencies were
installed at pinned versions, rather than the entire `requirements.txt`.
The console log and exact installed package/CUDA versions were not included in
the archived result directory; they should be preserved for any rerun.

## Results

Semantic metrics below are **micro** foreground scores accumulated across all
source-resolution pixels. Consecutive IoU and small-component count are
**macro** means over videos; only 24 videos have eligible consecutive-mask
comparisons. Values are rounded to four decimals.

| ID | Configuration | Foreground IoU | Dice | Precision | Recall | Consecutive IoU | Small components/frame |
|---|---|---:|---:|---:|---:|---:|---:|
| A0 | SegFormer, no temporal processing | 0.5027 | 0.6690 | 0.7698 | 0.5916 | 0.7867 | 0.5014 |
| A1 | U-Net ResNet34, no temporal processing | 0.4201 | 0.5916 | 0.8633 | 0.4500 | 0.7956 | 0.2342 |
| A2 | Equal-weight ensemble, no temporal processing | 0.4701 | 0.6395 | 0.8912 | 0.4987 | 0.8555 | 0.1581 |
| A3 | SegFormer + fixed smoothing | **0.5063** | **0.6723** | 0.7805 | 0.5904 | **0.8859** | 0.4083 |
| A4 | SegFormer + motion-adaptive weighting | 0.5043 | 0.6704 | 0.7738 | 0.5915 | 0.8350 | 0.4531 |
| A5 | SegFormer + component cleanup | 0.5027 | 0.6690 | 0.7700 | 0.5915 | 0.7980 | 0.0002 |
| A6 | SegFormer + adaptive weighting and cleanup | 0.5043 | 0.6704 | 0.7740 | 0.5913 | 0.8445 | 0.0001 |

A3 improved foreground IoU by **0.0037** and mean consecutive-mask IoU by
**0.0992** over A0. Its 1,397 empty predictions were 92 more than A0's 1,305;
the small recall decrease (0.5916 to 0.5904) matters when choosing a setting
for detection. Per-video foreground IoU improved on 12 videos, fell on six,
tied on 16, and was undefined on two. Thus the aggregate gain was not
universal.

A4 and A6 each improved foreground IoU by about **0.0016** over A0, less than
fixed smoothing. A5 almost eliminated small predicted components (0.5014 to
0.0002 per frame) while changing foreground IoU by only +0.00001 at the
dataset level. This supports cleanup as a fragmentation control, not an
accuracy improvement. The A2 ensemble again had the highest precision but
lower foreground IoU than SegFormer, consistent with the
[fusion experiment](ensemble_fusion_01.md).

## Motion and Runtime Evidence

`RipVIS-NR-020` had 381 motion-estimation fallbacks among 428 eligible pairs.
Across the full validation split there were 387 fallbacks, so this video
accounts for 98.4% of them. All 429 labelled frames in this video have empty
foreground ground truth. Its outcomes distinguish the two temporal methods:

| `RipVIS-NR-020` configuration | False-positive pixels | Empty predictions | Consecutive IoU |
|---|---:|---:|---:|
| A0 raw SegFormer | 1,972,928 | 291 | 0.6397 |
| A3 fixed smoothing | 898,736 | 346 | 0.8570 |
| A4 motion-adaptive weighting | 1,958,057 | 292 | 0.6445 |

The implementation assigns a motion score of 1.0 to each failed estimate;
scores at or above 0.75 set the adaptive smoothing radius to zero. Thus the
381 fallback frames receive their raw probability maps in A4, rather than
temporally averaged maps. This directly explains why A4 performs much like A0
on those frames, although the archived summary does not retain per-pair match
diagnostics needed to explain *why* feature matching failed. The video has
zero ground-truth positives, so its foreground IoU is 0.0 for A0, A3, and A4;
the false-positive count is more informative here.

This one video does not account for the entire dataset-level difference.
Excluding it, A0/A3/A4 foreground IoU is respectively 0.5042/0.5070/0.5058,
and their macro consecutive IoU over the remaining 23 eligible videos is
0.7931/0.8872/0.8433. These are post-hoc sensitivity calculations, not a new
predeclared evaluation or an independent test set. The exclusion recomputes
micro IoU from summed per-video true positives, false positives, and false
negatives; macro consecutive IoU is the mean of defined per-video values. No
algorithm change should be selected solely from this video.

The JSON reports estimated configuration runtimes from cached model inference
plus the stages each variant requires. They are **not** separately measured
end-to-end execution times for seven independent runs. In particular, do not
interpret small differences between A0 and A5 as a speed benefit of cleanup.

## Artifacts and Reproduction Boundary

The versioned formal output directory is
`outputs/marsp_ablation/marsp-ablation-val-full-20261001/`.

| Artifact | SHA-256 |
|---|---|
| `marsp_ablation_summary.json` | `58e27754ffb99fc68a75ae3fcb84467dd3e954420853843c4bd0b1841371680c` |
| `marsp_ablation_frame_metrics.csv` | `5418732b9d1bbd508846e4163f014a903f4874887d3c487da4d80d293f0a4bd5` |
| `marsp_ablation_video_metrics.csv` | `c4954dbf539d507ed23f8a341ff7082eb915828884ac46faf4b3f751a0ae6fbb` |

The run manifest records code-file hashes, checkpoint identities, annotation
identity, selected video IDs, parameters, and a clean Git revision. Generated
masks were recreated for the run; their individual hashes are not recorded in
the summary. The validation split was already used for checkpoint selection,
so this is a paired validation comparison, **not** an independent public-test
estimate or evidence of generalisation to new beaches.

## Decision and Follow-Up

Use A3 fixed smoothing as the current validation-supported temporal reference.
Retain motion-adaptive weighting as an experimental variant pending a focused
failure analysis. On a future run, retain per-pair keypoint, match, inlier,
fallback, and motion-score diagnostics to separate feature-matching failures
from high-motion decisions. Keep cleanup available when fragmentation is a
concern, but do not claim that it improves IoU. Before a final claim about
MARSP's broader benefit, evaluate on a genuinely separate held-out video set
or clearly label any further validation-only sensitivity study. Geometric
stabilisation remains outside this scoring protocol until coordinate alignment
is verified.
