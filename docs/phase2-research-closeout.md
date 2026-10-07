# SurfWatch Phase 2 Research Closeout

## Status as of 7 October 2026

The model and MARSP research experiments planned for Phase 2 are implemented
and evaluated on the full labelled RipVIS validation split: 4,349 frames from
36 videos. The frontend experiment-comparison and provenance extension remains
separate work. This is a validation-based research closeout, not an independent
test-set or deployment-performance claim.

| Planned study | Status | Evidence |
|---|---|---|
| Model-agnostic segmentation and backend comparison | Complete for SegFormer and U-Net; the existing upload workflow can select either adapter | [Adapter contract](segmentation-model-interface.md), [backend flow](frontend-backend-comparison-flow.md) |
| Train and evaluate U-Net ResNet34 | Complete | [Training record](experiments/unet_resnet34_cloud_run_01.md), [matched evaluation](experiments/segformer_reproduction_02.md) |
| Compare U-Net with SegFormer | Complete | [Corrected matched evaluation](experiments/segformer_reproduction_02.md) on identical source frames |
| Implement and evaluate multi-model ensemble fusion | Complete | [Formal fusion study](experiments/ensemble_fusion_01.md) |
| Conduct model and MARSP ablations | Complete | [A0-A6 study](experiments/marsp_ablation_01.md) |
| Evaluate confidence and uncertainty | Complete as descriptive validation analysis; no fitted calibration | [Formal uncertainty study](experiments/model_uncertainty_01.md) |
| Improve temporal aggregation and small-component handling | Compared fixed and motion-adaptive smoothing and cleanup; fixed smoothing is the validation-supported reference, cleanup remains optional | [A0-A6 study](experiments/marsp_ablation_01.md), [decision log](decision-log.md) |
| Broaden held-out video evaluation | Complete across all 36 labelled validation videos; independent quantitative assessment remains unavailable | [Dataset and split strategy](split-strategy.md), [study limits](experiments/marsp_ablation_01.md) |
| Consolidate runtime evidence | Available stage timings summarized; paired end-to-end throughput is not established | [Runtime evidence](phase2-runtime-evidence.md) |
| Extend frontend experiment comparison and provenance | Outstanding | [Current frontend contract](frontend-backend-comparison-flow.md) |

U-Net's selected epoch-10 checkpoint reached validation mean IoU 0.7407 at
training resolution. The reproduced SegFormer epoch-4 checkpoint reached
0.7565. These training metrics are not directly comparable with the
source-resolution foreground scores below.

## Main Findings

All figures below are source-resolution, foreground-specific validation metrics
unless stated otherwise. The invalid 17 September SegFormer checkpoint result
is excluded; the corrected 23 September reproduction is the comparison basis.

| Configuration | Foreground IoU | Dice | Precision | Recall | Consecutive-mask IoU |
|---|---:|---:|---:|---:|---:|
| SegFormer-B0, frame-level | 0.5027 | 0.6690 | 0.7698 | 0.5916 | 0.7867 |
| U-Net ResNet34, frame-level | 0.4201 | 0.5916 | 0.8633 | 0.4500 | 0.7956 |
| Equal-weight ensemble, frame-level | 0.4701 | 0.6395 | **0.8912** | 0.4987 | 0.8555 |
| SegFormer with fixed smoothing | **0.5063** | **0.6723** | 0.7805 | 0.5904 | **0.8859** |
| SegFormer with motion-adaptive weighting | 0.5043 | 0.6704 | 0.7738 | 0.5915 | 0.8350 |

Fixed smoothing raised consecutive-mask IoU by 0.0992 over raw SegFormer and
foreground IoU by 0.0037, but recall decreased slightly. Motion-adaptive
weighting helped less. Cleanup reduced mean small predicted components from
0.5014 to 0.0002 per frame with essentially no foreground-IoU improvement.
These are paired validation observations, not evidence that every beach video
will benefit.

The 50/50 ensemble had the best pooled probability scores: all-pixel Brier
0.02581, NLL 0.12287, and foreground ECE 0.01787, versus SegFormer's
0.02728, 0.13667, and 0.01864. Its fixed-threshold foreground IoU was lower
than SegFormer's (0.4701 versus 0.5027). Probability quality and mask quality
therefore support different conclusions; no ensemble or fitted calibration is
promoted to the default path.

The A0-A6 evaluator also reports stage-accumulated runtime estimates on the
same 4,349-frame run: 144.7 seconds for raw SegFormer (A0), 231.5 for fixed
smoothing (A3), and 348.3 for motion-adaptive weighting (A4). These reuse
cached inference and omit shared video I/O, so they are useful for identifying
relative processing cost but are **not** independently measured end-to-end
latencies or deployable FPS figures. See the [runtime summary](phase2-runtime-evidence.md).

## Decision and Remaining Boundary

Retain SegFormer as the primary segmentation reference, with fixed temporal
smoothing as the current validation-supported temporal setting. Keep the
equal-weight ensemble as a comparator and small-component cleanup as an
optional fragmentation control. Do not claim that motion-adaptive weighting,
ensemble fusion, or calibration has independently generalised to unseen
beaches. The [decision log](decision-log.md) records the supporting choices.

The official validation split also selected the model checkpoints. Public
RipVIS test videos have no released masks in the local dataset, so a new
quantitative test score would require additional labelled, video-disjoint
material. The available public-test video can support qualitative inspection,
not a numerical generalisation claim. Motion-estimation fallback on
`RipVIS-NR-020` is documented in the ablation record; a deeper root-cause
study is deferred because it would not change the current supported setting.

The frontend extension and final report/presentation packaging remain to be
completed. A same-environment end-to-end baseline-versus-MARSP timing
benchmark also remains unavailable; the stage-time summary must not be
presented as one.
