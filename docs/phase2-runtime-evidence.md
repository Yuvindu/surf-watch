# Phase 2 Runtime Evidence

## Comparable Scope

The full A0-A6 ablation evaluated 4,349 labelled frames from 36 RipVIS
validation videos on one Windows RTX 5050 (8 GB VRAM) system at clean revision
`342d37302f649933dc5000bda9951ed982c83b30`. The evaluator cached model
probability maps and accumulated model inference, motion estimation, and
postprocessing times for each configuration. Its JSON and CSV files are
versioned under `outputs/marsp_ablation/marsp-ablation-val-full-20261001/`;
the [experiment record](experiments/marsp_ablation_01.md) covers the input,
parameters, provenance, and missing environment details.

| Configuration | Cached inference (s) | Motion estimation (s) | Postprocessing (s) | Stage sum (s) |
|---|---:|---:|---:|---:|
| A0: SegFormer, frame-level | 87.4 | 0.0 | 57.3 | 144.7 |
| A3: SegFormer + fixed smoothing | 87.4 | 0.0 | 144.1 | 231.5 |
| A4: SegFormer + motion-adaptive weighting | 87.4 | 179.3 | 81.6 | 348.3 |
| A5: SegFormer + cleanup | 87.4 | 0.0 | 55.8 | 143.2 |

These are **stage-accumulated estimates**, not four separately executed
end-to-end workflows. They exclude shared video decoding, some data handling,
and output rendering; small differences such as A0 versus A5 are not evidence
of a real-world speed gain. The large motion-estimation term makes adaptive
weighting more expensive in this evaluation, while fixed smoothing improves
temporal stability without that term.

The [uncertainty evaluation](experiments/model_uncertainty_01.md) took 2,323.7
seconds on a different macOS CPU environment. That elapsed time cannot be
compared directly with this RTX 5050 ablation to calculate MARSP overhead.

## Outstanding Benchmark Boundary

The earlier profiling subtasks do not leave a consolidated, same-input,
same-environment end-to-end baseline-versus-MARSP timing artifact in this
repository. A defensible deployment-throughput or percentage-overhead claim
therefore remains unavailable. The final report should use the stage sums only
with the qualifications above, or run a fresh paired benchmark if such a claim
is essential. This limitation does not invalidate the paired semantic and
temporal ablation results.
