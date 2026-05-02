# MARSP Cloud Evaluation 01

## Objective
Evaluate the integrated MARSP pipeline on a broader subset of RipVIS videos and assess whether motion-adaptive temporal aggregation produces measurable temporal stability gains in segmentation outputs.

## Pipeline configuration
- Pipeline: motion compensation → SegFormer frame-level inference → motion-adaptive temporal aggregation
- Segmentation checkpoint: `checkpoints/best_model.pt`
- Aggregation window size: `5`
- Mask threshold: `0.5`
- Stability comparison: raw stabilised mask sequence vs temporally aggregated stabilised mask sequence
- Scope of metric comparison: temporal aggregation effect after motion compensation, not original-vs-stabilised comparison

## Videos evaluated
- `RipVIS-006`
- `RipVIS-016`
- `RipVIS-029`
- `RipVIS-032`
- `RipVIS-041`
- `RipVIS-051`

## Evaluation method
Visual inspection alone made the compensation and aggregation effects difficult to judge consistently, so sequence-level temporal stability metrics were used to compare the raw and aggregated mask videos. These metrics were computed on the stabilised branch of the pipeline, meaning they assess whether temporal aggregation improves stability after motion compensation has already been applied.

The following metrics were computed for each video:
- mean consecutive-frame IoU
- mean consecutive-frame Dice
- mean frame-to-frame pixel change rate
- standard deviation of mask area
- mean absolute frame-to-frame mask area change
- mean small connected-component count

Interpretation:
- **Higher** consecutive-frame IoU and Dice indicate better temporal consistency
- **Lower** pixel change rate indicates less flicker
- **Lower** mask-area variation indicates smoother temporal behaviour
- **Lower** mean absolute area change indicates fewer abrupt frame-to-frame changes

## Per-video results

| Video | Δ Mean IoU | Δ Mean Dice | Δ Pixel Change Rate | Δ Area Std (px) | Δ Mean Abs Area Change (px) | Δ Small Blob Count | Summary |
|---|---:|---:|---:|---:|---:|---:|---|
| RipVIS-006 | +0.01299 | +0.00682 | -0.000983 | -37.40 | -333.83 | +0.00810 | Modest but consistent stability improvement |
| RipVIS-016 | +0.01363 | +0.00704 | -0.000974 | -112.46 | -1120.46 | +0.04038 | Modest but clearly measurable stability improvement |
| RipVIS-029 | +0.00660 | +0.00340 | -0.000458 | -82.92 | -388.22 | +0.04887 | Modest but consistent improvement |
| RipVIS-032 | +0.00843 | +0.00433 | -0.000263 | -43.47 | -251.36 | +0.01108 | Modest improvement |
| RipVIS-041 | +0.00632 | +0.00320 | -0.001003 | -70.66 | -883.91 | +0.02756 | Modest but clearly measurable reduction in flicker and abrupt change |
| RipVIS-051 | +0.00591 | +0.00308 | -0.000490 | -31.44 | -182.70 | +0.02228 | Modest improvement despite subtle visual difference |

## Aggregate findings
Across all 6 evaluated videos:

- Consecutive-frame IoU improved in **6/6** cases
- Consecutive-frame Dice improved in **6/6** cases
- Pixel change rate decreased in **6/6** cases
- Mask area variation decreased in **6/6** cases
- Mean absolute mask area change decreased in **6/6** cases
- Small connected-component count increased slightly in **6/6** cases
    
## Interpretation
The results show that motion-adaptive temporal aggregation consistently improved temporal stability across the evaluated MARSP subset, but the gains were modest in absolute magnitude rather than dramatic.

The clearest repeated pattern was:
- smoother frame-to-frame mask behaviour
- less flicker
- fewer abrupt changes in total predicted rip-current area

This indicates that the aggregation stage is providing useful incremental stabilisation even when the difference is subtle by visual inspection alone.

The increase in small connected-component count suggests that the current aggregation step may preserve or introduce a small number of tiny isolated regions in some frames. However, these increases were small in absolute terms and did not outweigh the broader gains in stability.

## Conclusion
The integrated MARSP pipeline produced consistent temporal stability improvements across all evaluated RipVIS videos. Although the visual effect was often subtle, quantitative sequence-level analysis showed that motion-adaptive temporal aggregation made segmentation outputs more stable over time. The improvements were modest rather than dramatic, but they were repeatable across every evaluated video.

This supports the conclusion that MARSP is a promising extension to the SurfWatch baseline pipeline, particularly for reducing flicker and smoothing frame-to-frame prediction behaviour, while also showing that the current gains should be described as incremental.

## Limitations
- Evaluation was based on a subset of completed videos rather than the full RipVIS dataset
- The analysis focused on temporal stability rather than final segmentation accuracy against frame-level ground truth
- Qualitative visual differences remained subtle in many cases, which is why metric-based analysis was necessary

## Recommendation
- Keep motion-adaptive temporal aggregation in the MARSP pipeline
- Report MARSP as a **promising, technically validated, and incrementally beneficial stability-enhancing extension**
- If further work is possible, add a lightweight post-aggregation cleanup step for tiny connected components
- Future evaluation can expand to a larger subset or add direct accuracy-based comparison where practical
