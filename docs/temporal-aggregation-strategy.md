# Temporal Aggregation Strategy for SurfWatch

## Objective
To define the initial short-window temporal aggregation strategy for SurfWatch segmentation outputs in the MARSP pipeline.

## Motivation
Frame-level segmentation predictions are currently generated independently. This can lead to flicker, unstable rip-current boundaries, and inconsistent region shape across adjacent frames. Temporal aggregation is introduced to improve prediction stability over time.

## Selected initial method
The initial temporal aggregation method for SurfWatch is:

**Sliding-window probability averaging**

This method was selected over exponential moving average and majority voting because it preserves frame-level probability information while remaining simple and interpretable.

## Input format
The input to temporal aggregation is a time-ordered sequence of frame-level rip-current probability maps produced by the segmentation model before binary thresholding.

For the binary segmentation setting, each frame contributes:
- one rip-current probability map
- shape conceptually represented as `H x W`

## Output format
For each frame, temporal aggregation produces:
- a temporally smoothed rip-current probability map
- a binary rip-current mask after thresholding

## Aggregation window strategy
The initial MARSP design uses a centered window of **5 frames**:

- `t-2`
- `t-1`
- `t`
- `t+1`
- `t+2`

For each target frame `t`, the aggregated probability map is computed as the mean of the available probability maps within the window.

## Aggregation rule
For each pixel location:

`aggregated_prob_t = mean(probability maps in the local temporal window)`

This reduces frame-to-frame fluctuations and improves short-term consistency in the predicted rip-current region.

## Thresholding rule
The aggregated probability map is converted into a binary segmentation mask using a threshold of **0.5**:

- probability `>= 0.5` → rip current
- probability `< 0.5` → background

This threshold is sufficient for the initial MARSP implementation and may be tuned later if needed.

## Edge handling
At the beginning and end of a video, a full centered window may not be available.

Initial handling:
- use only the valid frames available within the sequence
- do not pad with synthetic or duplicated frames in the first version

Examples:
- frame `0` uses frames `[0, 1, 2]`
- frame `1` uses frames `[0, 1, 2, 3]`

## Edge cases
### Boundary frames
Use reduced valid windows when full context is unavailable.

### Noisy isolated predictions
Short-window averaging should suppress one-frame prediction spikes and drops.

### Sudden scene changes
This first design assumes continuous footage. Scene-cut-aware aggregation is left for later refinement.

### Upstream motion compensation instability
Temporal aggregation operates on available segmentation outputs and does not require every frame to have successful motion compensation.

### Over-smoothing risk
A short window of 5 frames is used to reduce flicker without excessively suppressing genuine local changes.

## Rationale
Sliding-window probability averaging is preferred for the first MARSP version because:
- it preserves confidence information better than majority voting
- it is easier to reason about than more advanced fusion strategies
- it directly targets the flicker and instability observed in frame-level segmentation outputs

## Future alternatives
The following methods may be evaluated later:
- exponential moving average
- confidence-weighted temporal fusion
- majority voting over binary masks

## Initial implementation note
The first MARSP implementation should apply temporal aggregation after frame-level segmentation inference and before final binary mask export or downstream video result packaging.