# Motion Compensation Prototype 01

## Objective
Implement and evaluate a feature-based partial affine motion compensation module for SurfWatch video preprocessing.

## Method
The prototype uses:
- ORB feature detection
- brute-force descriptor matching with Hamming distance
- Lowe’s ratio test for match filtering
- RANSAC-based partial affine transform estimation
- moving-average trajectory smoothing
- frame warping for stabilised output generation

The choice of partial affine estimation was preferred over full homography for the first integrated SurfWatch version because it is less likely to introduce unnecessary perspective distortion while still handling translation, rotation, and mild scale variation.

## Test video
- Input video: `../RipVIS/train/videos/RipVIS-051.mp4`
- Split: RipVIS train
- Resolution: `1280x720`
- Total frames: `404`
- Frame rate: `29.97 fps`

## Output files
- Stabilised video: `outputs/motion_compensation/RipVIS-051_stabilised.mp4`
- Comparison video: `outputs/motion_compensation/RipVIS-051_comparison.mp4`
- Results file: `outputs/motion_compensation/RipVIS-051_results.json`

## Quantitative results
- Method: `ORB + BFMatcher + RANSAC partial affine`
- Average good matches per frame: `1473.0`
- Average inliers per frame: `1460.1`
- Transform success rate: `100.0%`
- Fallback frames: `0`
- Mean horizontal correction: `2.95 px`
- Mean vertical correction: `0.907 px`
- Mean rotational correction: `0.028°`
- Smoothing radius: `15`

## Interpretation
The prototype successfully estimated partial affine motion transforms for all frame transitions in the selected sample video without fallback. The high average number of good matches and inliers suggests stable feature correspondence and robust transform estimation on this clip.

The correction magnitudes indicate that the method is compensating for relatively small frame-to-frame motion rather than severe camera shake. This suggests that the selected sample video contains mild but measurable motion suitable for feasibility testing.

## Qualitative observations
Visual comparison between the original and stabilised outputs did not show a dramatic difference by eye. However, this is not necessarily a negative result. For this clip, the motion appears to be relatively mild, so the compensation likely reduces subtle jitter and drift rather than producing an obvious stabilisation effect.

To assess downstream usefulness, baseline segmentation inference was run on both the original and stabilised videos. No major visible difference in prediction behaviour was observed in this first comparison, which suggests that the effect of motion compensation on SurfWatch segmentation stability may be limited or subtle for this particular sample.

## Feasibility observations
### What worked
- ORB matching remained stable across the full clip
- Partial affine estimation succeeded for all tested frame pairs
- Stabilised and side-by-side comparison outputs were generated successfully
- Fallback logic was not triggered on this sample

### Limitations
- Improvement was not visually dramatic on the selected video
- Downstream segmentation differences were not clearly obvious in the first comparison
- This result may reflect the relatively mild motion characteristics of the sample clip rather than the general value of motion compensation
- Additional evaluation is needed on harder clips with stronger camera movement, drift, or feature-poor conditions

## Conclusion
The motion compensation prototype is technically successful and feasible for integration into the SurfWatch preprocessing pipeline. However, on the tested sample video, the qualitative benefit appears subtle rather than dramatic. This suggests that motion compensation may still be useful, but its practical value should be assessed further on more challenging footage and in relation to temporal aggregation and prediction stability rather than visual appearance alone.

## Next steps
- Test the prototype on one or more more challenging RipVIS clips
- Compare segmentation stability on clips with stronger motion
- Evaluate whether motion compensation provides more benefit when combined with temporal aggregation in the MARSP pipeline