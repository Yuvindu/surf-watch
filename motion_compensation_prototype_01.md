# Motion Compensation Prototype — Experiment 01

**Ticket:** SCRUM-54  
**Parent:** SCRUM-46 Motion Compensation Investigation  
**Assignee:** Alan Bello  
**Date:** 2026-04-18  
**Status:** Complete ✅

---

## 1. Objective

Prototype a frame-to-frame motion compensation pipeline for SurfWatch that:
- Estimates camera motion between consecutive frames
- Stabilises shaky video for downstream rip current detection
- Is structured as a reusable, importable module

---

## 2. Approach

### Why Partial Affine instead of Full Homography?

| Property | Partial Affine (4 DOF) | Full Homography (8 DOF) |
|---|---|---|
| Translation | ✅ | ✅ |
| Rotation | ✅ | ✅ |
| Uniform scale | ✅ | ✅ |
| Perspective warp | ❌ | ✅ |
| Risk of distortion | Low | High |
| Suitable for ocean footage | ✅ | ⚠️ |

Partial affine was chosen because ocean/beach footage is approximately planar — full homography can introduce unwanted perspective distortion on wide-angle drone shots.

### Pipeline

```
Input video
    │
    ▼
[1] ORB feature detection (1000 keypoints/frame)
    │
    ▼
[2] Brute-force matching + Lowe ratio test (0.72)
    │
    ▼
[3] Partial affine estimation via RANSAC (threshold 5px)
    │   → Fallback to identity if < 6 inliers
    ▼
[4] Cumulative trajectory accumulation
    │
    ▼
[5] Moving average smoothing (radius = 20 frames)
    │
    ▼
[6] Per-frame correction = smoothed − raw
    │
    ▼
[7] cv2.warpAffine with BORDER_REFLECT_101
    │
    ▼
Stabilised output video
```

---

## 3. Dataset

| Property | Value |
|---|---|
| Source | RipVIS dataset — Dumitriu et al., CVPRW 2023 |
| Video | RipVIS-016.mp4 |
| Resolution | 1920 × 1080 (processed at 640 × 360) |
| Frame rate | 30 fps |
| Total frames | 941 |
| Duration | ~31 seconds |

---

## 4. Results

| Metric | Value |
|---|---|
| Successful transforms | 940 / 940 |
| Failed transforms | 0 |
| Fallback (identity) used | 0 |
| **Success rate** | **100%** |
| Avg inliers per frame | 765.7 |
| Avg horizontal correction | 0.09 px |
| Avg vertical correction | 0.07 px |
| Processing time | 63.3 s |

---

## 5. Module Structure

```
src/
  preprocessing/
    motion_compensation.py     ← reusable module (import this)
        MotionCompensator      ← main public class
        MotionCompensationConfig
        MotionCompensationResult
        FrameTransform

scripts/
  run_motion_compensation.py   ← CLI runner

docs/
  experiments/
    motion_compensation_prototype_01.md   ← this file

outputs/
  motion_compensation/
    *_stabilised.mp4           ← stabilised video output
    *_report.json              ← metrics report
```

---

## 6. How to Use

### As a module (import into other components):
```python
from src.preprocessing.motion_compensation import MotionCompensator

mc = MotionCompensator("RipVIS-016.mp4")
result = mc.run("outputs/motion_compensation/stabilised.mp4")
result.print_summary()
```

### With custom settings:
```python
from src.preprocessing.motion_compensation import (
    MotionCompensator, MotionCompensationConfig
)
config = MotionCompensationConfig(
    smoothing_radius = 30,
    max_features     = 1500,
    show_comparison  = False,
)
mc = MotionCompensator("video.mp4", config=config)
mc.run("outputs/motion_compensation/out.mp4")
```

### From the command line:
```bash
python scripts/run_motion_compensation.py --input RipVIS-016.mp4
python scripts/run_motion_compensation.py --input RipVIS-016.mp4 --smooth 30
python scripts/run_motion_compensation.py --input RipVIS-016.mp4 --no-comparison
```

---

## 7. Feasibility Observations

### ✅ What worked well
- Partial affine transform handled drone footage with no perspective distortion
- 100% success rate — zero fallbacks needed on the RipVIS test clip
- Moving average smoothing removed jitter while preserving intentional pans
- Module is cleanly importable with a well-defined public API

### ⚠️ Limitations
- Featureless regions (flat water, overcast sky) may reduce keypoint density
- Scene cuts would break continuity — a cut detector should be added
- Processing at full 1080p is slow — consider downscaling before processing

### 🔵 Recommendations for MARSP Integration
1. Pass stabilised frames directly into the segmentation pipeline
2. Consider partial affine as a preprocessing step before any model inference
3. Add scene-cut detection before this module in the pipeline

---

## 8. References

Dumitriu, A., Tatui, F., Miron, F., Ionescu, R. T., & Timofte, R. (2023).
Rip Current Segmentation: A Novel Benchmark and YOLOv8 Baseline Results.
*CVPR Workshops*, pp. 1261–1271.
