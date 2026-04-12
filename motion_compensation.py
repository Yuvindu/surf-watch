"""
motion_compensation.py
───────────────────────
SCRUM-47 · SurfWatch – Frame-to-frame motion compensation prototype
Approach : ORB feature detection → brute-force matching → homography estimation
           → inverse-warp frame alignment

Usage:
    python motion_compensation.py --input test_surf_video.mp4
    python motion_compensation.py --input my_surf_clip.mp4 --output stabilised.mp4

Dependencies:
    pip install opencv-python numpy
"""

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────
ORB_N_FEATURES = 2000        # Number of ORB keypoints to detect per frame
MATCH_RATIO_THRESH = 0.75    # Lowe's ratio test threshold
MIN_GOOD_MATCHES = 10        # Minimum matches needed to compute homography
RANSAC_REPROJ_THRESH = 5.0   # RANSAC reprojection threshold (pixels)
SMOOTHING_RADIUS = 15        # Frame radius for trajectory smoothing


# ─────────────────────────────────────────────────────────────────────────────
# Helper utilities
# ─────────────────────────────────────────────────────────────────────────────
def moving_average(curve: np.ndarray, radius: int) -> np.ndarray:
    """Smooth a 1-D trajectory with a uniform sliding window."""
    kernel_size = 2 * radius + 1
    kernel = np.ones(kernel_size) / kernel_size
    # Pad edges with the boundary value to avoid artefacts
    padded = np.pad(curve, (radius, radius), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def smooth_trajectory(trajectory: np.ndarray, radius: int) -> np.ndarray:
    """Apply moving-average smoothing to each column of a trajectory matrix."""
    smoothed = np.copy(trajectory)
    for col in range(trajectory.shape[1]):
        smoothed[:, col] = moving_average(trajectory[:, col], radius)
    return smoothed


# ─────────────────────────────────────────────────────────────────────────────
# Core motion estimation
# ─────────────────────────────────────────────────────────────────────────────
def estimate_frame_transform(
    orb: cv2.ORB,
    matcher: cv2.BFMatcher,
    prev_gray: np.ndarray,
    curr_gray: np.ndarray,
) -> tuple[np.ndarray | None, dict]:
    """
    Estimate the 2-D affine/homography transform from prev → curr using ORB.

    Returns
    -------
    transform : 3×3 homography matrix, or None if estimation failed
    info      : diagnostics dict (keypoints, matches, inliers)
    """
    kp1, des1 = orb.detectAndCompute(prev_gray, None)
    kp2, des2 = orb.detectAndCompute(curr_gray, None)

    info = {"kp_prev": len(kp1), "kp_curr": len(kp2),
            "good_matches": 0, "inliers": 0, "success": False}

    if des1 is None or des2 is None or len(kp1) < MIN_GOOD_MATCHES:
        return None, info

    # Ratio test (Lowe 2004)
    raw_matches = matcher.knnMatch(des1, des2, k=2)
    good = [m for m, n in raw_matches if m.distance < MATCH_RATIO_THRESH * n.distance]
    info["good_matches"] = len(good)

    if len(good) < MIN_GOOD_MATCHES:
        return None, info

    pts1 = np.float32([kp1[m.queryIdx].pt for m in good])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in good])

    H, mask = cv2.findHomography(pts2, pts1, cv2.RANSAC, RANSAC_REPROJ_THRESH)
    if H is None:
        return None, info

    inliers = int(mask.sum()) if mask is not None else 0
    info["inliers"] = inliers
    info["success"] = inliers >= MIN_GOOD_MATCHES
    return H, info


# ─────────────────────────────────────────────────────────────────────────────
# Cumulative trajectory accumulation + smoothing
# ─────────────────────────────────────────────────────────────────────────────
def accumulate_trajectory(transforms: list[np.ndarray | None]) -> np.ndarray:
    """Convert frame-to-frame transforms into a cumulative camera trajectory."""
    n = len(transforms)
    traj = np.zeros((n, 3))   # [tx, ty, angle]
    cumulative = np.eye(3, dtype=np.float64)

    for i, H in enumerate(transforms):
        if H is not None:
            cumulative = cumulative @ H
        # Decompose cumulative homography → tx, ty, rotation
        traj[i, 0] = cumulative[0, 2]                              # tx
        traj[i, 1] = cumulative[1, 2]                              # ty
        traj[i, 2] = np.arctan2(cumulative[1, 0], cumulative[0, 0])  # angle
    return traj


# ─────────────────────────────────────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────────────────────────────────────
def run(input_path: str, output_path: str) -> dict:
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"[INFO] Input  : {input_path}  ({width}×{height} @ {fps:.1f} fps, {total} frames)")
    print(f"[INFO] Output : {output_path}")
    print(f"[INFO] Approach: ORB feature matching + Homography + trajectory smoothing")
    print()

    orb = cv2.ORB_create(nfeatures=ORB_N_FEATURES)
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)

    # ── Pass 1: collect frame transforms ──────────────────────────────────────
    print("[PASS 1/2] Estimating inter-frame transforms …")
    frames = []
    transforms = []
    diag_list = []
    t0 = time.time()

    ret, frame = cap.read()
    if not ret:
        raise RuntimeError("Video has no readable frames.")
    frames.append(frame)
    prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    failed = 0
    for i in range(1, total):
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
        curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        H, info = estimate_frame_transform(orb, matcher, prev_gray, curr_gray)
        transforms.append(H)
        diag_list.append(info)
        if not info["success"]:
            failed += 1

        prev_gray = curr_gray
        if (i % 30) == 0:
            print(f"  … frame {i}/{total-1}  "
                  f"matches={info['good_matches']}  inliers={info['inliers']}")

    cap.release()
    elapsed_pass1 = time.time() - t0
    n_frames = len(frames)
    print(f"  Done. {n_frames} frames read, {failed} transforms failed/skipped "
          f"({elapsed_pass1:.1f}s)\n")

    # ── Smooth trajectory ─────────────────────────────────────────────────────
    print("[SMOOTH ] Computing & smoothing cumulative trajectory …")
    trajectory = accumulate_trajectory(transforms)
    smoothed = smooth_trajectory(trajectory, SMOOTHING_RADIUS)
    correction = smoothed - trajectory   # per-frame correction vector

    # ── Pass 2: write stabilised frames ───────────────────────────────────────
    print("[PASS 2/2] Warping frames …")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    t0 = time.time()
    dx_vals, dy_vals, da_vals = [], [], []

    for i, frame in enumerate(frames):
        dx = correction[i, 0] if i < len(correction) else 0
        dy = correction[i, 1] if i < len(correction) else 0
        da = correction[i, 2] if i < len(correction) else 0
        dx_vals.append(abs(dx))
        dy_vals.append(abs(dy))
        da_vals.append(abs(da))

        # Build correction transform
        cos_a, sin_a = np.cos(da), np.sin(da)
        M = np.array([
            [cos_a, -sin_a, dx],
            [sin_a,  cos_a, dy],
        ], dtype=np.float64)

        stabilised = cv2.warpAffine(frame, M, (width, height),
                                    flags=cv2.INTER_LINEAR,
                                    borderMode=cv2.BORDER_REFLECT_101)

        # ── Side-by-side comparison overlay ──────────────────────────────────
        comparison = np.hstack([frame, stabilised])
        comparison = cv2.resize(comparison, (width, height))

        label_orig = "ORIGINAL"
        label_stab = "STABILISED"
        cv2.putText(comparison, label_orig, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.putText(comparison, label_stab, (width // 2 + 10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(comparison, f"Frame {i+1}/{n_frames}", (10, height - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        writer.write(comparison)

    writer.release()
    elapsed_pass2 = time.time() - t0
    print(f"  Done. ({elapsed_pass2:.1f}s)\n")

    # ── Compile results ────────────────────────────────────────────────────────
    avg_matches = np.mean([d["good_matches"] for d in diag_list]) if diag_list else 0
    avg_inliers = np.mean([d["inliers"] for d in diag_list]) if diag_list else 0
    success_rate = np.mean([d["success"] for d in diag_list]) * 100 if diag_list else 0

    results = {
        "input": input_path,
        "output": output_path,
        "resolution": f"{width}×{height}",
        "fps": fps,
        "total_frames": n_frames,
        "approach": "ORB feature matching + RANSAC homography + trajectory smoothing",
        "orb_n_features": ORB_N_FEATURES,
        "smoothing_radius_frames": SMOOTHING_RADIUS,
        "avg_good_matches_per_frame": round(float(avg_matches), 1),
        "avg_inliers_per_frame": round(float(avg_inliers), 1),
        "transform_success_rate_pct": round(float(success_rate), 1),
        "failed_transforms": failed,
        "mean_correction_tx_px": round(float(np.mean(dx_vals)), 2),
        "mean_correction_ty_px": round(float(np.mean(dy_vals)), 2),
        "mean_correction_angle_deg": round(float(np.degrees(np.mean(da_vals))), 3),
        "processing_time_s": round(elapsed_pass1 + elapsed_pass2, 1),
    }
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="SCRUM-47 · SurfWatch motion compensation prototype"
    )
    parser.add_argument("--input",  default="test_surf_video.mp4",
                        help="Path to input video (default: test_surf_video.mp4)")
    parser.add_argument("--output", default="stabilised_output.mp4",
                        help="Path for stabilised output video")
    args = parser.parse_args()

    results = run(args.input, args.output)

    print("=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    for k, v in results.items():
        print(f"  {k:<35} {v}")
    print()

    results_path = "results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[OK] Results saved → {results_path}")
    print(f"[OK] Stabilised video saved → {args.output}")
