from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np


@dataclass
class MotionCompensationConfig:
    orb_n_features: int = 2000
    match_ratio_thresh: float = 0.75
    min_good_matches: int = 10
    ransac_reproj_thresh: float = 5.0
    smoothing_radius: int = 15
    border_mode: int = cv2.BORDER_REFLECT_101


def moving_average(curve: np.ndarray, radius: int) -> np.ndarray:
    kernel_size = 2 * radius + 1
    kernel = np.ones(kernel_size, dtype=np.float64) / kernel_size
    padded = np.pad(curve, (radius, radius), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def smooth_trajectory(trajectory: np.ndarray, radius: int) -> np.ndarray:
    smoothed = np.copy(trajectory)
    for col in range(trajectory.shape[1]):
        smoothed[:, col] = moving_average(trajectory[:, col], radius)
    return smoothed


def affine_to_params(M: np.ndarray) -> np.ndarray:
    dx = float(M[0, 2])
    dy = float(M[1, 2])
    da = float(np.arctan2(M[1, 0], M[0, 0]))
    return np.array([dx, dy, da], dtype=np.float64)


def params_to_affine(dx: float, dy: float, da: float) -> np.ndarray:
    cos_a = np.cos(da)
    sin_a = np.sin(da)
    return np.array(
        [
            [cos_a, -sin_a, dx],
            [sin_a,  cos_a, dy],
        ],
        dtype=np.float64,
    )


def estimate_partial_affine_transform(
    orb: cv2.ORB,
    matcher: cv2.BFMatcher,
    prev_gray: np.ndarray,
    curr_gray: np.ndarray,
    config: MotionCompensationConfig,
) -> tuple[Optional[np.ndarray], dict]:
    kp1, des1 = orb.detectAndCompute(prev_gray, None)
    kp2, des2 = orb.detectAndCompute(curr_gray, None)

    info = {
        "kp_prev": len(kp1),
        "kp_curr": len(kp2),
        "good_matches": 0,
        "inliers": 0,
        "success": False,
        "fallback": False,
    }

    if des1 is None or des2 is None or len(kp1) < config.min_good_matches:
        info["fallback"] = True
        return None, info

    raw_matches = matcher.knnMatch(des1, des2, k=2)
    good = [m for m, n in raw_matches if m.distance < config.match_ratio_thresh * n.distance]
    info["good_matches"] = len(good)

    if len(good) < config.min_good_matches:
        info["fallback"] = True
        return None, info

    pts1 = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    pts2 = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    M, inliers = cv2.estimateAffinePartial2D(
        pts2,
        pts1,
        method=cv2.RANSAC,
        ransacReprojThreshold=config.ransac_reproj_thresh,
    )

    if M is None:
        info["fallback"] = True
        return None, info

    inlier_count = int(inliers.sum()) if inliers is not None else 0
    info["inliers"] = inlier_count
    info["success"] = inlier_count >= config.min_good_matches

    if not info["success"]:
        info["fallback"] = True
        return None, info

    return M, info


def accumulate_trajectory(transforms: list[Optional[np.ndarray]]) -> np.ndarray:
    trajectory = np.zeros((len(transforms), 3), dtype=np.float64)
    cumulative = np.array([0.0, 0.0, 0.0], dtype=np.float64)

    for i, M in enumerate(transforms):
        if M is not None:
            cumulative += affine_to_params(M)
        trajectory[i] = cumulative

    return trajectory


def stabilise_video(
    input_path: str,
    output_path: str,
    comparison_output_path: Optional[str],
    config: MotionCompensationConfig,
) -> dict:
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    orb = cv2.ORB_create(nfeatures=config.orb_n_features)
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)

    frames: list[np.ndarray] = []
    transforms: list[Optional[np.ndarray]] = []
    diagnostics: list[dict] = []

    ret, first = cap.read()
    if not ret:
        raise RuntimeError("Video has no readable frames.")

    frames.append(first)
    prev_gray = cv2.cvtColor(first, cv2.COLOR_BGR2GRAY)

    for i in range(1, total):
        ret, frame = cap.read()
        if not ret:
            break

        frames.append(frame)
        curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        M, info = estimate_partial_affine_transform(
            orb, matcher, prev_gray, curr_gray, config
        )

        if M is not None:
            dx, dy, da = affine_to_params(M)
            translation_mag = float(np.sqrt(dx ** 2 + dy ** 2))
            rotation_deg = float(np.degrees(da))
        else:
            dx, dy, da = 0.0, 0.0, 0.0
            translation_mag = 0.0
            rotation_deg = 0.0

        frame_metadata = {
            "frame_index": i,
            "kp_prev": info["kp_prev"],
            "kp_curr": info["kp_curr"],
            "good_matches": info["good_matches"],
            "inliers": info["inliers"],
            "success": info["success"],
            "fallback": info["fallback"],
            "translation_x_px": float(dx),
            "translation_y_px": float(dy),
            "translation_magnitude_px": float(translation_mag),
            "rotation_deg": float(rotation_deg),
        }

        transforms.append(M)
        diagnostics.append(frame_metadata)
        prev_gray = curr_gray

        if i % 100 == 0:
            print(
                f"Processed frame {i}/{total} | "
                f"matches={info['good_matches']} inliers={info['inliers']} fallback={info['fallback']}"
            )

    cap.release()

    trajectory = accumulate_trajectory(transforms)
    smoothed = smooth_trajectory(trajectory, config.smoothing_radius)
    correction = smoothed - trajectory

    writer = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    comparison_writer = None
    if comparison_output_path is not None:
        comparison_writer = cv2.VideoWriter(
            comparison_output_path,
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width * 2, height),
        )

    fallback_count = 0
    dx_vals, dy_vals, da_vals = [], [], []

    for i, frame in enumerate(frames):
        if i == 0:
            stabilized = frame.copy()
        else:
            dx, dy, da = correction[i - 1]
            dx_vals.append(abs(dx))
            dy_vals.append(abs(dy))
            da_vals.append(abs(da))

            M = params_to_affine(dx, dy, da)
            stabilized = cv2.warpAffine(
                frame,
                M,
                (width, height),
                flags=cv2.INTER_LINEAR,
                borderMode=config.border_mode,
            )

        writer.write(stabilized)

        if comparison_writer is not None:
            comparison = np.hstack([frame, stabilized])
            cv2.putText(comparison, "Original", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            cv2.putText(comparison, "Stabilised", (width + 10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            comparison_writer.write(comparison)

    writer.release()
    if comparison_writer is not None:
        comparison_writer.release()

    for d in diagnostics:
        if d["fallback"]:
            fallback_count += 1

    avg_matches = float(np.mean([d["good_matches"] for d in diagnostics])) if diagnostics else 0.0
    avg_inliers = float(np.mean([d["inliers"] for d in diagnostics])) if diagnostics else 0.0
    success_rate = float(np.mean([d["success"] for d in diagnostics]) * 100.0) if diagnostics else 0.0

    return {
        "input": input_path,
        "output": output_path,
        "comparison_output": comparison_output_path,
        "total_frames": len(frames),
        "fps": fps,
        "resolution": f"{width}x{height}",
        "method": "ORB + BFMatcher + RANSAC partial affine",
        "avg_good_matches_per_frame": round(avg_matches, 1),
        "avg_inliers_per_frame": round(avg_inliers, 1),
        "transform_success_rate_pct": round(success_rate, 1),
        "fallback_frames": fallback_count,
        "mean_correction_tx_px": round(float(np.mean(dx_vals)) if dx_vals else 0.0, 3),
        "mean_correction_ty_px": round(float(np.mean(dy_vals)) if dy_vals else 0.0, 3),
        "mean_correction_angle_deg": round(float(np.degrees(np.mean(da_vals))) if da_vals else 0.0, 3),
        "smoothing_radius": config.smoothing_radius,
        "per_frame_metadata": diagnostics,
    }