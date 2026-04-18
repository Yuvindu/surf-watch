"""
motion_compensation.py
───────────────────────────────────────────────────────────────
src/preprocessing/motion_compensation.py

SCRUM-54 · SurfWatch – Motion Compensation Module

A reusable preprocessing component that:
  - Reads any supported video file
  - Estimates motion between consecutive frames using ORB
  - Computes a PARTIAL AFFINE transform (4 DOF: tx, ty, rotation, scale)
  - Warps frames into a stable sequence using trajectory smoothing
  - Falls back safely (identity transform) when matching fails
  - Saves a stabilised output video for review

Importable API:
    from src.preprocessing.motion_compensation import MotionCompensator
    mc = MotionCompensator(input_path="video.mp4")
    result = mc.run(output_path="outputs/motion_compensation/stabilised.mp4")
"""

import cv2
import numpy as np
import time
from dataclasses import dataclass
from typing import Optional


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class MotionCompensationConfig:
    """
    All tunable parameters for the motion compensation pipeline.
    Centralised here so other modules can import and reuse.
    """
    max_features: int       = 1000   # ORB keypoints per frame
    match_ratio: float      = 0.72   # Lowe ratio test threshold
    min_match_count: int    = 6      # minimum matches to attempt estimation
    ransac_threshold: float = 5.0   # RANSAC reprojection error (pixels)
    smoothing_radius: int   = 20     # moving average half-window (frames)
    crop_ratio: float       = 0.03   # border crop to hide warp artefacts
    show_comparison: bool   = True   # side-by-side output (original + stabilised)


SUPPORTED_FORMATS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".webm", ".flv"}


# ─────────────────────────────────────────────────────────────────────────────
# Frame Transform
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class FrameTransform:
    """
    Estimated motion between two consecutive frames.
    Partial affine (4 DOF) is used instead of full homography (8 DOF)
    to prevent perspective distortion on ocean/surf footage.
    """
    dx: float       = 0.0
    dy: float       = 0.0
    angle: float    = 0.0
    scale: float    = 1.0
    inliers: int    = 0
    matches: int    = 0
    success: bool   = False
    fallback: bool  = False

    @staticmethod
    def identity() -> "FrameTransform":
        """Safe fallback: no transform applied."""
        return FrameTransform(success=False, fallback=True)


# ─────────────────────────────────────────────────────────────────────────────
# Result
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class MotionCompensationResult:
    """Full metrics returned after processing."""
    input_path: str             = ""
    output_path: str            = ""
    resolution: str             = ""
    fps: float                  = 0.0
    total_frames: int           = 0
    successful_transforms: int  = 0
    failed_transforms: int      = 0
    fallback_transforms: int    = 0
    success_rate_pct: float     = 0.0
    avg_inliers: float          = 0.0
    avg_correction_dx_px: float = 0.0
    avg_correction_dy_px: float = 0.0
    processing_time_s: float    = 0.0

    def to_dict(self) -> dict:
        return {
            "input":                  self.input_path,
            "output":                 self.output_path,
            "resolution":             self.resolution,
            "fps":                    round(self.fps, 2),
            "total_frames":           self.total_frames,
            "successful_transforms":  self.successful_transforms,
            "failed_transforms":      self.failed_transforms,
            "fallback_transforms":    self.fallback_transforms,
            "success_rate_pct":       round(self.success_rate_pct, 1),
            "avg_inliers":            round(self.avg_inliers, 1),
            "avg_correction_dx_px":   round(self.avg_correction_dx_px, 2),
            "avg_correction_dy_px":   round(self.avg_correction_dy_px, 2),
            "processing_time_s":      round(self.processing_time_s, 1),
        }

    def print_summary(self):
        print(f"\n{'='*55}")
        print(f"  MOTION COMPENSATION RESULTS")
        print(f"{'='*55}")
        for k, v in self.to_dict().items():
            print(f"  {k:<35} {v}")
        print(f"{'='*55}\n")


# ─────────────────────────────────────────────────────────────────────────────
# Internal components
# ─────────────────────────────────────────────────────────────────────────────

class _MotionEstimator:
    """
    Estimates inter-frame motion using ORB + partial affine (RANSAC).
    Partial affine gives 4 DOF: translation (tx, ty), rotation, uniform scale.
    Falls back to identity if estimation fails at any step.
    """

    def __init__(self, cfg: MotionCompensationConfig):
        self.cfg     = cfg
        self.orb     = cv2.ORB_create(nfeatures=cfg.max_features,
                                      fastThreshold=10)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    def estimate(self, prev_gray: np.ndarray,
                 curr_gray: np.ndarray) -> FrameTransform:
        cfg = self.cfg

        # Step 1 – detect keypoints
        kp1, des1 = self.orb.detectAndCompute(prev_gray, None)
        kp2, des2 = self.orb.detectAndCompute(curr_gray, None)

        if (des1 is None or des2 is None
                or len(kp1) < cfg.min_match_count
                or len(kp2) < cfg.min_match_count):
            return FrameTransform.identity()

        # Step 2 – match with Lowe ratio test
        try:
            raw = self.matcher.knnMatch(des1, des2, k=2)
        except cv2.error:
            return FrameTransform.identity()

        good = [m for m, n in raw
                if m.distance < cfg.match_ratio * n.distance]

        if len(good) < cfg.min_match_count:
            return FrameTransform.identity()

        pts1 = np.float32([kp1[m.queryIdx].pt for m in good])
        pts2 = np.float32([kp2[m.trainIdx].pt for m in good])

        # Step 3 – partial affine (4 DOF) via RANSAC
        M, mask = cv2.estimateAffinePartial2D(
            pts1, pts2,
            method=cv2.RANSAC,
            ransacReprojThreshold=cfg.ransac_threshold,
        )

        if M is None or mask is None:
            return FrameTransform.identity()

        inliers = int(mask.sum())
        if inliers < cfg.min_match_count:
            return FrameTransform.identity()

        return FrameTransform(
            dx      = float(M[0, 2]),
            dy      = float(M[1, 2]),
            angle   = float(np.degrees(np.arctan2(M[1, 0], M[0, 0]))),
            scale   = float(np.sqrt(M[0, 0]**2 + M[1, 0]**2)),
            inliers = inliers,
            matches = len(good),
            success = True,
            fallback= False,
        )


class _TrajectorySmoother:
    """Smooths cumulative camera trajectory with a moving average."""

    @staticmethod
    def accumulate(transforms: list) -> np.ndarray:
        n    = len(transforms)
        traj = np.zeros((n, 3))
        tx = ty = angle = 0.0
        for i, t in enumerate(transforms):
            if t.success:
                tx    += t.dx
                ty    += t.dy
                angle += t.angle
            traj[i] = [tx, ty, angle]
        return traj

    @staticmethod
    def smooth(traj: np.ndarray, radius: int) -> np.ndarray:
        kernel   = np.ones(2 * radius + 1) / (2 * radius + 1)
        smoothed = np.copy(traj)
        for col in range(traj.shape[1]):
            padded            = np.pad(traj[:, col], (radius, radius), mode="edge")
            smoothed[:, col]  = np.convolve(padded, kernel, mode="valid")
        return smoothed


class _FrameWarper:
    """Applies affine correction to each frame."""

    def __init__(self, width: int, height: int, crop_ratio: float):
        self.width      = width
        self.height     = height
        self.crop_ratio = crop_ratio

    def warp(self, frame: np.ndarray, correction: np.ndarray) -> np.ndarray:
        dx, dy, angle = correction
        cos_a = np.cos(np.radians(angle))
        sin_a = np.sin(np.radians(angle))
        M     = np.array([[cos_a, -sin_a, dx],
                          [sin_a,  cos_a, dy]], dtype=np.float64)
        warped = cv2.warpAffine(frame, M, (self.width, self.height),
                                flags=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_REFLECT_101)
        if self.crop_ratio > 0:
            h, w   = warped.shape[:2]
            dy_    = int(h * self.crop_ratio)
            dx_    = int(w * self.crop_ratio)
            warped = cv2.resize(warped[dy_:h-dy_, dx_:w-dx_],
                                (w, h), interpolation=cv2.INTER_LINEAR)
        return warped


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

class MotionCompensator:
    """
    Main reusable motion compensation component for SurfWatch.

    Usage
    -----
    >>> from src.preprocessing.motion_compensation import MotionCompensator
    >>> mc = MotionCompensator("video.mp4")
    >>> result = mc.run("outputs/motion_compensation/stabilised.mp4")
    >>> result.print_summary()

    With custom config:
    >>> from src.preprocessing.motion_compensation import (
    ...     MotionCompensator, MotionCompensationConfig)
    >>> cfg = MotionCompensationConfig(smoothing_radius=30, max_features=1500)
    >>> mc  = MotionCompensator("video.mp4", config=cfg)
    >>> mc.run("outputs/motion_compensation/stabilised.mp4")
    """

    def __init__(self, input_path: str,
                 config: Optional[MotionCompensationConfig] = None,
                 smoothing_radius: Optional[int]   = None,
                 max_features: Optional[int]       = None,
                 crop_ratio: Optional[float]       = None,
                 show_comparison: Optional[bool]   = None):

        suffix = str(input_path).lower().rsplit(".", 1)[-1]
        if f".{suffix}" not in SUPPORTED_FORMATS:
            raise ValueError(
                f"Unsupported format '.{suffix}'. "
                f"Supported: {SUPPORTED_FORMATS}"
            )

        self.input_path = input_path
        self.config     = config or MotionCompensationConfig()

        if smoothing_radius is not None:
            self.config.smoothing_radius = smoothing_radius
        if max_features is not None:
            self.config.max_features = max_features
        if crop_ratio is not None:
            self.config.crop_ratio = crop_ratio
        if show_comparison is not None:
            self.config.show_comparison = show_comparison

    def run(self, output_path: str = "stabilised.mp4") -> MotionCompensationResult:
        """
        Execute the full pipeline:
          1. Read video
          2. Estimate inter-frame transforms (partial affine)
          3. Accumulate + smooth trajectory
          4. Warp frames with corrected transforms
          5. Write stabilised output video

        Returns MotionCompensationResult with all metrics.
        """
        cfg     = self.config
        t_start = time.time()

        # ── Open video ────────────────────────────────────────────────────────
        cap = cv2.VideoCapture(self.input_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {self.input_path}")

        fps    = cap.get(cv2.CAP_PROP_FPS) or 30.0
        width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        print(f"\n{'='*55}")
        print(f"  SurfWatch | Motion Compensation | SCRUM-54")
        print(f"{'='*55}")
        print(f"  Input      : {self.input_path}")
        print(f"  Resolution : {width}x{height} @ {fps:.1f}fps")
        print(f"  Frames     : {total}")
        print(f"  Transform  : Partial Affine (4 DOF)")
        print(f"  Smoothing  : {cfg.smoothing_radius} frames")
        print(f"  Output     : {output_path}")
        print(f"{'='*55}\n")

        # ── Pass 1: estimate transforms ───────────────────────────────────────
        print("[1/3] Estimating inter-frame transforms...")
        estimator  = _MotionEstimator(cfg)
        transforms = []
        frames     = []

        ret, frame = cap.read()
        if not ret:
            raise RuntimeError("Cannot read first frame!")
        frames.append(frame)
        prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        failed = fallback = 0
        inlier_counts = []

        for i in range(1, total):
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
            curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            t         = estimator.estimate(prev_gray, curr_gray)
            transforms.append(t)

            if not t.success:
                failed += 1
                if t.fallback:
                    fallback += 1
            else:
                inlier_counts.append(t.inliers)

            prev_gray = curr_gray
            if i % 100 == 0:
                print(f"  frame {i}/{total-1} | "
                      f"matches={t.matches} | "
                      f"inliers={t.inliers} | "
                      f"success={t.success}")

        cap.release()
        n = len(frames)
        print(f"\n  Done — {n} frames | {failed} failed | {fallback} fallback\n")

        # ── Pass 2: smooth trajectory ─────────────────────────────────────────
        print("[2/3] Smoothing trajectory...")
        smoother    = _TrajectorySmoother()
        traj        = smoother.accumulate(transforms)
        smoothed    = smoother.smooth(traj, cfg.smoothing_radius)
        corrections = smoothed - traj

        # ── Pass 3: warp and write output ─────────────────────────────────────
        print("[3/3] Warping and writing output...")
        out_w   = width * 2 if cfg.show_comparison else width
        warper  = _FrameWarper(width, height, cfg.crop_ratio)
        writer  = cv2.VideoWriter(
            output_path,
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps, (out_w, height)
        )

        dx_vals, dy_vals = [], []

        for i, frame in enumerate(frames):
            corr = corrections[i] if i < len(corrections) else np.zeros(3)
            dx_vals.append(abs(corr[0]))
            dy_vals.append(abs(corr[1]))

            stabilised = warper.warp(frame, corr)

            if cfg.show_comparison:
                canvas = np.hstack([frame, stabilised])
                self._label(canvas, width, height, i + 1, n)
            else:
                canvas = stabilised

            writer.write(canvas)

        writer.release()

        result = MotionCompensationResult(
            input_path            = self.input_path,
            output_path           = output_path,
            resolution            = f"{width}x{height}",
            fps                   = fps,
            total_frames          = n,
            successful_transforms = len(inlier_counts),
            failed_transforms     = failed,
            fallback_transforms   = fallback,
            success_rate_pct      = len(inlier_counts) / max(n - 1, 1) * 100,
            avg_inliers           = float(np.mean(inlier_counts)) if inlier_counts else 0,
            avg_correction_dx_px  = float(np.mean(dx_vals)) if dx_vals else 0,
            avg_correction_dy_px  = float(np.mean(dy_vals)) if dy_vals else 0,
            processing_time_s     = time.time() - t_start,
        )
        return result

    @staticmethod
    def _label(canvas: np.ndarray, w: int, h: int, idx: int, total: int):
        font = cv2.FONT_HERSHEY_SIMPLEX
        s    = max(h / 600, 0.6)
        for text, x, color in [
            ("ORIGINAL",   10,      (80,  80, 220)),
            ("STABILISED", w + 10,  (80, 220,  80)),
        ]:
            cv2.putText(canvas, text, (x, 34), font, s, (0,0,0), 3, cv2.LINE_AA)
            cv2.putText(canvas, text, (x, 34), font, s, color,   2, cv2.LINE_AA)
        cv2.putText(canvas, f"Frame {idx}/{total}",
                    (10, h - 10), font, s * 0.8, (180,180,180), 1, cv2.LINE_AA)
