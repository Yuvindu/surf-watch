"""
stabiliser.py
─────────────────────────────────────────────────────────────────
SCRUM-54 · SurfWatch – Video Stabilisation Module

A reusable, importable module that:
  ✅ Reads any video file
  ✅ Estimates motion between consecutive frames using ORB
  ✅ Computes a PARTIAL AFFINE transform (4 DOF: tx, ty, rotation, scale)
  ✅ Warps frames into a stable sequence using trajectory smoothing
  ✅ Falls back safely when feature matching fails
  ✅ Saves a stabilised output video for review

Usage as a module (import into other scripts):
──────────────────────────────────────────────
    from stabiliser import Stabiliser

    stab = Stabiliser(input_path="my_video.mp4")
    stab.run(output_path="stabilised.mp4")

    # Or with custom settings:
    stab = Stabiliser(
        input_path="my_video.mp4",
        smoothing_radius=25,
        max_features=1000,
        min_match_count=6,
    )
    result = stab.run(output_path="stabilised.mp4")
    print(result)  # dict with all metrics

Usage as a standalone script:
──────────────────────────────
    python stabiliser.py --input my_video.mp4 --output stabilised.mp4
    python stabiliser.py --input clip.mp4 --output out.mp4 --smooth 30 --features 1500
"""

import cv2
import numpy as np
import json
import time
import argparse
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional


# ─────────────────────────────────────────────────────────────────────────────
# Data classes — clean interfaces for inter-component communication
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class FrameTransform:
    """
    Represents the estimated motion between two consecutive frames.
    Uses partial affine (4 DOF) instead of full homography (8 DOF)
    to prevent perspective distortion on surf/beach footage.
    """
    dx: float = 0.0          # horizontal translation (pixels)
    dy: float = 0.0          # vertical translation (pixels)
    angle: float = 0.0       # rotation (degrees)
    scale: float = 1.0       # scale factor
    inliers: int = 0         # number of RANSAC inliers
    matches: int = 0         # number of feature matches before RANSAC
    success: bool = False    # whether estimation succeeded
    fallback: bool = False   # True if identity transform was used

    def to_matrix(self) -> np.ndarray:
        """Convert to 2x3 affine matrix for cv2.warpAffine."""
        angle_rad = np.radians(self.angle)
        cos_a = np.cos(angle_rad) * self.scale
        sin_a = np.sin(angle_rad) * self.scale
        return np.array([
            [cos_a, -sin_a, self.dx],
            [sin_a,  cos_a, self.dy],
        ], dtype=np.float64)

    @staticmethod
    def identity() -> "FrameTransform":
        """Returns a no-op transform (fallback when matching fails)."""
        return FrameTransform(success=False, fallback=True)


@dataclass
class StabiliserConfig:
    """All tunable parameters in one place for easy reuse."""
    max_features: int    = 1000   # ORB keypoints to detect per frame
    match_ratio: float   = 0.72   # Lowe's ratio test threshold
    min_match_count: int = 6      # minimum matches to attempt estimation
    ransac_threshold: float = 5.0 # RANSAC reprojection threshold (pixels)
    smoothing_radius: int  = 20   # moving average window half-width (frames)
    crop_ratio: float      = 0.03 # border crop to hide warp artefacts (0=off)
    show_comparison: bool  = True # side-by-side original vs stabilised output


@dataclass
class StabiliserResult:
    """
    Full result returned by Stabiliser.run().
    Can be serialised to JSON for reporting.
    """
    input_path: str = ""
    output_path: str = ""
    resolution: str = ""
    fps: float = 0.0
    total_frames: int = 0
    successful_transforms: int = 0
    failed_transforms: int = 0
    fallback_transforms: int = 0
    success_rate_pct: float = 0.0
    avg_inliers: float = 0.0
    avg_dx_px: float = 0.0
    avg_dy_px: float = 0.0
    processing_time_s: float = 0.0

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
            "avg_correction_dx_px":   round(self.avg_dx_px, 2),
            "avg_correction_dy_px":   round(self.avg_dy_px, 2),
            "processing_time_s":      round(self.processing_time_s, 1),
        }

    def print_summary(self):
        print(f"\n{'='*55}")
        print(f"  STABILISATION RESULTS")
        print(f"{'='*55}")
        for k, v in self.to_dict().items():
            print(f"  {k:<30} {v}")
        print(f"{'='*55}\n")


# ─────────────────────────────────────────────────────────────────────────────
# Core motion estimator
# ─────────────────────────────────────────────────────────────────────────────

class MotionEstimator:
    """
    Estimates inter-frame motion using ORB feature matching
    and partial affine transform estimation.

    Partial affine (4 DOF) is preferred over full homography (8 DOF)
    because it prevents perspective distortion on ocean/beach footage
    where the scene is approximately planar.
    """

    def __init__(self, config: StabiliserConfig):
        self.config = config
        self.orb = cv2.ORB_create(
            nfeatures=config.max_features,
            fastThreshold=10,
        )
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    def estimate(self, prev_gray: np.ndarray,
                 curr_gray: np.ndarray) -> FrameTransform:
        """
        Estimate motion from prev_gray → curr_gray.

        Strategy:
          1. Detect ORB keypoints in both frames
          2. Match descriptors using brute-force + Lowe's ratio test
          3. Estimate PARTIAL AFFINE transform using RANSAC
          4. Fall back to identity transform if any step fails

        Returns a FrameTransform with all motion parameters.
        """
        cfg = self.config

        # ── Step 1: Detect keypoints ──────────────────────────────────────────
        kp1, des1 = self.orb.detectAndCompute(prev_gray, None)
        kp2, des2 = self.orb.detectAndCompute(curr_gray, None)

        if (des1 is None or des2 is None
                or len(kp1) < cfg.min_match_count
                or len(kp2) < cfg.min_match_count):
            return FrameTransform.identity()

        # ── Step 2: Match descriptors with Lowe's ratio test ─────────────────
        try:
            raw_matches = self.matcher.knnMatch(des1, des2, k=2)
        except cv2.error:
            return FrameTransform.identity()

        good = [
            m for m, n in raw_matches
            if m.distance < cfg.match_ratio * n.distance
        ]

        if len(good) < cfg.min_match_count:
            return FrameTransform.identity()

        pts1 = np.float32([kp1[m.queryIdx].pt for m in good])
        pts2 = np.float32([kp2[m.trainIdx].pt for m in good])

        # ── Step 3: Partial affine estimation (RANSAC) ────────────────────────
        # estimateAffinePartial2D gives translation + rotation + uniform scale
        # This is safer than findHomography for ocean footage
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

        # ── Step 4: Decompose matrix into interpretable components ────────────
        dx    = float(M[0, 2])
        dy    = float(M[1, 2])
        angle = float(np.degrees(np.arctan2(M[1, 0], M[0, 0])))
        scale = float(np.sqrt(M[0, 0]**2 + M[1, 0]**2))

        return FrameTransform(
            dx=dx, dy=dy, angle=angle, scale=scale,
            inliers=inliers, matches=len(good),
            success=True, fallback=False,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Trajectory smoother
# ─────────────────────────────────────────────────────────────────────────────

class TrajectorySmoother:
    """
    Accumulates frame-to-frame transforms into a cumulative trajectory,
    then smooths it with a moving average to remove jitter while
    preserving intentional camera motion.
    """

    @staticmethod
    def accumulate(transforms: list[FrameTransform]) -> np.ndarray:
        """
        Convert per-frame transforms to cumulative trajectory.
        Returns (N, 3) array of [tx, ty, angle] per frame.
        """
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
    def smooth(trajectory: np.ndarray, radius: int) -> np.ndarray:
        """
        Apply moving average smoothing to each trajectory dimension.
        Larger radius = smoother but may lose intentional motion.
        """
        kernel_size = 2 * radius + 1
        kernel      = np.ones(kernel_size) / kernel_size
        smoothed    = np.copy(trajectory)

        for col in range(trajectory.shape[1]):
            padded          = np.pad(trajectory[:, col], (radius, radius), mode="edge")
            smoothed[:, col] = np.convolve(padded, kernel, mode="valid")

        return smoothed

    @staticmethod
    def compute_corrections(trajectory: np.ndarray,
                            smoothed: np.ndarray) -> np.ndarray:
        """
        Compute per-frame correction = smoothed - raw.
        This is what gets applied to each frame during warping.
        """
        return smoothed - trajectory


# ─────────────────────────────────────────────────────────────────────────────
# Frame warper
# ─────────────────────────────────────────────────────────────────────────────

class FrameWarper:
    """
    Applies correction transforms to frames and optionally crops borders.
    """

    def __init__(self, width: int, height: int, crop_ratio: float = 0.0):
        self.width      = width
        self.height     = height
        self.crop_ratio = crop_ratio

    def warp(self, frame: np.ndarray,
             correction: np.ndarray) -> np.ndarray:
        """
        Apply a [tx, ty, angle] correction to a frame using partial affine.
        Uses REFLECT border mode to avoid black edges.
        """
        dx, dy, angle = correction
        angle_rad = np.radians(angle)
        cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)

        M = np.array([
            [cos_a, -sin_a, dx],
            [sin_a,  cos_a, dy],
        ], dtype=np.float64)

        warped = cv2.warpAffine(
            frame, M, (self.width, self.height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101,
        )

        if self.crop_ratio > 0:
            warped = self._crop_and_resize(warped)

        return warped

    def _crop_and_resize(self, frame: np.ndarray) -> np.ndarray:
        """Crop borders to hide warp artefacts, then resize back."""
        h, w  = frame.shape[:2]
        dy    = int(h * self.crop_ratio)
        dx    = int(w * self.crop_ratio)
        cropped = frame[dy:h-dy, dx:w-dx]
        return cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)


# ─────────────────────────────────────────────────────────────────────────────
# Main Stabiliser class — the public API of this module
# ─────────────────────────────────────────────────────────────────────────────

class Stabiliser:
    """
    Main reusable stabilisation component.

    Example
    -------
    >>> from stabiliser import Stabiliser, StabiliserConfig
    >>> stab = Stabiliser("input.mp4")
    >>> result = stab.run("output.mp4")
    >>> result.print_summary()

    Or with custom config:
    >>> config = StabiliserConfig(smoothing_radius=30, max_features=1500)
    >>> stab = Stabiliser("input.mp4", config=config)
    >>> result = stab.run("output.mp4")
    """

    SUPPORTED_FORMATS = {
        ".mp4", ".avi", ".mov", ".mkv", ".wmv", ".webm", ".flv"
    }

    def __init__(self, input_path: str,
                 config: Optional[StabiliserConfig] = None,
                 # Convenience shortcuts (override config fields)
                 smoothing_radius: Optional[int] = None,
                 max_features: Optional[int] = None,
                 min_match_count: Optional[int] = None,
                 crop_ratio: Optional[float] = None,
                 show_comparison: Optional[bool] = None):

        self.input_path = input_path
        self.config     = config or StabiliserConfig()

        # Apply convenience overrides
        if smoothing_radius is not None:
            self.config.smoothing_radius = smoothing_radius
        if max_features is not None:
            self.config.max_features = max_features
        if min_match_count is not None:
            self.config.min_match_count = min_match_count
        if crop_ratio is not None:
            self.config.crop_ratio = crop_ratio
        if show_comparison is not None:
            self.config.show_comparison = show_comparison

        # Validate input
        suffix = Path(input_path).suffix.lower()
        if suffix not in self.SUPPORTED_FORMATS:
            raise ValueError(
                f"Unsupported format '{suffix}'. "
                f"Supported: {self.SUPPORTED_FORMATS}"
            )

    def run(self, output_path: str = "stabilised.mp4") -> StabiliserResult:
        """
        Run the full stabilisation pipeline.

        Steps:
          1. Read video
          2. Estimate inter-frame transforms (partial affine)
          3. Accumulate + smooth trajectory
          4. Compute per-frame corrections
          5. Warp frames and write output

        Returns StabiliserResult with all metrics.
        """
        t_start = time.time()
        cfg     = self.config

        # ── Open video ────────────────────────────────────────────────────────
        cap = cv2.VideoCapture(self.input_path)
        if not cap.isOpened():
            raise FileNotFoundError(
                f"Cannot open video: {self.input_path}"
            )

        fps    = cap.get(cv2.CAP_PROP_FPS) or 30.0
        width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        print(f"\n{'='*55}")
        print(f"  SurfWatch Stabiliser | SCRUM-54")
        print(f"{'='*55}")
        print(f"  Input      : {self.input_path}")
        print(f"  Resolution : {width}x{height} @ {fps:.1f}fps")
        print(f"  Frames     : {total}")
        print(f"  Smoothing  : radius={cfg.smoothing_radius} frames")
        print(f"  Transform  : Partial Affine (4 DOF)")
        print(f"  Output     : {output_path}")
        print(f"{'='*55}\n")

        # ── Pass 1: Estimate transforms ───────────────────────────────────────
        print("[PASS 1/2] Estimating inter-frame transforms...")
        estimator  = MotionEstimator(cfg)
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

            transform = estimator.estimate(prev_gray, curr_gray)
            transforms.append(transform)

            if not transform.success:
                failed += 1
                if transform.fallback:
                    fallback += 1
            else:
                inlier_counts.append(transform.inliers)

            prev_gray = curr_gray

            if i % 100 == 0:
                print(f"  frame {i}/{total-1}  "
                      f"matches={transform.matches}  "
                      f"inliers={transform.inliers}  "
                      f"success={transform.success}")

        cap.release()
        n_frames = len(frames)
        print(f"\n  Done. {n_frames} frames | "
              f"{failed} failed | {fallback} fallback\n")

        # ── Smooth trajectory ─────────────────────────────────────────────────
        print("[SMOOTH ] Smoothing trajectory...")
        smoother   = TrajectorySmoother()
        trajectory = smoother.accumulate(transforms)
        smoothed   = smoother.smooth(trajectory, cfg.smoothing_radius)
        corrections = smoother.compute_corrections(trajectory, smoothed)

        # ── Pass 2: Warp and write output ─────────────────────────────────────
        print("[PASS 2/2] Warping frames...")
        out_width  = width * 2 if cfg.show_comparison else width
        warper     = FrameWarper(width, height, cfg.crop_ratio)
        fourcc     = cv2.VideoWriter_fourcc(*"mp4v")
        writer     = cv2.VideoWriter(
            output_path, fourcc, fps, (out_width, height)
        )

        dx_vals, dy_vals = [], []

        for i, frame in enumerate(frames):
            correction = corrections[i] if i < len(corrections) \
                         else np.zeros(3)
            dx_vals.append(abs(correction[0]))
            dy_vals.append(abs(correction[1]))

            stabilised = warper.warp(frame, correction)

            if cfg.show_comparison:
                # Side-by-side: original (left) + stabilised (right)
                canvas = np.hstack([frame, stabilised])
                self._draw_labels(canvas, width, height, i+1, n_frames)
            else:
                canvas = stabilised

            writer.write(canvas)

        writer.release()

        # ── Build result ──────────────────────────────────────────────────────
        n_success = len(inlier_counts)
        result    = StabiliserResult(
            input_path=self.input_path,
            output_path=output_path,
            resolution=f"{width}x{height}",
            fps=fps,
            total_frames=n_frames,
            successful_transforms=n_success,
            failed_transforms=failed,
            fallback_transforms=fallback,
            success_rate_pct=(n_success / max(n_frames-1, 1)) * 100,
            avg_inliers=float(np.mean(inlier_counts)) if inlier_counts else 0,
            avg_dx_px=float(np.mean(dx_vals)) if dx_vals else 0,
            avg_dy_px=float(np.mean(dy_vals)) if dy_vals else 0,
            processing_time_s=time.time() - t_start,
        )
        return result

    @staticmethod
    def _draw_labels(canvas: np.ndarray, width: int, height: int,
                     frame_idx: int, total: int):
        """Draw ORIGINAL / STABILISED labels on side-by-side output."""
        font  = cv2.FONT_HERSHEY_SIMPLEX
        scale = max(height / 600, 0.6)

        def put(text, x, y, color):
            cv2.putText(canvas, text, (x, y), font,
                        scale, (0,0,0), 3, cv2.LINE_AA)
            cv2.putText(canvas, text, (x, y), font,
                        scale, color, 2, cv2.LINE_AA)

        put("ORIGINAL",   10, 34, (80, 80, 220))
        put("STABILISED", width + 10, 34, (80, 220, 80))
        put(f"Frame {frame_idx}/{total}", 10, height - 10, (200, 200, 200))


# ─────────────────────────────────────────────────────────────────────────────
# Standalone entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="SurfWatch Stabiliser — SCRUM-54",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python stabiliser.py --input clip.mp4
  python stabiliser.py --input clip.mp4 --output out.mp4 --smooth 30
  python stabiliser.py --input clip.mp4 --no-comparison --crop 0.04
        """
    )
    parser.add_argument("--input",         required=True,
                        help="Input video path")
    parser.add_argument("--output",        default="stabilised.mp4",
                        help="Output video path (default: stabilised.mp4)")
    parser.add_argument("--smooth",        type=int, default=20,
                        help="Smoothing radius in frames (default: 20)")
    parser.add_argument("--features",      type=int, default=1000,
                        help="Max ORB features per frame (default: 1000)")
    parser.add_argument("--min-matches",   type=int, default=6,
                        help="Min matches to attempt estimation (default: 6)")
    parser.add_argument("--crop",          type=float, default=0.03,
                        help="Border crop ratio (default: 0.03)")
    parser.add_argument("--no-comparison", action="store_true",
                        help="Output stabilised only (no side-by-side)")
    parser.add_argument("--save-report",   action="store_true",
                        help="Save results as JSON report")
    args = parser.parse_args()

    config = StabiliserConfig(
        smoothing_radius = args.smooth,
        max_features     = args.features,
        min_match_count  = args.min_matches,
        crop_ratio       = args.crop,
        show_comparison  = not args.no_comparison,
    )

    stab   = Stabiliser(args.input, config=config)
    result = stab.run(args.output)
    result.print_summary()

    if args.save_report:
        report_path = Path(args.output).stem + "_report.json"
        with open(report_path, "w") as f:
            json.dump(result.to_dict(), f, indent=2)
        print(f"[OK] Report saved → {report_path}")


if __name__ == "__main__":
    main()
