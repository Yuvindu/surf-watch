"""
wave_analyzer.py
─────────────────
SCRUM-54 · SurfWatch – Wave Detection & Danger Analysis System

Features:
  ✅ Detects wave crests line by line per frame
  ✅ Measures wave height in pixels per frame
  ✅ Classifies danger: SAFE (green) / MODERATE (yellow) / DANGEROUS (red)
  ✅ Shows measurements on every frame
  ✅ Flags dangerous frames with alerts
  ✅ Works on both videos AND images
  ✅ Generates full danger report with frame-by-frame stats

Usage:
    # Video
    python wave_analyzer.py --input RipVIS-016.mp4

    # Image
    python wave_analyzer.py --image surf_real_1.jpg

    # All images in folder
    python wave_analyzer.py --folder .
"""

import argparse
import json
import cv2
import numpy as np
from pathlib import Path

# ── Danger thresholds (in pixels — auto-scaled to frame height) ───────────────
# These represent wave height as % of frame height
SAFE_THRESH       = 0.15   # < 15% frame height = SAFE
MODERATE_THRESH   = 0.30   # 15-30% = MODERATE
# > 30% = DANGEROUS

# Colours (BGR)
C_SAFE      = (0,   200,  0)
C_MODERATE  = (0,   200, 200)
C_DANGER    = (0,    0,  220)
C_WHITE     = (255, 255, 255)
C_BLACK     = (0,     0,   0)
C_DARK      = (20,   20,  20)
C_YELLOW    = (0,   220, 220)
C_TEAL      = (180, 180,   0)
C_CREST     = (255, 180,   0)   # Wave crest line colour


# ── Wave detection core ───────────────────────────────────────────────────────
def detect_horizon(gray):
    """
    Estimate the horizon line using horizontal gradient analysis.
    Returns y-coordinate of estimated horizon.
    """
    h, w = gray.shape
    # Look in middle 60% of frame vertically
    y_start = int(h * 0.2)
    y_end   = int(h * 0.8)

    # Compute horizontal Sobel (detects horizontal edges = horizon)
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)
    sobel_x = np.abs(sobel_x)

    # Sum edge strength per row
    row_strength = sobel_x[y_start:y_end].sum(axis=1)
    best_row = int(np.argmax(row_strength)) + y_start
    return best_row


def detect_wave_crests(gray, horizon_y):
    """
    Detect wave crest lines below the horizon using:
    1. Gaussian blur to smooth noise
    2. Canny edge detection
    3. Horizontal line filtering
    Returns list of (y, x_start, x_end, strength) for each crest
    """
    h, w = gray.shape

    # Focus on ocean area (below horizon, above bottom 10%)
    ocean_top    = max(horizon_y - 20, 0)
    ocean_bottom = int(h * 0.92)
    ocean_region = gray[ocean_top:ocean_bottom, :]

    # Enhance contrast for wave detection
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    enhanced = clahe.apply(ocean_region)

    # Blur to reduce noise
    blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)

    # Edge detection
    edges = cv2.Canny(blurred, 30, 90)

    # Find horizontal wave lines using HoughLines
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=40,
        minLineLength=w // 6,
        maxLineGap=30
    )

    crests = []
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            # Only near-horizontal lines (wave crests)
            angle = abs(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
            if angle < 20 or angle > 160:
                abs_y = int((y1 + y2) / 2) + ocean_top
                length = abs(x2 - x1)
                crests.append((abs_y, min(x1, x2), max(x1, x2), length))

    # Sort by y position (top to bottom)
    crests.sort(key=lambda c: c[0])
    return crests, edges, ocean_top


def compute_wave_metrics(crests, horizon_y, frame_h, frame_w):
    """
    Compute wave height, count, and danger level from detected crests.
    """
    if not crests:
        return {
            "wave_count": 0,
            "max_wave_height_px": 0,
            "max_wave_height_pct": 0.0,
            "avg_wave_spacing_px": 0,
            "danger_level": "SAFE",
            "danger_color": C_SAFE,
        }

    # Wave height = distance from highest crest to horizon
    highest_crest_y = min(c[0] for c in crests)
    max_wave_height = abs(horizon_y - highest_crest_y)
    max_wave_pct    = max_wave_height / frame_h

    # Average spacing between crests
    if len(crests) > 1:
        ys = sorted([c[0] for c in crests])
        spacings = [abs(ys[i+1] - ys[i]) for i in range(len(ys)-1)]
        avg_spacing = int(np.mean(spacings))
    else:
        avg_spacing = 0

    # Danger classification
    if max_wave_pct < SAFE_THRESH:
        danger = "SAFE"
        color  = C_SAFE
    elif max_wave_pct < MODERATE_THRESH:
        danger = "MODERATE"
        color  = C_MODERATE
    else:
        danger = "DANGEROUS"
        color  = C_DANGER

    return {
        "wave_count": len(crests),
        "max_wave_height_px": int(max_wave_height),
        "max_wave_height_pct": round(float(max_wave_pct * 100), 1),
        "avg_wave_spacing_px": avg_spacing,
        "danger_level": danger,
        "danger_color": color,
    }


# ── Visualisation ─────────────────────────────────────────────────────────────
def draw_text_bg(img, text, pos, scale=0.55, thickness=1,
                 fg=C_WHITE, bg=C_DARK, padding=4):
    (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    x, y = pos
    cv2.rectangle(img, (x - padding, y - h - padding),
                  (x + w + padding, y + padding), bg, -1)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                scale, fg, thickness, cv2.LINE_AA)


def annotate_frame(frame, crests, horizon_y, metrics, frame_idx=None, total=None):
    """
    Draw all wave analysis annotations on a frame.
    """
    h, w = frame.shape[:2]
    annotated = frame.copy()
    danger_color = metrics["danger_color"]
    danger_level = metrics["danger_level"]

    # ── Horizon line ──────────────────────────────────────────────────────────
    cv2.line(annotated, (0, horizon_y), (w, horizon_y), C_TEAL, 1, cv2.LINE_AA)
    draw_text_bg(annotated, "HORIZON", (8, horizon_y - 6), 0.38, 1, C_TEAL, C_DARK)

    # ── Wave crest lines ──────────────────────────────────────────────────────
    for i, (y, x1, x2, strength) in enumerate(crests):
        # Draw crest line
        cv2.line(annotated, (x1, y), (x2, y), C_CREST, 2, cv2.LINE_AA)

        # Wave number label
        draw_text_bg(annotated, f"W{i+1}", (max(x1 - 28, 0), y + 4),
                     0.35, 1, C_CREST, C_DARK)

        # Height measurement line (from crest to horizon)
        mid_x = (x1 + x2) // 2
        cv2.line(annotated, (mid_x, horizon_y), (mid_x, y),
                 danger_color, 1, cv2.LINE_AA)

        # Height label
        height_px = abs(horizon_y - y)
        draw_text_bg(annotated, f"{height_px}px",
                     (mid_x + 4, (horizon_y + y) // 2),
                     0.35, 1, danger_color, C_DARK)

    # ── Danger banner ─────────────────────────────────────────────────────────
    banner_h = 36
    overlay  = annotated.copy()
    cv2.rectangle(overlay, (0, 0), (w, banner_h), danger_color, -1)
    cv2.addWeighted(overlay, 0.35, annotated, 0.65, 0, annotated)
    cv2.rectangle(annotated, (0, 0), (w, banner_h), danger_color, 2)

    # Danger icon
    icon = "!!" if danger_level == "DANGEROUS" else "!" if danger_level == "MODERATE" else "OK"
    cv2.putText(annotated, f"[{icon}] {danger_level}",
                (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                C_WHITE, 2, cv2.LINE_AA)

    # Frame counter
    if frame_idx is not None and total is not None:
        draw_text_bg(annotated, f"Frame {frame_idx}/{total}",
                     (w - 130, 22), 0.45, 1, C_WHITE, C_DARK)

    # ── Stats panel (bottom left) ─────────────────────────────────────────────
    stats = [
        f"Waves detected : {metrics['wave_count']}",
        f"Max height     : {metrics['max_wave_height_px']}px  ({metrics['max_wave_height_pct']}%)",
        f"Wave spacing   : {metrics['avg_wave_spacing_px']}px",
        f"Danger level   : {danger_level}",
    ]
    panel_x, panel_y = 8, h - len(stats) * 22 - 16
    panel_w, panel_h = 300, len(stats) * 22 + 16
    overlay = annotated.copy()
    cv2.rectangle(overlay, (panel_x, panel_y),
                  (panel_x + panel_w, panel_y + panel_h), C_DARK, -1)
    cv2.addWeighted(overlay, 0.75, annotated, 0.25, 0, annotated)
    cv2.rectangle(annotated, (panel_x, panel_y),
                  (panel_x + panel_w, panel_y + panel_h), danger_color, 1)

    for i, s in enumerate(stats):
        ty = panel_y + 16 + i * 22
        c = danger_color if "Danger" in s else C_WHITE
        cv2.putText(annotated, s, (panel_x + 8, ty),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.44, c, 1, cv2.LINE_AA)

    # ── Progress bar ──────────────────────────────────────────────────────────
    if frame_idx is not None and total is not None:
        pct = frame_idx / total
        cv2.rectangle(annotated, (0, h - 5), (w, h), C_DARK, -1)
        cv2.rectangle(annotated, (0, h - 5),
                      (int(w * pct), h), danger_color, -1)

    # ── DANGEROUS frame flash ──────────────────────────────────────────────────
    if danger_level == "DANGEROUS":
        # Red border flash
        cv2.rectangle(annotated, (0, 0), (w - 1, h - 1), C_DANGER, 6)
        draw_text_bg(annotated, "!! DANGEROUS WAVE DETECTED !!",
                     (w // 2 - 180, h // 2),
                     0.75, 2, C_WHITE, C_DANGER, 8)

    return annotated


# ── Wave history graph ────────────────────────────────────────────────────────
def draw_wave_history(frame, history, frame_idx, W, H):
    """Draw a mini wave height history graph in the top-right corner."""
    box_w, box_h = 200, 70
    x0 = W - box_w - 8
    y0 = 45

    cv2.rectangle(frame, (x0, y0), (x0 + box_w, y0 + box_h), C_DARK, -1)
    cv2.rectangle(frame, (x0, y0), (x0 + box_w, y0 + box_h), C_TEAL, 1)
    draw_text_bg(frame, "wave height history", (x0 + 4, y0 + 12),
                 0.35, 1, C_TEAL, C_DARK)

    n = min(len(history), box_w - 10)
    if n < 2:
        return

    vals = history[-n:]
    max_val = max(max(vals), 1)
    pad = 18

    for i in range(1, n):
        x1 = x0 + pad + int((i - 1) / n * (box_w - 2 * pad))
        x2 = x0 + pad + int(i / n * (box_w - 2 * pad))
        y1 = y0 + box_h - pad - int(vals[i-1] / max_val * (box_h - 2 * pad))
        y2 = y0 + box_h - pad - int(vals[i] / max_val * (box_h - 2 * pad))

        pct = vals[i] / max_val
        r = int(255 * min(pct * 2, 1.0))
        g = int(255 * min(2 - pct * 2, 1.0))
        cv2.line(frame, (x1, y1), (x2, y2), (0, g, r), 1, cv2.LINE_AA)

    # Threshold lines
    safe_y = y0 + box_h - pad - int(SAFE_THRESH / 0.5 * (box_h - 2 * pad))
    mod_y  = y0 + box_h - pad - int(MODERATE_THRESH / 0.5 * (box_h - 2 * pad))
    cv2.line(frame, (x0 + pad, safe_y), (x0 + box_w - pad, safe_y),
             C_SAFE, 1)
    cv2.line(frame, (x0 + pad, mod_y), (x0 + box_w - pad, mod_y),
             C_MODERATE, 1)


# ── Process video ─────────────────────────────────────────────────────────────
def process_video(input_path: str, output_path: str) -> dict:
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open: {input_path}")

    fps    = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W      = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H      = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"\n{'='*60}")
    print(f"  SurfWatch Wave Analyzer | SCRUM-54")
    print(f"{'='*60}")
    print(f"  Input  : {input_path}  ({W}x{H} @ {fps:.1f}fps, {total} frames)")
    print(f"  Output : {output_path}")
    print(f"{'='*60}\n")

    writer = cv2.VideoWriter(output_path,
                             cv2.VideoWriter_fourcc(*"mp4v"),
                             fps, (W, H))

    frame_results = []
    height_history = []
    dangerous_frames = []
    safe_count = moderate_count = danger_count = 0

    for i in range(total):
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect horizon and waves
        horizon_y          = detect_horizon(gray)
        crests, edges, _   = detect_wave_crests(gray, horizon_y)
        metrics            = compute_wave_metrics(crests, horizon_y, H, W)

        # Track history
        height_history.append(metrics["max_wave_height_px"])

        # Count danger levels
        level = metrics["danger_level"]
        if level == "SAFE":
            safe_count += 1
        elif level == "MODERATE":
            moderate_count += 1
        else:
            danger_count += 1
            dangerous_frames.append(i + 1)

        # Annotate frame
        annotated = annotate_frame(frame, crests, horizon_y,
                                   metrics, i + 1, total)
        draw_wave_history(annotated, height_history, i, W, H)

        writer.write(annotated)

        result = {
            "frame": i + 1,
            "horizon_y": horizon_y,
            "wave_count": metrics["wave_count"],
            "max_wave_height_px": metrics["max_wave_height_px"],
            "max_wave_height_pct": metrics["max_wave_height_pct"],
            "danger_level": level,
        }
        frame_results.append(result)

        if (i + 1) % 100 == 0:
            print(f"  frame {i+1}/{total}  "
                  f"waves={metrics['wave_count']}  "
                  f"height={metrics['max_wave_height_px']}px  "
                  f"danger={level}")

    cap.release()
    writer.release()

    results = {
        "input": input_path,
        "output": output_path,
        "total_frames": total,
        "safe_frames": safe_count,
        "moderate_frames": moderate_count,
        "dangerous_frames_count": danger_count,
        "dangerous_frame_numbers": dangerous_frames[:20],
        "max_wave_height_px_overall": max(height_history) if height_history else 0,
        "avg_wave_height_px": round(float(np.mean(height_history)), 1) if height_history else 0,
        "overall_danger": "DANGEROUS" if danger_count > total * 0.3
                          else "MODERATE" if moderate_count > total * 0.3
                          else "SAFE",
        "frame_results": frame_results[:50],  # First 50 frames in report
    }
    return results


# ── Process image ─────────────────────────────────────────────────────────────
def process_image(img_path: str, output_dir: Path) -> dict:
    img = cv2.imread(img_path)
    if img is None:
        print(f"  [WARN] Cannot read: {img_path}")
        return {}

    H, W = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    print(f"  Analysing: {Path(img_path).name}  ({W}x{H})")

    horizon_y        = detect_horizon(gray)
    crests, edges, _ = detect_wave_crests(gray, horizon_y)
    metrics          = compute_wave_metrics(crests, horizon_y, H, W)

    annotated = annotate_frame(img, crests, horizon_y, metrics)

    # Save annotated image
    stem = Path(img_path).stem
    out_path = output_dir / f"wave_analysis_{stem}.jpg"
    cv2.imwrite(str(out_path), annotated)

    level = metrics["danger_level"]
    color_name = "GREEN" if level == "SAFE" else \
                 "YELLOW" if level == "MODERATE" else "RED"

    print(f"  [{color_name}] {level} | "
          f"waves={metrics['wave_count']} | "
          f"height={metrics['max_wave_height_px']}px "
          f"({metrics['max_wave_height_pct']}%) | "
          f"→ {out_path.name}")

    return {
        "file": img_path,
        "output": str(out_path),
        "horizon_y": horizon_y,
        "wave_count": metrics["wave_count"],
        "max_wave_height_px": metrics["max_wave_height_px"],
        "max_wave_height_pct": metrics["max_wave_height_pct"],
        "avg_wave_spacing_px": metrics["avg_wave_spacing_px"],
        "danger_level": level,
    }


# ── Entry point ───────────────────────────────────────────────────────────────
SUPPORTED_VIDEOS = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.webm', '.flv'}
SUPPORTED_IMAGES = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="SurfWatch Wave Analyzer — SCRUM-54"
    )
    parser.add_argument("--input",  help="Video file to analyse")
    parser.add_argument("--image",  nargs="+", help="Image file(s) to analyse")
    parser.add_argument("--folder", help="Folder of images to analyse")
    parser.add_argument("--output", default="wave_analysis_output.mp4",
                        help="Output video path")
    parser.add_argument("--output-dir", default=".",
                        help="Output folder for image results")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    all_results = {}

    # ── Video mode ────────────────────────────────────────────────────────────
    if args.input:
        results = process_video(args.input, args.output)
        all_results["video"] = results

        print(f"\n{'='*60}")
        print(f"VIDEO RESULTS SUMMARY")
        print(f"{'='*60}")
        print(f"  Total frames     : {results['total_frames']}")
        print(f"  SAFE frames      : {results['safe_frames']}")
        print(f"  MODERATE frames  : {results['moderate_frames']}")
        print(f"  DANGEROUS frames : {results['dangerous_frames_count']}")
        print(f"  Max wave height  : {results['max_wave_height_px_overall']}px")
        print(f"  Avg wave height  : {results['avg_wave_height_px']}px")
        print(f"  Overall danger   : {results['overall_danger']}")
        if results['dangerous_frame_numbers']:
            print(f"  Danger at frames : {results['dangerous_frame_numbers'][:10]}")

    # ── Image mode ────────────────────────────────────────────────────────────
    images = []
    if args.image:
        images = args.image
    elif args.folder:
        folder = Path(args.folder)
        images = sorted([
            str(p) for p in folder.iterdir()
            if p.suffix.lower() in SUPPORTED_IMAGES
        ])

    if images:
        print(f"\n{'='*60}")
        print(f"  SurfWatch Wave Analyzer | IMAGE MODE")
        print(f"{'='*60}")
        img_results = []
        for img_path in images:
            if Path(img_path).suffix.lower() not in SUPPORTED_IMAGES:
                continue
            r = process_image(img_path, output_dir)
            if r:
                img_results.append(r)
        all_results["images"] = img_results

    # ── Save report ───────────────────────────────────────────────────────────
    report_path = output_dir / "wave_analysis_report.json"
    # Remove non-serializable color tuples before saving
    def clean(obj):
        if isinstance(obj, dict):
            return {k: clean(v) for k, v in obj.items()
                    if k != "danger_color"}
        if isinstance(obj, list):
            return [clean(i) for i in obj]
        return obj

    with open(report_path, "w") as f:
        json.dump(clean(all_results), f, indent=2)

    print(f"\n[OK] Report saved → {report_path}")
    if args.input:
        print(f"[OK] Annotated video → {args.output}")
