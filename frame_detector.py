"""
frame_detector.py
─────────────────
SCRUM-47 · SurfWatch – Frame-by-Frame Motion Detection Viewer
Shows ORB keypoints, motion vectors, and match info on every frame
like a research paper visualization.

Usage:
    python frame_detector.py --input RipVIS-016.mp4
    python frame_detector.py --input RipVIS-016.mp4 --output detected_output.mp4

Controls (if running live):
    Q = quit
"""

import argparse
import cv2
import numpy as np
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────
ORB_N_FEATURES   = 500
MATCH_RATIO      = 0.75
MIN_MATCHES      = 10
RANSAC_THRESH    = 5.0
FRAME_LIMIT      = 941   # process all frames
RESIZE_W, RESIZE_H = 960, 540  # output resolution per panel

# Colors (BGR)
COLOR_KP_PREV    = (0,   255, 255)   # yellow  – keypoints on prev frame
COLOR_KP_CURR    = (0,   255,   0)   # green   – keypoints on curr frame
COLOR_MATCH_LINE = (255,  80,  80)   # blue    – match lines
COLOR_INLIER     = (0,   200,   0)   # green   – inlier matches
COLOR_OUTLIER    = (0,    0,  200)   # red     – outlier matches
COLOR_TEXT       = (255, 255, 255)
COLOR_BOX        = (30,   30,  30)
COLOR_BORDER_OK  = (0,   220,   0)
COLOR_BORDER_FAIL= (0,     0, 220)


def draw_panel_label(img, text, pos=(10, 30), color=COLOR_TEXT):
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 4)
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)


def draw_stats_box(canvas, stats: dict, x, y):
    """Draw a semi-transparent stats box."""
    lines = [
        f"Frame    : {stats['frame']}",
        f"KP prev  : {stats['kp_prev']}",
        f"KP curr  : {stats['kp_curr']}",
        f"Matches  : {stats['matches']}",
        f"Inliers  : {stats['inliers']}",
        f"Ratio    : {stats['ratio']:.1%}",
        f"dX       : {stats['dx']:+.1f} px",
        f"dY       : {stats['dy']:+.1f} px",
        f"Angle    : {stats['angle']:+.2f} deg",
        f"Status   : {'OK' if stats['ok'] else 'FAILED'}",
    ]
    padding = 10
    line_h  = 22
    box_w   = 220
    box_h   = len(lines) * line_h + 2 * padding

    # Dark background
    overlay = canvas.copy()
    cv2.rectangle(overlay, (x, y), (x + box_w, y + box_h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.75, canvas, 0.25, 0, canvas)
    cv2.rectangle(canvas, (x, y), (x + box_w, y + box_h),
                  COLOR_BORDER_OK if stats['ok'] else COLOR_BORDER_FAIL, 1)

    for i, line in enumerate(lines):
        ty = y + padding + i * line_h + 14
        color = COLOR_BORDER_OK if (i == 9 and stats['ok']) else \
                COLOR_BORDER_FAIL if (i == 9 and not stats['ok']) else COLOR_TEXT
        cv2.putText(canvas, line, (x + padding, ty),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 1)


def process_video(input_path: str, output_path: str):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open: {input_path}")

    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps    = cap.get(cv2.CAP_PROP_FPS) or 30.0
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"[INFO] Input  : {input_path}  ({orig_w}x{orig_h} @ {fps:.1f}fps, {total} frames)")
    print(f"[INFO] Output : {output_path}")
    print(f"[INFO] Processing every frame with ORB detection visualization...\n")

    # Output is 2-panel side by side
    out_w = RESIZE_W * 2
    out_h = RESIZE_H
    writer = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (out_w, out_h)
    )

    orb     = cv2.ORB_create(nfeatures=ORB_N_FEATURES)
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)

    ret, first = cap.read()
    if not ret:
        raise RuntimeError("Cannot read first frame!")

    prev_frame = cv2.resize(first, (RESIZE_W, RESIZE_H))
    prev_gray  = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    kp_prev, des_prev = orb.detectAndCompute(prev_gray, None)

    frame_idx  = 1
    total_ok   = 0
    total_fail = 0

    while True:
        ret, raw = cap.read()
        if not ret:
            break

        curr_frame = cv2.resize(raw, (RESIZE_W, RESIZE_H))
        curr_gray  = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
        kp_curr, des_curr = orb.detectAndCompute(curr_gray, None)

        # ── Match ──────────────────────────────────────────────────────────
        dx, dy, angle = 0.0, 0.0, 0.0
        good, inliers_mask = [], None
        ok = False

        if des_prev is not None and des_curr is not None and len(kp_prev) >= MIN_MATCHES:
            raw_matches = matcher.knnMatch(des_prev, des_curr, k=2)
            good = [m for m, n in raw_matches
                    if m.distance < MATCH_RATIO * n.distance]

            if len(good) >= MIN_MATCHES:
                pts1 = np.float32([kp_prev[m.queryIdx].pt for m in good])
                pts2 = np.float32([kp_curr[m.trainIdx].pt for m in good])
                H, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, RANSAC_THRESH)
                if H is not None and mask is not None:
                    inliers_mask = mask.ravel()
                    n_inliers = int(inliers_mask.sum())
                    ok = n_inliers >= MIN_MATCHES
                    if ok:
                        total_ok += 1
                        dx    =  H[0, 2]
                        dy    =  H[1, 2]
                        angle =  np.degrees(np.arctan2(H[1, 0], H[0, 0]))
                else:
                    inliers_mask = np.zeros(len(good), dtype=np.uint8)

        if not ok:
            total_fail += 1

        n_inliers = int(inliers_mask.sum()) if inliers_mask is not None else 0
        ratio     = n_inliers / len(good) if good else 0.0

        # ── Build LEFT panel: prev frame with keypoints ────────────────────
        left = prev_frame.copy()
        for kp in kp_prev:
            cv2.circle(left, (int(kp.pt[0]), int(kp.pt[1])), 3, COLOR_KP_PREV, -1)
        draw_panel_label(left, "Previous frame  |  ORB keypoints (yellow)")

        # ── Build RIGHT panel: curr frame with matches ─────────────────────
        right = curr_frame.copy()
        for kp in kp_curr:
            cv2.circle(right, (int(kp.pt[0]), int(kp.pt[1])), 3, COLOR_KP_CURR, 1)

        # Draw inlier / outlier match lines
        if good and inliers_mask is not None:
            for i, m in enumerate(good):
                pt1 = tuple(map(int, kp_prev[m.queryIdx].pt))
                pt2 = tuple(map(int, kp_curr[m.trainIdx].pt))
                color = COLOR_INLIER if inliers_mask[i] else COLOR_OUTLIER
                thickness = 2 if inliers_mask[i] else 1
                cv2.line(right, pt1, pt2, color, thickness)

        draw_panel_label(right,
            f"Current frame  |  Inliers={n_inliers} (green)  Outliers={len(good)-n_inliers} (red)")

        # ── Combine panels ─────────────────────────────────────────────────
        canvas = np.hstack([left, right])

        # Stats box
        stats = dict(
            frame   = frame_idx,
            kp_prev = len(kp_prev),
            kp_curr = len(kp_curr),
            matches = len(good),
            inliers = n_inliers,
            ratio   = ratio,
            dx      = dx,
            dy      = dy,
            angle   = angle,
            ok      = ok,
        )
        draw_stats_box(canvas, stats, x=RESIZE_W - 240, y=RESIZE_H - 260)

        # Progress bar at bottom
        progress = frame_idx / total
        bar_w    = int(out_w * progress)
        cv2.rectangle(canvas, (0, out_h - 6), (out_w, out_h), (50, 50, 50), -1)
        cv2.rectangle(canvas, (0, out_h - 6), (bar_w, out_h),
                      COLOR_BORDER_OK if ok else COLOR_BORDER_FAIL, -1)

        # Title bar
        title = (f"SCRUM-47 | SurfWatch Motion Detection | "
                 f"Frame {frame_idx}/{total} | "
                 f"OK={total_ok}  FAIL={total_fail}")
        cv2.rectangle(canvas, (0, 0), (out_w, 28), (20, 20, 20), -1)
        cv2.putText(canvas, title, (8, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLOR_TEXT, 1)

        writer.write(canvas)

        # Update prev
        prev_frame = curr_frame
        prev_gray  = curr_gray
        kp_prev    = kp_curr
        des_prev   = des_curr

        frame_idx += 1
        if frame_idx % 100 == 0:
            print(f"  … frame {frame_idx}/{total}  inliers={n_inliers}  "
                  f"ok={total_ok}  fail={total_fail}")

    cap.release()
    writer.release()

    print(f"\n{'='*60}")
    print(f"FRAME-BY-FRAME DETECTION COMPLETE")
    print(f"{'='*60}")
    print(f"  Total frames processed : {frame_idx}")
    print(f"  Successful detections  : {total_ok}")
    print(f"  Failed detections      : {total_fail}")
    print(f"  Success rate           : {total_ok/frame_idx*100:.1f}%")
    print(f"  Output saved           : {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",  default="RipVIS-016.mp4")
    parser.add_argument("--output", default="frame_detection_output.mp4")
    args = parser.parse_args()
    process_video(args.input, args.output)
