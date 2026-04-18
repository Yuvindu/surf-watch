"""
image_frame_detector.py
───────────────────────
SCRUM-47 · SurfWatch – Image Frame Motion Detection
Runs ORB keypoint detection and motion analysis on still images
just like the video frame detector, but for .jpg .jpeg .png .bmp .tiff .webp

Usage:
    # Single image
    python image_frame_detector.py --input surf_real_1.jpg

    # Multiple images (compares each to the next)
    python image_frame_detector.py --input surf_real_1.jpg surf_real_2.jpg surf_real_3.jpg

    # Whole folder of images
    python image_frame_detector.py --folder ./my_images/

Output:
    detected_<filename>.jpg  – annotated image with keypoints + stats
    image_detection_report.json – full results
"""

import argparse
import json
import cv2
import numpy as np
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────
ORB_N_FEATURES   = 1500
MATCH_RATIO      = 0.72
MIN_MATCHES      = 8
RANSAC_THRESH    = 4.0
SUPPORTED_IMAGES = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}

# Colours (BGR)
C_WHITE  = (255, 255, 255)
C_BLACK  = (0,   0,   0)
C_GREEN  = (0,   220, 0)
C_RED    = (0,   0,   220)
C_YELLOW = (0,   220, 220)
C_TEAL   = (180, 180, 0)
C_BLUE   = (220, 100, 0)
C_DARK   = (20,  20,  20)
C_PINK   = (180, 80,  180)


def draw_text_bg(img, text, pos, scale=0.55, thickness=1,
                 fg=C_WHITE, bg=C_DARK, padding=4):
    (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    x, y = pos
    cv2.rectangle(img, (x - padding, y - h - padding),
                  (x + w + padding, y + padding), bg, -1)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                scale, fg, thickness, cv2.LINE_AA)


def detect_single_image(img_path: str, orb, output_dir: Path) -> dict:
    """Detect and annotate keypoints on a single image."""
    img = cv2.imread(img_path)
    if img is None:
        print(f"  [WARN] Cannot read: {img_path}")
        return {}

    H, W = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kp, des = orb.detectAndCompute(gray, None)

    # Draw keypoints
    annotated = img.copy()
    for k in kp:
        x, y = int(k.pt[0]), int(k.pt[1])
        r = max(2, int(k.size / 4))
        # Colour by response strength
        strength = min(k.response / 50.0, 1.0)
        color = (
            int(255 * (1 - strength)),
            int(255 * strength),
            int(128 * strength)
        )
        cv2.circle(annotated, (x, y), r, color, 1, cv2.LINE_AA)
        cv2.circle(annotated, (x, y), 2, C_YELLOW, -1)

        # Draw orientation line
        angle_rad = np.radians(k.angle)
        ex = int(x + r * np.cos(angle_rad))
        ey = int(y + r * np.sin(angle_rad))
        cv2.line(annotated, (x, y), (ex, ey), C_TEAL, 1, cv2.LINE_AA)

    # Stats box
    name = Path(img_path).name
    lines = [
        f"File     : {name}",
        f"Size     : {W} x {H} px",
        f"Keypoints: {len(kp)}",
        f"Descriptor: ORB ({ORB_N_FEATURES} max)",
        f"Status   : {'OK' if len(kp) >= MIN_MATCHES else 'TOO FEW KP'}",
    ]
    box_x, box_y = 10, 10
    box_w, box_h = 300, len(lines) * 22 + 16
    overlay = annotated.copy()
    cv2.rectangle(overlay, (box_x, box_y),
                  (box_x + box_w, box_y + box_h), C_DARK, -1)
    cv2.addWeighted(overlay, 0.75, annotated, 0.25, 0, annotated)
    cv2.rectangle(annotated, (box_x, box_y),
                  (box_x + box_w, box_y + box_h), C_GREEN, 1)

    for i, line in enumerate(lines):
        ty = box_y + 16 + i * 22
        color = C_GREEN if 'OK' in line else C_RED if 'FEW' in line else C_WHITE
        cv2.putText(annotated, line, (box_x + 8, ty),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.46, color, 1, cv2.LINE_AA)

    # Title bar
    cv2.rectangle(annotated, (0, H - 28), (W, H), C_DARK, -1)
    cv2.putText(annotated,
                f"SurfWatch | SCRUM-47 | ORB Keypoint Detection | {name}",
                (8, H - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, C_WHITE, 1, cv2.LINE_AA)

    # Save
    stem = Path(img_path).stem
    out_path = output_dir / f"detected_{stem}.jpg"
    cv2.imwrite(str(out_path), annotated)
    print(f"  [OK] {name} → {len(kp)} keypoints → {out_path.name}")

    return {
        "file": img_path,
        "width": W,
        "height": H,
        "keypoints": len(kp),
        "sufficient_for_matching": len(kp) >= MIN_MATCHES,
        "output": str(out_path),
    }


def detect_image_pair(img1_path: str, img2_path: str,
                      orb, matcher, output_dir: Path) -> dict:
    """Compare two images: detect keypoints + match + estimate transform."""
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)
    if img1 is None or img2 is None:
        print(f"  [WARN] Cannot read image pair")
        return {}

    # Resize img2 to match img1 if needed
    h1, w1 = img1.shape[:2]
    img2 = cv2.resize(img2, (w1, h1))

    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    kp1, des1 = orb.detectAndCompute(gray1, None)
    kp2, des2 = orb.detectAndCompute(gray2, None)

    good, inliers_mask = [], None
    H_mat, dx, dy, angle = None, 0.0, 0.0, 0.0
    ok = False

    if (des1 is not None and des2 is not None
            and len(kp1) >= MIN_MATCHES and len(kp2) >= MIN_MATCHES):
        raw = matcher.knnMatch(des1, des2, k=2)
        good = [m for m, n in raw if m.distance < MATCH_RATIO * n.distance]

        if len(good) >= MIN_MATCHES:
            pts1 = np.float32([kp1[m.queryIdx].pt for m in good])
            pts2 = np.float32([kp2[m.trainIdx].pt for m in good])
            H_mat, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, RANSAC_THRESH)
            if H_mat is not None and mask is not None:
                inliers_mask = mask.ravel()
                n_inliers = int(inliers_mask.sum())
                ok = n_inliers >= MIN_MATCHES
                if ok:
                    dx    = float(H_mat[0, 2])
                    dy    = float(H_mat[1, 2])
                    angle = float(np.degrees(np.arctan2(H_mat[1, 0], H_mat[0, 0])))

    n_inliers = int(inliers_mask.sum()) if inliers_mask is not None else 0

    # ── Build side-by-side comparison ────────────────────────────────────────
    left  = img1.copy()
    right = img2.copy()

    # Draw keypoints on left
    for k in kp1:
        cv2.circle(left, (int(k.pt[0]), int(k.pt[1])), 3, C_YELLOW, 1)

    # Draw inlier/outlier matches on right
    if good and inliers_mask is not None:
        for i, m in enumerate(good):
            pt1 = tuple(map(int, kp1[m.queryIdx].pt))
            pt2 = tuple(map(int, kp2[m.trainIdx].pt))
            color = C_GREEN if inliers_mask[i] else C_RED
            thick = 2 if inliers_mask[i] else 1
            cv2.line(right, pt1, pt2, color, thick, cv2.LINE_AA)

    # Labels
    draw_text_bg(left, f"Image 1: {Path(img1_path).name}", (10, 28), 0.55, 1, C_YELLOW)
    draw_text_bg(left, f"KP: {len(kp1)}", (10, 52), 0.5, 1, C_WHITE)
    draw_text_bg(right, f"Image 2: {Path(img2_path).name}", (10, 28), 0.55, 1, C_YELLOW)
    draw_text_bg(right, f"Inliers: {n_inliers}  Outliers: {len(good)-n_inliers}",
                 (10, 52), 0.5, 1, C_GREEN if ok else C_RED)

    # Stats box on right
    stats = [
        f"Matches : {len(good)}",
        f"Inliers : {n_inliers}",
        f"dX      : {dx:+.1f} px",
        f"dY      : {dy:+.1f} px",
        f"Angle   : {angle:+.2f} deg",
        f"Status  : {'MATCH OK' if ok else 'NO MATCH'}",
    ]
    bx, by = 10, h1 - len(stats) * 22 - 20
    overlay = right.copy()
    cv2.rectangle(overlay, (bx, by),
                  (bx + 220, by + len(stats) * 22 + 12), C_DARK, -1)
    cv2.addWeighted(overlay, 0.75, right, 0.25, 0, right)
    cv2.rectangle(right, (bx, by),
                  (bx + 220, by + len(stats) * 22 + 12),
                  C_GREEN if ok else C_RED, 1)
    for i, s in enumerate(stats):
        c = C_GREEN if ('OK' in s) else C_RED if ('NO' in s) else C_WHITE
        cv2.putText(right, s, (bx + 8, by + 16 + i * 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.46, c, 1, cv2.LINE_AA)

    canvas = np.hstack([left, right])

    # Title bar
    title = (f"SurfWatch | SCRUM-47 | Image Pair Matching | "
             f"{'SUCCESS' if ok else 'FAILED'}")
    cv2.rectangle(canvas, (0, 0), (w1 * 2, 22), C_DARK, -1)
    cv2.putText(canvas, title, (8, 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, C_WHITE, 1, cv2.LINE_AA)

    # Save
    stem1 = Path(img1_path).stem
    stem2 = Path(img2_path).stem
    out_path = output_dir / f"match_{stem1}_vs_{stem2}.jpg"
    cv2.imwrite(str(out_path), canvas)
    print(f"  [OK] {Path(img1_path).name} vs {Path(img2_path).name} "
          f"→ inliers={n_inliers} → {out_path.name}")

    return {
        "image1": img1_path,
        "image2": img2_path,
        "kp1": len(kp1),
        "kp2": len(kp2),
        "good_matches": len(good),
        "inliers": n_inliers,
        "dx_px": round(dx, 2),
        "dy_px": round(dy, 2),
        "angle_deg": round(angle, 3),
        "success": ok,
        "output": str(out_path),
    }


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="SurfWatch Image Frame Detector — SCRUM-47"
    )
    parser.add_argument("--input", nargs="+",
                        help="One or more image files")
    parser.add_argument("--folder",
                        help="Folder containing images to process")
    parser.add_argument("--output-dir", default=".",
                        help="Output directory (default: current folder)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    # Collect images
    images = []
    if args.folder:
        folder = Path(args.folder)
        images = sorted([
            str(p) for p in folder.iterdir()
            if p.suffix.lower() in SUPPORTED_IMAGES
        ])
        print(f"[INFO] Found {len(images)} images in {folder}")
    elif args.input:
        images = args.input
    else:
        # Default: use all supported images in current folder
        images = sorted([
            str(p) for p in Path(".").iterdir()
            if p.suffix.lower() in SUPPORTED_IMAGES
        ])
        print(f"[INFO] Found {len(images)} images in current folder")

    if not images:
        print("[ERROR] No images found! Use --input or --folder")
        return

    print(f"\n{'='*60}")
    print(f"  SurfWatch Image Frame Detector | SCRUM-47")
    print(f"{'='*60}")
    print(f"  Images     : {len(images)}")
    print(f"  Formats    : {SUPPORTED_IMAGES}")
    print(f"  Output dir : {output_dir}")
    print(f"{'='*60}\n")

    orb     = cv2.ORB_create(nfeatures=ORB_N_FEATURES, fastThreshold=10)
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    results = {"single_image_results": [], "pair_results": []}

    # ── Single image detection ────────────────────────────────────────────────
    print("[STEP 1] Single image keypoint detection:")
    for img_path in images:
        suffix = Path(img_path).suffix.lower()
        if suffix not in SUPPORTED_IMAGES:
            print(f"  [SKIP] Unsupported: {img_path}")
            continue
        r = detect_single_image(img_path, orb, output_dir)
        if r:
            results["single_image_results"].append(r)

    # ── Pair matching ─────────────────────────────────────────────────────────
    if len(images) >= 2:
        print(f"\n[STEP 2] Pairwise image matching ({len(images)-1} pairs):")
        for i in range(len(images) - 1):
            suffix1 = Path(images[i]).suffix.lower()
            suffix2 = Path(images[i+1]).suffix.lower()
            if suffix1 not in SUPPORTED_IMAGES or suffix2 not in SUPPORTED_IMAGES:
                continue
            r = detect_image_pair(images[i], images[i+1], orb, matcher, output_dir)
            if r:
                results["pair_results"].append(r)

    # ── Summary ───────────────────────────────────────────────────────────────
    n_single  = len(results["single_image_results"])
    n_pairs   = len(results["pair_results"])
    n_success = sum(1 for r in results["pair_results"] if r.get("success"))

    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    print(f"  Images processed : {n_single}")
    print(f"  Pairs matched    : {n_pairs}")
    print(f"  Successful pairs : {n_success}/{n_pairs}")
    if n_pairs > 0:
        print(f"  Success rate     : {n_success/n_pairs*100:.1f}%")
    print(f"  Output folder    : {output_dir}")

    report_path = output_dir / "image_detection_report.json"
    with open(report_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[OK] Report saved → {report_path}")


if __name__ == "__main__":
    main()
