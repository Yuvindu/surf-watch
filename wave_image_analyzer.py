"""
wave_image_analyzer.py - SCRUM-54 SurfWatch
Accurately detects wave lines in still images using:
- Adaptive brightness thresholding per image
- Only draws lines where actual foam/waves exist
- No lines on sand or sky
"""

import cv2
import numpy as np
import json
import argparse
from pathlib import Path

SUPPORTED_IMAGES = {'.jpg','.jpeg','.png','.bmp','.tiff','.webp'}
WAVE_COLORS = [
    (0,   220, 255),
    (0,   255, 180),
    (255, 220,   0),
    (180, 255,   0),
    (0,   180, 255),
    (100, 255, 220),
]


def find_wave_region(img):
    """
    Find where ocean/waves actually are by:
    1. Finding brightest foam blobs (breaking waves)
    2. Finding water colour region
    Returns (water_top, water_bot, wave_band_top, wave_band_bot)
    """
    H, W = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hsv  = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # ── Find water region by colour ───────────────────────────────────────────
    water_mask = cv2.inRange(hsv,
                             np.array([80, 15, 20]),
                             np.array([170, 255, 255]))
    foam_thresh = max(int(np.percentile(gray, 92)), 150)
    foam_mask   = (gray > foam_thresh).astype(np.uint8) * 255

    combined = cv2.bitwise_or(water_mask, foam_mask)
    kernel   = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)

    rows = combined.sum(axis=1) / 255
    valid_rows = np.where(rows > W * 0.06)[0]

    if len(valid_rows) == 0:
        return int(H*0.1), int(H*0.75)

    water_top = max(int(valid_rows.min()) - 10, 0)
    water_bot = min(int(valid_rows.max()) + 10, H)
    return water_top, water_bot


def detect_wave_lines(img, water_top, water_bot):
    """
    Detect wave lines ONLY within the water region.
    Uses adaptive threshold to find local brightness peaks (foam crests).
    Returns list of (y, x1, x2) for each detected wave.
    """
    H, W = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Work only in water region
    roi = gray[water_top:water_bot, :]
    roi_h = water_bot - water_top
    if roi_h < 10:
        return []

    # Adaptive brightness threshold — finds foam relative to this image
    blur     = cv2.GaussianBlur(roi, (7, 3), 0)
    adaptive = cv2.adaptiveThreshold(
        blur, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        blockSize=51,
        C=-8
    )

    # Also use global foam threshold
    foam_thresh = max(int(np.percentile(roi, 88)), 130)
    global_foam = (blur > foam_thresh).astype(np.uint8) * 255

    # Combine both
    combined = cv2.bitwise_or(adaptive, global_foam)

    # Clean up
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 3))
    combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel)
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE,
                                cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(9,5)))

    # Find row-wise foam density
    row_density = combined.sum(axis=1) / 255
    if row_density.max() == 0:
        return []
    row_density = row_density / row_density.max()

    # Find peaks (wave crests) — only where there's actual foam
    min_spacing = max(int(roi_h * 0.06), 15)
    foam_threshold = 0.20
    peaks = []
    last  = -min_spacing * 2

    for y in range(len(row_density)):
        if row_density[y] > foam_threshold and y - last >= min_spacing:
            peaks.append(y)
            last = y
        elif peaks and y - last < min_spacing and row_density[y] > row_density[peaks[-1]]:
            peaks[-1] = y
            last = y

    # For each peak, find the actual x extent of the foam
    waves = []
    for py in peaks:
        abs_y = py + water_top

        # Find where there is actual foam on this row (and nearby rows)
        band = combined[max(py-4,0):min(py+5,roi_h), :]
        col_sums = band.sum(axis=0) / 255
        foam_cols = np.where(col_sums > 0.3)[0]

        if len(foam_cols) == 0:
            continue

        # Group foam columns into continuous segments
        segments = []
        seg_start = foam_cols[0]
        seg_end   = foam_cols[0]
        for j in range(1, len(foam_cols)):
            if foam_cols[j] - foam_cols[j-1] < 30:  # allow gap of 30px
                seg_end = foam_cols[j]
            else:
                if seg_end - seg_start > W * 0.04:
                    segments.append((seg_start, seg_end))
                seg_start = foam_cols[j]
                seg_end   = foam_cols[j]
        if seg_end - seg_start > W * 0.04:
            segments.append((seg_start, seg_end))

        for x1, x2 in segments:
            waves.append((abs_y, int(x1), int(x2)))

    # Deduplicate waves within 20px
    deduped = []
    for w in waves:
        too_close = any(abs(w[0]-d[0]) < 20 and
                       not (w[1] > d[2]+50 or w[2] < d[1]-50)
                       for d in deduped)
        if not too_close:
            deduped.append(w)

    return deduped


def classify_danger(waves, water_top, water_bot, H):
    n = len(waves)
    water_h = max(water_bot - water_top, 1)
    if n == 0:
        return "SAFE", (0, 200, 0)
    density = n / (water_h / 100)
    if density < 1.5:
        return "SAFE",      (0, 200,   0)
    elif density < 3.5:
        return "MODERATE",  (0, 200, 200)
    else:
        return "DANGEROUS", (0,   0, 220)


def annotate(img, waves, water_top, water_bot, danger, color):
    H, W = img.shape[:2]
    out  = img.copy()

    # Water boundary
    cv2.line(out, (0, water_top), (W, water_top), (0,220,220), 3, cv2.LINE_AA)
    cv2.line(out, (0, water_bot), (W, water_bot), (0,150,150), 2, cv2.LINE_AA)

    # Wave lines — only where foam actually is
    for i, (y, x1, x2) in enumerate(waves):
        wc = WAVE_COLORS[i % len(WAVE_COLORS)]
        # Thick line along actual foam
        cv2.line(out, (x1, y), (x2, y), wc, 4, cv2.LINE_AA)
        # Small dot at each end
        cv2.circle(out, (x1, y), 5, wc, -1, cv2.LINE_AA)
        cv2.circle(out, (x2, y), 5, wc, -1, cv2.LINE_AA)
        # Label
        label_x = max(x1, 10)
        cv2.putText(out, f"W{i+1}", (label_x, y-12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, wc, 2, cv2.LINE_AA)

    # Danger banner
    bh = 50
    cv2.rectangle(out, (0,0), (W, bh), (0,0,0), -1)
    cv2.rectangle(out, (0,0), (W, bh), color, 3)
    icon = "!!" if danger=="DANGEROUS" else "!" if danger=="MODERATE" else "OK"
    cv2.putText(out,
                f"[{icon}]  {danger}  |  Waves detected: {len(waves)}",
                (14, 34), cv2.FONT_HERSHEY_SIMPLEX,
                0.9, color, 2, cv2.LINE_AA)

    # Legend
    legend = [("Water boundary", (0,220,220)),
              ("Wave crest line", (0,200,255)),
              (f"Status: {danger}", color)]
    lw, lh = 320, len(legend)*34+16
    lx, ly = W-lw-12, H-lh-12
    cv2.rectangle(out, (lx,ly), (lx+lw,ly+lh), (0,0,0), -1)
    cv2.rectangle(out, (lx,ly), (lx+lw,ly+lh), (60,60,60), 1)
    for j, (txt, col) in enumerate(legend):
        ty = ly+22+j*34
        cv2.line(out, (lx+8,ty-5),(lx+42,ty-5), col, 4)
        cv2.putText(out, txt, (lx+52,ty),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                    (255,255,255), 1, cv2.LINE_AA)

    if danger == "DANGEROUS":
        cv2.rectangle(out, (0,0),(W-1,H-1),(0,0,220),10)

    return out


def process_image(img_path, output_dir):
    img = cv2.imread(img_path)
    if img is None:
        print(f"  [WARN] Cannot read: {img_path}"); return {}
    H, W    = img.shape[:2]
    wt, wb  = find_wave_region(img)
    waves   = detect_wave_lines(img, wt, wb)
    danger, color = classify_danger(waves, wt, wb, H)
    out     = annotate(img, waves, wt, wb, danger, color)
    stem    = Path(img_path).stem
    path    = Path(output_dir) / f"wave_{stem}.jpg"
    cv2.imwrite(str(path), out, [cv2.IMWRITE_JPEG_QUALITY, 97])
    print(f"  [{danger}] {Path(img_path).name} | "
          f"waves={len(waves)} | → {path.name}")
    return {"file": img_path, "output": str(path),
            "waves": len(waves), "danger": danger,
            "water_top": wt, "water_bot": wb}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", nargs="+", required=True)
    parser.add_argument("--output-dir", default=".")
    args = parser.parse_args()
    Path(args.output_dir).mkdir(exist_ok=True)
    results = []
    print(f"\n{'='*55}\n  SurfWatch Wave Image Analyzer\n{'='*55}")
    for p in args.image:
        if Path(p).suffix.lower() in SUPPORTED_IMAGES:
            r = process_image(p, args.output_dir)
            if r: results.append(r)
    with open(Path(args.output_dir)/"wave_image_report.json","w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[OK] Report → wave_image_report.json")
