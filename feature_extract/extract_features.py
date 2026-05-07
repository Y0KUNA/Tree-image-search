#!/usr/bin/env python3
"""Extract image features from a directory of normalized images and save to CSV.

Outputs a CSV with columns: file_name, folder, <features...>

Features extracted:
- RGB histogram (8 bins per channel -> 24 values)
- Edge count and edge ratio (Canny)
- Shape features from largest contour (height/width ratio, area, convexity, defect_count)
- LBP texture histogram (uniform LBP)
- Green area pixel count and ratio (HSV green mask)

"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import pandas as pd
from skimage.feature import local_binary_pattern
from tqdm import tqdm


def rgb_histogram(img: np.ndarray, bins: int = 8) -> List[float]:
    # img assumed BGR (OpenCV). Convert to RGB order
    chans = cv2.split(img)
    feats = []
    total = img.shape[0] * img.shape[1]
    for ch in chans[::-1]:  # convert BGR->RGB by reversing
        hist = cv2.calcHist([ch], [0], None, [bins], [0, 256]).flatten()
        hist = hist / (hist.sum() + 1e-9)
        feats.extend(hist.tolist())
    return feats


def edge_features(gray: np.ndarray) -> Tuple[int, float]:
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)
    edge_count = int(np.count_nonzero(edges))
    ratio = edge_count / (gray.shape[0] * gray.shape[1])
    return edge_count, ratio


def shape_features(img: np.ndarray, gray: np.ndarray) -> Tuple[float, int, float, int]:
    # threshold to separate foreground (assume white-ish background)
    # use Otsu to be adaptive
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # find contours and pick the largest
    contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0.0, 0, 0.0, 0
    c = max(contours, key=cv2.contourArea)
    area = float(cv2.contourArea(c))
    x, y, w, h = cv2.boundingRect(c)
    hw_ratio = float(h) / float(w) if w > 0 else 0.0
    hull = cv2.convexHull(c)
    hull_area = float(cv2.contourArea(hull)) if hull is not None else 0.0
    convexity = (hull_area / area) if area > 0 else 0.0

    # convexity defects (approximate branch count)
    defects_count = 0
    try:
        hull_idx = cv2.convexHull(c, returnPoints=False)
        if hull_idx is not None and len(hull_idx) > 2:
            defects = cv2.convexityDefects(c, hull_idx)
            if defects is not None:
                defects_count = int(defects.shape[0])
    except Exception:
        defects_count = 0

    return hw_ratio, int(area), convexity, defects_count


def lbp_features(gray: np.ndarray, P: int = 8, R: int = 1, bins: int = 10) -> List[float]:
    lbp = local_binary_pattern(gray, P, R, method='uniform')
    # compute histogram
    (hist, _) = np.histogram(lbp.ravel(), bins=bins, range=(0, bins))
    hist = hist.astype('float')
    hist /= (hist.sum() + 1e-9)
    return hist.tolist()


def green_area_ratio(img: np.ndarray) -> Tuple[int, float]:
    # img is BGR
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # typical green range
    lower = np.array([35, 40, 40])
    upper = np.array([85, 255, 255])
    mask = cv2.inRange(hsv, lower, upper)
    green_count = int(np.count_nonzero(mask))
    ratio = green_count / (img.shape[0] * img.shape[1])
    return green_count, ratio


def process_image(path: Path) -> Dict[str, object]:
    img = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f'Could not open image: {path}')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    feats: Dict[str, object] = {}

    # RGB histogram
    rgb_hist = rgb_histogram(img, bins=8)
    for i, v in enumerate(rgb_hist, 1):
        feats[f'rgb_hist_{i}'] = float(v)

    # edges
    edge_count, edge_ratio = edge_features(gray)
    feats['edge_count'] = int(edge_count)
    feats['edge_ratio'] = float(edge_ratio)

    # shape
    hw_ratio, area, convexity, defects = shape_features(img, gray)
    feats['hw_ratio'] = float(hw_ratio)
    feats['contour_area'] = int(area)
    feats['convexity'] = float(convexity)
    feats['defect_count'] = int(defects)

    # texture LBP
    lbp_hist = lbp_features(gray, P=8, R=1, bins=10)
    for i, v in enumerate(lbp_hist, 1):
        feats[f'lbp_{i}'] = float(v)

    # green area
    green_count, green_ratio = green_area_ratio(img)
    feats['green_count'] = int(green_count)
    feats['green_ratio'] = float(green_ratio)

    return feats


def gather_images(src: Path) -> List[Tuple[Path, str]]:
    items: List[Tuple[Path, str]] = []
    for root, dirs, files in os.walk(src):
        rel = Path(root).relative_to(src)
        folder = rel.parts[0] if len(rel.parts) >= 1 else ''
        for f in files:
            p = Path(root) / f
            ext = p.suffix.lower()
            if ext in {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}:
                items.append((p, folder))
    return items


def main() -> None:
    parser = argparse.ArgumentParser(description='Extract image features from normalized images and save CSV')
    parser.add_argument('--src', required=True, help='source directory (Data_clean)')
    parser.add_argument('--out', required=True, help='output CSV path')
    args = parser.parse_args()

    src = Path(args.src)
    out = Path(args.out)
    rows: List[Dict[str, object]] = []

    images = gather_images(src)
    if not images:
        print('No images found in', src)
        return

    for p, folder in tqdm(images, desc='Processing'):
        try:
            feats = process_image(p)
            row = {'file_name': p.name, 'folder': folder}
            row.update(feats)
            rows.append(row)
        except Exception as e:
            print(f'Failed {p}: {e}')

    if rows:
        df = pd.DataFrame(rows)
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out, index=False)
        print(f'Wrote features for {len(df)} images to {out}')


if __name__ == '__main__':
    main()
