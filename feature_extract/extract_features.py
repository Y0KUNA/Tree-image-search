
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import psycopg2
from psycopg2.extras import execute_values
from skimage.feature import local_binary_pattern
import pandas as pd
from tqdm import tqdm

# ── DB Config ──────────────────────────────────────────────────────────────────
DB_CONFIG = {
    'host':     'localhost',
    'port':     5432,
    'dbname':   'Tree_image_metadata',   # ← đổi
    'user':     'postgres',   # ← đổi
    'password': '12345678',   # ← đổi
}

INSERT_SQL = """
INSERT INTO plant_features (
    file_name, folder,
    rgb_hist_1, rgb_hist_2, rgb_hist_3, rgb_hist_4, rgb_hist_5, rgb_hist_6,
    rgb_hist_7, rgb_hist_8, rgb_hist_9, rgb_hist_10, rgb_hist_11, rgb_hist_12,
    rgb_hist_13, rgb_hist_14, rgb_hist_15, rgb_hist_16, rgb_hist_17, rgb_hist_18,
    rgb_hist_19, rgb_hist_20, rgb_hist_21, rgb_hist_22, rgb_hist_23, rgb_hist_24,
    edge_count, edge_ratio,
    hw_ratio, contour_area, convexity, defect_count,
    lbp_1, lbp_2, lbp_3, lbp_4, lbp_5, lbp_6, lbp_7, lbp_8, lbp_9, lbp_10,
    hu_1, hu_2, hu_3, hu_4, hu_5, hu_6, hu_7,
    green_count, green_ratio
 ) VALUES %s
ON CONFLICT (file_name, folder) DO UPDATE SET
    folder       = EXCLUDED.folder,
    rgb_hist_1   = EXCLUDED.rgb_hist_1,  rgb_hist_2  = EXCLUDED.rgb_hist_2,
    rgb_hist_3   = EXCLUDED.rgb_hist_3,  rgb_hist_4  = EXCLUDED.rgb_hist_4,
    rgb_hist_5   = EXCLUDED.rgb_hist_5,  rgb_hist_6  = EXCLUDED.rgb_hist_6,
    rgb_hist_7   = EXCLUDED.rgb_hist_7,  rgb_hist_8  = EXCLUDED.rgb_hist_8,
    rgb_hist_9   = EXCLUDED.rgb_hist_9,  rgb_hist_10 = EXCLUDED.rgb_hist_10,
    rgb_hist_11  = EXCLUDED.rgb_hist_11, rgb_hist_12 = EXCLUDED.rgb_hist_12,
    rgb_hist_13  = EXCLUDED.rgb_hist_13, rgb_hist_14 = EXCLUDED.rgb_hist_14,
    rgb_hist_15  = EXCLUDED.rgb_hist_15, rgb_hist_16 = EXCLUDED.rgb_hist_16,
    rgb_hist_17  = EXCLUDED.rgb_hist_17, rgb_hist_18 = EXCLUDED.rgb_hist_18,
    rgb_hist_19  = EXCLUDED.rgb_hist_19, rgb_hist_20 = EXCLUDED.rgb_hist_20,
    rgb_hist_21  = EXCLUDED.rgb_hist_21, rgb_hist_22 = EXCLUDED.rgb_hist_22,
    rgb_hist_23  = EXCLUDED.rgb_hist_23, rgb_hist_24 = EXCLUDED.rgb_hist_24,
    edge_count   = EXCLUDED.edge_count,  edge_ratio  = EXCLUDED.edge_ratio,
    hw_ratio     = EXCLUDED.hw_ratio,    contour_area = EXCLUDED.contour_area,
    convexity    = EXCLUDED.convexity,   defect_count = EXCLUDED.defect_count,
    lbp_1        = EXCLUDED.lbp_1,  lbp_2  = EXCLUDED.lbp_2,
    lbp_3        = EXCLUDED.lbp_3,  lbp_4  = EXCLUDED.lbp_4,
    lbp_5        = EXCLUDED.lbp_5,  lbp_6  = EXCLUDED.lbp_6,
    lbp_7        = EXCLUDED.lbp_7,  lbp_8  = EXCLUDED.lbp_8,
    lbp_9        = EXCLUDED.lbp_9,  lbp_10 = EXCLUDED.lbp_10,
    hu_1         = EXCLUDED.hu_1,   hu_2   = EXCLUDED.hu_2,
    hu_3         = EXCLUDED.hu_3,   hu_4   = EXCLUDED.hu_4,
    hu_5         = EXCLUDED.hu_5,   hu_6   = EXCLUDED.hu_6,
    hu_7         = EXCLUDED.hu_7,
    green_count  = EXCLUDED.green_count, green_ratio = EXCLUDED.green_ratio;
"""


def row_to_tuple(file_name: str, folder: str, feats: Dict) -> tuple:
    return (
        file_name, folder,
        feats['rgb_hist_1'],  feats['rgb_hist_2'],  feats['rgb_hist_3'],
        feats['rgb_hist_4'],  feats['rgb_hist_5'],  feats['rgb_hist_6'],
        feats['rgb_hist_7'],  feats['rgb_hist_8'],  feats['rgb_hist_9'],
        feats['rgb_hist_10'], feats['rgb_hist_11'], feats['rgb_hist_12'],
        feats['rgb_hist_13'], feats['rgb_hist_14'], feats['rgb_hist_15'],
        feats['rgb_hist_16'], feats['rgb_hist_17'], feats['rgb_hist_18'],
        feats['rgb_hist_19'], feats['rgb_hist_20'], feats['rgb_hist_21'],
        feats['rgb_hist_22'], feats['rgb_hist_23'], feats['rgb_hist_24'],
        feats['edge_count'],  feats['edge_ratio'],
        feats['hw_ratio'],    feats['contour_area'],
        feats['convexity'],   feats['defect_count'],
        feats['lbp_1'],  feats['lbp_2'],  feats['lbp_3'],
        feats['lbp_4'],  feats['lbp_5'],  feats['lbp_6'],
        feats['lbp_7'],  feats['lbp_8'],  feats['lbp_9'],  feats['lbp_10'],
        feats['hu_1'], feats['hu_2'], feats['hu_3'], feats['hu_4'], feats['hu_5'], feats['hu_6'], feats['hu_7'],
        feats['green_count'], feats['green_ratio'],
    )


# ── Feature extraction ─────────────────────────────────────────────────────────

TARGET_SIZE = (1080, 1080)


def rgb_histogram(img: np.ndarray, bins: int = 8) -> List[float]:
    chans = cv2.split(img)
    feats = []
    for ch in chans[::-1]:
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
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
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


def hu_moments_from_mask(gray: np.ndarray) -> List[float]:
    """Compute Hu moments from the largest contour mask. Returns 7 Hu moments (log-scaled)."""
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return [0.0] * 7
    c = max(contours, key=cv2.contourArea)
    mask = np.zeros_like(gray)
    cv2.drawContours(mask, [c], -1, color=255, thickness=-1)
    moments = cv2.moments(mask)
    hu = cv2.HuMoments(moments).flatten()
    # log scale for numerical stability/sign handling
    hu_log = []
    for v in hu:
        if v == 0:
            hu_log.append(0.0)
        else:
            hu_log.append(-1.0 * float(np.sign(v) * np.log10(abs(v))))
    # ensure length 7
    while len(hu_log) < 7:
        hu_log.append(0.0)
    return hu_log


def lbp_features(gray: np.ndarray, P: int = 8, R: int = 1, bins: int = 10) -> List[float]:
    lbp = local_binary_pattern(gray, P, R, method='uniform')
    hist, _ = np.histogram(lbp.ravel(), bins=bins, range=(0, bins))
    hist = hist.astype('float')
    hist /= (hist.sum() + 1e-9)
    return hist.tolist()


def green_area_ratio(img: np.ndarray) -> Tuple[int, float]:
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([35, 40, 40]), np.array([85, 255, 255]))
    green_count = int(np.count_nonzero(mask))
    return green_count, green_count / (img.shape[0] * img.shape[1])


def process_image(path: Path) -> Dict[str, object]:
    img = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f'Could not open image: {path}')
    img = cv2.resize(img, TARGET_SIZE, interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    feats: Dict[str, object] = {}
    for i, v in enumerate(rgb_histogram(img, bins=8), 1):
        feats[f'rgb_hist_{i}'] = float(v)
    edge_count, edge_ratio = edge_features(gray)
    feats['edge_count'] = int(edge_count)
    feats['edge_ratio'] = float(edge_ratio)
    hw_ratio, area, convexity, defects = shape_features(img, gray)
    feats['hw_ratio'] = float(hw_ratio)
    feats['contour_area'] = int(area)
    feats['convexity'] = float(convexity)
    feats['defect_count'] = int(defects)
    for i, v in enumerate(lbp_features(gray), 1):
        feats[f'lbp_{i}'] = float(v)
    # Hu moments
    hu = hu_moments_from_mask(gray)
    for i, v in enumerate(hu, 1):
        feats[f'hu_{i}'] = float(v)
    green_count, green_ratio = green_area_ratio(img)
    feats['green_count'] = int(green_count)
    feats['green_ratio'] = float(green_ratio)
    return feats


def gather_images(src: Path) -> List[Tuple[Path, str]]:
    items: List[Tuple[Path, str]] = []
    for root, _, files in os.walk(src):
        rel = Path(root).relative_to(src)
        folder = rel.parts[0] if len(rel.parts) >= 1 else ''
        for f in files:
            p = Path(root) / f
            if p.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}:
                items.append((p, folder))
    return items


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description='Extract features and insert into PostgreSQL')
    parser.add_argument('--src', required=True, help='Source directory (Data_clean)')
    parser.add_argument('--batch', type=int, default=50, help='Batch size khi insert (default: 50)')
    args = parser.parse_args()

    src = Path(args.src)
    images = gather_images(src)
    if not images:
        print('Không tìm thấy ảnh trong', src)
        return

    print(f'Tìm thấy {len(images)} ảnh — bắt đầu trích xuất...')

    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()

    batch: List[tuple] = []
    success, failed = 0, 0
    rows: List[Dict[str, object]] = []

    for p, folder in tqdm(images, desc='Processing'):
        try:
            feats = process_image(p)
            batch.append(row_to_tuple(p.name, folder, feats))
            # also collect a dict for CSV output
            row = {'file_name': p.name, 'folder': folder}
            row.update(feats)
            rows.append(row)
        except Exception as e:
            print(f'\nLỗi {p.name}: {e}')
            failed += 1
            continue

        # Insert theo batch
        if len(batch) >= args.batch:
            execute_values(cur, INSERT_SQL, batch)
            conn.commit()
            success += len(batch)
            batch.clear()

    # Insert phần còn lại
    if batch:
        execute_values(cur, INSERT_SQL, batch)
        conn.commit()
        success += len(batch)

    cur.close()
    conn.close()

    # write features.csv (overwrite)
    try:
        out_csv = Path(__file__).parent / 'features.csv'
        if rows:
            df_out = pd.DataFrame(rows)
            # order columns: file_name, folder, then sorted feature columns
            cols = [c for c in df_out.columns if c not in ('file_name', 'folder')]
            cols_sorted = sorted(cols)
            df_out = df_out[['file_name', 'folder', *cols_sorted]]
            df_out.to_csv(out_csv, index=False)
            print(f'Wrote features CSV to {out_csv}')
        else:
            print('No rows to write to CSV')
    except Exception as e:
        print(f'Failed to write features.csv: {e}')

    print(f'\nHoàn thành: {success} ảnh inserted/updated | {failed} lỗi')


if __name__ == '__main__':
    main()