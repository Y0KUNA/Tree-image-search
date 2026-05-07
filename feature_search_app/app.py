from __future__ import annotations

import os
from pathlib import Path
from typing import List
import sys

from flask import Flask, render_template, request, redirect, url_for, send_from_directory, abort
from werkzeug.utils import safe_join
import numpy as np
import pandas as pd

import sys

# ensure project root is on sys.path so local modules can be imported
APP_ROOT = Path(__file__).parent
project_root = str(APP_ROOT.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from feature_extract.extract_features import process_image


DATA_CLEAN = (APP_ROOT / '..' / 'Data_clean').resolve()
FEATURES_CSV = (APP_ROOT / '..' / 'feature_extract' / 'features.csv').resolve()

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = str(APP_ROOT / 'uploads')
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)


def load_features() -> pd.DataFrame:
    if not FEATURES_CSV.exists():
        return pd.DataFrame()
    df = pd.read_csv(FEATURES_CSV)
    return df


def feature_vector_from_row(row: pd.Series) -> np.ndarray:
    # exclude file_name and folder
    vals = row.drop(labels=['file_name', 'folder']).values.astype(float)
    return vals


def find_top_k(query_vec: np.ndarray, df: pd.DataFrame, k: int = 5) -> List[dict]:
    feats = df.apply(feature_vector_from_row, axis=1)
    # stack into matrix
    mat = np.stack(feats.values)
    # compute euclidean distance
    dists = np.linalg.norm(mat - query_vec, axis=1)
    # exclude exact-zero distances (likely same image). use small epsilon to avoid float issues
    eps = 1e-9
    valid_idx = np.where(dists > eps)[0]
    if valid_idx.size == 0:
        return []
    # sort only the valid indices by their distances
    sorted_valid = valid_idx[np.argsort(dists[valid_idx])]
    idx = sorted_valid[:k]
    results = []
    for i in idx:
        r = df.iloc[int(i)]
        results.append({
            'file_name': r['file_name'],
            'folder': r['folder'],
            'distance': float(dists[i]),
            'path': str(DATA_CLEAN.joinpath(r['folder'], r['file_name']))
        })
    return results


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'image' not in request.files:
            return redirect(request.url)
        f = request.files['image']
        if f.filename == '':
            return redirect(request.url)
        dest = Path(app.config['UPLOAD_FOLDER']) / f.filename
        f.save(str(dest))

        # extract features of uploaded image using same routine
        feats = process_image(dest)
        # build vector in same order as CSV
        df = load_features()
        if df.empty:
            return 'No features database found', 500
        # order columns
        cols = [c for c in df.columns if c not in ('file_name', 'folder')]
        query_vec = np.array([feats[c] for c in cols], dtype=float)

        results = find_top_k(query_vec, df, k=5)
        # adjust result paths to be relative (folder/file)
        for r in results:
            r['relpath'] = f"{r['folder']}/{r['file_name']}"
        return render_template('results.html', results=results, query_image=('uploads', dest.name))

    return render_template('index.html')


@app.route('/uploads/<path:filename>')
def uploaded_file(filename: str):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


@app.route('/data_clean/<path:filename>')
def data_clean_file(filename: str):
    # serve files from Data_clean safely
    base = str(DATA_CLEAN)
    full_path = safe_join(base, filename)
    if not full_path:
        abort(404)
    return send_from_directory(base, filename)


if __name__ == '__main__':
    app.run(debug=True, port=5000)
