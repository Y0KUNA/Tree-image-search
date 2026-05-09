from __future__ import annotations

import sys
import threading
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import psycopg2
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
import customtkinter as ctk
from tkinter import filedialog, messagebox

APP_ROOT = Path(__file__).parent
project_root = str(APP_ROOT.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from feature_extract.extract_features import process_image

DATA_CLEAN = (APP_ROOT / '..' / 'Data_clean').resolve()

DB_CONFIG = {
    'host':     'localhost',
    'port':     5432,
    'dbname':   'Tree_image_metadata',   # ← đổi
    'user':     'postgres',   # ← đổi
    'password': '12345678',   # ← đổi
}

# ── Appearance ─────────────────────────────────────────────────────────────────
ctk.set_appearance_mode('light')
ctk.set_default_color_theme('green')

# ── Feature helpers ────────────────────────────────────────────────────────────

def load_features() -> pd.DataFrame:
    """Tải toàn bộ dữ liệu từ PostgreSQL về DataFrame."""
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        df = pd.read_sql('SELECT * FROM plant_features ORDER BY file_name', conn)
        conn.close()
        return df
    except Exception as e:
        messagebox.showerror('Lỗi kết nối DB', f'Không thể tải dữ liệu từ database:\n{e}')
        return pd.DataFrame()


def get_feature_cols(df: pd.DataFrame) -> list:
    return [c for c in df.columns if c not in ('id', 'file_name', 'folder')]


def find_top_k(query_vec: np.ndarray, df: pd.DataFrame, k: int = 5) -> List[dict]:
    cols = get_feature_cols(df)
    mat = df[cols].values.astype(float)

    scaler = StandardScaler()
    mat_scaled = scaler.fit_transform(mat)
    query_scaled = scaler.transform(query_vec.reshape(1, -1))

    sims = cosine_similarity(query_scaled, mat_scaled)[0]

    eps = 1e-9
    valid_idx = np.where(sims < 1.0 - eps)[0]
    if valid_idx.size == 0:
        valid_idx = np.arange(len(sims))

    sorted_valid = valid_idx[np.argsort(-sims[valid_idx])]
    top_idx = sorted_valid[:k]

    results = []
    for i in top_idx:
        r = df.iloc[int(i)]
        results.append({
            'file_name': r['file_name'],
            'folder':    r['folder'],
            'similarity': round(float(sims[i]) * 100, 2),
            'path': DATA_CLEAN / r['folder'] / r['file_name'],
        })
    return results


def build_query_vec(dest: Path, df: pd.DataFrame) -> np.ndarray:
    cols = get_feature_cols(df)
    matched = df[df['file_name'] == dest.name]
    if not matched.empty:
        return matched.iloc[0][cols].values.astype(float)
    feats = process_image(dest)
    return np.array([feats[c] for c in cols], dtype=float)


# ── GUI ────────────────────────────────────────────────────────────────────────

THUMB_QUERY  = (220, 220)
THUMB_RESULT = (160, 160)


class ResultCard(ctk.CTkFrame):
    def __init__(self, master, rank: int, result: dict, **kwargs):
        super().__init__(master, corner_radius=10, fg_color='white',
                         border_width=1, border_color='#e0e0e0', **kwargs)

        try:
            img = Image.open(result['path']).convert('RGB')
            img.thumbnail(THUMB_RESULT, Image.LANCZOS)
            ctk_img = ctk.CTkImage(light_image=img, size=THUMB_RESULT)
            ctk.CTkLabel(self, image=ctk_img, text='').pack(side='left', padx=12, pady=12)
        except Exception:
            ctk.CTkLabel(self, text='[ảnh lỗi]', width=THUMB_RESULT[0]).pack(side='left', padx=12)

        info = ctk.CTkFrame(self, fg_color='transparent')
        info.pack(side='left', fill='both', expand=True, padx=(0, 16), pady=12)

        ctk.CTkLabel(info, text=f'#{rank}  {result["file_name"]}',
                     font=ctk.CTkFont(size=13, weight='bold'),
                     anchor='w').pack(fill='x')

        ctk.CTkLabel(info, text=f'📁  Nhóm: {result["folder"]}',
                     font=ctk.CTkFont(size=11), text_color='#666',
                     anchor='w').pack(fill='x', pady=(2, 8))

        sim = result['similarity']
        ctk.CTkLabel(info, text=f'Độ tương đồng: {sim}%',
                     font=ctk.CTkFont(size=11, weight='bold'),
                     text_color='#2e7d32', anchor='w').pack(fill='x')

        bar = ctk.CTkProgressBar(info, progress_color='#2e7d32',
                                 fg_color='#e0e0e0', height=8, corner_radius=4)
        bar.set(sim / 100)
        bar.pack(fill='x', pady=(4, 0))


class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title('Tìm kiếm cây tương đồng')
        self.geometry('1200x750')
        self.resizable(True, True)

        self.df = pd.DataFrame()
        self._selected_path: Path | None = None

        self._build_ui()
        # Tải DB sau khi UI sẵn sàng
        threading.Thread(target=self._load_db, daemon=True).start()

    # ── Build UI ───────────────────────────────────────────────────────────────

    def _build_ui(self):
        hdr = ctk.CTkFrame(self, corner_radius=0, fg_color='#2e7d32', height=56)
        hdr.pack(fill='x')
        hdr.pack_propagate(False)
        ctk.CTkLabel(hdr, text='🌿  Plant Similarity Search',
                     font=ctk.CTkFont(size=20, weight='bold'),
                     text_color='white').pack(expand=True)

        body = ctk.CTkFrame(self, fg_color='transparent')
        body.pack(fill='both', expand=True, padx=20, pady=16)

        # Left panel
        left = ctk.CTkFrame(body, width=270, corner_radius=12)
        left.pack(side='left', fill='y', padx=(0, 16))
        left.pack_propagate(False)

        ctk.CTkLabel(left, text='Ảnh truy vấn',
                     font=ctk.CTkFont(size=13, weight='bold')).pack(pady=(16, 6))

        self.query_img_label = ctk.CTkLabel(
            left, text='Chưa chọn ảnh\n\nNhấn "Chọn ảnh"\nđể bắt đầu',
            width=220, height=220, fg_color='#e8f5e9', corner_radius=8,
            font=ctk.CTkFont(size=11), text_color='#888'
        )
        self.query_img_label.pack(pady=(0, 8))

        self.file_label = ctk.CTkLabel(left, text='', font=ctk.CTkFont(size=10),
                                       text_color='#666', wraplength=240)
        self.file_label.pack()

        ctk.CTkButton(left, text='📂  Chọn ảnh', command=self._pick_image,
                      height=38, font=ctk.CTkFont(size=12, weight='bold'),
                      fg_color='#2e7d32', hover_color='#1b5e20',
                      corner_radius=8).pack(pady=(12, 6), padx=16, fill='x')

        self.search_btn = ctk.CTkButton(
            left, text='🔍  Tìm kiếm', command=self._search,
            height=38, font=ctk.CTkFont(size=12, weight='bold'),
            fg_color='#1565c0', hover_color='#0d47a1',
            corner_radius=8, state='disabled'
        )
        self.search_btn.pack(padx=16, fill='x')

        self.status_label = ctk.CTkLabel(left, text='Đang kết nối database...',
                                         font=ctk.CTkFont(size=10),
                                         text_color='#888', wraplength=240)
        self.status_label.pack(pady=(10, 0))

        # Right panel
        right = ctk.CTkFrame(body, fg_color='transparent')
        right.pack(side='left', fill='both', expand=True)

        ctk.CTkLabel(right, text='Kết quả tương đồng',
                     font=ctk.CTkFont(size=13, weight='bold')).pack(anchor='w', pady=(0, 8))

        self.scroll_frame = ctk.CTkScrollableFrame(right, fg_color='transparent',
                                                    corner_radius=0)
        self.scroll_frame.pack(fill='both', expand=True)

    # ── DB loading ─────────────────────────────────────────────────────────────

    def _load_db(self):
        """Chạy trong thread riêng — tải DB không làm đơ UI."""
        self.after(0, lambda: self.status_label.configure(text='Đang tải dữ liệu...'))
        df = load_features()
        self.df = df
        if df.empty:
            self.after(0, lambda: self.status_label.configure(
                text='❌ Không tải được dữ liệu', text_color='red'))
        else:
            self.after(0, lambda: self.status_label.configure(
                text=f'✅ Đã tải {len(df)} ảnh từ DB', text_color='#2e7d32'))

    # ── Actions ────────────────────────────────────────────────────────────────

    def _pick_image(self):
        path = filedialog.askopenfilename(
            title='Chọn ảnh cây',
            filetypes=[('Image files', '*.jpg *.jpeg *.png *.bmp *.tif *.tiff')]
        )
        if not path:
            return
        self._selected_path = Path(path)
        self.file_label.configure(text=self._selected_path.name)

        try:
            img = Image.open(self._selected_path).convert('RGB')
            img.thumbnail(THUMB_QUERY, Image.LANCZOS)
            ctk_img = ctk.CTkImage(light_image=img, size=img.size)
            self.query_img_label.configure(image=ctk_img, text='')
            self._query_img_ref = ctk_img
        except Exception as e:
            self.query_img_label.configure(text=f'Không đọc được ảnh\n{e}')

        self.search_btn.configure(state='normal')
        self._clear_results()

    def _search(self):
        if self._selected_path is None:
            return
        if self.df.empty:
            messagebox.showerror('Lỗi', 'Chưa có dữ liệu từ database.')
            return

        self.search_btn.configure(state='disabled')
        self.status_label.configure(text='Đang tìm kiếm...', text_color='#888')
        self._clear_results()

        threading.Thread(target=self._run_search, daemon=True).start()

    def _run_search(self):
        try:
            query_vec = build_query_vec(self._selected_path, self.df)
            results = find_top_k(query_vec, self.df, k=5)
            self.after(0, lambda: self._show_results(results))
        except Exception as e:
            self.after(0, lambda: messagebox.showerror('Lỗi tìm kiếm', str(e)))
            self.after(0, lambda: self.search_btn.configure(state='normal'))
            self.after(0, lambda: self.status_label.configure(text=''))

    def _show_results(self, results: List[dict]):
        self._clear_results()
        for rank, r in enumerate(results, 1):
            card = ResultCard(self.scroll_frame, rank=rank, result=r)
            card.pack(fill='x', pady=6, padx=4)

        self.status_label.configure(
            text=f'Tìm thấy {len(results)} kết quả', text_color='#2e7d32')
        self.search_btn.configure(state='normal')

    def _clear_results(self):
        for w in self.scroll_frame.winfo_children():
            w.destroy()


if __name__ == '__main__':
    app = App()
    app.mainloop()