import pandas as pd
import psycopg2
from pathlib import Path

CSV_PATH = Path(__file__).parent / 'feature_extract' / 'features.csv'

DB_CONFIG = {
    'host':     'localhost',
    'port':     5432,
    'dbname':   'Tree_image_metadata',   # ← đổi thành tên database của bạn
    'user':     'postgres',   # ← đổi thành username của bạn
    'password': '12345678',   # ← đổi thành password của bạn
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
    green_count, green_ratio
) VALUES (
    %(file_name)s, %(folder)s,
    %(rgb_hist_1)s, %(rgb_hist_2)s, %(rgb_hist_3)s, %(rgb_hist_4)s,
    %(rgb_hist_5)s, %(rgb_hist_6)s, %(rgb_hist_7)s, %(rgb_hist_8)s,
    %(rgb_hist_9)s, %(rgb_hist_10)s, %(rgb_hist_11)s, %(rgb_hist_12)s,
    %(rgb_hist_13)s, %(rgb_hist_14)s, %(rgb_hist_15)s, %(rgb_hist_16)s,
    %(rgb_hist_17)s, %(rgb_hist_18)s, %(rgb_hist_19)s, %(rgb_hist_20)s,
    %(rgb_hist_21)s, %(rgb_hist_22)s, %(rgb_hist_23)s, %(rgb_hist_24)s,
    %(edge_count)s, %(edge_ratio)s,
    %(hw_ratio)s, %(contour_area)s, %(convexity)s, %(defect_count)s,
    %(lbp_1)s, %(lbp_2)s, %(lbp_3)s, %(lbp_4)s, %(lbp_5)s,
    %(lbp_6)s, %(lbp_7)s, %(lbp_8)s, %(lbp_9)s, %(lbp_10)s,
    %(green_count)s, %(green_ratio)s
)

"""


def import_csv_to_db():
    df = pd.read_csv(CSV_PATH)
    print(f'Đọc được {len(df)} dòng từ CSV')

    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()

    success, skipped, failed = 0, 0, 0

    for _, row in df.iterrows():
        try:
            cur.execute(INSERT_SQL, row.to_dict())
            if cur.rowcount == 1:
                success += 1
            else:
                skipped += 1  # ON CONFLICT DO NOTHING
        except Exception as e:
            print(f'Lỗi dòng {row["file_name"]}: {e}')
            conn.rollback()
            failed += 1
            continue

        conn.commit()

    cur.close()
    conn.close()

    print(f'Hoàn thành: {success} thêm mới | {skipped} bỏ qua (trùng) | {failed} lỗi')


if __name__ == '__main__':
    import_csv_to_db()