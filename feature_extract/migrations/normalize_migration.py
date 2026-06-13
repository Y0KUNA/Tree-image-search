from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
FEATURE_EXTRACT_DIR = PROJECT_ROOT / 'feature_extract'
DEFAULT_INPUT_CSV = FEATURE_EXTRACT_DIR / 'features.csv'
DEFAULT_OUTPUT_CSV = FEATURE_EXTRACT_DIR / 'normalized_features.csv'

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


KEY_COLUMNS = ('file_name', 'folder')


def quote_ident(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in KEY_COLUMNS]


def standard_scale_values(values: pd.DataFrame) -> pd.DataFrame:
    try:
        from sklearn.preprocessing import StandardScaler

        scaler = StandardScaler()
        scaled = scaler.fit_transform(values.to_numpy(dtype=float))
        return pd.DataFrame(scaled, columns=values.columns, index=values.index)
    except ModuleNotFoundError:
        mean = values.mean(axis=0)
        std = values.std(axis=0, ddof=0).replace(0, 1)
        return (values - mean) / std


def standard_scale_csv(input_csv: Path, output_csv: Path) -> tuple[pd.DataFrame, list[str]]:
    if not input_csv.exists():
        raise FileNotFoundError(f'Input CSV not found: {input_csv}')

    df = pd.read_csv(input_csv)
    missing_keys = [c for c in KEY_COLUMNS if c not in df.columns]
    if missing_keys:
        raise ValueError(f'Missing required key columns in CSV: {missing_keys}')

    cols = feature_columns(df)
    if not cols:
        raise ValueError('No feature columns found to normalize')

    normalized = df.copy()
    normalized[cols] = normalized[cols].apply(pd.to_numeric, errors='raise')

    normalized[cols] = standard_scale_values(normalized[cols])
    normalized.to_csv(output_csv, index=False)

    return normalized, cols


def load_table_columns(cur) -> set[str]:
    cur.execute(
        """
        SELECT column_name
        FROM information_schema.columns
        WHERE table_schema = current_schema()
          AND table_name = 'plant_features'
        """
    )
    return {row[0] for row in cur.fetchall()}


def ensure_feature_columns_are_float(cur, cols: Sequence[str]) -> None:
    for col in cols:
        cur.execute(
            f'ALTER TABLE plant_features ALTER COLUMN {quote_ident(col)} TYPE DOUBLE PRECISION '
            f'USING {quote_ident(col)}::DOUBLE PRECISION'
        )


def update_database(df: pd.DataFrame, cols: Sequence[str], batch_size: int) -> int:
    import psycopg2
    from psycopg2.extras import execute_values

    from feature_extract.extract_features import DB_CONFIG

    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()

    try:
        table_cols = load_table_columns(cur)
        missing = [c for c in (*KEY_COLUMNS, *cols) if c not in table_cols]
        if missing:
            raise ValueError(f'Columns missing from plant_features: {missing}')

        ensure_feature_columns_are_float(cur, cols)

        data_cols = [*KEY_COLUMNS, *cols]
        set_clause = ', '.join(
            f'{quote_ident(col)} = data.{quote_ident(col)}'
            for col in cols
        )
        data_columns_sql = ', '.join(quote_ident(col) for col in data_cols)
        update_sql = f"""
            UPDATE plant_features AS pf
            SET {set_clause}
            FROM (VALUES %s) AS data ({data_columns_sql})
            WHERE pf.file_name = data.file_name
              AND pf.folder = data.folder
        """

        rows = [
            tuple(row[col] for col in data_cols)
            for _, row in df[data_cols].iterrows()
        ]

        updated = 0
        for start in range(0, len(rows), batch_size):
            batch = rows[start:start + batch_size]
            execute_values(cur, update_sql, batch)
            updated += cur.rowcount

        conn.commit()
        return updated
    except Exception:
        conn.rollback()
        raise
    finally:
        cur.close()
        conn.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Standard-scale features.csv, save normalized_features.csv, and update PostgreSQL.'
    )
    parser.add_argument('--input', type=Path, default=DEFAULT_INPUT_CSV, help='Path to features.csv')
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT_CSV, help='Path to normalized_features.csv')
    parser.add_argument('--batch', type=int, default=500, help='Database update batch size')
    parser.add_argument(
        '--skip-db',
        action='store_true',
        help='Only write normalized CSV; do not update PostgreSQL',
    )
    args = parser.parse_args()

    normalized_df, cols = standard_scale_csv(args.input, args.output)
    print(f'Wrote {len(normalized_df)} normalized rows to {args.output}')
    print(f'Normalized {len(cols)} feature columns with StandardScaler')

    if args.skip_db:
        return

    updated = update_database(normalized_df, cols, args.batch)
    print(f'Updated {updated} rows in plant_features')


if __name__ == '__main__':
    main()
