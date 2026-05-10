from __future__ import annotations

from pathlib import Path
import psycopg2
import importlib

MIGRATIONS_DIR = Path(__file__).parent / 'migrations'


def main():
    mod = importlib.import_module('feature_extract.extract_features')
    DB_CONFIG = getattr(mod, 'DB_CONFIG')

    sql_files = sorted(MIGRATIONS_DIR.glob('*.sql'))
    if not sql_files:
        print('No migrations found in', MIGRATIONS_DIR)
        return

    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()
    try:
        for f in sql_files:
            print('Applying', f.name)
            sql = f.read_text(encoding='utf-8')
            cur.execute(sql)
            conn.commit()
        print('Migrations applied successfully')
    finally:
        cur.close()
        conn.close()


if __name__ == '__main__':
    main()
