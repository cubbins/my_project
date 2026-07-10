import sqlite3
import pandas as pd
from pathlib import Path

DB_FILE = "research_archive.db"
OUT_DIR = Path("chatgpt_upload")
OUT_DIR.mkdir(exist_ok=True)

conn = sqlite3.connect(DB_FILE)

for table in ["ResearchRun", "SearchResult"]:
    df = pd.read_sql_query(f"SELECT * FROM {table}", conn)
    out_file = OUT_DIR / f"{table}.csv"
    df.to_csv(out_file, index=False, encoding="utf-8")
    print(f"Saved {out_file}")

conn.close()