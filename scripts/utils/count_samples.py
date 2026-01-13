import os
import pandas as pd
from pathlib import Path

data_dir = Path("data/raw/cicids2017_full")
csv_files = [f for f in data_dir.glob("*.csv")]

print(f"Found {len(csv_files)} CSV files")
total = 0

for f in csv_files:
    try:
        count = sum(1 for _ in open(f, encoding="utf-8", errors="ignore")) - 1
        total += count
        print(f"  {f.name}: {count:,} rows")
    except Exception as e:
        print(f"  {f.name}: Error - {e}")

print(f"\nTotal samples: {total:,}")
