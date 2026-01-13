import sqlite3
import os

db_path = 'src/database/cyberguard.db'

if not os.path.exists(db_path):
    print(f"Database bulunamadi: {db_path}")
    exit(1)

conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# Tablolari listele
cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
tables = cursor.fetchall()

print("DATABASE ANALIZI")
print("-" * 40)
print(f"Database: {db_path}")
print(f"Toplam Tablo: {len(tables)}")
print("-" * 40)

for table in tables:
    table_name = table[0]
    cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
    count = cursor.fetchone()[0]
    cursor.execute(f"PRAGMA table_info({table_name})")
    columns = [c[1] for c in cursor.fetchall()]
    print(f"{table_name}: {count} kayit")
    print(f"  Kolonlar: {columns}")
    print()

print("-" * 40)
conn.close()
