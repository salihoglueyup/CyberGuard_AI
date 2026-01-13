"""Database Index Oluşturma Scripti"""
import sqlite3
import os

db_path = 'src/database/cyberguard.db'

if not os.path.exists(db_path):
    print(f"❌ DB bulunamadı: {db_path}")
    exit(1)

conn = sqlite3.connect(db_path)
c = conn.cursor()

# Index oluştur
indexes = [
    ('idx_attacks_timestamp', 'attacks(timestamp)'),
    ('idx_attacks_type', 'attacks(attack_type)'),
    ('idx_attacks_source_ip', 'attacks(source_ip)'),
    ('idx_attacks_severity', 'attacks(severity)'),
    ('idx_attacks_blocked', 'attacks(blocked)'),
    ('idx_defences_timestamp', 'defences(timestamp)'),
    ('idx_defences_attack_id', 'defences(attack_id)'),
]

for name, columns in indexes:
    try:
        c.execute(f'CREATE INDEX IF NOT EXISTS {name} ON {columns}')
        print(f"✅ {name}")
    except Exception as e:
        print(f"⚠️ {name}: {e}")

conn.commit()
conn.close()

print("\n✅ Tüm indexler oluşturuldu!")
