import sqlite3
conn = sqlite3.connect('src/database/cyberguard.db')
c = conn.cursor()

# Timestamp formatını kontrol et
c.execute('SELECT timestamp FROM attacks LIMIT 3')
print("Timestamp örnekleri:")
for r in c.fetchall():
    print(f"  {r[0]}")

# Son 24 saat
c.execute("SELECT COUNT(*) FROM attacks WHERE timestamp > datetime('now', '-24 hours')")
print(f"\nSon 24 saat (datetime): {c.fetchone()[0]}")

# Son 7 gün
c.execute("SELECT COUNT(*) FROM attacks WHERE timestamp > datetime('now', '-7 days')")
print(f"Son 7 gün: {c.fetchone()[0]}")

# Tüm kayıtlar
c.execute("SELECT COUNT(*) FROM attacks")
print(f"Toplam: {c.fetchone()[0]}")

conn.close()
