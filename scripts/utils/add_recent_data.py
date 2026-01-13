"""
Quick script to add recent attack data
"""

import sqlite3
import random
from datetime import datetime, timedelta

db_path = "src/database/cyberguard.db"
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

attack_types = [
    "DDoS",
    "SQL Injection",
    "XSS",
    "Port Scan",
    "Brute Force",
    "Malware",
    "Phishing",
]
severities = ["low", "medium", "high", "critical"]

print("Son 7 gunluk 2000 saldiri ekleniyor...")

for i in range(2000):
    ts = datetime.now() - timedelta(hours=random.randint(0, 168))
    source_ip = f"{random.randint(1,255)}.{random.randint(1,255)}.{random.randint(1,255)}.{random.randint(1,255)}"
    dest_ip = f"10.0.{random.randint(0,255)}.{random.randint(1,254)}"
    port = random.choice([22, 80, 443, 3306, 8080, 21, 3389])
    attack_type = random.choice(attack_types)
    severity = random.choice(severities)
    packet_size = random.randint(100, 2000)
    blocked = random.choice([0, 1])

    cursor.execute(
        """
        INSERT INTO attacks (timestamp, source_ip, destination_ip, port, protocol, attack_type, severity, packet_size, blocked, is_anomaly, anomaly_score)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """,
        (
            ts.isoformat(),
            source_ip,
            dest_ip,
            port,
            "TCP",
            attack_type,
            severity,
            packet_size,
            blocked,
            random.choice([0, 1]),
            random.random(),
        ),
    )

conn.commit()
print("Tamamlandi! 2000 yeni saldiri eklendi.")

# Kontrol
cursor.execute(
    'SELECT COUNT(*) FROM attacks WHERE timestamp > datetime("now", "-24 hours")'
)
count = cursor.fetchone()[0]
print(f"Son 24 saatte toplam: {count} saldiri")

conn.close()
