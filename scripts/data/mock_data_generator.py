"""
Mock Data Generator - CyberGuard AI
Gerçekçi sentetik saldırı verisi oluşturur ve veritabanına kaydeder
"""

import sys
import os

# Proje kök dizinini ekle
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)  # scripts/ -> proje root
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Default DB path (mutlak)
DEFAULT_DB_PATH = os.path.join(project_root, 'src', 'database', 'cyberguard.db')

import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
from typing import List, Dict
import json


class MockDataGenerator:
    """
    Gerçekçi siber saldırı verisi oluşturur

    Desteklenen saldırı türleri:
    - DDoS
    - SQL Injection
    - XSS (Cross-Site Scripting)
    - Port Scan
    - Brute Force
    """

    def __init__(self, db_path: str = None):
        """
        Args:
            db_path: SQLite veritabanı yolu
        """
        self.db_path = db_path or DEFAULT_DB_PATH

        # Saldırı türleri ve özellikleri
        self.attack_types = {
            'DDoS': {
                'severity': ['critical', 'high'],
                'ports': [80, 443, 8080, 8443],
                'packet_size': (500, 1500),
                'burst': True
            },
            'SQL Injection': {
                'severity': ['critical', 'high'],
                'ports': [3306, 5432, 1433, 1521],
                'packet_size': (200, 1000),
                'burst': False
            },
            'XSS': {
                'severity': ['medium', 'high'],
                'ports': [80, 443, 8080],
                'packet_size': (100, 800),
                'burst': False
            },
            'Port Scan': {
                'severity': ['low', 'medium'],
                'ports': list(range(1, 1024)),
                'packet_size': (40, 100),
                'burst': True
            },
            'Brute Force': {
                'severity': ['medium', 'high'],
                'ports': [22, 21, 3389, 23],
                'packet_size': (50, 300),
                'burst': True
            }
        }

        # Severity weight (kaç tane oluşturulacak)
        self.severity_weights = {
            'low': 0.2,
            'medium': 0.3,
            'high': 0.35,
            'critical': 0.15
        }

        print(f"🎲 Mock Data Generator başlatıldı")
        print(f"📁 Database: {self.db_path}")

    def create_database(self):
        """Veritabanını oluştur"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # attacks tablosu
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS attacks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                source_ip TEXT NOT NULL,
                destination_ip TEXT NOT NULL,
                port INTEGER NOT NULL,
                protocol TEXT NOT NULL,
                attack_type TEXT NOT NULL,
                severity TEXT NOT NULL,
                packet_size INTEGER NOT NULL,
                blocked INTEGER NOT NULL,
                is_anomaly INTEGER DEFAULT 0,
                anomaly_score REAL DEFAULT 0.0,
                description TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        # defences tablosu
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS defences (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                attack_id INTEGER,
                attack_type TEXT NOT NULL,
                source_ip TEXT NOT NULL,
                defence_action TEXT NOT NULL,
                defence_method TEXT,
                success INTEGER DEFAULT 1,
                response_time_ms REAL,
                blocked_packets INTEGER DEFAULT 0,
                allowed_packets INTEGER DEFAULT 0,
                firewall_rule TEXT,
                mitigation_type TEXT,
                severity_before TEXT,
                severity_after TEXT,
                risk_score_before REAL,
                risk_score_after REAL,
                false_positive INTEGER DEFAULT 0,
                notes TEXT,
                operator TEXT DEFAULT 'AUTO',
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (attack_id) REFERENCES attacks(id)
            )
        ''')

        conn.commit()
        conn.close()

        print("✅ Database tabloları oluşturuldu (attacks + defences)")

    def generate_ip(self) -> str:
        """Random IP adresi oluştur"""
        # Gerçekçi private/public IP'ler
        ip_ranges = [
            (192, 168, random.randint(0, 255), random.randint(1, 254)),  # Private
            (10, random.randint(0, 255), random.randint(0, 255), random.randint(1, 254)),  # Private
            (172, random.randint(16, 31), random.randint(0, 255), random.randint(1, 254)),  # Private
            (random.randint(1, 223), random.randint(0, 255), random.randint(0, 255), random.randint(1, 254))  # Public
        ]

        octets = random.choice(ip_ranges)
        return f"{octets[0]}.{octets[1]}.{octets[2]}.{octets[3]}"

    def generate_timestamp(
            self,
            start_date: datetime,
            end_date: datetime
    ) -> str:
        """Random timestamp oluştur"""
        time_delta = end_date - start_date
        random_seconds = random.randint(0, int(time_delta.total_seconds()))
        random_time = start_date + timedelta(seconds=random_seconds)
        return random_time.isoformat()

    def generate_attack_record(
            self,
            attack_type: str,
            timestamp: str
    ) -> Dict:
        """
        Tek bir saldırı kaydı oluştur

        Args:
            attack_type: Saldırı türü
            timestamp: Zaman damgası

        Returns:
            Saldırı kaydı dictionary
        """
        attack_config = self.attack_types[attack_type]

        # Severity seç (weighted)
        severity = random.choice(attack_config['severity'])

        # Port seç
        if isinstance(attack_config['ports'], list):
            if len(attack_config['ports']) > 100:  # Port scan gibi
                port = random.choice(attack_config['ports'])
            else:
                port = random.choice(attack_config['ports'])
        else:
            port = attack_config['ports']

        # Packet size
        packet_size = random.randint(*attack_config['packet_size'])

        # Protocol
        protocol = random.choice(['TCP', 'UDP', 'ICMP'])

        # Blocked (critical/high severity'ler daha çok bloklanır)
        blocked_probability = {
            'low': 0.3,
            'medium': 0.5,
            'high': 0.7,
            'critical': 0.9
        }
        blocked = 1 if random.random() < blocked_probability[severity] else 0

        # Description
        descriptions = {
            'DDoS': [
                "High volume traffic detected",
                "SYN flood attack pattern",
                "UDP flood from multiple sources",
                "HTTP flood attack"
            ],
            'SQL Injection': [
                "SQL injection attempt in query parameter",
                "Union-based SQL injection detected",
                "Blind SQL injection pattern",
                "OR 1=1 pattern detected"
            ],
            'XSS': [
                "Script tag injection attempt",
                "Event handler injection detected",
                "JavaScript injection in URL",
                "DOM-based XSS pattern"
            ],
            'Port Scan': [
                "Sequential port scanning detected",
                "SYN scan pattern identified",
                "Full connect scan attempt",
                "Stealth scan detected"
            ],
            'Brute Force': [
                "Multiple failed login attempts",
                "Dictionary attack pattern",
                "Credential stuffing detected",
                "SSH brute force attempt"
            ]
        }

        # Anomaly detection - yüksek severity ve büyük packet = daha yüksek anomaly
        is_anomaly = 1 if severity in ['critical', 'high'] and packet_size > 1000 else 0
        anomaly_score = random.uniform(0.7, 1.0) if is_anomaly else random.uniform(0.0, 0.4)
        
        return {
            'timestamp': timestamp,
            'source_ip': self.generate_ip(),
            'destination_ip': self.generate_ip(),
            'port': port,
            'protocol': protocol,
            'attack_type': attack_type,
            'severity': severity,
            'packet_size': packet_size,
            'blocked': blocked,
            'is_anomaly': is_anomaly,
            'anomaly_score': round(anomaly_score, 4),
            'description': random.choice(descriptions[attack_type])
        }

    def generate_defence_record(
            self,
            attack_record: Dict,
            attack_id: int = None
    ) -> Dict:
        """
        Bir saldırıya karşılık savunma kaydı oluştur

        Args:
            attack_record: İlgili saldırı kaydı
            attack_id: Veritabanındaki attack ID

        Returns:
            Savunma kaydı dictionary
        """
        # Defence actions
        defence_actions = ['BLOCK', 'ALLOW', 'QUARANTINE', 'THROTTLE', 'DROP', 'REDIRECT']
        
        # Defence methods
        defence_methods = {
            'DDoS': ['Rate Limiting', 'IP Blacklist', 'Traffic Shaping', 'CDN Protection'],
            'SQL Injection': ['WAF Rule', 'Input Sanitization', 'Query Blocking', 'Parameter Filtering'],
            'XSS': ['Content Security Policy', 'Output Encoding', 'Script Blocking', 'WAF Rule'],
            'Port Scan': ['Port Blocking', 'IP Blacklist', 'Honeypot Redirect', 'SYN Cookie'],
            'Brute Force': ['Account Lockout', 'CAPTCHA', 'IP Ban', 'Rate Limiting']
        }
        
        # Mitigation types
        mitigation_types = ['AUTO', 'MANUAL', 'ML_BASED', 'RULE_BASED', 'HYBRID']
        
        # Firewall rules
        firewall_rules = [
            'DENY_ALL_FROM_IP', 'RATE_LIMIT_100', 'GEO_BLOCK', 'DROP_SYN_FLOOD',
            'WAF_INJECTION', 'WHITELIST_ONLY', 'BLOCK_PORT_SCAN', 'TARPIT'
        ]
        
        attack_type = attack_record['attack_type']
        severity = attack_record['severity']
        
        # Success probability - daha tehlikeli saldırılar daha az başarılı savunma
        success_prob = {'low': 0.95, 'medium': 0.85, 'high': 0.75, 'critical': 0.65}
        success = 1 if random.random() < success_prob.get(severity, 0.8) else 0
        
        # Response time - critical daha uzun
        response_times = {'low': (10, 100), 'medium': (50, 300), 'high': (100, 500), 'critical': (200, 1000)}
        response_time = random.uniform(*response_times.get(severity, (50, 200)))
        
        # Blocked ve allowed packets
        total_packets = random.randint(10, 1000)
        blocked_ratio = 0.9 if success else 0.3
        blocked_packets = int(total_packets * blocked_ratio)
        allowed_packets = total_packets - blocked_packets
        
        # False positive
        false_positive = 1 if random.random() < 0.05 else 0  # 5% false positive
        
        # Risk scores
        risk_before = {'low': 0.25, 'medium': 0.5, 'high': 0.75, 'critical': 0.95}
        risk_score_before = risk_before.get(severity, 0.5) + random.uniform(-0.1, 0.1)
        risk_score_after = risk_score_before * (0.2 if success else 0.8)
        
        return {
            'timestamp': attack_record['timestamp'],
            'attack_id': attack_id,
            'attack_type': attack_type,
            'source_ip': attack_record['source_ip'],
            'defence_action': random.choice(['BLOCK', 'DROP']) if success else random.choice(['ALLOW', 'THROTTLE']),
            'defence_method': random.choice(defence_methods.get(attack_type, ['General Block'])),
            'success': success,
            'response_time_ms': round(response_time, 2),
            'blocked_packets': blocked_packets,
            'allowed_packets': allowed_packets,
            'firewall_rule': random.choice(firewall_rules),
            'mitigation_type': random.choice(mitigation_types),
            'severity_before': severity,
            'severity_after': 'low' if success else severity,
            'risk_score_before': round(risk_score_before, 4),
            'risk_score_after': round(max(0, risk_score_after), 4),
            'false_positive': false_positive,
            'notes': f"Auto-generated defence for {attack_type}",
            'operator': random.choice(['AUTO', 'SYSTEM', 'AI_ENGINE', 'FIREWALL'])
        }

    def generate_attack_burst(
            self,
            attack_type: str,
            base_timestamp: datetime,
            count: int = 10
    ) -> List[Dict]:
        """
        Saldırı patlaması oluştur (DDoS, Port Scan gibi)

        Args:
            attack_type: Saldırı türü
            base_timestamp: Başlangıç zamanı
            count: Kayıt sayısı

        Returns:
            Saldırı kayıtları listesi
        """
        records = []
        source_ip = self.generate_ip()  # Aynı kaynaktan
        dest_ip = self.generate_ip()  # Aynı hedefe

        for i in range(count):
            # Burst içinde zamanlar çok yakın (saniyeler içinde)
            timestamp = (base_timestamp + timedelta(seconds=random.randint(0, 60))).isoformat()

            record = self.generate_attack_record(attack_type, timestamp)
            record['source_ip'] = source_ip
            record['destination_ip'] = dest_ip

            records.append(record)

        return records

    def generate_data(
            self,
            num_records: int = 10000,
            start_date: datetime = None,
            end_date: datetime = None,
            attack_distribution: Dict[str, float] = None
    ) -> pd.DataFrame:
        """
        Mock veri oluştur

        Args:
            num_records: Toplam kayıt sayısı
            start_date: Başlangıç tarihi
            end_date: Bitiş tarihi
            attack_distribution: Saldırı türü dağılımı

        Returns:
            DataFrame
        """
        if start_date is None:
            start_date = datetime.now() - timedelta(days=30)
        if end_date is None:
            end_date = datetime.now()

        # Default dağılım (eşit)
        if attack_distribution is None:
            attack_distribution = {
                attack_type: 1.0 / len(self.attack_types)
                for attack_type in self.attack_types
            }

        print(f"\n🎲 {num_records:,} kayıt oluşturuluyor...")
        print(f"📅 Tarih aralığı: {start_date.date()} → {end_date.date()}")
        print(f"📊 Saldırı dağılımı: {attack_distribution}")

        records = []

        # Her saldırı türü için kayıt oluştur
        for attack_type, proportion in attack_distribution.items():
            count = max(1, int(num_records * proportion))  # Minimum 1 kayıt

            print(f"   ├─ {attack_type}: {count:,} kayıt")

            attack_config = self.attack_types[attack_type]

            if attack_config['burst']:
                # Burst pattern (DDoS, Port Scan)
                num_bursts = count // 20  # Her burst 20 kayıt
                remaining = count % 20

                for _ in range(num_bursts):
                    base_time = self.generate_timestamp(start_date, end_date)
                    base_datetime = datetime.fromisoformat(base_time)
                    burst_records = self.generate_attack_burst(
                        attack_type, base_datetime, count=20
                    )
                    records.extend(burst_records)

                # Kalan kayıtlar
                for _ in range(remaining):
                    timestamp = self.generate_timestamp(start_date, end_date)
                    records.append(self.generate_attack_record(attack_type, timestamp))

            else:
                # Normal pattern (SQL Injection, XSS, Brute Force)
                for _ in range(count):
                    timestamp = self.generate_timestamp(start_date, end_date)
                    records.append(self.generate_attack_record(attack_type, timestamp))

        print(f"✅ {len(records):,} kayıt oluşturuldu")

        # DataFrame'e çevir
        df = pd.DataFrame(records)

        # Timestamp'e göre sırala
        df = df.sort_values('timestamp').reset_index(drop=True)

        return df

    def save_to_database(self, df: pd.DataFrame):
        """
        DataFrame'i veritabanına kaydet

        Args:
            df: Kayıt DataFrame'i
        """
        print(f"\n💾 Veritabanına kaydediliyor: {self.db_path}")

        conn = sqlite3.connect(self.db_path)

        # Append mode (mevcut verileri korur)
        df.to_sql('attacks', conn, if_exists='append', index=False)

        conn.close()

        print(f"✅ {len(df):,} kayıt başarıyla kaydedildi")

    def get_statistics(self) -> Dict:
        """Veritabanı istatistiklerini getir"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Toplam kayıt
        cursor.execute("SELECT COUNT(*) FROM attacks")
        total_records = cursor.fetchone()[0]

        # Saldırı türü dağılımı
        cursor.execute("""
                       SELECT attack_type, COUNT(*) as count
                       FROM attacks
                       GROUP BY attack_type
                       """)
        attack_dist = dict(cursor.fetchall())

        # Severity dağılımı
        cursor.execute("""
                       SELECT severity, COUNT(*) as count
                       FROM attacks
                       GROUP BY severity
                       """)
        severity_dist = dict(cursor.fetchall())

        # Bloklanma oranı
        cursor.execute("""
                       SELECT SUM(blocked) * 100.0 / COUNT(*) as block_rate
                       FROM attacks
                       """)
        block_rate = cursor.fetchone()[0]

        # Tarih aralığı
        cursor.execute("""
                       SELECT MIN(timestamp), MAX(timestamp)
                       FROM attacks
                       """)
        date_range = cursor.fetchall()[0]

        conn.close()

        return {
            'total_records': total_records,
            'attack_distribution': attack_dist,
            'severity_distribution': severity_dist,
            'block_rate': block_rate,
            'date_range': date_range
        }

    def print_statistics(self):
        """İstatistikleri yazdır"""
        stats = self.get_statistics()

        print("\n" + "=" * 70)
        print("📊 VERİTABANI İSTATİSTİKLERİ")
        print("=" * 70)

        print(f"\n📈 Toplam Kayıt: {stats['total_records']:,}")

        print(f"\n🎯 Saldırı Türü Dağılımı:")
        for attack_type, count in stats['attack_distribution'].items():
            percentage = (count / stats['total_records']) * 100
            print(f"   ├─ {attack_type}: {count:,} ({percentage:.1f}%)")

        print(f"\n⚠️  Severity Dağılımı:")
        for severity, count in stats['severity_distribution'].items():
            percentage = (count / stats['total_records']) * 100
            print(f"   ├─ {severity}: {count:,} ({percentage:.1f}%)")

        block_rate = stats['block_rate'] or 0
        print(f"\n🛡️  Bloklanma Oranı: {block_rate:.1f}%")

        print(f"\n📅 Tarih Aralığı:")
        print(f"   ├─ Başlangıç: {stats['date_range'][0]}")
        print(f"   └─ Bitiş: {stats['date_range'][1]}")

        print("=" * 70 + "\n")

    def clear_database(self):
        """Veritabanını temizle"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("DELETE FROM attacks")
        conn.commit()

        count = cursor.rowcount
        conn.close()

        print(f"🗑️  {count:,} kayıt silindi")

    def export_to_csv(self, output_file: str = 'mock_attacks.csv'):
        """
        Veritabanını CSV'ye export et

        Args:
            output_file: CSV dosya yolu
        """
        conn = sqlite3.connect(self.db_path)
        df = pd.read_sql_query("SELECT * FROM attacks", conn)
        conn.close()

        df.to_csv(output_file, index=False)
        print(f"✅ CSV export: {output_file} ({len(df):,} kayıt)")


def main():
    """Ana fonksiyon"""

    print("\n" + "=" * 70)
    print("🎲 CYBERGUARD AI - MOCK DATA GENERATOR")
    print("=" * 70 + "\n")

    # Generator oluştur
    generator = MockDataGenerator()  # DEFAULT_DB_PATH kullanır

    # Database oluştur
    generator.create_database()

    while True:
        # Kullanıcı seçimi
        print("\n📋 Ana Menü:")
        print("1. 🗡️  Saldırı (attacks) verisi oluştur")
        print("2. 🛡️  Savunma (defences) verisi oluştur")
        print("3. 📊 İstatistikleri göster")
        print("4. 🗑️  Veritabanını temizle")
        print("5. 📥 CSV'ye export et")
        print("6. 🚪 Çıkış")

        choice = input("\nSeçiminiz (1-6): ").strip()

        if choice == '1':
            # Saldırı verisi oluştur
            num_records = int(input("\nKaç saldırı kaydı oluşturulsun? (örn: 1000): ") or "1000")
            days_ago = int(input("Kaç gün öncesine kadar? (örn: 30): ") or "30")

            start_date = datetime.now() - timedelta(days=days_ago)
            end_date = datetime.now()

            # Veri oluştur
            df = generator.generate_data(
                num_records=num_records,
                start_date=start_date,
                end_date=end_date
            )

            # Önizleme
            print("\n📊 Saldırı verisi önizleme (ilk 5 kayıt):")
            print(df.head().to_string())

            # Kaydet
            confirm = input("\n💾 Veritabanına kaydetmek istiyor musunuz? (E/H): ").strip().upper()
            if confirm == 'E':
                generator.save_to_database(df)
                print("✅ Saldırı verileri kaydedildi!")

        elif choice == '2':
            # Savunma verisi oluştur
            print("\n🛡️  SAVUNMA VERİSİ OLUŞTURMA")
            print("-" * 40)
            
            # Mevcut saldırı sayısı
            conn = sqlite3.connect(generator.db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM attacks")
            attack_count = cursor.fetchone()[0]
            
            if attack_count == 0:
                print("❌ Önce saldırı verisi oluşturmalısınız!")
                conn.close()
                continue
            
            print(f"📊 Mevcut saldırı sayısı: {attack_count:,}")
            num_defences = int(input(f"Kaç savunma kaydı oluşturulsun? (max {attack_count}): ") or str(min(1000, attack_count)))
            num_defences = min(num_defences, attack_count)
            
            # Random saldırıları al
            cursor.execute(f"SELECT * FROM attacks ORDER BY RANDOM() LIMIT {num_defences}")
            columns = [desc[0] for desc in cursor.description]
            attacks = cursor.fetchall()
            conn.close()
            
            print(f"\n🎲 {len(attacks)} savunma kaydı oluşturuluyor...")
            
            defence_records = []
            for attack_row in attacks:
                attack_dict = dict(zip(columns, attack_row))
                defence = generator.generate_defence_record(attack_dict, attack_dict.get('id'))
                defence_records.append(defence)
            
            df_defences = pd.DataFrame(defence_records)
            
            # Önizleme
            print("\n📊 Savunma verisi önizleme (ilk 5 kayıt):")
            print(df_defences[['timestamp', 'attack_type', 'defence_action', 'success', 'response_time_ms']].head().to_string())
            
            # Kaydet
            confirm = input("\n💾 Veritabanına kaydetmek istiyor musunuz? (E/H): ").strip().upper()
            if confirm == 'E':
                conn = sqlite3.connect(generator.db_path)
                df_defences.to_sql('defences', conn, if_exists='append', index=False)
                conn.close()
                print(f"✅ {len(df_defences):,} savunma kaydı kaydedildi!")
                
                # İstatistik
                conn = sqlite3.connect(generator.db_path)
                cursor = conn.cursor()
                cursor.execute("SELECT COUNT(*) FROM defences")
                total = cursor.fetchone()[0]
                cursor.execute("SELECT SUM(success) * 100.0 / COUNT(*) FROM defences")
                success_rate = cursor.fetchone()[0] or 0
                conn.close()
                print(f"📊 Toplam savunma: {total:,} | Başarı oranı: {success_rate:.1f}%")

        elif choice == '3':
            # İstatistikleri göster
            print("\n📊 Hangi tablonun istatistiklerini görmek istersiniz?")
            print("1. Saldırılar (attacks)")
            print("2. Savunmalar (defences)")
            print("3. Her ikisi")
            
            stat_choice = input("\nSeçiminiz (1-3): ").strip()
            
            if stat_choice in ['1', '3']:
                generator.print_statistics()
            
            if stat_choice in ['2', '3']:
                # Savunma istatistikleri
                conn = sqlite3.connect(generator.db_path)
                cursor = conn.cursor()
                
                cursor.execute("SELECT COUNT(*) FROM defences")
                total = cursor.fetchone()[0]
                
                if total > 0:
                    print("\n" + "=" * 70)
                    print("🛡️  SAVUNMA İSTATİSTİKLERİ")
                    print("=" * 70)
                    
                    print(f"\n📈 Toplam Savunma: {total:,}")
                    
                    cursor.execute("SELECT SUM(success) * 100.0 / COUNT(*) FROM defences")
                    success_rate = cursor.fetchone()[0] or 0
                    print(f"✅ Başarı Oranı: {success_rate:.1f}%")
                    
                    cursor.execute("SELECT AVG(response_time_ms) FROM defences")
                    avg_response = cursor.fetchone()[0] or 0
                    print(f"⏱️  Ortalama Tepki Süresi: {avg_response:.2f} ms")
                    
                    cursor.execute("SELECT SUM(blocked_packets), SUM(allowed_packets) FROM defences")
                    blocked, allowed = cursor.fetchone()
                    print(f"📦 Engellenen/İzin Verilen: {blocked or 0:,} / {allowed or 0:,}")
                    
                    cursor.execute("SELECT defence_action, COUNT(*) FROM defences GROUP BY defence_action")
                    actions = cursor.fetchall()
                    print("\n🎯 Savunma Aksiyonları:")
                    for action, count in actions:
                        print(f"   ├─ {action}: {count:,}")
                    
                    print("=" * 70)
                else:
                    print("❌ Henüz savunma verisi yok!")
                
                conn.close()

        elif choice == '4':
            # Temizle
            print("\n🗑️  Hangi tabloyu temizlemek istersiniz?")
            print("1. Saldırılar (attacks)")
            print("2. Savunmalar (defences)")
            print("3. Her ikisi")
            
            clear_choice = input("\nSeçiminiz (1-3): ").strip()
            confirm = input("\n⚠️  VERİLER SİLİNECEK! Emin misiniz? (EVET yazın): ").strip()
            
            if confirm == "EVET":
                conn = sqlite3.connect(generator.db_path)
                cursor = conn.cursor()
                
                if clear_choice in ['1', '3']:
                    cursor.execute("DELETE FROM attacks")
                    print("✅ Saldırı verileri temizlendi")
                
                if clear_choice in ['2', '3']:
                    cursor.execute("DELETE FROM defences")
                    print("✅ Savunma verileri temizlendi")
                
                conn.commit()
                conn.close()
            else:
                print("❌ İşlem iptal edildi")

        elif choice == '5':
            # CSV export
            print("\n📥 Hangi tabloyu export etmek istersiniz?")
            print("1. Saldırılar (attacks)")
            print("2. Savunmalar (defences)")
            
            export_choice = input("\nSeçiminiz (1-2): ").strip()
            
            if export_choice == '1':
                filename = input("Dosya adı (örn: attacks.csv): ").strip() or "attacks.csv"
                generator.export_to_csv(filename)
            elif export_choice == '2':
                filename = input("Dosya adı (örn: defences.csv): ").strip() or "defences.csv"
                conn = sqlite3.connect(generator.db_path)
                df = pd.read_sql_query("SELECT * FROM defences", conn)
                conn.close()
                df.to_csv(filename, index=False)
                print(f"✅ CSV export: {filename} ({len(df):,} kayıt)")

        elif choice == '6':
            print("\n👋 Çıkış yapılıyor...")
            break

        else:
            print("\n❌ Geçersiz seçim!")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  İşlem durduruldu!")
    except Exception as e:
        print(f"\n❌ HATA: {e}")
        import traceback

        traceback.print_exc()