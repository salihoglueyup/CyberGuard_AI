"""
Mock Veri Oluşturucu - CyberGuard AI
Sahte saldırı verileri, loglar ve tarama sonuçları üretir
"""

import random
import sqlite3
from datetime import datetime, timedelta
from typing import List, Dict
import json

class MockDataGenerator:
    """Sahte siber güvenlik verileri üretir"""

    # Saldırı türleri - AZALTILDI (6 tür)
    ATTACK_TYPES = [
        'DDoS',
        'Port Scan',
        'SQL Injection',
        'Brute Force',
        'XSS',
        'Malware'
    ]

    # Severity seviyeleri
    SEVERITIES = ['low', 'medium', 'high', 'critical']

    # Sahte IP'ler (zararlı görünümlü)
    MALICIOUS_IPS = [
        '192.168.1.{}'.format(i) for i in range(100, 200)
    ] + [
        '10.0.0.{}'.format(i) for i in range(1, 50)
    ] + [
        '172.16.0.{}'.format(i) for i in range(1, 30)
    ]

    # Sahte portlar
    PORTS = [22, 80, 443, 3306, 5432, 8080, 8443, 21, 25, 3389]

    def __init__(self, db_path: str = 'cyberguard.db'):
        """
        Args:
            db_path (str): Database dosya yolu
        """
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.cursor = self.conn.cursor()
        self._create_tables()

    def _create_tables(self):
        """Gerekli tabloları oluştur"""

        # Attacks tablosu
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS attacks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                attack_type TEXT NOT NULL,
                source_ip TEXT NOT NULL,
                destination_ip TEXT,
                port INTEGER,
                severity TEXT,
                status TEXT DEFAULT 'detected',
                description TEXT,
                blocked BOOLEAN DEFAULT 0
            )
        ''')

        # Logs tablosu
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                level TEXT,
                source TEXT,
                message TEXT,
                ip_address TEXT
            )
        ''')

        # Network scans tablosu
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS network_scans (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                scan_time DATETIME DEFAULT CURRENT_TIMESTAMP,
                ip_address TEXT NOT NULL,
                open_ports TEXT,
                os_info TEXT,
                vulnerabilities TEXT,
                risk_score INTEGER
            )
        ''')

        self.conn.commit()

    def generate_attacks(self, count: int = 100) -> List[Dict]:
        """
        Sahte saldırı verileri üret - DAHA PATTERN'LI

        Args:
            count (int): Üretilecek saldırı sayısı

        Returns:
            List[Dict]: Saldırı listesi
        """
        attacks = []
        now = datetime.now()

        for i in range(count):
            # Rastgele zaman (son 7 gün)
            random_time = now - timedelta(
                days=random.randint(0, 7),
                hours=random.randint(0, 23),
                minutes=random.randint(0, 59)
            )

            attack_type = random.choice(self.ATTACK_TYPES)

            # PATTERN EKLE: Saldırı türüne göre özellikler
            if attack_type == 'DDoS':
                port = random.choice([80, 443, 8080])  # Web portları
                severity = random.choice(['high', 'critical', 'critical'])  # Genelde çok yüksek
                source_ip_range = range(100, 120)
                blocked = 1  # Çoğu engellenir

            elif attack_type == 'Port Scan':
                port = random.choice([22, 80, 443, 3306, 8080])
                severity = random.choice(['low', 'low', 'medium'])
                source_ip_range = range(120, 140)
                blocked = random.choice([0, 0, 1])  # Genelde engellenmez

            elif attack_type == 'SQL Injection':
                port = random.choice([3306, 5432, 1433])  # Database portları
                severity = random.choice(['high', 'critical', 'high'])
                source_ip_range = range(140, 160)
                blocked = 1

            elif attack_type == 'Brute Force':
                port = random.choice([22, 3389, 21])  # SSH, RDP, FTP
                severity = random.choice(['medium', 'high', 'medium'])
                source_ip_range = range(160, 175)
                blocked = random.choice([0, 1])

            elif attack_type == 'XSS':
                port = random.choice([80, 443, 8080])
                severity = random.choice(['medium', 'high'])
                source_ip_range = range(175, 185)
                blocked = random.choice([0, 1])

            elif attack_type == 'Malware':
                port = random.choice([80, 443, 8080, 445])
                severity = random.choice(['critical', 'high', 'critical'])
                source_ip_range = range(185, 200)
                blocked = 1

            else:
                port = random.choice(self.PORTS)
                severity = random.choice(self.SEVERITIES)
                source_ip_range = range(100, 200)
                blocked = random.choice([0, 1])

            attack = {
                'timestamp': random_time.strftime('%Y-%m-%d %H:%M:%S'),
                'attack_type': attack_type,
                'source_ip': f'192.168.1.{random.choice(list(source_ip_range))}',
                'destination_ip': f'192.168.0.{random.randint(1, 50)}',
                'port': port,
                'severity': severity,
                'status': random.choice(['detected', 'blocked', 'investigating']),
                'description': self._generate_attack_description(attack_type),
                'blocked': blocked
            }

            attacks.append(attack)

        return attacks

    def generate_logs(self, count: int = 200) -> List[Dict]:
        """
        Sahte log verileri üret

        Args:
            count (int): Üretilecek log sayısı

        Returns:
            List[Dict]: Log listesi
        """
        logs = []
        now = datetime.now()

        log_messages = [
            'Firewall rule updated',
            'Suspicious activity detected',
            'Connection attempt from unknown source',
            'Authentication failed',
            'Port scan detected',
            'System backup completed',
            'IDS alert triggered',
            'Malware signature updated',
            'VPN connection established',
            'Access denied'
        ]

        for i in range(count):
            random_time = now - timedelta(
                days=random.randint(0, 7),
                hours=random.randint(0, 23),
                minutes=random.randint(0, 59)
            )

            log = {
                'timestamp': random_time.strftime('%Y-%m-%d %H:%M:%S'),
                'level': random.choices(
                    ['INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                    weights=[50, 30, 15, 5]
                )[0],
                'source': random.choice(['Firewall', 'IDS', 'IPS', 'VPN', 'Auth']),
                'message': random.choice(log_messages),
                'ip_address': random.choice(self.MALICIOUS_IPS)
            }

            logs.append(log)

        return logs

    def generate_network_scans(self, count: int = 50) -> List[Dict]:
        """
        Sahte network tarama sonuçları üret

        Args:
            count (int): Üretilecek tarama sayısı

        Returns:
            List[Dict]: Tarama listesi
        """
        scans = []
        now = datetime.now()

        os_list = ['Linux', 'Windows', 'MacOS', 'Unknown']
        vulnerabilities = [
            'CVE-2024-1234', 'CVE-2024-5678', 'CVE-2023-9012',
            'None', 'None', 'None'  # Çoğu temiz
        ]

        for i in range(count):
            random_time = now - timedelta(
                days=random.randint(0, 7),
                hours=random.randint(0, 23)
            )

            open_ports = random.sample(self.PORTS, k=random.randint(1, 5))
            vuln = random.choice(vulnerabilities)
            risk_score = random.randint(1, 10) if vuln != 'None' else random.randint(1, 4)

            scan = {
                'scan_time': random_time.strftime('%Y-%m-%d %H:%M:%S'),
                'ip_address': random.choice(self.MALICIOUS_IPS),
                'open_ports': json.dumps(open_ports),
                'os_info': random.choice(os_list),
                'vulnerabilities': vuln,
                'risk_score': risk_score
            }

            scans.append(scan)

        return scans

    def _generate_attack_description(self, attack_type: str) -> str:
        """Saldırı tipi için açıklama üret"""
        descriptions = {
            'DDoS': 'Multiple connection attempts detected',
            'Port Scan': 'Sequential port scanning activity',
            'SQL Injection': 'Malicious SQL query in request',
            'Brute Force': 'Multiple failed authentication attempts',
            'XSS': 'Cross-site scripting attempt detected',
            'MITM': 'Man-in-the-middle attack suspected',
            'Phishing': 'Suspicious email or link detected',
            'Ransomware': 'File encryption activity detected',
            'Malware': 'Malicious software behavior detected',
            'Zero Day': 'Unknown exploit pattern detected',
            'DNS Spoofing': 'DNS response manipulation detected',
            'ARP Poisoning': 'ARP cache poisoning detected'
        }
        return descriptions.get(attack_type, 'Suspicious activity detected')

    def insert_to_database(self, attacks: List[Dict] = None,
                          logs: List[Dict] = None,
                          scans: List[Dict] = None):
        """
        Üretilen verileri database'e ekle

        Args:
            attacks: Saldırı listesi
            logs: Log listesi
            scans: Tarama listesi
        """

        # Attacks ekle
        if attacks:
            for attack in attacks:
                self.cursor.execute('''
                    INSERT INTO attacks 
                    (timestamp, attack_type, source_ip, destination_ip, port, 
                     severity, status, description, blocked)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    attack['timestamp'],
                    attack['attack_type'],
                    attack['source_ip'],
                    attack['destination_ip'],
                    attack['port'],
                    attack['severity'],
                    attack['status'],
                    attack['description'],
                    attack['blocked']
                ))

        # Logs ekle
        if logs:
            for log in logs:
                self.cursor.execute('''
                    INSERT INTO logs 
                    (timestamp, level, source, message, ip_address)
                    VALUES (?, ?, ?, ?, ?)
                ''', (
                    log['timestamp'],
                    log['level'],
                    log['source'],
                    log['message'],
                    log['ip_address']
                ))

        # Scans ekle
        if scans:
            for scan in scans:
                self.cursor.execute('''
                    INSERT INTO network_scans 
                    (scan_time, ip_address, open_ports, os_info, 
                     vulnerabilities, risk_score)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (
                    scan['scan_time'],
                    scan['ip_address'],
                    scan['open_ports'],
                    scan['os_info'],
                    scan['vulnerabilities'],
                    scan['risk_score']
                ))

        self.conn.commit()
        print("✅ Veriler database'e eklendi!")

    def clear_database(self):
        """Tüm verileri temizle"""
        self.cursor.execute('DELETE FROM attacks')
        self.cursor.execute('DELETE FROM logs')
        self.cursor.execute('DELETE FROM network_scans')
        self.conn.commit()
        print("🗑️ Database temizlendi!")

    def generate_all(self, attack_count: int = 100,
                    log_count: int = 200,
                    scan_count: int = 50,
                    clear_first: bool = False):
        """
        Tüm mock verileri üret ve database'e ekle

        Args:
            attack_count: Saldırı sayısı
            log_count: Log sayısı
            scan_count: Tarama sayısı
            clear_first: Önce database'i temizle
        """
        if clear_first:
            self.clear_database()

        print("🔄 Mock veriler üretiliyor...")

        # Üret
        attacks = self.generate_attacks(attack_count)
        logs = self.generate_logs(log_count)
        scans = self.generate_network_scans(scan_count)

        print(f"✅ {len(attacks)} saldırı üretildi")
        print(f"✅ {len(logs)} log üretildi")
        print(f"✅ {len(scans)} tarama üretildi")

        # Database'e ekle
        self.insert_to_database(attacks, logs, scans)

        print("\n🎉 Mock veri oluşturma tamamlandı!")
        print(f"📊 Toplam: {attack_count + log_count + scan_count} kayıt")

    def close(self):
        """Database bağlantısını kapat"""
        self.conn.close()


# Test için
if __name__ == "__main__":
    generator = MockDataGenerator()

    # ÇOK DAHA FAZLA VERİ! 🔥🔥🔥
    generator.generate_all(
        attack_count=5000,   # 5000 saldırı! (6 tür = ~833 örnek/tür)
        log_count=10000,     # 10000 log!
        scan_count=2500,     # 2500 tarama!
        clear_first=True     # Önce temizle
    )

    generator.close()
    print("\n✅ Hazır! Şimdi modeli eğit!")