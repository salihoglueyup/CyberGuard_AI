"""
Generate Sample Data - CyberGuard AI
Hızlı test verisi oluştur (mock_data_generator'ın basit versiyonu)

Dosya Yolu: scripts/generate_sample_data.py
"""

import os
import sys
import sqlite3
import random
from datetime import datetime, timedelta
from typing import List, Dict

# Proje root
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)


class SampleDataGenerator:
    """
    Hızlı örnek veri oluşturucu
    
    mock_data_generator'dan daha basit, hızlı test için
    """
    
    ATTACK_TYPES = ['DDoS', 'SQL Injection', 'XSS', 'Port Scan', 'Brute Force', 'Malware']
    SEVERITIES = ['low', 'medium', 'high', 'critical']
    PROTOCOLS = ['TCP', 'UDP', 'ICMP']
    
    def __init__(self, db_path: str = None):
        """
        Args:
            db_path: Database yolu (None = proje root'taki cyberguard.db)
        """
        if db_path is None:
            db_path = os.path.join(project_root, 'src', 'database', 'cyberguard.db')
        
        self.db_path = db_path
        print(f"🎲 Sample Data Generator")
        print(f"📁 Database: {db_path}")
    
    def generate_quick_sample(self, count: int = 100) -> List[Dict]:
        """
        Hızlı örnek veri oluştur
        
        Args:
            count: Kayıt sayısı
            
        Returns:
            Kayıt listesi
        """
        print(f"\n⚡ {count} kayıt oluşturuluyor...")
        
        records = []
        now = datetime.now()
        
        for i in range(count):
            timestamp = now - timedelta(
                days=random.randint(0, 7),
                hours=random.randint(0, 23),
                minutes=random.randint(0, 59)
            )
            
            attack_type = random.choice(self.ATTACK_TYPES)
            
            record = {
                'timestamp': timestamp.isoformat(),
                'attack_type': attack_type,
                'source_ip': f"192.168.{random.randint(0,255)}.{random.randint(1,254)}",
                'destination_ip': f"10.0.{random.randint(0,255)}.{random.randint(1,254)}",
                'port': random.choice([22, 80, 443, 3306, 8080, 3389]),
                'protocol': random.choice(self.PROTOCOLS),
                'severity': random.choice(self.SEVERITIES),
                'packet_size': random.randint(64, 1500),
                'blocked': random.choice([0, 1]),
                'description': f"{attack_type} attempt detected"
            }
            
            records.append(record)
        
        print(f"✅ {len(records)} kayıt oluşturuldu")
        return records
    
    def save_to_database(self, records: List[Dict]) -> int:
        """
        Kayıtları veritabanına ekle
        
        Args:
            records: Kayıt listesi
            
        Returns:
            Eklenen kayıt sayısı
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Tablo yoksa oluştur
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS attacks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                attack_type TEXT NOT NULL,
                source_ip TEXT NOT NULL,
                destination_ip TEXT NOT NULL,
                port INTEGER NOT NULL,
                protocol TEXT NOT NULL,
                severity TEXT NOT NULL,
                packet_size INTEGER NOT NULL,
                blocked INTEGER NOT NULL,
                description TEXT
            )
        ''')
        
        # Kayıtları ekle
        for record in records:
            cursor.execute('''
                INSERT INTO attacks 
                (timestamp, attack_type, source_ip, destination_ip, port, 
                 protocol, severity, packet_size, blocked, description)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                record['timestamp'],
                record['attack_type'],
                record['source_ip'],
                record['destination_ip'],
                record['port'],
                record['protocol'],
                record['severity'],
                record['packet_size'],
                record['blocked'],
                record['description']
            ))
        
        conn.commit()
        conn.close()
        
        print(f"💾 {len(records)} kayıt veritabanına eklendi")
        return len(records)
    
    def get_db_stats(self) -> Dict:
        """Veritabanı istatistikleri"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT COUNT(*) FROM attacks")
        total = cursor.fetchone()[0]
        
        cursor.execute("SELECT attack_type, COUNT(*) FROM attacks GROUP BY attack_type")
        by_type = dict(cursor.fetchall())
        
        conn.close()
        
        return {'total': total, 'by_type': by_type}


def main():
    """Ana fonksiyon"""
    
    print("\n" + "=" * 60)
    print("⚡ CYBERGUARD AI - HIZLI ÖRNEK VERİ OLUŞTURUCU")
    print("=" * 60)
    
    generator = SampleDataGenerator()
    
    print("\n📋 Seçenekler:")
    print("  1. Hızlı test verisi oluştur (100 kayıt)")
    print("  2. Özel miktarda veri oluştur")
    print("  3. Mevcut veritabanı istatistikleri")
    print("  4. Çıkış")
    
    choice = input("\nSeçiminiz (1-4): ").strip()
    
    if choice == '1':
        records = generator.generate_quick_sample(100)
        confirm = input("\n💾 Veritabanına kaydetmek ister misiniz? (E/H): ").strip().upper()
        if confirm == 'E':
            generator.save_to_database(records)
            
    elif choice == '2':
        count = int(input("Kaç kayıt oluşturulsun? ").strip() or "500")
        records = generator.generate_quick_sample(count)
        confirm = input("\n💾 Veritabanına kaydetmek ister misiniz? (E/H): ").strip().upper()
        if confirm == 'E':
            generator.save_to_database(records)
            
    elif choice == '3':
        stats = generator.get_db_stats()
        print(f"\n📊 Veritabanı İstatistikleri:")
        print(f"   Toplam: {stats['total']:,} kayıt")
        if stats['by_type']:
            print(f"   Saldırı türleri:")
            for attack, count in stats['by_type'].items():
                print(f"      • {attack}: {count:,}")
                
    elif choice == '4':
        print("\n👋 Çıkış...")
    else:
        print("❌ Geçersiz seçim!")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️ İptal edildi!")
    except Exception as e:
        print(f"\n❌ HATA: {e}")
        import traceback
        traceback.print_exc()
