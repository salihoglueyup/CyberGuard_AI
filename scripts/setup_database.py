# scripts/setup_database.py

"""
Database kurulum scripti
İlk çalıştırmada tüm tabloları oluşturur ve test verileri ekler
"""

import sys
import os

# Parent directory'yi path'e ekle
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.utils.database import DatabaseManager
from datetime import datetime, timedelta
import random


def generate_sample_data(db: DatabaseManager, num_records: int = 100):
    """
    Test için örnek veri oluştur

    Args:
        db: DatabaseManager instance
        num_records: Oluşturulacak kayıt sayısı
    """

    print(f"\n📝 {num_records} adet örnek veri oluşturuluyor...\n")

    # Örnek IP'ler
    sample_ips = [
        '192.168.1.100', '192.168.1.101', '192.168.1.105',
        '10.0.0.23', '10.0.0.47', '172.16.0.8',
        '185.220.101.43', '194.26.29.45'
    ]

    # Saldırı türleri
    attack_types = [
        'DDoS', 'Port Scan', 'SQL Injection', 'XSS',
        'Brute Force', 'Malware', 'Phishing', 'MITM'
    ]

    # Severity seviyeleri
    severities = ['LOW', 'MEDIUM', 'HIGH', 'CRITICAL']

    # Protokoller
    protocols = ['TCP', 'UDP', 'ICMP', 'HTTP', 'HTTPS']

    # Saldırılar ekle
    print("⚔️  Saldırı kayıtları ekleniyor...")
    for i in range(num_records):
        timestamp = datetime.now() - timedelta(
            hours=random.randint(0, 168)  # Son 7 gün
        )

        attack_data = {
            'attack_type': random.choice(attack_types),
            'source_ip': random.choice(sample_ips),
            'destination_ip': '10.0.0.1',
            'source_port': random.randint(1024, 65535),
            'destination_port': random.choice([80, 443, 22, 21, 3306]),
            'protocol': random.choice(protocols),
            'severity': random.choice(severities),
            'confidence': round(random.uniform(0.7, 0.99), 2),
            'packet_count': random.randint(100, 10000),
            'bytes_transferred': random.randint(1000, 5000000),
            'duration': round(random.uniform(1, 300), 2),
            'blocked': random.choice([True, False]),
            'description': f'Test attack #{i + 1}'
        }

        db.add_attack(attack_data)

        if (i + 1) % 20 == 0:
            print(f"  ✓ {i + 1}/{num_records} saldırı eklendi")

    print(f"✅ {num_records} saldırı kaydı eklendi!\n")

    # Network logs ekle
    print("📡 Network log kayıtları ekleniyor...")
    for i in range(num_records * 2):
        log_data = {
            'source_ip': random.choice(sample_ips),
            'destination_ip': '10.0.0.1',
            'source_port': random.randint(1024, 65535),
            'destination_port': random.choice([80, 443, 22]),
            'protocol': random.choice(protocols),
            'packet_size': random.randint(64, 1500),
            'flags': random.choice(['SYN', 'ACK', 'FIN', 'PSH']),
            'service': random.choice(['http', 'https', 'ssh', 'ftp']),
            'is_attack': random.choice([True, False]),
            'prediction_confidence': round(random.uniform(0.6, 0.99), 2)
        }

        db.add_network_log(log_data)

    print(f"✅ {num_records * 2} network log kaydı eklendi!\n")

    # Scan results ekle
    print("🔍 Tarama sonuçları ekleniyor...")
    malware_types = ['Trojan', 'Virus', 'Worm', 'Ransomware', 'Spyware', 'Benign']
    file_extensions = ['.exe', '.dll', '.bat', '.ps1', '.sh']

    for i in range(50):
        is_malware = random.choice([True, False])
        scan_data = {
            'file_name': f'file_{i + 1}{random.choice(file_extensions)}',
            'file_path': f'/path/to/file_{i + 1}',
            'file_size': random.randint(1024, 10485760),
            'file_hash': f'hash_{i + 1}_' + ''.join(random.choices('abcdef0123456789', k=40)),
            'scan_type': 'full_scan',
            'is_malware': is_malware,
            'malware_type': random.choice(malware_types) if is_malware else 'Benign',
            'confidence': round(random.uniform(0.8, 0.99), 2),
            'risk_score': round(random.uniform(0, 10), 1),
            'quarantined': is_malware and random.choice([True, False]),
            'scan_duration': round(random.uniform(0.5, 5.0), 2)
        }

        db.add_scan_result(scan_data)

    print(f"✅ 50 tarama sonucu eklendi!\n")

    # Chat history ekle
    print("💬 Chat geçmişi ekleniyor...")
    sample_questions = [
        "Son 1 saatte kaç saldırı oldu?",
        "192.168.1.100 IP adresini analiz et",
        "Sistem durumu nedir?",
        "DDoS saldırısı nedir?",
        "En tehlikeli IP'leri listele"
    ]

    sample_responses = [
        "Son 1 saatte 23 saldırı tespit edildi.",
        "IP analizi tamamlandı. Risk skoru: 8.5/10",
        "Sistem normal çalışıyor. 156 paket analiz edildi.",
        "DDoS, dağıtık hizmet reddi saldırısıdır...",
        "En tehlikeli IP'ler listelendi."
    ]

    for i in range(30):
        chat_data = {
            'user_message': random.choice(sample_questions),
            'bot_response': random.choice(sample_responses),
            'intent': random.choice(['query_attacks', 'analyze_ip', 'ask_info']),
            'response_time': round(random.uniform(0.5, 2.0), 2),
            'user_id': 'test_user',
            'session_id': f'session_{random.randint(1, 5)}'
        }

        db.add_chat_message(chat_data)

    print(f"✅ 30 chat mesajı eklendi!\n")

    # IP Blacklist ekle
    print("🚫 IP Blacklist ekleniyor...")
    for ip in sample_ips[:3]:
        db.add_to_blacklist(
            ip_address=ip,
            reason="Çoklu saldırı denemesi",
            permanent=False,
            duration_hours=random.randint(24, 168)
        )

    print(f"✅ 3 IP blacklist'e eklendi!\n")

    # System metrics ekle
    print("📊 Sistem metrikleri ekleniyor...")
    metric_types = [
        'cpu_usage', 'memory_usage', 'disk_usage',
        'network_throughput', 'attack_rate'
    ]

    for _ in range(100):
        for metric_type in metric_types:
            db.add_metric(
                metric_type=metric_type,
                value=round(random.uniform(0, 100), 2),
                unit='%' if 'usage' in metric_type else 'Mbps'
            )

    print(f"✅ 500 metrik kaydı eklendi!\n")


def main():
    """Ana kurulum fonksiyonu"""

    print("=" * 60)
    print("🛡️  CYBERGUARD AI - DATABASE SETUP")
    print("=" * 60)

    # Database yolu
    db_path = "cyberguard.db"

    # Eğer database varsa soralım
    if os.path.exists(db_path):
        print(f"\n⚠️  Database zaten mevcut: {db_path}")
        response = input("Silip yeniden oluşturmak ister misiniz? (y/n): ")

        if response.lower() == 'y':
            os.remove(db_path)
            print("🗑️  Eski database silindi!")
        else:
            print("❌ Kurulum iptal edildi.")
            return

    # Database oluştur
    print("\n📦 Database oluşturuluyor...")
    db = DatabaseManager(db_path)

    # Örnek veri oluştur
    create_sample = input("\n🎲 Örnek veri oluşturulsun mu? (y/n): ")

    if create_sample.lower() == 'y':
        num_records = input("Kaç adet saldırı kaydı? (varsayılan: 100): ")
        num_records = int(num_records) if num_records.isdigit() else 100

        generate_sample_data(db, num_records)

    # İstatistikler
    print("\n" + "=" * 60)
    print("📊 DATABASE İSTATİSTİKLERİ")
    print("=" * 60)

    stats = db.get_database_stats()

    print(f"""
        📋 Tablolar:
        ├─ Attacks:         {stats['attacks']:,} kayıt
        ├─ Network Logs:    {stats['network_logs']:,} kayıt
        ├─ Scan Results:    {stats['scan_results']:,} kayıt
        ├─ Chat History:    {stats['chat_history']:,} kayıt
        ├─ System Metrics:  {stats['system_metrics']:,} kayıt
        └─ IP Blacklist:    {stats['ip_blacklist']:,} kayıt

        💾 Database Boyutu:  {stats['db_size_mb']} MB
        📍 Konum:            {os.path.abspath(db_path)}
        """)

    print("=" * 60)
    print("✅ DATABASE KURULUMU TAMAMLANDI!")
    print("=" * 60)

    print("""
        🚀 Sonraki Adımlar:

        1. Web uygulamasını başlat:
           streamlit run app/main.py

        2. Test scriptini çalıştır:
           python tests/test_database.py

        3. Modelleri eğit:
           python src/network_detection/train.py
        """)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n❌ Kurulum iptal edildi (Ctrl+C)")
    except Exception as e:
        print(f"\n\n❌ HATA: {str(e)}")
        import traceback

        traceback.print_exc()