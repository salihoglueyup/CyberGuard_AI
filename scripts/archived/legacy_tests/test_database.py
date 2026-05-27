# tests/test_database.py

"""
Database test dosyası
Tüm database fonksiyonlarını test eder
"""

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.utils.database import DatabaseManager
import unittest
from datetime import datetime


class TestDatabaseManager(unittest.TestCase):
    """Database Manager test sınıfı"""

    @classmethod
    def setUpClass(cls):
        """Test başlamadan önce"""
        cls.db = DatabaseManager("test_db.db")
        print("\n🧪 Test database oluşturuldu\n")

    @classmethod
    def tearDownClass(cls):
        """Test bittikten sonra"""
        if os.path.exists("test_db.db"):
            os.remove("test_db.db")
        print("\n🗑️  Test database silindi\n")

    def test_01_create_tables(self):
        """Tabloların oluşturulmasını test et"""
        print("Test 1: Tablo oluşturma")

        stats = self.db.get_database_stats()

        # Tüm tablolar 0 kayıtla başlamalı
        self.assertEqual(stats['attacks'], 0)
        self.assertEqual(stats['network_logs'], 0)

        print("  ✅ Tablolar başarıyla oluşturuldu")

    def test_02_add_attack(self):
        """Saldırı eklemeyi test et"""
        print("Test 2: Saldırı ekleme")

        attack_data = {
            'attack_type': 'DDoS',
            'source_ip': '192.168.1.100',
            'destination_ip': '10.0.0.1',
            'source_port': 54321,
            'destination_port': 80,
            'protocol': 'TCP',
            'severity': 'HIGH',
            'confidence': 0.95,
            'packet_count': 10000,
            'bytes_transferred': 5000000,
            'duration': 120.5,
            'blocked': True,
            'description': 'Test DDoS attack'
        }

        attack_id = self.db.add_attack(attack_data)

        self.assertIsNotNone(attack_id)
        self.assertGreater(attack_id, 0)

        print(f"  ✅ Saldırı eklendi (ID: {attack_id})")

    def test_03_get_recent_attacks(self):
        """Son saldırıları getirmeyi test et"""
        print("Test 3: Son saldırıları getir")

        # Birkaç saldırı daha ekle
        for i in range(5):
            self.db.add_attack({
                'attack_type': f'Test_{i}',
                'source_ip': f'192.168.1.{i}',
                'destination_ip': '10.0.0.1',
                'severity': 'MEDIUM',
                'confidence': 0.8
            })

        attacks = self.db.get_recent_attacks(limit=10)

        self.assertGreater(len(attacks), 0)
        self.assertLessEqual(len(attacks), 10)

        print(f"  ✅ {len(attacks)} saldırı getirildi")

    def test_04_get_attack_stats(self):
        """İstatistikleri test et"""
        print("Test 4: Saldırı istatistikleri")

        stats = self.db.get_attack_stats(24)

        self.assertIn('total', stats)
        self.assertIn('by_severity', stats)
        self.assertGreater(stats['total'], 0)

        print(f"  ✅ Toplam saldırı: {stats['total']}")

    def test_05_add_network_log(self):
        """Network log eklemeyi test et"""
        print("Test 5: Network log ekleme")

        log_data = {
            'source_ip': '192.168.1.50',
            'destination_ip': '10.0.0.1',
            'source_port': 12345,
            'destination_port': 80,
            'protocol': 'TCP',
            'packet_size': 1024,
            'is_attack': False,
            'prediction_confidence': 0.75
        }

        log_id = self.db.add_network_log(log_data)

        self.assertIsNotNone(log_id)
        print(f"  ✅ Network log eklendi (ID: {log_id})")

    def test_06_ip_history(self):
        """IP geçmişini test et"""
        print("Test 6: IP geçmişi")

        test_ip = '192.168.1.50'

        # Birkaç log daha ekle
        for _ in range(3):
            self.db.add_network_log({
                'source_ip': test_ip,
                'destination_ip': '10.0.0.1',
                'source_port': 12345,
                'destination_port': 80,
                'protocol': 'TCP',
                'packet_size': 1024
            })

        history = self.db.get_ip_history(test_ip)

        self.assertGreater(len(history), 0)
        print(f"  ✅ {len(history)} kayıt bulundu")

    def test_07_add_scan_result(self):
        """Tarama sonucu eklemeyi test et"""
        print("Test 7: Tarama sonucu ekleme")

        scan_data = {
            'file_name': 'malware.exe',
            'file_path': '/tmp/malware.exe',
            'file_size': 1024000,
            'file_hash': 'abcdef123456',
            'scan_type': 'full_scan',
            'is_malware': True,
            'malware_type': 'Trojan',
            'confidence': 0.97,
            'risk_score': 9.5,
            'scan_duration': 2.3
        }

        scan_id = self.db.add_scan_result(scan_data)

        self.assertIsNotNone(scan_id)
        print(f"  ✅ Tarama sonucu eklendi (ID: {scan_id})")

    def test_08_check_file_hash(self):
        """Hash kontrolünü test et"""
        print("Test 8: Dosya hash kontrolü")

        result = self.db.check_file_hash('abcdef123456')

        self.assertIsNotNone(result)
        self.assertEqual(result['file_hash'], 'abcdef123456')

        print(f"  ✅ Hash bulundu: {result['file_name']}")

    def test_09_blacklist_operations(self):
        """Blacklist işlemlerini test et"""
        print("Test 9: IP Blacklist")

        test_ip = '192.168.1.200'

        # IP'yi blacklist'e ekle
        bl_id = self.db.add_to_blacklist(
            ip_address=test_ip,
            reason="Test blacklist",
            permanent=False,
            duration_hours=24
        )

        self.assertIsNotNone(bl_id)

        # Blacklist'te mi kontrol et
        is_blocked = self.db.is_blacklisted(test_ip)
        self.assertTrue(is_blocked)

        print(f"  ✅ IP blacklist'e eklendi ve doğrulandı")

    def test_10_chat_operations(self):
        """Chat işlemlerini test et"""
        print("Test 10: Chat history")

        session_id = "test_session_123"

        # Mesaj ekle
        chat_id = self.db.add_chat_message({
            'user_message': "Test sorusu",
            'bot_response': "Test cevabı",
            'intent': 'test',
            'response_time': 1.5,
            'user_id': 'test_user',
            'session_id': session_id
        })

        self.assertIsNotNone(chat_id)

        # Geçmişi getir
        history = self.db.get_chat_history(session_id)

        self.assertGreater(len(history), 0)
        print(f"  ✅ Chat mesajı eklendi ve getirildi")

    def test_11_system_metrics(self):
        """Metrik işlemlerini test et"""
        print("Test 11: Sistem metrikleri")

        # Metrik ekle
        metric_id = self.db.add_metric(
            metric_type='cpu_usage',
            value=75.5,
            unit='%'
        )

        self.assertIsNotNone(metric_id)

        # Metrikleri getir
        metrics = self.db.get_metrics('cpu_usage', hours=24)

        self.assertGreater(len(metrics), 0)
        print(f"  ✅ {len(metrics)} metrik kaydı bulundu")

    def test_12_database_stats(self):
        """Database istatistiklerini test et"""
        print("Test 12: Database istatistikleri")

        stats = self.db.get_database_stats()

        self.assertIn('attacks', stats)
        self.assertIn('db_size_mb', stats)

        print(f"  ✅ Database boyutu: {stats['db_size_mb']} MB")

    def test_13_top_attackers(self):
        """En çok saldırı yapan IP'leri test et"""
        print("Test 13: En çok saldıran IP'ler")

        top_attackers = self.db.get_top_attackers(limit=5)

        self.assertIsInstance(top_attackers, list)

        if len(top_attackers) > 0:
            print(f"  ✅ {len(top_attackers)} saldırgan IP bulundu")
            print(f"     En aktif: {top_attackers[0]['source_ip']}")
        else:
            print(f"  ✅ Test geçti (veri yok)")


def run_tests():
    """Testleri çalıştır"""
    print("=" * 60)
    print("🧪 CYBERGUARD AI - DATABASE TESTLER")
    print("=" * 60)
    print()

    # Test suite oluştur
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestDatabaseManager)

    # Test runner
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    print("\n" + "=" * 60)

    if result.wasSuccessful():
        print("✅ TÜM TESTLER BAŞARILI!")
    else:
        print("❌ BAZI TESTLER BAŞARISIZ!")
        print(f"   Başarısız: {len(result.failures)}")
        print(f"   Hata: {len(result.errors)}")

    print("=" * 60)

    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)