"""
Database Initialization - CyberGuard AI
Veritabanı tablolarını oluştur ve başlangıç verilerini ekle

Dosya Yolu: src/database/init_db.py
"""

import os
import sys
import sqlite3
from datetime import datetime

# Proje root
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


def create_tables(db_path: str = None):
    """
    Tüm tabloları oluştur
    
    Args:
        db_path: Database dosya yolu
    """
    if db_path is None:
        db_path = os.path.join(project_root, 'src', 'database', 'cyberguard.db')
    
    print(f"📦 Database oluşturuluyor: {db_path}")
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Attacks tablosu
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS attacks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            attack_type TEXT NOT NULL,
            source_ip TEXT NOT NULL,
            destination_ip TEXT,
            port INTEGER,
            protocol TEXT,
            severity TEXT,
            packet_size INTEGER,
            blocked INTEGER DEFAULT 0,
            is_anomaly INTEGER DEFAULT 0,
            anomaly_score REAL DEFAULT 0.0,
            description TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    print("  ✓ attacks tablosu oluşturuldu (is_anomaly, anomaly_score dahil)")
    
    # Network logs tablosu
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS network_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            source_ip TEXT NOT NULL,
            destination_ip TEXT,
            source_port INTEGER,
            destination_port INTEGER,
            protocol TEXT,
            packet_size INTEGER,
            flags TEXT,
            service TEXT,
            is_attack INTEGER DEFAULT 0,
            prediction_confidence REAL,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    print("  ✓ network_logs tablosu oluşturuldu")
    
    # Scan results tablosu
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS scan_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            file_name TEXT NOT NULL,
            file_path TEXT,
            file_size INTEGER,
            file_hash TEXT,
            scan_type TEXT,
            is_malware INTEGER DEFAULT 0,
            malware_type TEXT,
            confidence REAL,
            risk_score REAL,
            quarantined INTEGER DEFAULT 0,
            scan_duration REAL,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    print("  ✓ scan_results tablosu oluşturuldu")
    
    # Chat history tablosu
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS chat_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            user_message TEXT NOT NULL,
            bot_response TEXT,
            intent TEXT,
            response_time REAL,
            user_id TEXT,
            session_id TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    print("  ✓ chat_history tablosu oluşturuldu")
    
    # System metrics tablosu
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS system_metrics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            metric_type TEXT NOT NULL,
            value REAL NOT NULL,
            unit TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    print("  ✓ system_metrics tablosu oluşturuldu")
    
    # IP blacklist tablosu
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS ip_blacklist (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ip_address TEXT NOT NULL UNIQUE,
            reason TEXT,
            permanent INTEGER DEFAULT 0,
            expires_at TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    print("  ✓ ip_blacklist tablosu oluşturuldu")
    
    # Defences tablosu - Savunma başarısı takibi
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
            operator TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (attack_id) REFERENCES attacks(id)
        )
    ''')
    print("  ✓ defences tablosu oluşturuldu")
    
    conn.commit()
    conn.close()
    
    print(f"✅ Database başarıyla oluşturuldu: {db_path}")


def main():
    """Ana fonksiyon"""
    print("\n" + "=" * 50)
    print("🛡️  CYBERGUARD AI - DATABASE INITIALIZATION")
    print("=" * 50 + "\n")
    
    create_tables()
    
    print("\n" + "=" * 50)
    print("✅ TAMAMLANDI")
    print("=" * 50)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"❌ HATA: {e}")
        import traceback
        traceback.print_exc()
