# src/utils/database.py

"""
Database yönetimi ve işlemleri
SQLite kullanarak tüm verileri saklama
"""

import sqlite3
import os
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
import json
import pandas as pd
from contextlib import contextmanager


class DatabaseManager:
    """
    CyberGuard AI Database Manager

    Tüm database işlemlerini yönetir:
    - Tablo oluşturma
    - CRUD operasyonları
    - Query'ler
    - Veri temizleme
    """

    def __init__(self, db_path: str = "src/database/cyberguard.db"):
        """
        Database manager'ı başlat

        Args:
            db_path (str): Database dosya yolu
        """
        self.db_path = db_path
        self.connection = None
        self._tables_created = False

        # Database klasörünü oluştur
        db_dir = os.path.dirname(db_path)
        if db_dir and not os.path.exists(db_dir):
            try:
                os.makedirs(db_dir, exist_ok=True)
            except Exception as e:
                print(f"⚠️ Database klasörü oluşturulamadı: {e}")

        # Database'i oluştur (yoksa) - hata yakalama ile
        try:
            self.create_tables()
        except Exception as e:
            print(f"⚠️ Database tabloları oluşturulurken hata (devam ediyor): {e}")

    @contextmanager
    def get_connection(self):
        """
        Database bağlantısı context manager

        Kullanım:
            with db.get_connection() as conn:
                cursor = conn.cursor()
                ...
        """
        conn = None
        try:
            conn = sqlite3.connect(self.db_path, timeout=30.0)
            conn.row_factory = sqlite3.Row  # Dict-like access
            yield conn
        except sqlite3.OperationalError as e:
            print(f"⚠️ SQLite bağlantı hatası: {e}")
            # Boş bir mock connection döndür
            raise
        finally:
            if conn:
                conn.close()

    def create_tables(self):
        """Tüm tabloları oluştur"""

        with self.get_connection() as conn:
            cursor = conn.cursor()

            # 1. ATTACKS TABLE
            cursor.execute(
                """
                           CREATE TABLE IF NOT EXISTS attacks
                           (
                               id
                               INTEGER
                               PRIMARY
                               KEY
                               AUTOINCREMENT,
                               timestamp
                               DATETIME
                               DEFAULT
                               CURRENT_TIMESTAMP,
                               attack_type
                               VARCHAR
                           (
                               50
                           ) NOT NULL,
                               source_ip VARCHAR
                           (
                               45
                           ) NOT NULL,
                               destination_ip VARCHAR
                           (
                               45
                           ),
                               source_port INTEGER,
                               destination_port INTEGER,
                               protocol VARCHAR
                           (
                               10
                           ),
                               severity VARCHAR
                           (
                               20
                           ) NOT NULL,
                               confidence FLOAT,
                               packet_count INTEGER,
                               bytes_transferred INTEGER,
                               duration FLOAT,
                               blocked BOOLEAN DEFAULT 0,
                               description TEXT,
                               model_prediction TEXT,
                               created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                               )
                           """
            )

            # Index'ler
            cursor.execute(
                """
                           CREATE INDEX IF NOT EXISTS idx_attacks_timestamp
                               ON attacks(timestamp)
                           """
            )
            cursor.execute(
                """
                           CREATE INDEX IF NOT EXISTS idx_attacks_source_ip
                               ON attacks(source_ip)
                           """
            )
            cursor.execute(
                """
                           CREATE INDEX IF NOT EXISTS idx_attacks_type
                               ON attacks(attack_type)
                           """
            )

            # 2. NETWORK LOGS TABLE
            cursor.execute(
                """
                           CREATE TABLE IF NOT EXISTS network_logs
                           (
                               id
                               INTEGER
                               PRIMARY
                               KEY
                               AUTOINCREMENT,
                               timestamp
                               DATETIME
                               DEFAULT
                               CURRENT_TIMESTAMP,
                               source_ip
                               VARCHAR
                           (
                               45
                           ) NOT NULL,
                               destination_ip VARCHAR
                           (
                               45
                           ) NOT NULL,
                               source_port INTEGER,
                               destination_port INTEGER,
                               protocol VARCHAR
                           (
                               10
                           ),
                               packet_size INTEGER,
                               flags VARCHAR
                           (
                               20
                           ),
                               service VARCHAR
                           (
                               50
                           ),
                               is_attack BOOLEAN DEFAULT 0,
                               prediction_confidence FLOAT,
                               features TEXT,
                               created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                               )
                           """
            )

            cursor.execute(
                """
                           CREATE INDEX IF NOT EXISTS idx_network_logs_timestamp
                               ON network_logs(timestamp)
                           """
            )

            # 3. SCAN RESULTS TABLE
            cursor.execute(
                """
                           CREATE TABLE IF NOT EXISTS scan_results
                           (
                               id
                               INTEGER
                               PRIMARY
                               KEY
                               AUTOINCREMENT,
                               timestamp
                               DATETIME
                               DEFAULT
                               CURRENT_TIMESTAMP,
                               file_name
                               VARCHAR
                           (
                               255
                           ) NOT NULL,
                               file_path TEXT,
                               file_size INTEGER,
                               file_hash VARCHAR
                           (
                               64
                           ),
                               scan_type VARCHAR
                           (
                               50
                           ),
                               is_malware BOOLEAN DEFAULT 0,
                               malware_type VARCHAR
                           (
                               50
                           ),
                               confidence FLOAT,
                               risk_score FLOAT,
                               quarantined BOOLEAN DEFAULT 0,
                               quarantine_path TEXT,
                               scan_duration FLOAT,
                               model_prediction TEXT,
                               created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                               )
                           """
            )

            cursor.execute(
                """
                           CREATE INDEX IF NOT EXISTS idx_scan_results_timestamp
                               ON scan_results(timestamp)
                           """
            )
            cursor.execute(
                """
                           CREATE INDEX IF NOT EXISTS idx_scan_results_file_hash
                               ON scan_results(file_hash)
                           """
            )

            # 4. CHAT HISTORY TABLE
            cursor.execute(
                """
                           CREATE TABLE IF NOT EXISTS chat_history
                           (
                               id
                               INTEGER
                               PRIMARY
                               KEY
                               AUTOINCREMENT,
                               timestamp
                               DATETIME
                               DEFAULT
                               CURRENT_TIMESTAMP,
                               user_message
                               TEXT
                               NOT
                               NULL,
                               bot_response
                               TEXT
                               NOT
                               NULL,
                               intent
                               VARCHAR
                           (
                               50
                           ),
                               context_used TEXT,
                               response_time FLOAT,
                               user_id VARCHAR
                           (
                               50
                           ),
                               session_id VARCHAR
                           (
                               100
                           ),
                               created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                               )
                           """
            )

            cursor.execute(
                """
                           CREATE INDEX IF NOT EXISTS idx_chat_history_timestamp
                               ON chat_history(timestamp)
                           """
            )
            cursor.execute(
                """
                           CREATE INDEX IF NOT EXISTS idx_chat_history_session
                               ON chat_history(session_id)
                           """
            )

            # 5. SYSTEM METRICS TABLE
            cursor.execute(
                """
                           CREATE TABLE IF NOT EXISTS system_metrics
                           (
                               id
                               INTEGER
                               PRIMARY
                               KEY
                               AUTOINCREMENT,
                               timestamp
                               DATETIME
                               DEFAULT
                               CURRENT_TIMESTAMP,
                               metric_type
                               VARCHAR
                           (
                               50
                           ) NOT NULL,
                               metric_value FLOAT NOT NULL,
                               metric_unit VARCHAR
                           (
                               20
                           ),
                               metadata TEXT,
                               created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                               )
                           """
            )

            cursor.execute(
                """
                           CREATE INDEX IF NOT EXISTS idx_system_metrics_timestamp
                               ON system_metrics(timestamp)
                           """
            )
            cursor.execute(
                """
                           CREATE INDEX IF NOT EXISTS idx_system_metrics_type
                               ON system_metrics(metric_type)
                           """
            )

            # 6. IP BLACKLIST TABLE
            cursor.execute(
                """
                           CREATE TABLE IF NOT EXISTS ip_blacklist
                           (
                               id
                               INTEGER
                               PRIMARY
                               KEY
                               AUTOINCREMENT,
                               ip_address
                               VARCHAR
                           (
                               45
                           ) NOT NULL UNIQUE,
                               reason TEXT,
                               attack_count INTEGER DEFAULT 1,
                               first_seen DATETIME DEFAULT CURRENT_TIMESTAMP,
                               last_seen DATETIME DEFAULT CURRENT_TIMESTAMP,
                               blocked_until DATETIME,
                               permanent BOOLEAN DEFAULT 0,
                               added_by VARCHAR
                           (
                               50
                           ),
                               created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                               )
                           """
            )

            cursor.execute(
                """
                           CREATE INDEX IF NOT EXISTS idx_ip_blacklist_ip
                               ON ip_blacklist(ip_address)
                           """
            )

            # 7. SYSTEM LOGS TABLE
            cursor.execute(
                """
                           CREATE TABLE IF NOT EXISTS system_logs
                           (
                               id INTEGER PRIMARY KEY AUTOINCREMENT,
                               timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                               level VARCHAR(20) NOT NULL,
                               source VARCHAR(50) NOT NULL,
                               message TEXT NOT NULL,
                               details TEXT,
                               user_id VARCHAR(50),
                               session_id VARCHAR(100),
                               created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                           )
                           """
            )

            cursor.execute(
                """
                           CREATE INDEX IF NOT EXISTS idx_system_logs_timestamp
                               ON system_logs(timestamp)
                           """
            )
            cursor.execute(
                """
                           CREATE INDEX IF NOT EXISTS idx_system_logs_level
                               ON system_logs(level)
                           """
            )

            # 8. FILE QUARANTINE TABLE
            cursor.execute(
                """
                           CREATE TABLE IF NOT EXISTS file_quarantine
                           (
                               id
                               INTEGER
                               PRIMARY
                               KEY
                               AUTOINCREMENT,
                               timestamp
                               DATETIME
                               DEFAULT
                               CURRENT_TIMESTAMP,
                               original_path
                               TEXT
                               NOT
                               NULL,
                               quarantine_path
                               TEXT
                               NOT
                               NULL,
                               file_name
                               VARCHAR
                           (
                               255
                           ) NOT NULL,
                               file_hash VARCHAR
                           (
                               64
                           ),
                               malware_type VARCHAR
                           (
                               50
                           ),
                               risk_score FLOAT,
                               quarantine_reason TEXT,
                               restored BOOLEAN DEFAULT 0,
                               deleted BOOLEAN DEFAULT 0,
                               created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                               )
                           """
            )

            cursor.execute(
                """
                           CREATE INDEX IF NOT EXISTS idx_file_quarantine_hash
                               ON file_quarantine(file_hash)
                           """
            )

            conn.commit()
            print("✅ Database tabloları oluşturuldu!")

    # ========================================
    # ATTACK OPERATIONS
    # ========================================

    def add_attack(self, attack_data: Dict) -> int:
        """
        Yeni saldırı kaydı ekle

        Args:
            attack_data (dict): Saldırı bilgileri

        Returns:
            int: Eklenen kaydın ID'si
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute(
                """
                           INSERT INTO attacks (attack_type, source_ip, destination_ip,
                                                source_port, destination_port, protocol,
                                                severity, confidence, packet_count,
                                                bytes_transferred, duration, blocked,
                                                description, model_prediction)
                           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                           """,
                (
                    attack_data.get("attack_type"),
                    attack_data.get("source_ip"),
                    attack_data.get("destination_ip"),
                    attack_data.get("source_port"),
                    attack_data.get("destination_port"),
                    attack_data.get("protocol"),
                    attack_data.get("severity"),
                    attack_data.get("confidence"),
                    attack_data.get("packet_count"),
                    attack_data.get("bytes_transferred"),
                    attack_data.get("duration"),
                    attack_data.get("blocked", False),
                    attack_data.get("description"),
                    json.dumps(attack_data.get("model_prediction", {})),
                ),
            )

            conn.commit()
            return cursor.lastrowid

    def get_attacks(
        self,
        hours: Optional[int] = 24,
        attack_type: Optional[str] = None,
        severity: Optional[str] = None,
        limit: int = 200000,  # Model verisi için yüksek limit
    ) -> List[Dict]:
        """
        Saldırıları filtreli olarak getir

        Args:
            hours (int): Kaç saat geriye bakılacak (None = tüm zamanlar)
            attack_type (str): Saldırı tipi filtresi
            severity (str): Ciddiyet filtresi
            limit (int): Maksimum kayıt sayısı

        Returns:
            list: Saldırı listesi
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()

            # Dinamik WHERE oluştur
            conditions = []
            params = []

            if hours is not None:
                conditions.append("timestamp > datetime('now', '-' || ? || ' hours')")
                params.append(hours)

            if attack_type:
                conditions.append("attack_type = ?")
                params.append(attack_type)

            if severity:
                conditions.append("severity = ?")
                params.append(severity)

            where_clause = " AND ".join(conditions) if conditions else "1=1"
            params.append(limit)

            cursor.execute(
                f"""
                SELECT *
                FROM attacks
                WHERE {where_clause}
                ORDER BY timestamp DESC
                LIMIT ?
            """,
                params,
            )

            rows = cursor.fetchall()
            return [dict(row) for row in rows]

    def get_recent_attacks(self, limit: int = 10, hours: int = 24) -> List[Dict]:
        """
        Son saldırıları getir

        Args:
            limit (int): Maksimum kayıt sayısı
            hours (int): Kaç saat geriye bakılacak

        Returns:
            list: Saldırı listesi
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute(
                """
                           SELECT *
                           FROM attacks
                           WHERE timestamp > datetime('now', '-' || ? || ' hours')
                           ORDER BY timestamp DESC
                               LIMIT ?
                           """,
                (hours, limit),
            )

            rows = cursor.fetchall()
            return [dict(row) for row in rows]

    def get_attack_stats(self, hours: Optional[int] = 24) -> Dict:
        """
        Saldırı istatistikleri

        Args:
            hours (int): Kaç saat geriye bakılacak (None = tüm zamanlar)

        Returns:
            dict: İstatistikler
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()

            # Zaman filtresi
            if hours is None:
                time_filter = "1=1"  # Tüm zamanlar
                params = ()
            else:
                time_filter = "timestamp > datetime('now', '-' || ? || ' hours')"
                params = (hours,)

            # Toplam saldırı
            cursor.execute(
                f"""
                SELECT COUNT(*) as total
                FROM attacks
                WHERE {time_filter}
            """,
                params,
            )
            total = cursor.fetchone()["total"]

            # Severity'ye göre
            cursor.execute(
                f"""
                SELECT severity,
                       COUNT(*) as count
                FROM attacks
                WHERE {time_filter}
                GROUP BY severity
            """,
                params,
            )
            by_severity = {row["severity"]: row["count"] for row in cursor.fetchall()}

            # Türe göre
            cursor.execute(
                f"""
                SELECT attack_type,
                       COUNT(*) as count
                FROM attacks
                WHERE {time_filter}
                GROUP BY attack_type
                ORDER BY count DESC
                LIMIT 10
            """,
                params,
            )
            by_type = {row["attack_type"]: row["count"] for row in cursor.fetchall()}

            # Engellenen/Engellenmeyen
            cursor.execute(
                f"""
                SELECT blocked,
                       COUNT(*) as count
                FROM attacks
                WHERE {time_filter}
                GROUP BY blocked
            """,
                params,
            )
            blocking_stats = {row["blocked"]: row["count"] for row in cursor.fetchall()}

            return {
                "total": total,
                "by_severity": by_severity,
                "by_type": by_type,
                "blocked": blocking_stats.get(1, 0),
                "not_blocked": blocking_stats.get(0, 0),
                "period_hours": hours if hours else "all",
            }

    def get_top_attackers(self, limit: int = 10, hours: int = 24) -> List[Dict]:
        """
        En çok saldırı yapan IP'ler

        Args:
            limit (int): Maksimum kayıt
            hours (int): Kaç saat geriye

        Returns:
            list: IP listesi
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute(
                """
                           SELECT source_ip,
                                  COUNT(*)                    as attack_count,
                                  COUNT(DISTINCT attack_type) as unique_attacks,
                                  MAX(timestamp)              as last_attack,
                                  AVG(confidence)             as avg_confidence
                           FROM attacks
                           WHERE timestamp > datetime('now', '-' || ? || ' hours')
                           GROUP BY source_ip
                           ORDER BY attack_count DESC
                               LIMIT ?
                           """,
                (hours, limit),
            )

            return [dict(row) for row in cursor.fetchall()]

    # ========================================
    # NETWORK LOG OPERATIONS
    # ========================================

    def add_network_log(self, log_data: Dict) -> int:
        """Network log ekle"""
        with self.get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute(
                """
                           INSERT INTO network_logs (source_ip, destination_ip, source_port,
                                                     destination_port, protocol, packet_size,
                                                     flags, service, is_attack,
                                                     prediction_confidence, features)
                           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                           """,
                (
                    log_data.get("source_ip"),
                    log_data.get("destination_ip"),
                    log_data.get("source_port"),
                    log_data.get("destination_port"),
                    log_data.get("protocol"),
                    log_data.get("packet_size"),
                    log_data.get("flags"),
                    log_data.get("service"),
                    log_data.get("is_attack", False),
                    log_data.get("prediction_confidence"),
                    json.dumps(log_data.get("features", {})),
                ),
            )

            conn.commit()
            return cursor.lastrowid

    def get_ip_history(self, ip_address: str, limit: int = 100) -> List[Dict]:
        """Bir IP'nin geçmişini getir"""
        with self.get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute(
                """
                           SELECT *
                           FROM network_logs
                           WHERE source_ip = ?
                           ORDER BY timestamp DESC
                               LIMIT ?
                           """,
                (ip_address, limit),
            )

            return [dict(row) for row in cursor.fetchall()]

    # ========================================
    # SCAN RESULTS OPERATIONS
    # ========================================

    def add_scan_result(self, scan_data: Dict) -> int:
        """Tarama sonucu ekle"""
        with self.get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute(
                """
                           INSERT INTO scan_results (file_name, file_path, file_size, file_hash,
                                                     scan_type, is_malware, malware_type,
                                                     confidence, risk_score, quarantined,
                                                     quarantine_path, scan_duration, model_prediction)
                           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                           """,
                (
                    scan_data.get("file_name"),
                    scan_data.get("file_path"),
                    scan_data.get("file_size"),
                    scan_data.get("file_hash"),
                    scan_data.get("scan_type"),
                    scan_data.get("is_malware", False),
                    scan_data.get("malware_type"),
                    scan_data.get("confidence"),
                    scan_data.get("risk_score"),
                    scan_data.get("quarantined", False),
                    scan_data.get("quarantine_path"),
                    scan_data.get("scan_duration"),
                    json.dumps(scan_data.get("model_prediction", {})),
                ),
            )

            conn.commit()
            return cursor.lastrowid

    def get_scan_history(self, limit: int = 50) -> List[Dict]:
        """Tarama geçmişi"""
        with self.get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute(
                """
                           SELECT *
                           FROM scan_results
                           ORDER BY timestamp DESC
                               LIMIT ?
                           """,
                (limit,),
            )

            return [dict(row) for row in cursor.fetchall()]

    def check_file_hash(self, file_hash: str) -> Optional[Dict]:
        """Dosya hash'i daha önce tarandı mı?"""
        with self.get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute(
                """
                           SELECT *
                           FROM scan_results
                           WHERE file_hash = ?
                           ORDER BY timestamp DESC
                               LIMIT 1
                           """,
                (file_hash,),
            )

            row = cursor.fetchone()
            return dict(row) if row else None

    # ========================================
    # CHAT HISTORY OPERATIONS
    # ========================================

    def add_chat_message(self, chat_data: Dict) -> int:
        """Chat mesajı ekle"""
        with self.get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute(
                """
                           INSERT INTO chat_history (user_message, bot_response, intent,
                                                     context_used, response_time, user_id, session_id)
                           VALUES (?, ?, ?, ?, ?, ?, ?)
                           """,
                (
                    chat_data.get("user_message"),
                    chat_data.get("bot_response"),
                    chat_data.get("intent"),
                    json.dumps(chat_data.get("context_used", {})),
                    chat_data.get("response_time"),
                    chat_data.get("user_id"),
                    chat_data.get("session_id"),
                ),
            )

            conn.commit()
            return cursor.lastrowid

    def get_chat_history(self, session_id: str, limit: int = 10) -> List[Dict]:
        """Session'a ait chat geçmişi"""
        with self.get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute(
                """
                           SELECT *
                           FROM chat_history
                           WHERE session_id = ?
                           ORDER BY timestamp DESC
                               LIMIT ?
                           """,
                (session_id, limit),
            )

            return [dict(row) for row in cursor.fetchall()]

    # ========================================
    # IP BLACKLIST OPERATIONS
    # ========================================

    def add_to_blacklist(
        self,
        ip_address: str,
        reason: str = "",
        permanent: bool = False,
        duration_hours: int = 24,
    ) -> int:
        """IP'yi blacklist'e ekle"""
        with self.get_connection() as conn:
            cursor = conn.cursor()

            blocked_until = None
            if not permanent:
                blocked_until = (
                    datetime.now() + timedelta(hours=duration_hours)
                ).strftime("%Y-%m-%d %H:%M:%S")

            cursor.execute(
                """
                INSERT OR REPLACE INTO ip_blacklist (
                    ip_address, reason, attack_count,
                    blocked_until, permanent, added_by
                ) VALUES (?, ?, 
                    COALESCE((SELECT attack_count + 1 FROM ip_blacklist WHERE ip_address = ?), 1),
                    ?, ?, 'system')
            """,
                (ip_address, reason, ip_address, blocked_until, permanent),
            )

            conn.commit()
            return cursor.lastrowid

    def is_blacklisted(self, ip_address: str) -> bool:
        """IP blacklist'te mi?"""
        with self.get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute(
                """
                           SELECT COUNT(*) as count
                           FROM ip_blacklist
                           WHERE ip_address = ?
                             AND (permanent = 1
                              OR blocked_until
                               > datetime('now'))
                           """,
                (ip_address,),
            )

            return cursor.fetchone()["count"] > 0

    def get_blacklist(self) -> List[Dict]:
        """Tüm blacklist'i getir"""
        with self.get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute(
                """
                           SELECT *
                           FROM ip_blacklist
                           WHERE permanent = 1
                              OR blocked_until > datetime('now')
                           ORDER BY last_seen DESC
                           """
            )

            return [dict(row) for row in cursor.fetchall()]

    # ========================================
    # SYSTEM METRICS OPERATIONS
    # ========================================

    def add_metric(
        self, metric_type: str, value: float, unit: str = "", metadata: Dict = None
    ) -> int:
        """Sistem metriği ekle"""
        with self.get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute(
                """
                           INSERT INTO system_metrics (metric_type, metric_value, metric_unit, metadata)
                           VALUES (?, ?, ?, ?)
                           """,
                (metric_type, value, unit, json.dumps(metadata or {})),
            )

            conn.commit()
            return cursor.lastrowid

    def get_metrics(self, metric_type: str, hours: int = 24) -> List[Dict]:
        """Belirli bir metriğin geçmişi"""
        with self.get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute(
                """
                           SELECT *
                           FROM system_metrics
                           WHERE metric_type = ?
                             AND timestamp
                               > datetime('now'
                               , '-' || ? || ' hours')
                           ORDER BY timestamp ASC
                           """,
                (metric_type, hours),
            )

            return [dict(row) for row in cursor.fetchall()]

    # ========================================
    # CLEANUP OPERATIONS
    # ========================================

    def cleanup_old_data(self, days: int = 30):
        """Eski verileri temizle"""
        with self.get_connection() as conn:
            cursor = conn.cursor()

            cutoff_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")

            tables = ["attacks", "network_logs", "chat_history", "system_metrics"]

            for table in tables:
                cursor.execute(
                    f"""
                    DELETE FROM {table}
                    WHERE timestamp < ?
                """,
                    (cutoff_date,),
                )

            conn.commit()
            print(f"✅ {days} günden eski veriler temizlendi!")

    def get_database_stats(self) -> Dict:
        """Database istatistikleri"""
        with self.get_connection() as conn:
            cursor = conn.cursor()

            stats = {}

            tables = [
                "attacks",
                "network_logs",
                "scan_results",
                "chat_history",
                "system_metrics",
                "ip_blacklist",
            ]

            for table in tables:
                cursor.execute(f"SELECT COUNT(*) as count FROM {table}")
                stats[table] = cursor.fetchone()["count"]

            # Database boyutu
            stats["db_size_mb"] = round(
                os.path.getsize(self.db_path) / (1024 * 1024), 2
            )

            return stats

    def export_to_csv(self, table_name: str, output_path: str):
        """Tabloyu CSV'ye export et"""
        with self.get_connection() as conn:
            df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
            df.to_csv(output_path, index=False)
            print(f"✅ {table_name} → {output_path}")


# Test fonksiyonu
if __name__ == "__main__":
    print("🗄️ Database Manager Test\n")

    # Database oluştur
    db = DatabaseManager("test_cyberguard.db")

    # Test verileri ekle
    print("\n📝 Test verileri ekleniyor...")

    # Saldırı ekle
    attack_id = db.add_attack(
        {
            "attack_type": "DDoS",
            "source_ip": "192.168.1.100",
            "destination_ip": "10.0.0.1",
            "source_port": 54321,
            "destination_port": 80,
            "protocol": "TCP",
            "severity": "HIGH",
            "confidence": 0.95,
            "packet_count": 10000,
            "bytes_transferred": 5000000,
            "duration": 120.5,
            "blocked": True,
            "description": "Massive DDoS attack detected",
        }
    )
    print(f"✅ Saldırı eklendi (ID: {attack_id})")

    # İstatistikler
    stats = db.get_attack_stats(24)
    print(f"\n📊 İstatistikler:\n{json.dumps(stats, indent=2)}")

    # Database bilgileri
    db_stats = db.get_database_stats()
    print(f"\n💾 Database İstatistikleri:\n{json.dumps(db_stats, indent=2)}")

    print("\n✅ Test tamamlandı!")
