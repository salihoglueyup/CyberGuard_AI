"""
Logs API Routes - CyberGuard AI
Sistem logları - Gerçek veritabanı entegrasyonu

Dosya Yolu: app/api/routes/logs.py
"""

from fastapi import APIRouter, Query
from typing import Optional
import sys
import os
from datetime import datetime

# Path düzeltmesi
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
sys.path.insert(0, project_root)

from src.utils.database import DatabaseManager

router = APIRouter()
db = DatabaseManager()

# Log seviyeleri ve kaynakları
LOG_LEVELS = ["info", "warning", "error", "debug", "critical"]
LOG_SOURCES = [
    "Frontend",
    "Backend",
    "Database",
    "WebSocket",
    "ML Engine",
    "Security",
    "API",
    "Scanner",
    "Network",
]


@router.get("/")
async def get_logs(
    page: int = Query(1, ge=1),
    limit: int = Query(50, ge=1, le=200),
    level: Optional[str] = None,
    source: Optional[str] = None,
    search: Optional[str] = None,
):
    """Tüm logları getir - Veritabanından"""
    try:
        with db.get_connection() as conn:
            cursor = conn.cursor()

            # Base query
            query = "SELECT * FROM system_logs WHERE 1=1"
            params = []

            if level and level != "all":
                query += " AND level = ?"
                params.append(level)

            if source and source != "all":
                query += " AND source = ?"
                params.append(source)

            if search:
                query += " AND (message LIKE ? OR source LIKE ?)"
                params.extend([f"%{search}%", f"%{search}%"])

            query += " ORDER BY timestamp DESC"

            # Count total
            count_query = query.replace("SELECT *", "SELECT COUNT(*)")
            cursor.execute(count_query, params)
            total = cursor.fetchone()[0]

            # Pagination
            query += f" LIMIT {limit} OFFSET {(page - 1) * limit}"
            cursor.execute(query, params)

            rows = cursor.fetchall()
            logs = []
            for row in rows:
                logs.append(
                    {
                        "id": row["id"],
                        "timestamp": row["timestamp"],
                        "level": row["level"],
                        "source": row["source"],
                        "message": row["message"],
                        "details": row["details"],
                        "user_id": row["user_id"],
                        "session_id": row["session_id"],
                    }
                )

            return {
                "success": True,
                "data": logs,
                "pagination": {
                    "page": page,
                    "limit": limit,
                    "total": total,
                    "pages": (total + limit - 1) // limit,
                },
            }
    except Exception as e:
        return {"success": False, "error": str(e), "data": []}


@router.get("/stats")
async def get_log_stats():
    """Log istatistikleri"""
    try:
        with db.get_connection() as conn:
            cursor = conn.cursor()

            # Total
            cursor.execute("SELECT COUNT(*) FROM system_logs")
            total = cursor.fetchone()[0]

            # By level
            cursor.execute(
                """
                SELECT level, COUNT(*) as count 
                FROM system_logs 
                GROUP BY level
            """
            )
            by_level = {row["level"]: row["count"] for row in cursor.fetchall()}

            # By source
            cursor.execute(
                """
                SELECT source, COUNT(*) as count 
                FROM system_logs 
                GROUP BY source
            """
            )
            by_source = {row["source"]: row["count"] for row in cursor.fetchall()}

            return {
                "success": True,
                "data": {
                    "total": total,
                    "by_level": by_level,
                    "by_source": by_source,
                    "levels": LOG_LEVELS,
                    "sources": LOG_SOURCES,
                },
            }
    except Exception as e:
        return {"success": False, "error": str(e)}


@router.post("/")
async def add_log(level: str, source: str, message: str, details: Optional[str] = None):
    """Yeni log ekle - Veritabanına"""
    try:
        with db.get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute(
                """
                INSERT INTO system_logs (level, source, message, details, timestamp)
                VALUES (?, ?, ?, ?, ?)
            """,
                (
                    level if level in LOG_LEVELS else "info",
                    source if source in LOG_SOURCES else "Backend",
                    message,
                    details,
                    datetime.now().isoformat(),
                ),
            )

            conn.commit()
            log_id = cursor.lastrowid

            return {
                "success": True,
                "data": {
                    "id": log_id,
                    "level": level,
                    "source": source,
                    "message": message,
                    "timestamp": datetime.now().isoformat(),
                },
            }
    except Exception as e:
        return {"success": False, "error": str(e)}


@router.delete("/")
async def clear_logs():
    """Tüm logları temizle"""
    try:
        with db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM system_logs")
            conn.commit()
            deleted = cursor.rowcount

        return {"success": True, "message": f"{deleted} log silindi"}
    except Exception as e:
        return {"success": False, "error": str(e)}


@router.delete("/{log_id}")
async def delete_log(log_id: int):
    """Belirli bir logu sil"""
    try:
        with db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM system_logs WHERE id = ?", (log_id,))
            conn.commit()

        return {"success": True, "message": "Log silindi"}
    except Exception as e:
        return {"success": False, "error": str(e)}


@router.get("/sources")
async def get_log_sources():
    """Mevcut log kaynaklarını getir"""
    return {"success": True, "data": LOG_SOURCES}


@router.get("/levels")
async def get_log_levels():
    """Mevcut log seviyelerini getir"""
    return {"success": True, "data": LOG_LEVELS}


# Helper function for other modules to log
def log_event(level: str, source: str, message: str, details: str = None):
    """Sistem logu ekle (çağırılabilir fonksiyon)"""
    try:
        with db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO system_logs (level, source, message, details, timestamp)
                VALUES (?, ?, ?, ?, ?)
            """,
                (level, source, message, details, datetime.now().isoformat()),
            )
            conn.commit()
    except:
        pass  # Log hatası uygulama akışını bozmamalı
