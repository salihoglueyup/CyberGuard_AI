"""
Database API Routes - CyberGuard AI
Veritabanı yönetimi ve sorgu endpoint'leri

Dosya Yolu: app/api/routes/database.py
"""

from fastapi import APIRouter, Query
from pydantic import BaseModel
from typing import Optional, List
import sys
import os

# Path düzeltmesi
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
sys.path.insert(0, project_root)

from src.utils.database import DatabaseManager

router = APIRouter()
db = DatabaseManager()


class SQLQuery(BaseModel):
    query: str


@router.get("/stats")
async def get_database_stats():
    """Veritabanı istatistikleri"""
    try:
        stats = db.get_database_stats()

        # Dosya boyutunu da ekle
        db_path = db.db_path
        if os.path.exists(db_path):
            size = os.path.getsize(db_path)
            stats["size"] = size

        # Tablo sayılarını düzenle
        tables = {}
        if "attacks" in stats:
            tables["attacks"] = stats.get("attacks", 0)
        if "network_logs" in stats:
            tables["network_logs"] = stats.get("network_logs", 0)
        if "scan_results" in stats:
            tables["scan_results"] = stats.get("scan_results", 0)
        if "system_logs" in stats:
            tables["system_logs"] = stats.get("system_logs", 0)
        if "chat_history" in stats:
            tables["chat_history"] = stats.get("chat_history", 0)
        if "system_metrics" in stats:
            tables["system_metrics"] = stats.get("system_metrics", 0)

        stats["tables"] = tables

        return {"success": True, "data": stats}

    except Exception as e:
        import traceback

        traceback.print_exc()
        return {"success": False, "error": str(e), "data": {}}


@router.get("/table/{table_name}")
async def get_table_data(
    table_name: str, limit: int = Query(50, ge=1, le=1000), offset: int = Query(0, ge=0)
):
    """Belirli bir tablonun verilerini getir"""
    try:
        # Güvenlik: İzin verilen tablolar
        allowed_tables = [
            "attacks",
            "network_logs",
            "scan_results",
            "system_logs",
            "chat_history",
            "system_metrics",
            "ip_blacklist",
            "file_quarantine",
            "defences",
        ]

        if table_name not in allowed_tables:
            return {
                "success": False,
                "error": f"Tablo bulunamadı: {table_name}",
                "data": [],
            }

        with db.get_connection() as conn:
            cursor = conn.cursor()

            # Tablo varlığını kontrol et
            cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
                (table_name,),
            )
            if not cursor.fetchone():
                return {
                    "success": False,
                    "error": f"Tablo mevcut değil: {table_name}",
                    "data": [],
                }

            # Veriyi çek
            cursor.execute(
                f"SELECT * FROM {table_name} ORDER BY id DESC LIMIT ? OFFSET ?",
                (limit, offset),
            )

            rows = cursor.fetchall()
            columns = [desc[0] for desc in cursor.description]

            data = [dict(zip(columns, row)) for row in rows]

            return {"success": True, "data": data, "count": len(data)}

    except Exception as e:
        import traceback

        traceback.print_exc()
        return {"success": False, "error": str(e), "data": []}


@router.post("/query")
async def execute_query(query: SQLQuery):
    """SQL sorgusu çalıştır (sadece SELECT)"""
    try:
        sql = query.query.strip()

        # Güvenlik: Sadece SELECT izin ver
        if not sql.upper().startswith("SELECT"):
            return {"success": False, "error": "Sadece SELECT sorguları desteklenir"}

        # Tehlikeli kelimeleri kontrol et
        dangerous = [
            "DROP",
            "DELETE",
            "INSERT",
            "UPDATE",
            "ALTER",
            "CREATE",
            "TRUNCATE",
            "--",
            ";--",
        ]
        for word in dangerous:
            if word.upper() in sql.upper():
                return {
                    "success": False,
                    "error": f"Güvenlik: '{word}' kullanımı yasak",
                }

        with db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(sql)

            rows = cursor.fetchall()
            columns = [desc[0] for desc in cursor.description]

            data = [dict(zip(columns, row)) for row in rows]

            return {"success": True, "data": data, "count": len(data)}

    except Exception as e:
        return {"success": False, "error": str(e), "data": []}


@router.get("/tables")
async def list_tables():
    """Tüm tabloları listele"""
    try:
        with db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
            )
            tables = [row[0] for row in cursor.fetchall()]

            # Her tablonun satır sayısını al
            table_info = []
            for table in tables:
                if not table.startswith("sqlite_"):
                    cursor.execute(f"SELECT COUNT(*) FROM {table}")
                    count = cursor.fetchone()[0]
                    table_info.append({"name": table, "rows": count})

            return {"success": True, "data": table_info}

    except Exception as e:
        return {"success": False, "error": str(e), "data": []}


@router.get("/table/{table_name}/schema")
async def get_table_schema(table_name: str):
    """Tablo şemasını getir"""
    try:
        with db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(f"PRAGMA table_info({table_name})")

            columns = []
            for row in cursor.fetchall():
                columns.append(
                    {
                        "cid": row[0],
                        "name": row[1],
                        "type": row[2],
                        "notnull": row[3],
                        "default": row[4],
                        "pk": row[5],
                    }
                )

            return {"success": True, "data": columns}

    except Exception as e:
        return {"success": False, "error": str(e), "data": []}
