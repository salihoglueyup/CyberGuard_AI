"""
Attacks API Routes - CyberGuard AI
GERÇEK VERİTABANI ENTEGRASYONU

Dosya Yolu: app/api/routes/attacks.py
"""

from fastapi import APIRouter, Query
from typing import Optional
import sys
import os
from datetime import datetime, timedelta

# Path düzeltmesi
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
sys.path.insert(0, project_root)

from src.utils.database import DatabaseManager

router = APIRouter()

# Global database instance
db = DatabaseManager()


@router.get("/")
async def get_attacks(
    page: int = Query(1, ge=1),
    limit: int = Query(1000, ge=1, le=200000),  # Model verisi için yüksek limit
    hours: Optional[int] = Query(None),  # None = tüm zamanlar
    attack_type: Optional[str] = None,
    severity: Optional[str] = None,
):
    """Saldırı listesi - Gerçek veritabanından"""
    try:
        # Gerçek veritabanı sorgusu
        attacks = db.get_attacks(
            hours=hours, attack_type=attack_type, severity=severity, limit=limit * page
        )

        # Sayfalama
        start_idx = (page - 1) * limit
        end_idx = start_idx + limit
        paginated = attacks[start_idx:end_idx]

        result = []
        for attack in paginated:
            result.append(
                {
                    "id": attack.get("id"),
                    "timestamp": attack.get("timestamp"),
                    "attack_type": attack.get("attack_type"),
                    "source_ip": attack.get("source_ip"),
                    "destination_ip": attack.get("destination_ip", "10.0.0.1"),
                    "source_port": attack.get("source_port"),
                    "destination_port": attack.get("destination_port"),
                    "protocol": attack.get("protocol"),
                    "severity": attack.get("severity", "medium").lower(),
                    "confidence": attack.get("confidence", 0.85),
                    "blocked": attack.get("blocked", False),
                    "description": attack.get("description", ""),
                }
            )

        return {
            "success": True,
            "data": result,
            "pagination": {"page": page, "limit": limit, "total": len(attacks)},
        }
    except Exception as e:
        return {"success": False, "error": str(e), "data": []}


@router.get("/stats")
async def get_attack_stats(hours: Optional[int] = Query(None)):
    """Saldırı istatistikleri"""
    try:
        attacks = db.get_attacks(hours=hours)

        total = len(attacks)
        blocked = len([a for a in attacks if a.get("blocked")])
        not_blocked = total - blocked
        block_rate = round((blocked / total * 100) if total > 0 else 0, 1)

        # Severity dağılımı
        by_severity = {}
        for a in attacks:
            sev = a.get("severity", "MEDIUM").upper()
            by_severity[sev] = by_severity.get(sev, 0) + 1

        # Attack type dağılımı
        by_type = {}
        for a in attacks:
            atype = a.get("attack_type", "Unknown")
            by_type[atype] = by_type.get(atype, 0) + 1

        return {
            "success": True,
            "data": {
                "total": total,
                "blocked": blocked,
                "not_blocked": not_blocked,
                "block_rate": block_rate,
                "by_severity": by_severity,
                "by_type": by_type,
            },
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


@router.get("/by-type")
async def get_attacks_by_type(hours: Optional[int] = Query(None)):
    """Saldırı türü dağılımı"""
    try:
        attacks = db.get_attacks(hours=hours)

        by_type = {}
        for a in attacks:
            atype = a.get("attack_type", "Unknown")
            by_type[atype] = by_type.get(atype, 0) + 1

        result = [{"name": k, "value": v} for k, v in by_type.items()]
        result.sort(key=lambda x: x["value"], reverse=True)

        return {"success": True, "data": result}
    except Exception as e:
        return {"success": False, "error": str(e), "data": []}


@router.get("/by-severity")
async def get_attacks_by_severity(hours: Optional[int] = Query(None)):
    """Severity dağılımı"""
    try:
        attacks = db.get_attacks(hours=hours)

        by_severity = {}
        for a in attacks:
            sev = a.get("severity", "MEDIUM").upper()
            by_severity[sev] = by_severity.get(sev, 0) + 1

        # Sıralı döndür
        order = ["CRITICAL", "HIGH", "MEDIUM", "LOW"]
        result = []
        for sev in order:
            if sev in by_severity:
                result.append({"name": sev, "value": by_severity[sev]})

        return {"success": True, "data": result}
    except Exception as e:
        return {"success": False, "error": str(e), "data": []}


@router.get("/top-ips")
async def get_top_ips(limit: int = Query(10), hours: int = Query(24)):
    """En çok saldırı yapan IP'ler"""
    try:
        attacks = db.get_attacks(hours=hours)

        ip_counts = {}
        for a in attacks:
            ip = a.get("source_ip", "Unknown")
            ip_counts[ip] = ip_counts.get(ip, 0) + 1

        # Sırala ve limit uygula
        sorted_ips = sorted(ip_counts.items(), key=lambda x: x[1], reverse=True)
        result = [{"source_ip": ip, "count": count} for ip, count in sorted_ips[:limit]]

        return {"success": True, "data": result}
    except Exception as e:
        return {"success": False, "error": str(e), "data": []}


@router.get("/timeline")
async def get_attack_timeline(hours: int = Query(24)):
    """Saatlik saldırı timeline"""
    try:
        attacks = db.get_attacks(hours=hours)

        # Saatlik gruplama
        hourly = {}
        for a in attacks:
            ts = a.get("timestamp", "")
            if ts:
                try:
                    if isinstance(ts, str):
                        dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                    else:
                        dt = ts
                    hour_key = dt.strftime("%H:00")
                    hourly[hour_key] = hourly.get(hour_key, 0) + 1
                except:
                    pass

        # 24 saat için sonuç oluştur
        result = []
        for h in range(24):
            hour_str = f"{h:02d}:00"
            result.append({"hour": hour_str, "count": hourly.get(hour_str, 0)})

        return {"success": True, "data": result}
    except Exception as e:
        return {"success": False, "error": str(e), "data": []}


@router.get("/recent")
async def get_recent_attacks(limit: int = Query(10)):
    """Son saldırılar"""
    try:
        attacks = db.get_attacks(hours=24, limit=limit)

        result = []
        for a in attacks[:limit]:
            result.append(
                {
                    "id": a.get("id"),
                    "timestamp": a.get("timestamp"),
                    "attack_type": a.get("attack_type"),
                    "source_ip": a.get("source_ip"),
                    "severity": a.get("severity", "medium").lower(),
                    "blocked": a.get("blocked", False),
                }
            )

        return {"success": True, "data": result}
    except Exception as e:
        return {"success": False, "error": str(e), "data": []}


@router.get("/search/{query}")
async def search_attacks(query: str, limit: int = Query(50)):
    """Saldırı arama"""
    try:
        attacks = db.get_attacks(hours=168)  # Son 7 gün

        result = []
        query_lower = query.lower()
        for a in attacks:
            # IP, type veya description'da ara
            if (
                query_lower in a.get("source_ip", "").lower()
                or query_lower in a.get("attack_type", "").lower()
                or query_lower in a.get("description", "").lower()
            ):
                result.append(
                    {
                        "id": a.get("id"),
                        "timestamp": a.get("timestamp"),
                        "attack_type": a.get("attack_type"),
                        "source_ip": a.get("source_ip"),
                        "severity": a.get("severity", "medium").lower(),
                        "blocked": a.get("blocked", False),
                    }
                )

        return {
            "success": True,
            "query": query,
            "data": result[:limit],
            "count": len(result[:limit]),
        }
    except Exception as e:
        return {"success": False, "error": str(e), "data": []}
