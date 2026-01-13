"""
Alerts API - REAL DATA VERSION
Security alert management with persistence
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Optional
from datetime import datetime, timedelta
import os
import json
import uuid

router = APIRouter()

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
data_dir = os.path.join(project_root, "data")
alerts_file = os.path.join(data_dir, "alerts.json")

ALERTS = []


class AlertCreate(BaseModel):
    title: str
    message: str
    severity: str = "medium"
    source: str = "system"
    metadata: Optional[Dict] = None


def load_alerts():
    global ALERTS
    if os.path.exists(alerts_file):
        try:
            with open(alerts_file, "r") as f:
                ALERTS = json.load(f)
        except:
            pass


def save_alerts():
    os.makedirs(os.path.dirname(alerts_file), exist_ok=True)
    with open(alerts_file, "w") as f:
        json.dump(ALERTS[-1000:], f, indent=2, default=str)


load_alerts()


@router.get("/status")
async def get_status():
    active = len([a for a in ALERTS if not a.get("acknowledged")])
    by_severity = {"critical": 0, "high": 0, "medium": 0, "low": 0}
    for a in ALERTS:
        if not a.get("acknowledged"):
            sev = a.get("severity", "medium")
            if sev in by_severity:
                by_severity[sev] += 1
    return {
        "success": True,
        "data": {"total": len(ALERTS), "active": active, "by_severity": by_severity},
    }


@router.get("")
async def get_alerts(
    severity: Optional[str] = None,
    acknowledged: Optional[bool] = None,
    limit: int = 100,
):
    filtered = ALERTS.copy()
    if severity:
        filtered = [a for a in filtered if a.get("severity") == severity]
    if acknowledged is not None:
        filtered = [a for a in filtered if a.get("acknowledged") == acknowledged]
    filtered.sort(key=lambda x: x.get("created_at", ""), reverse=True)
    return {
        "success": True,
        "data": {"alerts": filtered[:limit], "total": len(filtered)},
    }


@router.post("")
async def create_alert(alert: AlertCreate):
    new_alert = {
        "id": f"ALT-{str(uuid.uuid4())[:8]}",
        "title": alert.title,
        "message": alert.message,
        "severity": alert.severity,
        "source": alert.source,
        "metadata": alert.metadata or {},
        "created_at": datetime.now().isoformat(),
        "acknowledged": False,
        "acknowledged_at": None,
        "acknowledged_by": None,
    }
    ALERTS.append(new_alert)
    save_alerts()
    return {"success": True, "data": new_alert}


@router.put("/{alert_id}/acknowledge")
async def acknowledge_alert(alert_id: str, user: str = "system"):
    for a in ALERTS:
        if a.get("id") == alert_id:
            a["acknowledged"] = True
            a["acknowledged_at"] = datetime.now().isoformat()
            a["acknowledged_by"] = user
            save_alerts()
            return {"success": True, "message": "Alert acknowledged"}
    raise HTTPException(status_code=404)


@router.put("/acknowledge-all")
async def acknowledge_all(user: str = "system"):
    count = 0
    for a in ALERTS:
        if not a.get("acknowledged"):
            a["acknowledged"] = True
            a["acknowledged_at"] = datetime.now().isoformat()
            a["acknowledged_by"] = user
            count += 1
    save_alerts()
    return {"success": True, "message": f"Acknowledged {count} alerts"}


@router.delete("/{alert_id}")
async def delete_alert(alert_id: str):
    global ALERTS
    for i, a in enumerate(ALERTS):
        if a.get("id") == alert_id:
            ALERTS.pop(i)
            save_alerts()
            return {"success": True}
    raise HTTPException(status_code=404)


@router.get("/stats")
async def get_stats():
    by_severity = {"critical": 0, "high": 0, "medium": 0, "low": 0}
    by_source = {}
    for a in ALERTS:
        sev = a.get("severity", "medium")
        if sev in by_severity:
            by_severity[sev] += 1
        src = a.get("source", "unknown")
        by_source[src] = by_source.get(src, 0) + 1

    cutoff = datetime.now() - timedelta(hours=24)
    recent = sum(
        1
        for a in ALERTS
        if datetime.fromisoformat(a.get("created_at", "2000-01-01")) > cutoff
    )

    return {
        "success": True,
        "data": {
            "total": len(ALERTS),
            "by_severity": by_severity,
            "by_source": by_source,
            "recent_24h": recent,
        },
    }
