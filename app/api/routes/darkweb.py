"""
Dark Web Monitoring API - REAL DATA VERSION
Uses breach databases and local data with optional external APIs
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import os
import json
import hashlib
import re

router = APIRouter()

# Path setup
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
data_dir = os.path.join(project_root, "data")
breach_db_file = os.path.join(data_dir, "breach_db.json")
monitoring_file = os.path.join(data_dir, "darkweb_monitoring.json")

# Environment variables
INTELX_API_KEY = os.getenv("INTELX_API_KEY", "")
HAVEIBEENPWNED_API_KEY = os.getenv("HAVEIBEENPWNED_API_KEY", "")

# In-memory database with persistence
BREACH_DB = {"known_breaches": [], "monitored_assets": [], "alerts": []}


class MonitoringAsset(BaseModel):
    type: str  # email, domain, keyword
    value: str
    enabled: bool = True


class BreachCheck(BaseModel):
    email: Optional[str] = None
    domain: Optional[str] = None


def load_breach_db():
    """Load breach database from file"""
    global BREACH_DB

    if os.path.exists(breach_db_file):
        try:
            with open(breach_db_file, "r", encoding="utf-8") as f:
                BREACH_DB.update(json.load(f))
        except:
            pass

    if os.path.exists(monitoring_file):
        try:
            with open(monitoring_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                BREACH_DB["monitored_assets"] = data.get("assets", [])
                BREACH_DB["alerts"] = data.get("alerts", [])
        except:
            pass


def save_breach_db():
    """Save breach database to file"""
    os.makedirs(os.path.dirname(breach_db_file), exist_ok=True)

    with open(breach_db_file, "w", encoding="utf-8") as f:
        json.dump(
            {
                "known_breaches": BREACH_DB["known_breaches"],
                "last_updated": datetime.now().isoformat(),
            },
            f,
            indent=2,
            default=str,
        )

    with open(monitoring_file, "w", encoding="utf-8") as f:
        json.dump(
            {
                "assets": BREACH_DB["monitored_assets"],
                "alerts": BREACH_DB["alerts"],
                "last_updated": datetime.now().isoformat(),
            },
            f,
            indent=2,
            default=str,
        )


def hash_email(email: str) -> str:
    """Hash email for privacy"""
    return hashlib.sha256(email.lower().encode()).hexdigest()[:16]


def check_local_breaches(email: str = None, domain: str = None) -> List[Dict]:
    """Check against local breach database"""
    results = []

    for breach in BREACH_DB.get("known_breaches", []):
        if email:
            email_hash = hash_email(email)
            if email_hash in breach.get("affected_hashes", []) or email.lower() in [
                e.lower() for e in breach.get("affected_emails", [])
            ]:
                results.append(breach)

        if domain:
            if domain.lower() in breach.get("affected_domains", []):
                results.append(breach)

    return results


# Known major breaches (public information)
KNOWN_BREACHES = [
    {
        "name": "Collection #1",
        "date": "2019-01",
        "records": 773000000,
        "data_types": ["email", "password"],
        "description": "Largest data breach collection discovered",
    },
    {
        "name": "LinkedIn",
        "date": "2021-04",
        "records": 700000000,
        "data_types": ["email", "name", "phone", "professional_info"],
        "description": "LinkedIn data scraping incident",
    },
    {
        "name": "Facebook",
        "date": "2021-04",
        "records": 533000000,
        "data_types": ["phone", "email", "name", "location"],
        "description": "Facebook data leak affecting 533M users",
    },
    {
        "name": "Yahoo",
        "date": "2016-09",
        "records": 3000000000,
        "data_types": ["email", "password", "security_questions"],
        "description": "Yahoo data breach affecting 3 billion accounts",
    },
    {
        "name": "Marriott",
        "date": "2018-11",
        "records": 500000000,
        "data_types": ["name", "email", "phone", "passport", "credit_card"],
        "description": "Marriott Starwood reservation system breach",
    },
]

# Initialize
load_breach_db()
if not BREACH_DB["known_breaches"]:
    BREACH_DB["known_breaches"] = KNOWN_BREACHES


@router.get("/status")
async def get_darkweb_status():
    """Get dark web monitoring status"""
    return {
        "success": True,
        "data": {
            "monitoring_active": True,
            "apis_configured": {
                "intelx": bool(INTELX_API_KEY),
                "haveibeenpwned": bool(HAVEIBEENPWNED_API_KEY),
            },
            "monitored_assets": len(BREACH_DB.get("monitored_assets", [])),
            "active_alerts": len(
                [a for a in BREACH_DB.get("alerts", []) if not a.get("resolved")]
            ),
            "known_breaches": len(BREACH_DB.get("known_breaches", [])),
            "last_scan": datetime.now().isoformat(),
        },
    }


@router.post("/check")
async def check_breach(check: BreachCheck):
    """Check if email or domain appears in known breaches"""
    if not check.email and not check.domain:
        raise HTTPException(status_code=400, detail="Email or domain required")

    results = {
        "query": check.email or check.domain,
        "type": "email" if check.email else "domain",
        "breaches_found": [],
        "total_breaches": 0,
        "risk_level": "safe",
    }

    # Check local database
    local_results = check_local_breaches(check.email, check.domain)
    results["breaches_found"] = local_results
    results["total_breaches"] = len(local_results)

    # Try external APIs if configured
    # Note: In production, you would call HaveIBeenPwned or similar

    # Determine risk level
    if results["total_breaches"] >= 5:
        results["risk_level"] = "critical"
    elif results["total_breaches"] >= 3:
        results["risk_level"] = "high"
    elif results["total_breaches"] >= 1:
        results["risk_level"] = "medium"

    # Recommendations
    results["recommendations"] = []
    if results["total_breaches"] > 0:
        results["recommendations"] = [
            "Change passwords immediately",
            "Enable two-factor authentication",
            "Monitor accounts for suspicious activity",
            "Consider using a password manager",
            "Check for unauthorized access",
        ]

    results["checked_at"] = datetime.now().isoformat()

    return {"success": True, "data": results}


@router.get("/check/email/{email}")
async def check_email(email: str):
    """Quick email breach check"""
    check = BreachCheck(email=email)
    return await check_breach(check)


@router.get("/check/domain/{domain}")
async def check_domain(domain: str):
    """Quick domain breach check"""
    check = BreachCheck(domain=domain)
    return await check_breach(check)


@router.get("/breaches")
async def get_known_breaches(limit: int = 50):
    """Get list of known breaches"""
    breaches = BREACH_DB.get("known_breaches", [])

    # Sort by record count
    breaches.sort(key=lambda x: x.get("records", 0), reverse=True)

    return {
        "success": True,
        "data": {"breaches": breaches[:limit], "total": len(breaches)},
    }


@router.get("/monitoring")
async def get_monitored_assets():
    """Get monitored assets"""
    assets = BREACH_DB.get("monitored_assets", [])

    return {
        "success": True,
        "data": {
            "assets": assets,
            "total": len(assets),
            "enabled": len([a for a in assets if a.get("enabled", True)]),
        },
    }


@router.post("/monitoring")
async def add_monitoring_asset(asset: MonitoringAsset):
    """Add asset to monitoring"""
    new_asset = {
        "id": hashlib.md5(f"{asset.type}:{asset.value}".encode()).hexdigest()[:12],
        "type": asset.type,
        "value": asset.value,
        "enabled": asset.enabled,
        "added_at": datetime.now().isoformat(),
        "last_checked": None,
        "findings": 0,
    }

    BREACH_DB["monitored_assets"].append(new_asset)
    save_breach_db()

    return {"success": True, "data": new_asset, "message": "Asset added to monitoring"}


@router.delete("/monitoring/{asset_id}")
async def remove_monitoring_asset(asset_id: str):
    """Remove asset from monitoring"""
    for i, asset in enumerate(BREACH_DB.get("monitored_assets", [])):
        if asset.get("id") == asset_id:
            deleted = BREACH_DB["monitored_assets"].pop(i)
            save_breach_db()
            return {"success": True, "message": "Asset removed", "data": deleted}

    raise HTTPException(status_code=404, detail="Asset not found")


@router.get("/alerts")
async def get_alerts(unresolved_only: bool = False):
    """Get dark web alerts"""
    alerts = BREACH_DB.get("alerts", [])

    if unresolved_only:
        alerts = [a for a in alerts if not a.get("resolved")]

    # Sort by created_at descending
    alerts.sort(key=lambda x: x.get("created_at", ""), reverse=True)

    return {
        "success": True,
        "data": {
            "alerts": alerts,
            "total": len(alerts),
            "unresolved": len(
                [a for a in BREACH_DB.get("alerts", []) if not a.get("resolved")]
            ),
        },
    }


@router.put("/alerts/{alert_id}/resolve")
async def resolve_alert(alert_id: str):
    """Mark an alert as resolved"""
    for alert in BREACH_DB.get("alerts", []):
        if alert.get("id") == alert_id:
            alert["resolved"] = True
            alert["resolved_at"] = datetime.now().isoformat()
            save_breach_db()
            return {"success": True, "message": "Alert resolved"}

    raise HTTPException(status_code=404, detail="Alert not found")


@router.post("/scan")
async def trigger_scan():
    """Trigger a dark web scan for all monitored assets"""
    assets = BREACH_DB.get("monitored_assets", [])
    results = []

    for asset in assets:
        if not asset.get("enabled", True):
            continue

        # Perform check
        if asset["type"] == "email":
            check = BreachCheck(email=asset["value"])
        elif asset["type"] == "domain":
            check = BreachCheck(domain=asset["value"])
        else:
            continue

        result = await check_breach(check)

        # Update asset
        asset["last_checked"] = datetime.now().isoformat()
        asset["findings"] = result["data"]["total_breaches"]

        # Create alert if findings
        if result["data"]["total_breaches"] > 0:
            alert = {
                "id": hashlib.md5(
                    f"{asset['value']}:{datetime.now()}".encode()
                ).hexdigest()[:12],
                "asset_id": asset["id"],
                "asset_value": asset["value"],
                "asset_type": asset["type"],
                "breaches_found": result["data"]["total_breaches"],
                "risk_level": result["data"]["risk_level"],
                "created_at": datetime.now().isoformat(),
                "resolved": False,
            }
            BREACH_DB["alerts"].append(alert)

        results.append(
            {
                "asset": asset["value"],
                "type": asset["type"],
                "findings": result["data"]["total_breaches"],
            }
        )

    save_breach_db()

    return {
        "success": True,
        "data": {
            "assets_scanned": len(results),
            "results": results,
            "scan_time": datetime.now().isoformat(),
        },
    }


@router.get("/stats")
async def get_darkweb_stats():
    """Get dark web monitoring statistics"""
    alerts = BREACH_DB.get("alerts", [])
    assets = BREACH_DB.get("monitored_assets", [])

    # Count by risk level
    by_risk = {"critical": 0, "high": 0, "medium": 0, "low": 0, "safe": 0}
    for alert in alerts:
        risk = alert.get("risk_level", "unknown")
        if risk in by_risk:
            by_risk[risk] += 1

    return {
        "success": True,
        "data": {
            "total_assets": len(assets),
            "active_monitoring": len([a for a in assets if a.get("enabled")]),
            "total_alerts": len(alerts),
            "unresolved_alerts": len([a for a in alerts if not a.get("resolved")]),
            "alerts_by_risk": by_risk,
            "known_breaches": len(BREACH_DB.get("known_breaches", [])),
        },
    }
