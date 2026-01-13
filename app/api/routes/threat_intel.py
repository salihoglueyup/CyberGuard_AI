"""
Threat Intelligence API - REAL DATA VERSION
Uses OTX, AbuseIPDB, and local threat database
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import os
import json
import hashlib

# Try to import requests for external APIs
try:
    import requests

    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

router = APIRouter()

# Path setup
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
data_dir = os.path.join(project_root, "data")
threat_db_file = os.path.join(data_dir, "threat_intel.json")
ioc_file = os.path.join(data_dir, "iocs.json")

# Environment variables for API keys
VIRUSTOTAL_API_KEY = os.getenv("VIRUSTOTAL_API_KEY", "")
ABUSEIPDB_API_KEY = os.getenv("ABUSEIPDB_API_KEY", "")
ALIENVAULT_API_KEY = os.getenv("ALIENVAULT_API_KEY", "")

# In-memory threat database with persistence
THREAT_DB = {
    "malicious_ips": {},
    "malicious_domains": {},
    "malicious_hashes": {},
    "iocs": [],
    "last_updated": None,
}


class IOCSubmit(BaseModel):
    type: str  # ip, domain, hash, url
    value: str
    source: str = "manual"
    confidence: int = 80
    tags: List[str] = []


class ThreatQuery(BaseModel):
    indicator: str
    type: Optional[str] = None  # auto-detect if not provided


def load_threat_db():
    """Load threat database from file"""
    global THREAT_DB

    if os.path.exists(threat_db_file):
        try:
            with open(threat_db_file, "r", encoding="utf-8") as f:
                loaded = json.load(f)
                THREAT_DB.update(loaded)
        except:
            pass

    if os.path.exists(ioc_file):
        try:
            with open(ioc_file, "r", encoding="utf-8") as f:
                THREAT_DB["iocs"] = json.load(f)
        except:
            pass


def save_threat_db():
    """Save threat database to file"""
    os.makedirs(os.path.dirname(threat_db_file), exist_ok=True)

    THREAT_DB["last_updated"] = datetime.now().isoformat()

    with open(threat_db_file, "w", encoding="utf-8") as f:
        json.dump(THREAT_DB, f, indent=2, default=str)


def detect_ioc_type(indicator: str) -> str:
    """Auto-detect IOC type"""
    import re

    # IP address
    ip_pattern = r"^(\d{1,3}\.){3}\d{1,3}$"
    if re.match(ip_pattern, indicator):
        return "ip"

    # Domain
    domain_pattern = r"^[a-zA-Z0-9][a-zA-Z0-9-]{0,61}[a-zA-Z0-9]?\.[a-zA-Z]{2,}$"
    if re.match(domain_pattern, indicator):
        return "domain"

    # MD5 hash
    if len(indicator) == 32 and all(c in "0123456789abcdefABCDEF" for c in indicator):
        return "md5"

    # SHA256 hash
    if len(indicator) == 64 and all(c in "0123456789abcdefABCDEF" for c in indicator):
        return "sha256"

    # URL
    if indicator.startswith(("http://", "https://")):
        return "url"

    return "unknown"


def check_abuseipdb(ip: str) -> Optional[Dict]:
    """Check IP against AbuseIPDB"""
    if not ABUSEIPDB_API_KEY or not REQUESTS_AVAILABLE:
        return None

    try:
        headers = {"Key": ABUSEIPDB_API_KEY, "Accept": "application/json"}
        response = requests.get(
            f"https://api.abuseipdb.com/api/v2/check",
            params={"ipAddress": ip, "maxAgeInDays": 90},
            headers=headers,
            timeout=10,
        )
        if response.status_code == 200:
            data = response.json().get("data", {})
            return {
                "source": "AbuseIPDB",
                "ip": ip,
                "abuse_score": data.get("abuseConfidenceScore", 0),
                "is_public": data.get("isPublic", True),
                "country": data.get("countryCode", "XX"),
                "isp": data.get("isp", "Unknown"),
                "total_reports": data.get("totalReports", 0),
                "last_reported": data.get("lastReportedAt"),
            }
    except Exception as e:
        print(f"AbuseIPDB error: {e}")

    return None


def check_virustotal_hash(file_hash: str) -> Optional[Dict]:
    """Check hash against VirusTotal"""
    if not VIRUSTOTAL_API_KEY or not REQUESTS_AVAILABLE:
        return None

    try:
        headers = {"x-apikey": VIRUSTOTAL_API_KEY}
        response = requests.get(
            f"https://www.virustotal.com/api/v3/files/{file_hash}",
            headers=headers,
            timeout=10,
        )
        if response.status_code == 200:
            data = response.json().get("data", {}).get("attributes", {})
            stats = data.get("last_analysis_stats", {})
            return {
                "source": "VirusTotal",
                "hash": file_hash,
                "malicious": stats.get("malicious", 0),
                "suspicious": stats.get("suspicious", 0),
                "harmless": stats.get("harmless", 0),
                "undetected": stats.get("undetected", 0),
                "detection_ratio": f"{stats.get('malicious', 0)}/{sum(stats.values())}",
                "file_type": data.get("type_description", "Unknown"),
                "file_name": data.get("meaningful_name", "Unknown"),
            }
    except Exception as e:
        print(f"VirusTotal error: {e}")

    return None


def check_local_db(indicator: str, ioc_type: str) -> Optional[Dict]:
    """Check against local threat database"""
    if ioc_type == "ip":
        if indicator in THREAT_DB.get("malicious_ips", {}):
            return THREAT_DB["malicious_ips"][indicator]
    elif ioc_type == "domain":
        if indicator in THREAT_DB.get("malicious_domains", {}):
            return THREAT_DB["malicious_domains"][indicator]
    elif ioc_type in ["md5", "sha256", "hash"]:
        if indicator in THREAT_DB.get("malicious_hashes", {}):
            return THREAT_DB["malicious_hashes"][indicator]

    # Check IOCs list
    for ioc in THREAT_DB.get("iocs", []):
        if ioc.get("value") == indicator:
            return ioc

    return None


# Initialize on module load
load_threat_db()


@router.get("/status")
async def get_threat_intel_status():
    """Get threat intelligence status"""
    return {
        "success": True,
        "data": {
            "apis_configured": {
                "virustotal": bool(VIRUSTOTAL_API_KEY),
                "abuseipdb": bool(ABUSEIPDB_API_KEY),
                "alienvault": bool(ALIENVAULT_API_KEY),
            },
            "local_db": {
                "malicious_ips": len(THREAT_DB.get("malicious_ips", {})),
                "malicious_domains": len(THREAT_DB.get("malicious_domains", {})),
                "malicious_hashes": len(THREAT_DB.get("malicious_hashes", {})),
                "iocs_total": len(THREAT_DB.get("iocs", [])),
            },
            "last_updated": THREAT_DB.get("last_updated"),
            "requests_available": REQUESTS_AVAILABLE,
        },
    }


@router.post("/lookup")
async def lookup_indicator(query: ThreatQuery):
    """Look up an indicator against threat intelligence sources"""
    indicator = query.indicator.strip().lower()
    ioc_type = query.type or detect_ioc_type(indicator)

    results = {
        "indicator": indicator,
        "type": ioc_type,
        "sources_checked": [],
        "findings": [],
        "risk_score": 0,
        "verdict": "unknown",
    }

    # Check local database first
    local_result = check_local_db(indicator, ioc_type)
    if local_result:
        results["sources_checked"].append("local_db")
        results["findings"].append(
            {"source": "Local Database", "data": local_result, "malicious": True}
        )
        results["risk_score"] = max(
            results["risk_score"], local_result.get("confidence", 80)
        )

    # Check external APIs based on type
    if ioc_type == "ip":
        abuseipdb_result = check_abuseipdb(indicator)
        if abuseipdb_result:
            results["sources_checked"].append("abuseipdb")
            is_malicious = abuseipdb_result.get("abuse_score", 0) > 50
            results["findings"].append(
                {
                    "source": "AbuseIPDB",
                    "data": abuseipdb_result,
                    "malicious": is_malicious,
                }
            )
            if is_malicious:
                results["risk_score"] = max(
                    results["risk_score"], abuseipdb_result["abuse_score"]
                )

    elif ioc_type in ["md5", "sha256"]:
        vt_result = check_virustotal_hash(indicator)
        if vt_result:
            results["sources_checked"].append("virustotal")
            malicious_count = vt_result.get("malicious", 0)
            is_malicious = malicious_count > 5
            results["findings"].append(
                {"source": "VirusTotal", "data": vt_result, "malicious": is_malicious}
            )
            if is_malicious:
                results["risk_score"] = max(
                    results["risk_score"], min(100, malicious_count * 2)
                )

    # Determine verdict
    if results["risk_score"] >= 80:
        results["verdict"] = "malicious"
    elif results["risk_score"] >= 50:
        results["verdict"] = "suspicious"
    elif results["risk_score"] >= 20:
        results["verdict"] = "low_risk"
    elif results["findings"]:
        results["verdict"] = "clean"
    else:
        results["verdict"] = "unknown"

    results["checked_at"] = datetime.now().isoformat()

    return {"success": True, "data": results}


@router.get("/lookup/ip/{ip}")
async def lookup_ip(ip: str):
    """Quick IP lookup"""
    query = ThreatQuery(indicator=ip, type="ip")
    return await lookup_indicator(query)


@router.get("/lookup/hash/{file_hash}")
async def lookup_hash(file_hash: str):
    """Quick hash lookup"""
    ioc_type = "sha256" if len(file_hash) == 64 else "md5"
    query = ThreatQuery(indicator=file_hash, type=ioc_type)
    return await lookup_indicator(query)


@router.post("/ioc")
async def submit_ioc(ioc: IOCSubmit):
    """Submit a new IOC to local database"""
    new_ioc = {
        "type": ioc.type,
        "value": ioc.value.strip().lower(),
        "source": ioc.source,
        "confidence": ioc.confidence,
        "tags": ioc.tags,
        "added_at": datetime.now().isoformat(),
        "id": hashlib.md5(ioc.value.encode()).hexdigest()[:12],
    }

    # Add to appropriate category
    if ioc.type == "ip":
        THREAT_DB["malicious_ips"][new_ioc["value"]] = new_ioc
    elif ioc.type == "domain":
        THREAT_DB["malicious_domains"][new_ioc["value"]] = new_ioc
    elif ioc.type in ["md5", "sha256", "hash"]:
        THREAT_DB["malicious_hashes"][new_ioc["value"]] = new_ioc

    # Also add to general IOCs list
    THREAT_DB["iocs"].append(new_ioc)

    save_threat_db()

    return {"success": True, "data": new_ioc, "message": "IOC added to local database"}


@router.get("/iocs")
async def get_iocs(ioc_type: Optional[str] = None, limit: int = 100):
    """Get IOCs from local database"""
    iocs = THREAT_DB.get("iocs", [])

    if ioc_type:
        iocs = [i for i in iocs if i.get("type") == ioc_type]

    # Sort by added_at descending
    iocs.sort(key=lambda x: x.get("added_at", ""), reverse=True)

    return {
        "success": True,
        "data": {
            "iocs": iocs[:limit],
            "total": len(iocs),
        },
    }


@router.delete("/ioc/{ioc_id}")
async def delete_ioc(ioc_id: str):
    """Delete an IOC from local database"""
    for i, ioc in enumerate(THREAT_DB.get("iocs", [])):
        if ioc.get("id") == ioc_id:
            deleted = THREAT_DB["iocs"].pop(i)

            # Also remove from category dicts
            value = deleted.get("value", "")
            THREAT_DB["malicious_ips"].pop(value, None)
            THREAT_DB["malicious_domains"].pop(value, None)
            THREAT_DB["malicious_hashes"].pop(value, None)

            save_threat_db()
            return {"success": True, "message": "IOC deleted"}

    raise HTTPException(status_code=404, detail="IOC not found")


@router.get("/feeds")
async def get_threat_feeds():
    """Get available threat intelligence feeds"""
    feeds = [
        {
            "name": "AbuseIPDB",
            "type": "ip_reputation",
            "configured": bool(ABUSEIPDB_API_KEY),
            "url": "https://abuseipdb.com",
        },
        {
            "name": "VirusTotal",
            "type": "file_hash",
            "configured": bool(VIRUSTOTAL_API_KEY),
            "url": "https://virustotal.com",
        },
        {
            "name": "AlienVault OTX",
            "type": "threat_intel",
            "configured": bool(ALIENVAULT_API_KEY),
            "url": "https://otx.alienvault.com",
        },
        {
            "name": "Local Database",
            "type": "all",
            "configured": True,
            "ioc_count": len(THREAT_DB.get("iocs", [])),
        },
    ]

    return {"success": True, "data": {"feeds": feeds}}


@router.get("/stats")
async def get_threat_stats():
    """Get threat intelligence statistics"""
    iocs = THREAT_DB.get("iocs", [])

    # Count by type
    by_type = {}
    for ioc in iocs:
        t = ioc.get("type", "unknown")
        by_type[t] = by_type.get(t, 0) + 1

    # Count by source
    by_source = {}
    for ioc in iocs:
        s = ioc.get("source", "unknown")
        by_source[s] = by_source.get(s, 0) + 1

    # Recent IOCs (last 24h)
    cutoff = datetime.now() - timedelta(hours=24)
    recent = 0
    for ioc in iocs:
        try:
            added = datetime.fromisoformat(ioc.get("added_at", "2000-01-01"))
            if added > cutoff:
                recent += 1
        except:
            pass

    return {
        "success": True,
        "data": {
            "total_iocs": len(iocs),
            "by_type": by_type,
            "by_source": by_source,
            "recent_24h": recent,
            "malicious_ips": len(THREAT_DB.get("malicious_ips", {})),
            "malicious_domains": len(THREAT_DB.get("malicious_domains", {})),
            "malicious_hashes": len(THREAT_DB.get("malicious_hashes", {})),
        },
    }
