"""
SIEM Integration API - REAL DATA VERSION
Integration with Splunk, Elastic, QRadar, and Sentinel
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from datetime import datetime
import os
import json

# Try to import requests
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
siem_config_file = os.path.join(data_dir, "siem_config.json")
siem_events_file = os.path.join(data_dir, "siem_events.json")

# Environment variables for SIEM connections
SPLUNK_HOST = os.getenv("SPLUNK_HOST", "")
SPLUNK_TOKEN = os.getenv("SPLUNK_TOKEN", "")
ELASTIC_HOST = os.getenv("ELASTIC_HOST", "")
ELASTIC_API_KEY = os.getenv("ELASTIC_API_KEY", "")
QRADAR_HOST = os.getenv("QRADAR_HOST", "")
QRADAR_TOKEN = os.getenv("QRADAR_TOKEN", "")
SENTINEL_WORKSPACE_ID = os.getenv("SENTINEL_WORKSPACE_ID", "")
SENTINEL_KEY = os.getenv("SENTINEL_KEY", "")

# In-memory config and events
SIEM_CONFIG = {"connections": [], "active_rules": [], "forwarding_enabled": False}
SIEM_EVENTS = []


class SIEMConnection(BaseModel):
    name: str
    type: str  # splunk, elastic, qradar, sentinel
    host: str
    token: Optional[str] = None
    enabled: bool = True


class ForwardEvent(BaseModel):
    event_type: str
    severity: str
    message: str
    metadata: Optional[Dict] = None


def load_config():
    """Load SIEM config from file"""
    global SIEM_CONFIG, SIEM_EVENTS

    if os.path.exists(siem_config_file):
        try:
            with open(siem_config_file, "r", encoding="utf-8") as f:
                SIEM_CONFIG.update(json.load(f))
        except:
            pass

    if os.path.exists(siem_events_file):
        try:
            with open(siem_events_file, "r", encoding="utf-8") as f:
                SIEM_EVENTS = json.load(f)
        except:
            pass


def save_config():
    """Save SIEM config to file"""
    os.makedirs(os.path.dirname(siem_config_file), exist_ok=True)

    with open(siem_config_file, "w", encoding="utf-8") as f:
        json.dump(SIEM_CONFIG, f, indent=2, default=str)


def save_events():
    """Save SIEM events to file"""
    os.makedirs(os.path.dirname(siem_events_file), exist_ok=True)

    with open(siem_events_file, "w", encoding="utf-8") as f:
        json.dump(SIEM_EVENTS[-1000:], f, indent=2, default=str)


def test_connection(conn_type: str, host: str, token: str = None) -> Dict:
    """Test SIEM connection"""
    if not REQUESTS_AVAILABLE:
        return {"success": False, "error": "requests library not available"}

    if not host:
        return {"success": False, "error": "Host not configured"}

    try:
        if conn_type == "splunk":
            response = requests.get(
                f"{host}/services/server/info",
                headers={"Authorization": f"Bearer {token}"},
                verify=False,
                timeout=10,
            )
            return {
                "success": response.status_code == 200,
                "status_code": response.status_code,
            }

        elif conn_type == "elastic":
            response = requests.get(
                f"{host}/_cluster/health",
                headers={"Authorization": f"ApiKey {token}"},
                timeout=10,
            )
            return {
                "success": response.status_code == 200,
                "status_code": response.status_code,
            }

        elif conn_type == "qradar":
            response = requests.get(
                f"{host}/api/system/about",
                headers={"SEC": token},
                verify=False,
                timeout=10,
            )
            return {
                "success": response.status_code == 200,
                "status_code": response.status_code,
            }

        else:
            return {"success": False, "error": f"Unknown SIEM type: {conn_type}"}

    except requests.exceptions.RequestException as e:
        return {"success": False, "error": str(e)}


def forward_to_siem(event: Dict, connection: Dict) -> bool:
    """Forward event to SIEM"""
    if not REQUESTS_AVAILABLE or not connection.get("enabled"):
        return False

    try:
        conn_type = connection.get("type")
        host = connection.get("host")
        token = connection.get("token")

        if conn_type == "splunk":
            response = requests.post(
                f"{host}/services/collector/event",
                headers={"Authorization": f"Splunk {token}"},
                json={"event": event},
                verify=False,
                timeout=10,
            )
            return response.status_code == 200

        elif conn_type == "elastic":
            response = requests.post(
                f"{host}/cyberguard-events/_doc",
                headers={
                    "Authorization": f"ApiKey {token}",
                    "Content-Type": "application/json",
                },
                json=event,
                timeout=10,
            )
            return response.status_code in [200, 201]

        return False

    except:
        return False


# Initialize
load_config()


@router.get("/status")
async def get_siem_status():
    """Get SIEM integration status"""
    configured = {
        "splunk": bool(SPLUNK_HOST and SPLUNK_TOKEN),
        "elastic": bool(ELASTIC_HOST and ELASTIC_API_KEY),
        "qradar": bool(QRADAR_HOST and QRADAR_TOKEN),
        "sentinel": bool(SENTINEL_WORKSPACE_ID and SENTINEL_KEY),
    }

    return {
        "success": True,
        "data": {
            "status": "active" if any(configured.values()) else "not_configured",
            "platforms_configured": configured,
            "connections": len(SIEM_CONFIG.get("connections", [])),
            "forwarding_enabled": SIEM_CONFIG.get("forwarding_enabled", False),
            "events_forwarded": len(SIEM_EVENTS),
            "requests_available": REQUESTS_AVAILABLE,
        },
    }


@router.get("/platforms")
async def get_siem_platforms():
    """Get available SIEM platforms"""
    platforms = [
        {
            "id": "splunk",
            "name": "Splunk Enterprise",
            "logo": "🔴",
            "description": "Enterprise SIEM için Splunk entegrasyonu",
            "configured": bool(SPLUNK_HOST and SPLUNK_TOKEN),
            "features": ["Log Aggregation", "Real-time Search", "Dashboards", "Alerts"],
            "documentation": "https://docs.splunk.com",
        },
        {
            "id": "elastic",
            "name": "Elastic SIEM",
            "logo": "🟡",
            "description": "Elasticsearch tabanlı güvenlik izleme",
            "configured": bool(ELASTIC_HOST and ELASTIC_API_KEY),
            "features": ["ELK Stack", "Machine Learning", "SIEM", "Observability"],
            "documentation": "https://www.elastic.co/security",
        },
        {
            "id": "qradar",
            "name": "IBM QRadar",
            "logo": "🔵",
            "description": "IBM kurumsal SIEM çözümü",
            "configured": bool(QRADAR_HOST and QRADAR_TOKEN),
            "features": ["Threat Intelligence", "Incident Forensics", "Risk Manager"],
            "documentation": "https://www.ibm.com/qradar",
        },
        {
            "id": "sentinel",
            "name": "Microsoft Sentinel",
            "logo": "🟢",
            "description": "Azure bulut-native SIEM",
            "configured": bool(SENTINEL_WORKSPACE_ID and SENTINEL_KEY),
            "features": ["Cloud-native", "AI Analytics", "SOAR", "Azure Integration"],
            "documentation": "https://docs.microsoft.com/azure/sentinel",
        },
        {
            "id": "wazuh",
            "name": "Wazuh",
            "logo": "⚪",
            "description": "Açık kaynak güvenlik izleme",
            "configured": False,
            "features": [
                "Open Source",
                "Intrusion Detection",
                "Log Analysis",
                "Compliance",
            ],
            "documentation": "https://wazuh.com/docs",
        },
    ]

    return {
        "success": True,
        "data": {
            "platforms": platforms,
            "total": len(platforms),
            "configured_count": len([p for p in platforms if p["configured"]]),
        },
    }


@router.get("/rules")
async def get_siem_rules():
    """Get SIEM forwarding rules"""
    # Get rules from config or return defaults
    rules = SIEM_CONFIG.get("active_rules", [])

    if not rules:
        # Default rules
        rules = [
            {
                "id": "RULE-001",
                "name": "Critical Threat Alert",
                "description": "Forward all critical severity events",
                "enabled": True,
                "condition": "severity == 'critical'",
                "action": "forward_all",
                "priority": 1,
                "created_at": datetime.now().isoformat(),
            },
            {
                "id": "RULE-002",
                "name": "Malware Detection",
                "description": "Forward malware detection events",
                "enabled": True,
                "condition": "event_type == 'malware'",
                "action": "forward_all",
                "priority": 2,
                "created_at": datetime.now().isoformat(),
            },
            {
                "id": "RULE-003",
                "name": "Network Intrusion",
                "description": "Forward network intrusion events",
                "enabled": True,
                "condition": "event_type == 'intrusion'",
                "action": "forward_all",
                "priority": 3,
                "created_at": datetime.now().isoformat(),
            },
            {
                "id": "RULE-004",
                "name": "Authentication Failures",
                "description": "Forward failed login attempts",
                "enabled": False,
                "condition": "event_type == 'auth_failure' AND count > 5",
                "action": "forward_splunk",
                "priority": 4,
                "created_at": datetime.now().isoformat(),
            },
            {
                "id": "RULE-005",
                "name": "Data Exfiltration",
                "description": "Forward data exfiltration alerts",
                "enabled": True,
                "condition": "event_type == 'exfiltration'",
                "action": "forward_all",
                "priority": 1,
                "created_at": datetime.now().isoformat(),
            },
        ]
        SIEM_CONFIG["active_rules"] = rules
        save_config()

    return {
        "success": True,
        "data": {
            "rules": rules,
            "total": len(rules),
            "active_count": len([r for r in rules if r.get("enabled")]),
        },
    }


@router.get("/connections")
async def get_connections():
    """Get SIEM connections"""
    return {
        "success": True,
        "data": {"connections": SIEM_CONFIG.get("connections", [])},
    }


@router.post("/connections")
async def add_connection(conn: SIEMConnection):
    """Add a new SIEM connection"""
    new_conn = {
        "id": f"SIEM-{len(SIEM_CONFIG.get('connections', [])) + 1}",
        "name": conn.name,
        "type": conn.type,
        "host": conn.host,
        "token": conn.token,
        "enabled": conn.enabled,
        "created_at": datetime.now().isoformat(),
    }

    SIEM_CONFIG.setdefault("connections", []).append(new_conn)
    save_config()

    return {"success": True, "data": new_conn, "message": "Connection added"}


@router.post("/connections/{conn_id}/test")
async def test_siem_connection(conn_id: str):
    """Test a SIEM connection"""
    for conn in SIEM_CONFIG.get("connections", []):
        if conn.get("id") == conn_id:
            result = test_connection(conn["type"], conn["host"], conn.get("token"))
            return {"success": True, "data": result}

    raise HTTPException(status_code=404, detail="Connection not found")


@router.put("/connections/{conn_id}")
async def update_connection(conn_id: str, enabled: bool = None):
    """Update a SIEM connection"""
    for conn in SIEM_CONFIG.get("connections", []):
        if conn.get("id") == conn_id:
            if enabled is not None:
                conn["enabled"] = enabled
            conn["updated_at"] = datetime.now().isoformat()
            save_config()
            return {"success": True, "data": conn}

    raise HTTPException(status_code=404, detail="Connection not found")


@router.delete("/connections/{conn_id}")
async def delete_connection(conn_id: str):
    """Delete a SIEM connection"""
    connections = SIEM_CONFIG.get("connections", [])

    for i, conn in enumerate(connections):
        if conn.get("id") == conn_id:
            deleted = connections.pop(i)
            save_config()
            return {"success": True, "message": "Connection deleted", "data": deleted}

    raise HTTPException(status_code=404, detail="Connection not found")


@router.post("/forward")
async def forward_event(event: ForwardEvent):
    """Forward an event to all enabled SIEM connections"""
    event_data = {
        "timestamp": datetime.now().isoformat(),
        "event_type": event.event_type,
        "severity": event.severity,
        "message": event.message,
        "metadata": event.metadata or {},
        "source": "CyberGuard AI",
    }

    # Record event
    SIEM_EVENTS.append(event_data)
    save_events()

    # Forward to all enabled connections
    results = []
    for conn in SIEM_CONFIG.get("connections", []):
        if conn.get("enabled"):
            success = forward_to_siem(event_data, conn)
            results.append({"connection": conn["name"], "success": success})

    return {
        "success": True,
        "data": {"event": event_data, "forwarding_results": results},
    }


@router.get("/events")
async def get_forwarded_events(limit: int = 100):
    """Get forwarded events"""
    sorted_events = sorted(
        SIEM_EVENTS, key=lambda x: x.get("timestamp", ""), reverse=True
    )

    return {
        "success": True,
        "data": {"events": sorted_events[:limit], "total": len(SIEM_EVENTS)},
    }


@router.put("/forwarding")
async def toggle_forwarding(enabled: bool):
    """Enable/disable event forwarding"""
    SIEM_CONFIG["forwarding_enabled"] = enabled
    save_config()

    return {
        "success": True,
        "message": f"Forwarding {'enabled' if enabled else 'disabled'}",
    }


@router.get("/stats")
async def get_siem_stats():
    """Get SIEM integration statistics"""
    connections = SIEM_CONFIG.get("connections", [])

    by_type = {}
    for conn in connections:
        t = conn.get("type", "unknown")
        by_type[t] = by_type.get(t, 0) + 1

    by_severity = {}
    for event in SIEM_EVENTS:
        s = event.get("severity", "unknown")
        by_severity[s] = by_severity.get(s, 0) + 1

    return {
        "success": True,
        "data": {
            "total_connections": len(connections),
            "active_connections": len([c for c in connections if c.get("enabled")]),
            "connections_by_type": by_type,
            "total_events_forwarded": len(SIEM_EVENTS),
            "events_by_severity": by_severity,
            "forwarding_enabled": SIEM_CONFIG.get("forwarding_enabled", False),
        },
    }
