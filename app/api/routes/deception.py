"""
Deception Technology API - REAL DATA VERSION
Manages honeypots with persistent logging
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import os
import json
import uuid
import socket

router = APIRouter()

# Path setup
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
data_dir = os.path.join(project_root, "data")
honeypot_file = os.path.join(data_dir, "honeypots.json")
captures_file = os.path.join(data_dir, "honeypot_captures.json")

# In-memory store with persistence
HONEYPOTS = []
CAPTURES = []


class HoneypotCreate(BaseModel):
    name: str
    type: str  # ssh, http, ftp, rdp, smb, telnet
    port: int
    ip: str = "0.0.0.0"
    enabled: bool = True


class CaptureReport(BaseModel):
    honeypot_id: str
    attacker_ip: str
    attack_type: str
    data_captured: Optional[str] = None
    metadata: Optional[Dict] = None


def load_data():
    """Load honeypot data from files"""
    global HONEYPOTS, CAPTURES

    if os.path.exists(honeypot_file):
        try:
            with open(honeypot_file, "r", encoding="utf-8") as f:
                HONEYPOTS = json.load(f)
        except:
            pass

    if os.path.exists(captures_file):
        try:
            with open(captures_file, "r", encoding="utf-8") as f:
                CAPTURES = json.load(f)
        except:
            pass

    # Initialize default honeypots if none exist
    if not HONEYPOTS:
        HONEYPOTS = [
            {
                "id": "hp-ssh-01",
                "name": "SSH Honeypot",
                "type": "ssh",
                "port": 2222,
                "ip": "0.0.0.0",
                "enabled": True,
                "created_at": datetime.now().isoformat(),
            },
            {
                "id": "hp-http-01",
                "name": "HTTP Honeypot",
                "type": "http",
                "port": 8888,
                "ip": "0.0.0.0",
                "enabled": True,
                "created_at": datetime.now().isoformat(),
            },
            {
                "id": "hp-ftp-01",
                "name": "FTP Honeypot",
                "type": "ftp",
                "port": 2121,
                "ip": "0.0.0.0",
                "enabled": True,
                "created_at": datetime.now().isoformat(),
            },
            {
                "id": "hp-telnet-01",
                "name": "Telnet Honeypot",
                "type": "telnet",
                "port": 2323,
                "ip": "0.0.0.0",
                "enabled": True,
                "created_at": datetime.now().isoformat(),
            },
        ]
        save_data()


def save_data():
    """Save honeypot data to files"""
    os.makedirs(os.path.dirname(honeypot_file), exist_ok=True)

    with open(honeypot_file, "w", encoding="utf-8") as f:
        json.dump(HONEYPOTS, f, indent=2, default=str)

    with open(captures_file, "w", encoding="utf-8") as f:
        json.dump(CAPTURES[-1000:], f, indent=2, default=str)  # Keep last 1000


def get_honeypot_stats(honeypot_id: str) -> Dict:
    """Get statistics for a honeypot"""
    hp_captures = [c for c in CAPTURES if c.get("honeypot_id") == honeypot_id]

    # Count by attack type
    by_type = {}
    for c in hp_captures:
        t = c.get("attack_type", "unknown")
        by_type[t] = by_type.get(t, 0) + 1

    # Unique attackers
    unique_ips = set(c.get("attacker_ip") for c in hp_captures)

    # Recent captures (last 24h)
    cutoff = datetime.now() - timedelta(hours=24)
    recent = 0
    for c in hp_captures:
        try:
            ts = datetime.fromisoformat(c.get("timestamp", "2000-01-01"))
            if ts > cutoff:
                recent += 1
        except:
            pass

    return {
        "total_captures": len(hp_captures),
        "unique_attackers": len(unique_ips),
        "captures_24h": recent,
        "by_attack_type": by_type,
    }


def check_port_available(port: int) -> bool:
    """Check if a port is available"""
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(1)
        result = sock.connect_ex(("127.0.0.1", port))
        sock.close()
        return result != 0  # Port is available if connection failed
    except:
        return True


# Initialize
load_data()


@router.get("/status")
async def get_deception_status():
    """Get deception technology status"""
    active = [h for h in HONEYPOTS if h.get("enabled")]

    # Count total captures
    total_captures = len(CAPTURES)

    # Count unique attackers
    unique_attackers = len(set(c.get("attacker_ip") for c in CAPTURES))

    return {
        "success": True,
        "data": {
            "status": "active" if active else "inactive",
            "honeypots_total": len(HONEYPOTS),
            "honeypots_active": len(active),
            "total_captures": total_captures,
            "unique_attackers": unique_attackers,
        },
    }


@router.get("/honeypots")
async def get_honeypots():
    """Get all honeypots"""
    result = []

    for hp in HONEYPOTS:
        stats = get_honeypot_stats(hp["id"])
        result.append(
            {**hp, "stats": stats, "port_available": check_port_available(hp["port"])}
        )

    return {"success": True, "data": {"honeypots": result, "total": len(result)}}


@router.get("/honeypots/{honeypot_id}")
async def get_honeypot(honeypot_id: str):
    """Get specific honeypot"""
    for hp in HONEYPOTS:
        if hp.get("id") == honeypot_id:
            stats = get_honeypot_stats(honeypot_id)
            return {
                "success": True,
                "data": {
                    **hp,
                    "stats": stats,
                    "recent_captures": [
                        c for c in CAPTURES if c.get("honeypot_id") == honeypot_id
                    ][-10:],
                },
            }

    raise HTTPException(status_code=404, detail="Honeypot not found")


@router.post("/honeypots")
async def create_honeypot(hp: HoneypotCreate):
    """Create a new honeypot"""
    # Check port availability
    if not check_port_available(hp.port):
        raise HTTPException(status_code=400, detail=f"Port {hp.port} is already in use")

    new_hp = {
        "id": f"hp-{hp.type}-{str(uuid.uuid4())[:8]}",
        "name": hp.name,
        "type": hp.type,
        "port": hp.port,
        "ip": hp.ip,
        "enabled": hp.enabled,
        "created_at": datetime.now().isoformat(),
    }

    HONEYPOTS.append(new_hp)
    save_data()

    return {"success": True, "data": new_hp, "message": "Honeypot created"}


@router.put("/honeypots/{honeypot_id}")
async def update_honeypot(honeypot_id: str, enabled: bool = None, name: str = None):
    """Update a honeypot"""
    for hp in HONEYPOTS:
        if hp.get("id") == honeypot_id:
            if enabled is not None:
                hp["enabled"] = enabled
            if name is not None:
                hp["name"] = name
            hp["updated_at"] = datetime.now().isoformat()
            save_data()
            return {"success": True, "data": hp, "message": "Honeypot updated"}

    raise HTTPException(status_code=404, detail="Honeypot not found")


@router.delete("/honeypots/{honeypot_id}")
async def delete_honeypot(honeypot_id: str):
    """Delete a honeypot"""
    global HONEYPOTS

    for i, hp in enumerate(HONEYPOTS):
        if hp.get("id") == honeypot_id:
            deleted = HONEYPOTS.pop(i)
            save_data()
            return {"success": True, "message": "Honeypot deleted", "data": deleted}

    raise HTTPException(status_code=404, detail="Honeypot not found")


@router.get("/captures")
async def get_captures(honeypot_id: Optional[str] = None, limit: int = 100):
    """Get captured attacks"""
    filtered = CAPTURES.copy()

    if honeypot_id:
        filtered = [c for c in filtered if c.get("honeypot_id") == honeypot_id]

    # Sort by timestamp descending
    filtered.sort(key=lambda x: x.get("timestamp", ""), reverse=True)

    return {
        "success": True,
        "data": {"captures": filtered[:limit], "total": len(filtered)},
    }


@router.post("/captures")
async def report_capture(capture: CaptureReport):
    """Report a new capture (used by honeypot services)"""
    # Verify honeypot exists
    hp = None
    for h in HONEYPOTS:
        if h.get("id") == capture.honeypot_id:
            hp = h
            break

    if not hp:
        raise HTTPException(status_code=404, detail="Honeypot not found")

    new_capture = {
        "id": str(uuid.uuid4())[:12],
        "honeypot_id": capture.honeypot_id,
        "honeypot_type": hp.get("type"),
        "honeypot_port": hp.get("port"),
        "attacker_ip": capture.attacker_ip,
        "attack_type": capture.attack_type,
        "data_captured": capture.data_captured,
        "metadata": capture.metadata or {},
        "timestamp": datetime.now().isoformat(),
    }

    CAPTURES.append(new_capture)
    save_data()

    return {"success": True, "data": new_capture, "message": "Capture recorded"}


@router.get("/stats")
async def get_deception_stats():
    """Get deception technology statistics"""
    # By honeypot type
    by_type = {}
    for hp in HONEYPOTS:
        t = hp.get("type", "unknown")
        by_type[t] = by_type.get(t, 0) + 1

    # By attack type
    attack_types = {}
    for c in CAPTURES:
        t = c.get("attack_type", "unknown")
        attack_types[t] = attack_types.get(t, 0) + 1

    # Top attackers
    attacker_counts = {}
    for c in CAPTURES:
        ip = c.get("attacker_ip", "unknown")
        attacker_counts[ip] = attacker_counts.get(ip, 0) + 1

    top_attackers = sorted(attacker_counts.items(), key=lambda x: x[1], reverse=True)[
        :10
    ]

    return {
        "success": True,
        "data": {
            "honeypots_by_type": by_type,
            "captures_by_attack_type": attack_types,
            "top_attackers": [
                {"ip": ip, "count": count} for ip, count in top_attackers
            ],
            "total_honeypots": len(HONEYPOTS),
            "active_honeypots": len([h for h in HONEYPOTS if h.get("enabled")]),
            "total_captures": len(CAPTURES),
            "unique_attackers": len(set(c.get("attacker_ip") for c in CAPTURES)),
        },
    }


@router.get("/dashboard")
async def get_deception_dashboard():
    """Get dashboard data for deception technology"""
    active = [h for h in HONEYPOTS if h.get("enabled")]

    # Recent captures
    recent = sorted(CAPTURES, key=lambda x: x.get("timestamp", ""), reverse=True)[:5]

    # Captures in last 24h
    cutoff = datetime.now() - timedelta(hours=24)
    captures_24h = 0
    for c in CAPTURES:
        try:
            ts = datetime.fromisoformat(c.get("timestamp", "2000-01-01"))
            if ts > cutoff:
                captures_24h += 1
        except:
            pass

    return {
        "success": True,
        "data": {
            "active_honeypots": len(active),
            "total_captures": len(CAPTURES),
            "captures_24h": captures_24h,
            "unique_attackers": len(set(c.get("attacker_ip") for c in CAPTURES)),
            "recent_captures": recent,
            "honeypots": [
                {
                    "id": h["id"],
                    "name": h["name"],
                    "type": h["type"],
                    "port": h["port"],
                    "enabled": h.get("enabled", False),
                    "captures": len(
                        [c for c in CAPTURES if c.get("honeypot_id") == h["id"]]
                    ),
                }
                for h in HONEYPOTS
            ],
        },
    }
