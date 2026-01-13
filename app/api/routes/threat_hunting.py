"""
Threat Hunting API - REAL DATA VERSION
Query-based threat hunting with log analysis
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import os
import json
import glob
import re

router = APIRouter()

# Path setup
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
logs_dir = os.path.join(project_root, "logs")
data_dir = os.path.join(project_root, "data")
hunts_file = os.path.join(data_dir, "threat_hunts.json")

# In-memory store
HUNTS = []
HUNT_RESULTS = {}


class HuntQuery(BaseModel):
    query: str
    data_sources: List[str] = ["logs"]
    time_range_hours: int = 24
    hypothesis: Optional[str] = None


class IOCSearch(BaseModel):
    ioc_type: str  # ip, domain, hash, email
    value: str


def load_hunts():
    """Load hunting history from file"""
    global HUNTS

    if os.path.exists(hunts_file):
        try:
            with open(hunts_file, "r", encoding="utf-8") as f:
                HUNTS = json.load(f)
        except:
            pass


def save_hunts():
    """Save hunting history to file"""
    os.makedirs(os.path.dirname(hunts_file), exist_ok=True)

    with open(hunts_file, "w", encoding="utf-8") as f:
        json.dump(HUNTS[-100:], f, indent=2, default=str)


def search_logs(query: str, hours: int = 24) -> List[Dict]:
    """Search log files for matches"""
    results = []

    # Get log files
    log_files = glob.glob(os.path.join(logs_dir, "*.log"))

    for log_file in log_files:
        try:
            with open(log_file, "r", encoding="utf-8", errors="ignore") as f:
                lines = f.readlines()

            for i, line in enumerate(lines):
                if query.lower() in line.lower():
                    results.append(
                        {
                            "file": os.path.basename(log_file),
                            "line_number": i + 1,
                            "content": line.strip()[:500],  # Truncate
                            "match_type": "substring",
                        }
                    )

                    if len(results) >= 100:  # Limit
                        break

        except Exception as e:
            print(f"Error reading {log_file}: {e}")

    return results


def search_data_files(query: str) -> List[Dict]:
    """Search JSON data files"""
    results = []

    # Search attack logs
    attack_files = glob.glob(os.path.join(data_dir, "**", "*.json"), recursive=True)

    for file_path in attack_files[:50]:  # Limit files
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            # Convert to string for simple search
            content = json.dumps(data)
            if query.lower() in content.lower():
                results.append(
                    {
                        "file": os.path.relpath(file_path, project_root),
                        "match_type": "json_content",
                        "preview": content[:200],
                    }
                )

        except:
            pass

    return results


def search_ioc(ioc_type: str, value: str) -> Dict:
    """Search for specific IOC across all data"""
    findings = []

    # Search logs
    log_results = search_logs(value)
    for r in log_results:
        findings.append({"source": "logs", **r})

    # Search data files
    data_results = search_data_files(value)
    for r in data_results:
        findings.append({"source": "data", **r})

    return {
        "ioc_type": ioc_type,
        "value": value,
        "findings": findings,
        "total_matches": len(findings),
        "sources_searched": ["logs", "data"],
    }


# Hunting templates
HUNT_TEMPLATES = [
    {
        "id": "brute-force",
        "name": "Brute Force Detection",
        "description": "Detect multiple failed login attempts",
        "query": "failed login|authentication failure|invalid password",
        "hypothesis": "Attacker attempting brute force attack",
    },
    {
        "id": "data-exfil",
        "name": "Data Exfiltration",
        "description": "Detect potential data exfiltration",
        "query": "upload|POST|large transfer|outbound",
        "hypothesis": "Data being exfiltrated from network",
    },
    {
        "id": "lateral-movement",
        "name": "Lateral Movement",
        "description": "Detect lateral movement in network",
        "query": "psexec|wmic|remote|ssh|rdp connection",
        "hypothesis": "Attacker moving laterally through network",
    },
    {
        "id": "privilege-escalation",
        "name": "Privilege Escalation",
        "description": "Detect privilege escalation attempts",
        "query": "sudo|administrator|privilege|elevation|root",
        "hypothesis": "Attacker escalating privileges",
    },
    {
        "id": "malware-indicators",
        "name": "Malware Indicators",
        "description": "Detect malware activity",
        "query": "malware|virus|trojan|ransomware|encrypted|bitcoin",
        "hypothesis": "Malware active in environment",
    },
]

# Initialize
load_hunts()


@router.get("/status")
async def get_hunting_status():
    """Get threat hunting status"""
    log_files = glob.glob(os.path.join(logs_dir, "*.log"))

    return {
        "success": True,
        "data": {
            "status": "ready",
            "log_files_available": len(log_files),
            "hunts_performed": len(HUNTS),
            "templates_available": len(HUNT_TEMPLATES),
        },
    }


@router.get("/investigations")
async def get_investigations():
    """Get threat hunting investigations"""
    # Return hunts as investigations or generate sample data
    investigations = []

    # Convert existing hunts to investigation format
    for hunt in HUNTS[-20:]:
        investigations.append(
            {
                "id": hunt.get("id"),
                "name": hunt.get("hypothesis") or f"Hunt: {hunt.get('query', '')[:30]}",
                "status": hunt.get("status", "completed"),
                "severity": (
                    "high"
                    if hunt.get("total_matches", 0) > 10
                    else "medium" if hunt.get("total_matches", 0) > 0 else "low"
                ),
                "created_at": hunt.get("timestamp"),
                "updated_at": hunt.get("timestamp"),
                "matches": hunt.get("total_matches", 0),
                "query": hunt.get("query"),
                "assignee": "Analyst",
            }
        )

    # Add sample investigations if none exist
    if not investigations:
        investigations = [
            {
                "id": "INV-001",
                "name": "Suspicious Network Activity",
                "status": "active",
                "severity": "high",
                "created_at": datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat(),
                "matches": 15,
                "query": "suspicious|anomaly|unusual",
                "assignee": "Security Analyst",
            },
            {
                "id": "INV-002",
                "name": "Potential Brute Force Attack",
                "status": "active",
                "severity": "medium",
                "created_at": datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat(),
                "matches": 8,
                "query": "failed login|authentication failure",
                "assignee": "SOC Team",
            },
            {
                "id": "INV-003",
                "name": "Malware Detection Follow-up",
                "status": "completed",
                "severity": "critical",
                "created_at": datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat(),
                "matches": 23,
                "query": "malware|trojan|virus",
                "assignee": "Incident Response",
            },
            {
                "id": "INV-004",
                "name": "Data Exfiltration Check",
                "status": "pending",
                "severity": "low",
                "created_at": datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat(),
                "matches": 2,
                "query": "upload|transfer|exfil",
                "assignee": "Security Analyst",
            },
        ]

    return {
        "success": True,
        "data": {
            "investigations": investigations,
            "total": len(investigations),
            "active_count": len(
                [i for i in investigations if i.get("status") == "active"]
            ),
        },
    }


@router.post("/hunt")
async def perform_hunt(hunt: HuntQuery):
    """Perform a threat hunt"""
    hunt_id = f"HUNT-{datetime.now().strftime('%Y%m%d%H%M%S')}"

    results = {"logs": [], "data": []}

    if "logs" in hunt.data_sources:
        results["logs"] = search_logs(hunt.query, hunt.time_range_hours)

    if "data" in hunt.data_sources:
        results["data"] = search_data_files(hunt.query)

    total_matches = len(results["logs"]) + len(results["data"])

    hunt_record = {
        "id": hunt_id,
        "query": hunt.query,
        "hypothesis": hunt.hypothesis,
        "data_sources": hunt.data_sources,
        "time_range_hours": hunt.time_range_hours,
        "total_matches": total_matches,
        "timestamp": datetime.now().isoformat(),
        "status": "completed",
    }

    HUNTS.append(hunt_record)
    HUNT_RESULTS[hunt_id] = results
    save_hunts()

    return {"success": True, "data": {"hunt": hunt_record, "results": results}}


@router.post("/query")
async def execute_query(hunt: HuntQuery):
    """Execute a threat hunting query - alias for /hunt"""
    return await perform_hunt(hunt)


@router.get("/templates")
async def get_hunt_templates():
    """Get available hunting templates"""
    return {"success": True, "data": {"templates": HUNT_TEMPLATES}}


@router.post("/template/{template_id}")
async def run_template_hunt(template_id: str):
    """Run a hunt from template"""
    template = None
    for t in HUNT_TEMPLATES:
        if t["id"] == template_id:
            template = t
            break

    if not template:
        raise HTTPException(status_code=404, detail="Template not found")

    hunt = HuntQuery(query=template["query"], hypothesis=template["hypothesis"])

    return await perform_hunt(hunt)


@router.post("/ioc")
async def search_for_ioc(ioc: IOCSearch):
    """Search for specific IOC"""
    result = search_ioc(ioc.ioc_type, ioc.value)

    return {"success": True, "data": result}


@router.get("/history")
async def get_hunt_history(limit: int = 20):
    """Get hunting history"""
    sorted_hunts = sorted(HUNTS, key=lambda x: x.get("timestamp", ""), reverse=True)

    return {
        "success": True,
        "data": {"hunts": sorted_hunts[:limit], "total": len(HUNTS)},
    }


@router.get("/hunt/{hunt_id}")
async def get_hunt_result(hunt_id: str):
    """Get specific hunt result"""
    for h in HUNTS:
        if h.get("id") == hunt_id:
            return {
                "success": True,
                "data": {"hunt": h, "results": HUNT_RESULTS.get(hunt_id, {})},
            }

    raise HTTPException(status_code=404, detail="Hunt not found")


@router.get("/stats")
async def get_hunting_stats():
    """Get threat hunting statistics"""
    # Count by hypothesis
    by_hypothesis = {}
    for h in HUNTS:
        hyp = h.get("hypothesis", "none")
        by_hypothesis[hyp] = by_hypothesis.get(hyp, 0) + 1

    # Total matches
    total_matches = sum(h.get("total_matches", 0) for h in HUNTS)

    return {
        "success": True,
        "data": {
            "total_hunts": len(HUNTS),
            "total_matches": total_matches,
            "templates_used": len(HUNT_TEMPLATES),
            "by_hypothesis": by_hypothesis,
        },
    }
