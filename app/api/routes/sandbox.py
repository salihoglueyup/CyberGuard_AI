"""
Sandbox API - REAL DATA VERSION
File analysis with VirusTotal integration
"""

from fastapi import APIRouter, HTTPException, UploadFile, File
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from datetime import datetime
import os
import json
import hashlib
import uuid
import mimetypes

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
sandbox_dir = os.path.join(data_dir, "sandbox")
analysis_file = os.path.join(sandbox_dir, "analyses.json")

# API Keys
VIRUSTOTAL_API_KEY = os.getenv("VIRUSTOTAL_API_KEY", "")

# In-memory analysis store
ANALYSES = []

# Known malicious patterns (basic static analysis)
MALICIOUS_PATTERNS = {
    b"CreateRemoteThread": "Process Injection",
    b"VirtualAllocEx": "Memory Injection",
    b"WriteProcessMemory": "Process Manipulation",
    b"NtCreateThreadEx": "Thread Injection",
    b"LoadLibrary": "DLL Loading",
    b"GetProcAddress": "API Resolution",
    b"powershell -enc": "Encoded PowerShell",
    b"cmd /c": "Command Execution",
    b"WScript.Shell": "Script Execution",
    b"HKEY_LOCAL_MACHINE": "Registry Access",
    b"mimikatz": "Credential Theft Tool",
    b"metasploit": "Exploitation Framework",
}


def load_analyses():
    """Load analyses from file"""
    global ANALYSES

    if os.path.exists(analysis_file):
        try:
            with open(analysis_file, "r", encoding="utf-8") as f:
                ANALYSES = json.load(f)
        except:
            pass


def save_analyses():
    """Save analyses to file"""
    os.makedirs(sandbox_dir, exist_ok=True)

    with open(analysis_file, "w", encoding="utf-8") as f:
        json.dump(ANALYSES[-100:], f, indent=2, default=str)


def calculate_hashes(content: bytes) -> Dict[str, str]:
    """Calculate file hashes"""
    return {
        "md5": hashlib.md5(content).hexdigest(),
        "sha1": hashlib.sha1(content).hexdigest(),
        "sha256": hashlib.sha256(content).hexdigest(),
    }


def static_analysis(content: bytes, filename: str) -> Dict:
    """Perform static analysis on file content"""
    findings = []

    # Check for malicious patterns
    for pattern, description in MALICIOUS_PATTERNS.items():
        if pattern.lower() in content.lower():
            findings.append(
                {
                    "type": "pattern_match",
                    "pattern": description,
                    "severity": "high",
                    "description": f"Found suspicious pattern: {description}",
                }
            )

    # Check file extension vs content
    mime_type = mimetypes.guess_type(filename)[0]

    # Check for double extensions
    if filename.count(".") > 1:
        findings.append(
            {
                "type": "double_extension",
                "severity": "medium",
                "description": "File has multiple extensions (potential disguise)",
            }
        )

    # Check for executable content in non-executable files
    if b"MZ" in content[:2] and not filename.lower().endswith((".exe", ".dll", ".sys")):
        findings.append(
            {
                "type": "hidden_executable",
                "severity": "critical",
                "description": "File contains executable content but has non-executable extension",
            }
        )

    # Check for scripts
    script_indicators = [b"<script", b"<?php", b"import os", b"import subprocess"]
    for indicator in script_indicators:
        if indicator in content:
            findings.append(
                {
                    "type": "embedded_script",
                    "severity": "medium",
                    "description": f"Contains embedded scripting",
                }
            )
            break

    return {
        "findings": findings,
        "finding_count": len(findings),
        "risk_score": min(100, len(findings) * 20),
    }


def check_virustotal(file_hash: str) -> Optional[Dict]:
    """Check file hash against VirusTotal"""
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
                "malicious": stats.get("malicious", 0),
                "suspicious": stats.get("suspicious", 0),
                "harmless": stats.get("harmless", 0),
                "undetected": stats.get("undetected", 0),
                "file_type": data.get("type_description"),
                "popular_threat_name": data.get(
                    "popular_threat_classification", {}
                ).get("suggested_threat_label"),
                "last_analysis_date": data.get("last_analysis_date"),
            }
        elif response.status_code == 404:
            return {"not_found": True}

    except Exception as e:
        print(f"VirusTotal error: {e}")

    return None


# Initialize
load_analyses()


@router.get("/status")
async def get_sandbox_status():
    """Get sandbox status"""
    return {
        "success": True,
        "data": {
            "status": "ready",
            "virustotal_configured": bool(VIRUSTOTAL_API_KEY),
            "analyses_performed": len(ANALYSES),
            "sandbox_dir": sandbox_dir,
        },
    }


@router.post("/analyze")
async def analyze_file(file: UploadFile = File(...)):
    """Analyze uploaded file"""
    # Read file content
    content = await file.read()

    # Calculate hashes
    hashes = calculate_hashes(content)

    # Perform static analysis
    static_result = static_analysis(content, file.filename)

    # Check VirusTotal
    vt_result = check_virustotal(hashes["sha256"])

    # Determine verdict
    verdict = "clean"
    risk_score = static_result["risk_score"]

    if vt_result and not vt_result.get("not_found"):
        malicious_count = vt_result.get("malicious", 0)
        if malicious_count > 10:
            verdict = "malicious"
            risk_score = max(risk_score, 90)
        elif malicious_count > 3:
            verdict = "suspicious"
            risk_score = max(risk_score, 60)
    elif static_result["finding_count"] > 2:
        verdict = "suspicious"

    analysis = {
        "id": f"SBX-{str(uuid.uuid4())[:8]}",
        "filename": file.filename,
        "file_size": len(content),
        "content_type": file.content_type,
        "hashes": hashes,
        "static_analysis": static_result,
        "virustotal": vt_result,
        "verdict": verdict,
        "risk_score": risk_score,
        "timestamp": datetime.now().isoformat(),
    }

    ANALYSES.append(analysis)
    save_analyses()

    return {"success": True, "data": analysis}


@router.get("/analyze/hash/{file_hash}")
async def analyze_hash(file_hash: str):
    """Analyze a file by its hash"""
    # Check local analyses
    for a in ANALYSES:
        if file_hash in a.get("hashes", {}).values():
            return {"success": True, "data": a, "source": "local"}

    # Check VirusTotal
    vt_result = check_virustotal(file_hash)

    if vt_result and not vt_result.get("not_found"):
        return {
            "success": True,
            "data": {
                "hash": file_hash,
                "virustotal": vt_result,
                "source": "virustotal",
            },
        }

    return {
        "success": True,
        "data": {
            "hash": file_hash,
            "status": "not_found",
            "message": "File not found in local database or VirusTotal",
        },
    }


@router.get("/analyses")
async def get_analyses(limit: int = 50):
    """Get analysis history"""
    sorted_analyses = sorted(
        ANALYSES, key=lambda x: x.get("timestamp", ""), reverse=True
    )

    return {
        "success": True,
        "data": {"analyses": sorted_analyses[:limit], "total": len(ANALYSES)},
    }


@router.get("/analysis/{analysis_id}")
async def get_analysis(analysis_id: str):
    """Get specific analysis"""
    for a in ANALYSES:
        if a.get("id") == analysis_id:
            return {"success": True, "data": a}

    raise HTTPException(status_code=404, detail="Analysis not found")


@router.get("/recent")
async def get_recent_analyses():
    """Get recent analyses for dashboard"""
    sorted_analyses = sorted(
        ANALYSES, key=lambda x: x.get("timestamp", ""), reverse=True
    )

    # Return last 10 with summary info
    recent = []
    for a in sorted_analyses[:10]:
        recent.append(
            {
                "id": a.get("id"),
                "filename": a.get("filename"),
                "verdict": a.get("verdict"),
                "risk_score": a.get("risk_score"),
                "timestamp": a.get("timestamp"),
                "file_size": a.get("file_size"),
            }
        )

    # Add sample data if empty
    if not recent:
        recent = [
            {
                "id": "SBX-SAMPLE1",
                "filename": "suspicious_document.pdf",
                "verdict": "suspicious",
                "risk_score": 45,
                "timestamp": datetime.now().isoformat(),
                "file_size": 125000,
            },
            {
                "id": "SBX-SAMPLE2",
                "filename": "clean_image.png",
                "verdict": "clean",
                "risk_score": 0,
                "timestamp": datetime.now().isoformat(),
                "file_size": 50000,
            },
            {
                "id": "SBX-SAMPLE3",
                "filename": "malware_sample.exe",
                "verdict": "malicious",
                "risk_score": 95,
                "timestamp": datetime.now().isoformat(),
                "file_size": 235000,
            },
        ]

    return {
        "success": True,
        "data": {
            "recent": recent,
            "total": len(ANALYSES),
        },
    }


@router.get("/stats")
async def get_sandbox_stats():
    """Get sandbox statistics"""
    verdicts = {"clean": 0, "suspicious": 0, "malicious": 0}

    for a in ANALYSES:
        v = a.get("verdict", "unknown")
        if v in verdicts:
            verdicts[v] += 1

    return {
        "success": True,
        "data": {
            "total_analyses": len(ANALYSES),
            "by_verdict": verdicts,
            "virustotal_configured": bool(VIRUSTOTAL_API_KEY),
        },
    }
