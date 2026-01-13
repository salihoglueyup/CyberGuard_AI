"""
Scanner API Routes - CyberGuard AI
Dosya yükleme ve tarama - VirusTotal API Entegrasyonu

Dosya Yolu: app/api/routes/scanner.py
"""

from fastapi import APIRouter, UploadFile, File, HTTPException
from typing import Optional, List
import sys
import os
import hashlib
from datetime import datetime
import httpx
import asyncio

# Path düzeltmesi
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
sys.path.insert(0, project_root)

from src.utils.database import DatabaseManager

router = APIRouter()
db = DatabaseManager()

# Upload dizini
UPLOAD_DIR = os.path.join(project_root, "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

# VirusTotal API
VIRUSTOTAL_API_KEY = os.getenv("VIRUSTOTAL_API_KEY", "")
VIRUSTOTAL_URL = "https://www.virustotal.com/api/v3"

# Tehlikeli dosya uzantıları
DANGEROUS_EXTENSIONS = [
    ".exe",
    ".dll",
    ".bat",
    ".cmd",
    ".ps1",
    ".vbs",
    ".js",
    ".jar",
    ".msi",
    ".scr",
]
SUSPICIOUS_EXTENSIONS = [
    ".zip",
    ".rar",
    ".7z",
    ".iso",
    ".pdf",
    ".doc",
    ".docx",
    ".xls",
    ".xlsx",
]


def calculate_file_hash(file_path: str) -> dict:
    """Dosya hash'lerini hesapla"""
    hashes = {"md5": "", "sha1": "", "sha256": ""}

    try:
        with open(file_path, "rb") as f:
            content = f.read()
            hashes["md5"] = hashlib.md5(content).hexdigest()
            hashes["sha1"] = hashlib.sha1(content).hexdigest()
            hashes["sha256"] = hashlib.sha256(content).hexdigest()
    except:
        pass

    return hashes


async def check_virustotal_hash(file_hash: str) -> dict:
    """VirusTotal'da hash kontrolü"""
    if not VIRUSTOTAL_API_KEY:
        return {"error": "API key not configured", "available": False}

    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{VIRUSTOTAL_URL}/files/{file_hash}",
                headers={"x-apikey": VIRUSTOTAL_API_KEY},
                timeout=30,
            )

            if response.status_code == 200:
                data = response.json()
                stats = (
                    data.get("data", {})
                    .get("attributes", {})
                    .get("last_analysis_stats", {})
                )

                return {
                    "available": True,
                    "found": True,
                    "malicious": stats.get("malicious", 0),
                    "suspicious": stats.get("suspicious", 0),
                    "harmless": stats.get("harmless", 0),
                    "undetected": stats.get("undetected", 0),
                    "total_engines": sum(stats.values()),
                    "detection_ratio": f"{stats.get('malicious', 0)}/{sum(stats.values())}",
                    "threat_names": data.get("data", {})
                    .get("attributes", {})
                    .get("popular_threat_classification", {}),
                }
            elif response.status_code == 404:
                return {
                    "available": True,
                    "found": False,
                    "message": "Hash not found in VirusTotal",
                }
            else:
                return {
                    "available": False,
                    "error": f"API error: {response.status_code}",
                }
    except Exception as e:
        return {"available": False, "error": str(e)}


async def check_virustotal_url(url: str) -> dict:
    """VirusTotal'da URL kontrolü"""
    if not VIRUSTOTAL_API_KEY:
        return {"error": "API key not configured", "available": False}

    try:
        import base64

        url_id = base64.urlsafe_b64encode(url.encode()).decode().strip("=")

        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{VIRUSTOTAL_URL}/urls/{url_id}",
                headers={"x-apikey": VIRUSTOTAL_API_KEY},
                timeout=30,
            )

            if response.status_code == 200:
                data = response.json()
                stats = (
                    data.get("data", {})
                    .get("attributes", {})
                    .get("last_analysis_stats", {})
                )

                return {
                    "available": True,
                    "found": True,
                    "malicious": stats.get("malicious", 0),
                    "suspicious": stats.get("suspicious", 0),
                    "harmless": stats.get("harmless", 0),
                    "undetected": stats.get("undetected", 0),
                    "categories": data.get("data", {})
                    .get("attributes", {})
                    .get("categories", {}),
                }
            elif response.status_code == 404:
                return {"available": True, "found": False}
            else:
                return {
                    "available": False,
                    "error": f"API error: {response.status_code}",
                }
    except Exception as e:
        return {"available": False, "error": str(e)}


def analyze_file_local(file_path: str, filename: str) -> dict:
    """Yerel dosya analizi (fallback)"""
    ext = os.path.splitext(filename)[1].lower()
    file_size = os.path.getsize(file_path) if os.path.exists(file_path) else 0

    threat_level = "safe"
    threats_found = []
    confidence = 0.0

    if ext in DANGEROUS_EXTENSIONS:
        threat_level = "dangerous"
        confidence = 0.85  # Fixed high confidence for dangerous extensions
        threats_found = [
            {"name": "Potential Malware", "type": "executable", "severity": "high"},
            {
                "name": "Suspicious Code Pattern",
                "type": "heuristic",
                "severity": "medium",
            },
        ]
    elif ext in SUSPICIOUS_EXTENSIONS:
        threat_level = "suspicious"
        confidence = 0.55  # Fixed medium confidence for suspicious extensions
        threats_found = [
            {"name": "Unknown Content", "type": "archive", "severity": "low"}
        ]
    else:
        threat_level = "safe"
        confidence = 0.95  # Fixed high confidence for safe files

    return {
        "threat_level": threat_level,
        "confidence": round(confidence, 4),
        "threats_found": threats_found,
        "is_malware": threat_level == "dangerous",
        "file_size": file_size,
        "extension": ext,
    }


@router.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """Dosya yükle ve tara (VirusTotal + Yerel)"""
    try:
        # Dosyayı kaydet
        file_path = os.path.join(UPLOAD_DIR, file.filename)

        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)

        # Hash hesapla
        hashes = calculate_file_hash(file_path)

        # Yerel analiz
        local_analysis = analyze_file_local(file_path, file.filename)

        # VirusTotal kontrolü
        vt_result = await check_virustotal_hash(hashes["sha256"])

        # Sonucu birleştir
        if vt_result.get("available") and vt_result.get("found"):
            # VirusTotal sonucu var
            malicious = vt_result.get("malicious", 0)
            suspicious = vt_result.get("suspicious", 0)

            if malicious > 0:
                threat_level = "dangerous"
                confidence = min(0.99, malicious / vt_result.get("total_engines", 70))
            elif suspicious > 0:
                threat_level = "suspicious"
                confidence = min(0.8, suspicious / vt_result.get("total_engines", 70))
            else:
                threat_level = "safe"
                confidence = 0.95

            analysis = {
                "threat_level": threat_level,
                "confidence": round(confidence, 4),
                "is_malware": threat_level == "dangerous",
                "source": "virustotal",
                "virustotal": vt_result,
                "file_size": local_analysis["file_size"],
                "extension": local_analysis["extension"],
            }
        else:
            # VirusTotal bulunamadı, yerel analiz kullan
            analysis = local_analysis
            analysis["source"] = "local"
            analysis["virustotal"] = vt_result

        # Veritabanına kaydet
        scan_data = {
            "filename": file.filename,
            "file_hash": hashes["sha256"],
            "file_size": analysis["file_size"],
            "threat_level": analysis["threat_level"],
            "confidence": analysis["confidence"],
            "scan_source": analysis.get("source", "local"),
        }

        # DB'ye kaydet (scan_results tablosu varsa)
        try:
            db.add_scan_result(scan_data)
        except:
            pass  # Tablo yoksa geç

        result = {
            "id": int(datetime.now().timestamp()),
            "filename": file.filename,
            "file_size": analysis["file_size"],
            "upload_time": datetime.now().isoformat(),
            "scan_time": datetime.now().isoformat(),
            "hashes": hashes,
            "analysis": analysis,
            "status": "completed",
        }

        # Dosyayı sil (güvenlik için)
        if os.path.exists(file_path):
            os.remove(file_path)

        return {"success": True, "data": result}
    except Exception as e:
        return {"success": False, "error": str(e)}


@router.post("/scan-url")
async def scan_url(url: str):
    """URL tara (VirusTotal + Yerel)"""
    try:
        # VirusTotal kontrolü
        vt_result = await check_virustotal_url(url)

        if vt_result.get("available") and vt_result.get("found"):
            malicious = vt_result.get("malicious", 0)
            suspicious = vt_result.get("suspicious", 0)

            if malicious > 0:
                threat_level = "dangerous"
                is_safe = False
            elif suspicious > 0:
                threat_level = "suspicious"
                is_safe = False
            else:
                threat_level = "safe"
                is_safe = True

            result = {
                "id": int(datetime.now().timestamp()),
                "url": url,
                "scan_time": datetime.now().isoformat(),
                "is_safe": is_safe,
                "threat_level": threat_level,
                "source": "virustotal",
                "virustotal": vt_result,
                "confidence": 0.95,
            }
        else:
            # Yerel analiz (fallback)
            is_suspicious = any(
                word in url.lower()
                for word in ["malware", "phishing", "hack", "crack", "keygen"]
            )

            result = {
                "id": int(datetime.now().timestamp()),
                "url": url,
                "scan_time": datetime.now().isoformat(),
                "is_safe": not is_suspicious,
                "threat_level": "dangerous" if is_suspicious else "safe",
                "source": "local",
                "virustotal": vt_result,
                "confidence": 0.7,  # Fixed confidence for local analysis
            }

        return {"success": True, "data": result}
    except Exception as e:
        return {"success": False, "error": str(e)}


@router.get("/results")
async def get_scan_results(limit: int = 20):
    """Tarama sonuçlarını DB'den getir"""
    try:
        results = db.get_scan_history(limit=limit)
        return {"success": True, "data": results, "total": len(results)}
    except:
        return {"success": True, "data": [], "total": 0}


@router.get("/stats")
async def get_scanner_stats():
    """Tarayıcı istatistikleri"""
    try:
        results = db.get_scan_results(limit=1000)

        total = len(results)
        safe = len([r for r in results if r.get("threat_level") == "safe"])
        suspicious = len([r for r in results if r.get("threat_level") == "suspicious"])
        dangerous = len([r for r in results if r.get("threat_level") == "dangerous"])

        return {
            "success": True,
            "data": {
                "total_scans": total,
                "safe": safe,
                "suspicious": suspicious,
                "dangerous": dangerous,
                "detection_rate": round(
                    (dangerous + suspicious) / max(total, 1) * 100, 1
                ),
                "virustotal_enabled": bool(VIRUSTOTAL_API_KEY),
            },
        }
    except:
        return {
            "success": True,
            "data": {
                "total_scans": 0,
                "safe": 0,
                "suspicious": 0,
                "dangerous": 0,
                "detection_rate": 0,
                "virustotal_enabled": bool(VIRUSTOTAL_API_KEY),
            },
        }


@router.get("/api-status")
async def get_api_status():
    """VirusTotal API durumu"""
    return {
        "success": True,
        "data": {
            "virustotal_configured": bool(VIRUSTOTAL_API_KEY),
            "api_key_preview": (
                VIRUSTOTAL_API_KEY[:8] + "..." if VIRUSTOTAL_API_KEY else None
            ),
        },
    }
