"""
Reports API Routes - CyberGuard AI
Rapor oluşturma ve yönetimi

Dosya Yolu: app/api/routes/reports.py
"""

from fastapi import APIRouter, Response
from pydantic import BaseModel
from typing import Optional, List
from datetime import datetime
import json
import os
import sys
import uuid

# Path düzeltmesi
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
sys.path.insert(0, project_root)

from src.utils.database import DatabaseManager

router = APIRouter()
db = DatabaseManager()

# Reports dizini
REPORTS_DIR = os.path.join(project_root, "reports")
os.makedirs(REPORTS_DIR, exist_ok=True)


class ReportRequest(BaseModel):
    type: str  # dashboard, threat, model, security
    title: Optional[str] = None
    date_range: Optional[str] = "24h"
    include_charts: Optional[bool] = True


@router.get("")
async def get_reports():
    """Tüm raporları listele"""
    try:
        reports = []

        if os.path.exists(REPORTS_DIR):
            for filename in os.listdir(REPORTS_DIR):
                if filename.endswith(".json"):
                    filepath = os.path.join(REPORTS_DIR, filename)
                    try:
                        with open(filepath, "r", encoding="utf-8") as f:
                            meta = json.load(f)
                            reports.append(meta)
                    except:
                        pass

        # Tarihe göre sırala (en yeni önce)
        reports.sort(key=lambda x: x.get("createdAt", ""), reverse=True)

        return {"success": True, "data": reports}

    except Exception as e:
        return {"success": False, "error": str(e), "data": []}


@router.post("/generate")
async def generate_report(request: ReportRequest):
    """Yeni rapor oluştur"""
    try:
        report_id = str(uuid.uuid4())[:8]
        now = datetime.now()

        # Rapor verilerini topla
        report_data = {
            "id": report_id,
            "type": request.type,
            "title": request.title or f"{request.type.title()} Raporu",
            "createdAt": now.isoformat(),
            "createdBy": "Sistem",
            "status": "completed",
        }

        # Tipe göre veri topla
        if request.type == "dashboard":
            stats = db.get_database_stats()
            report_data["data"] = {
                "total_attacks": stats.get("attacks", 0),
                "total_scans": stats.get("scan_results", 0),
                "total_logs": stats.get("system_logs", 0),
                "summary": "Dashboard özet raporu",
            }

        elif request.type == "threat":
            attack_stats = db.get_attack_stats(hours=24)
            report_data["data"] = {
                "total_threats": attack_stats.get("total", 0),
                "by_type": attack_stats.get("by_type", {}),
                "by_severity": attack_stats.get("by_severity", {}),
                "summary": "Tehdit analiz raporu",
            }

        elif request.type == "model":
            # Model registry'den veri al
            registry_path = os.path.join(project_root, "models", "model_registry.json")
            models = []
            if os.path.exists(registry_path):
                with open(registry_path, "r") as f:
                    registry = json.load(f)
                    models = list(registry.get("models", {}).values())

            report_data["data"] = {
                "total_models": len(models),
                "models": [
                    {
                        "name": m.get("name"),
                        "accuracy": m.get("metrics", {}).get("accuracy"),
                    }
                    for m in models[:5]
                ],
                "summary": "ML model performans raporu",
            }

        elif request.type == "security":
            scans = db.get_scan_history(limit=10)
            report_data["data"] = {
                "total_scans": len(scans),
                "malware_detected": sum(1 for s in scans if s.get("is_malware")),
                "summary": "Güvenlik tarama raporu",
            }

        # Boyut hesapla
        report_json = json.dumps(report_data)
        size_kb = len(report_json.encode("utf-8")) / 1024
        report_data["size"] = f"{size_kb:.1f} KB"
        report_data["name"] = (
            f"CyberGuard_{request.type}_{now.strftime('%Y%m%d_%H%M')}.pdf"
        )
        report_data["typeLabel"] = {
            "dashboard": "Dashboard Raporu",
            "threat": "Tehdit Raporu",
            "model": "Model Raporu",
            "security": "Güvenlik Raporu",
        }.get(request.type, "Rapor")

        # Meta dosyası kaydet
        meta_path = os.path.join(REPORTS_DIR, f"{report_id}.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False)

        return {"success": True, "message": "Rapor oluşturuldu", "data": report_data}

    except Exception as e:
        import traceback

        traceback.print_exc()
        return {"success": False, "error": str(e)}


@router.get("/{report_id}")
async def get_report(report_id: str):
    """Belirli bir raporu getir"""
    try:
        meta_path = os.path.join(REPORTS_DIR, f"{report_id}.json")

        if not os.path.exists(meta_path):
            return {"success": False, "error": "Rapor bulunamadı"}

        with open(meta_path, "r", encoding="utf-8") as f:
            report = json.load(f)

        return {"success": True, "data": report}

    except Exception as e:
        return {"success": False, "error": str(e)}


@router.delete("/{report_id}")
async def delete_report(report_id: str):
    """Rapor sil"""
    try:
        meta_path = os.path.join(REPORTS_DIR, f"{report_id}.json")

        if os.path.exists(meta_path):
            os.remove(meta_path)
            return {"success": True, "message": "Rapor silindi"}

        return {"success": False, "error": "Rapor bulunamadı"}

    except Exception as e:
        return {"success": False, "error": str(e)}


@router.get("/{report_id}/download")
async def download_report(report_id: str):
    """Rapor indir (JSON olarak)"""
    try:
        meta_path = os.path.join(REPORTS_DIR, f"{report_id}.json")

        if not os.path.exists(meta_path):
            return {"success": False, "error": "Rapor bulunamadı"}

        with open(meta_path, "r", encoding="utf-8") as f:
            report = json.load(f)

        # JSON response olarak döndür
        content = json.dumps(report, indent=2, ensure_ascii=False)

        return Response(
            content=content,
            media_type="application/json",
            headers={
                "Content-Disposition": f"attachment; filename={report.get('name', 'report.json')}"
            },
        )

    except Exception as e:
        return {"success": False, "error": str(e)}


@router.get("/stats/summary")
async def get_reports_stats():
    """Rapor istatistikleri"""
    try:
        reports = []

        if os.path.exists(REPORTS_DIR):
            for filename in os.listdir(REPORTS_DIR):
                if filename.endswith(".json"):
                    filepath = os.path.join(REPORTS_DIR, filename)
                    try:
                        with open(filepath, "r", encoding="utf-8") as f:
                            reports.append(json.load(f))
                    except:
                        pass

        type_counts = {}
        for r in reports:
            t = r.get("type", "other")
            type_counts[t] = type_counts.get(t, 0) + 1

        return {
            "success": True,
            "data": {"total": len(reports), "by_type": type_counts},
        }

    except Exception as e:
        return {"success": False, "error": str(e)}
