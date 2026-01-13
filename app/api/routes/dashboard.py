"""
Dashboard API Routes - CyberGuard AI
GERÇEK VERİTABANI + MODEL BİLGİLERİ

Dosya Yolu: app/api/routes/dashboard.py
"""

from fastapi import APIRouter, Query
from typing import Optional
import sys
import os
import json
from datetime import datetime

# Path düzeltmesi
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
sys.path.insert(0, project_root)
models_dir = os.path.join(project_root, "models")
registry_path = os.path.join(models_dir, "model_registry.json")

from src.utils.database import DatabaseManager

router = APIRouter()
db = DatabaseManager()


def load_registry():
    """Model registry yükle"""
    if os.path.exists(registry_path):
        with open(registry_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"models": []}


def get_model_folders():
    """Model klasörlerini listele"""
    folders = []
    if os.path.exists(models_dir):
        for name in os.listdir(models_dir):
            path = os.path.join(models_dir, name)
            if os.path.isdir(path) and not name.startswith("."):
                folders.append(name)
    return folders


@router.get("/stats")
async def get_dashboard_stats(hours: int = Query(24)):
    """Dashboard istatistikleri - Gerçek saldırı + Model bilgileri"""
    try:
        # Model bilgileri
        registry = load_registry()
        models = registry.get("models", [])
        model_folders = get_model_folders()

        total_models = len(models)
        deployed = len([m for m in models if m.get("status") == "deployed"])
        training = len([m for m in models if m.get("status") == "training"])

        # En iyi model
        best_model = None
        best_accuracy = 0
        for m in models:
            acc = m.get("metrics", {}).get("accuracy", 0)
            if acc > best_accuracy:
                best_accuracy = acc
                best_model = m.get("name", m.get("model_name"))

        # Framework dağılımı
        frameworks = {}
        for m in models:
            fw = m.get("framework", "tensorflow")
            frameworks[fw] = frameworks.get(fw, 0) + 1

        # Gerçek saldırı verileri
        attacks = db.get_attacks(hours=hours)
        total_attacks = len(attacks)
        blocked = len([a for a in attacks if a.get("blocked")])
        not_blocked = total_attacks - blocked
        block_rate = round(
            (blocked / total_attacks * 100) if total_attacks > 0 else 0, 1
        )

        # Severity dağılımı
        by_severity = {}
        for a in attacks:
            sev = a.get("severity", "MEDIUM").upper()
            by_severity[sev] = by_severity.get(sev, 0) + 1

        return {
            "success": True,
            "data": {
                # Saldırı istatistikleri
                "total_attacks": total_attacks,
                "blocked": blocked,
                "not_blocked": not_blocked,
                "block_rate": block_rate,
                "critical": by_severity.get("CRITICAL", 0),
                "by_severity": by_severity,
                "by_type": frameworks,
                "period_hours": hours,
                # Model istatistikleri
                "total_models": total_models,
                "deployed_models": deployed,
                "training_models": training,
                "best_model": best_model,
                "best_accuracy": best_accuracy,
                "model_folders": len(model_folders),
                "unique_ips": len(model_folders),
            },
        }
    except Exception as e:
        import traceback

        traceback.print_exc()
        return {"success": False, "error": str(e)}


@router.get("/summary")
async def get_dashboard_summary():
    """Dashboard özet"""
    try:
        registry = load_registry()
        models = registry.get("models", [])
        model_folders = get_model_folders()

        # Gerçek saldırı sayısı
        attacks = db.get_attacks(hours=24)

        # Toplam disk kullanımı
        total_size = 0
        for folder in model_folders:
            folder_path = os.path.join(models_dir, folder)
            for root, dirs, files in os.walk(folder_path):
                for f in files:
                    try:
                        total_size += os.path.getsize(os.path.join(root, f))
                    except:
                        pass

        return {
            "success": True,
            "data": {
                "total_attacks": len(attacks),
                "total_defences": len([a for a in attacks if a.get("blocked")]),
                "total_models": len(models),
                "total_logs": len(model_folders),
                "database_size_mb": round(total_size / (1024 * 1024), 2),
            },
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


@router.get("/hourly-trend")
async def get_hourly_trend(hours: int = Query(24)):
    """Saatlik saldırı trendi - Gerçek veri"""
    try:
        attacks = db.get_attacks(hours=hours)

        # Saatlik gruplama
        hourly = {}
        hourly_blocked = {}
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
                    if a.get("blocked"):
                        hourly_blocked[hour_key] = hourly_blocked.get(hour_key, 0) + 1
                except:
                    pass

        # 24 saat için sonuç oluştur
        result = []
        for h in range(24):
            hour_str = f"{h:02d}:00"
            result.append(
                {
                    "hour": hour_str,
                    "count": hourly.get(hour_str, 0),
                    "blocked": hourly_blocked.get(hour_str, 0),
                }
            )

        return {"success": True, "data": result}
    except Exception as e:
        return {"success": False, "error": str(e), "data": []}


@router.get("/recent-attacks")
async def get_recent_attacks(limit: int = Query(10)):
    """Son saldırılar - Gerçek veri"""
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
                    "destination_ip": a.get("destination_ip", "10.0.0.1"),
                    "port": a.get("destination_port", 80),
                    "severity": a.get("severity", "medium").lower(),
                    "blocked": a.get("blocked", False),
                }
            )

        return {"success": True, "data": result, "count": len(result)}
    except Exception as e:
        return {"success": False, "error": str(e), "data": []}


@router.get("/system/metrics")
async def get_system_metrics():
    """Sistem metrikleri - CPU, Memory, Disk"""
    try:
        import psutil

        # CPU
        cpu_percent = psutil.cpu_percent(interval=0.1)
        cpu_count = psutil.cpu_count()

        # Memory
        memory = psutil.virtual_memory()
        memory_percent = memory.percent
        memory_used_gb = round(memory.used / (1024**3), 2)
        memory_total_gb = round(memory.total / (1024**3), 2)

        # Disk
        disk = psutil.disk_usage("/")
        disk_percent = disk.percent
        disk_used_gb = round(disk.used / (1024**3), 2)
        disk_total_gb = round(disk.total / (1024**3), 2)

        # GPU (optional)
        gpu_percent = 0
        gpu_name = None
        try:
            import GPUtil

            gpus = GPUtil.getGPUs()
            if gpus:
                gpu_percent = gpus[0].load * 100
                gpu_name = gpus[0].name
        except:
            pass

        return {
            "success": True,
            "data": {
                "cpu": {
                    "percent": cpu_percent,
                    "cores": cpu_count,
                },
                "memory": {
                    "percent": memory_percent,
                    "used_gb": memory_used_gb,
                    "total_gb": memory_total_gb,
                },
                "disk": {
                    "percent": disk_percent,
                    "used_gb": disk_used_gb,
                    "total_gb": disk_total_gb,
                },
                "gpu": {
                    "percent": gpu_percent,
                    "name": gpu_name,
                },
            },
        }
    except ImportError:
        return {
            "success": False,
            "error": "psutil not installed",
            "data": {
                "cpu": {"percent": 0, "cores": 0},
                "memory": {"percent": 0, "used_gb": 0, "total_gb": 0},
                "disk": {"percent": 0, "used_gb": 0, "total_gb": 0},
                "gpu": {"percent": 0, "name": None},
            },
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


@router.get("/model-performance")
async def get_model_performance():
    """Model performans metrikleri - gerçek registry verileri"""
    try:
        registry = load_registry()
        models = registry.get("models", [])

        if not models:
            return {
                "success": True,
                "data": {
                    "avg_accuracy": 0,
                    "avg_f1": 0,
                    "avg_precision": 0,
                    "avg_recall": 0,
                    "total_models": 0,
                    "by_attack": {},
                },
            }

        # Ortalama metrikler
        accuracies = []
        f1_scores = []
        precisions = []
        recalls = []

        for m in models:
            metrics = m.get("metrics", {})
            if metrics.get("accuracy"):
                accuracies.append(metrics["accuracy"])
            if metrics.get("f1_score"):
                f1_scores.append(metrics["f1_score"])
            if metrics.get("precision"):
                precisions.append(metrics["precision"])
            if metrics.get("recall"):
                recalls.append(metrics["recall"])

        avg_accuracy = sum(accuracies) / len(accuracies) if accuracies else 0
        avg_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0
        avg_precision = sum(precisions) / len(precisions) if precisions else 0
        avg_recall = sum(recalls) / len(recalls) if recalls else 0

        # Attack-specific performance (from registry or default)
        by_attack = {}
        for m in models:
            attack_perf = m.get("attack_performance", {})
            for attack, perf in attack_perf.items():
                if attack not in by_attack:
                    by_attack[attack] = []
                by_attack[attack].append(perf)

        # Average by attack
        attack_avg = {}
        for attack, perfs in by_attack.items():
            attack_avg[attack] = sum(perfs) / len(perfs) if perfs else 0

        return {
            "success": True,
            "data": {
                "avg_accuracy": round(avg_accuracy, 4),
                "avg_f1": round(avg_f1, 4),
                "avg_precision": round(avg_precision, 4),
                "avg_recall": round(avg_recall, 4),
                "total_models": len(models),
                "by_attack": attack_avg,
                "best_model": (
                    max(
                        models, key=lambda x: x.get("metrics", {}).get("accuracy", 0)
                    ).get("name", "Unknown")
                    if models
                    else None
                ),
            },
        }
    except Exception as e:
        import traceback

        traceback.print_exc()
        return {"success": False, "error": str(e)}
