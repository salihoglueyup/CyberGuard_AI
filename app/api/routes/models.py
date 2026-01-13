"""
Models API Routes - CyberGuard AI
ML Model yönetimi ve tahmin

Dosya Yolu: app/api/routes/models.py
"""

from fastapi import APIRouter, Query
from pydantic import BaseModel
from typing import Optional, Dict, Any, List
import sys
import os
import json

# Path düzeltmesi - app artık ana dizinde
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
sys.path.insert(0, project_root)

router = APIRouter()
models_dir = os.path.join(project_root, "models")
registry_path = os.path.join(models_dir, "model_registry.json")


class PredictionRequest(BaseModel):
    source_ip: str = "192.168.1.100"
    destination_ip: str = "10.0.0.1"
    port: int = 80
    severity: str = "medium"
    blocked: int = 0
    status: str = "detected"


def load_registry():
    """Model registry yükle"""
    if os.path.exists(registry_path):
        with open(registry_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"models": []}


@router.get("/")
async def list_models():
    """Tüm modelleri listele"""
    try:
        registry = load_registry()

        models = []
        for m in registry.get("models", []):
            metrics = m.get("metrics", {})
            training = m.get("training_config", {})

            models.append(
                {
                    "id": m.get("id", m.get("model_id", "")),
                    "name": m.get("name", m.get("model_name", "Unknown")),
                    "status": m.get("status", "unknown"),
                    "framework": m.get("framework", "tensorflow"),
                    "model_type": m.get("model_type", "neural_network"),
                    "accuracy": metrics.get("accuracy", 0),
                    "f1_score": metrics.get("f1_score", 0),
                    "precision": metrics.get("precision", 0),
                    "recall": metrics.get("recall", 0),
                    "train_samples": training.get("train_samples", 0),
                    "test_samples": training.get("test_samples", 0),
                    "epochs": training.get("epochs", 0),
                    "created_at": m.get("created_at", ""),
                    "description": m.get("description", ""),
                }
            )

        return {"success": True, "data": models, "count": len(models)}

    except Exception as e:
        import traceback

        traceback.print_exc()
        return {"success": False, "error": str(e), "data": []}


@router.get("/deployed")
async def list_deployed_models():
    """Deploy edilmiş modelleri listele"""
    try:
        registry = load_registry()

        deployed = []
        for m in registry.get("models", []):
            if m.get("status") == "deployed":
                metrics = m.get("metrics", {})
                deployed.append(
                    {
                        "id": m.get("id", m.get("model_id", "")),
                        "name": m.get("name", m.get("model_name", "Unknown")),
                        "accuracy": metrics.get("accuracy", 0),
                        "f1_score": metrics.get("f1_score", 0),
                        "train_samples": m.get("training_config", {}).get(
                            "train_samples", 0
                        ),
                    }
                )

        return {"success": True, "data": deployed, "count": len(deployed)}

    except Exception as e:
        return {"success": False, "error": str(e), "data": []}


@router.get("/stats")
async def get_models_stats():
    """Model istatistikleri"""
    try:
        registry = load_registry()
        models = registry.get("models", [])

        total = len(models)
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
            fw = m.get("framework", "unknown")
            frameworks[fw] = frameworks.get(fw, 0) + 1

        return {
            "success": True,
            "data": {
                "total_models": total,
                "deployed": deployed,
                "training": training,
                "best_model": best_model,
                "best_accuracy": best_accuracy,
                "frameworks": frameworks,
            },
        }

    except Exception as e:
        return {"success": False, "error": str(e)}


@router.get("/{model_id}")
async def get_model_info(model_id: str):
    """Model detayları"""
    try:
        registry = load_registry()

        for m in registry.get("models", []):
            if m.get("id") == model_id or m.get("model_id") == model_id:
                return {"success": True, "data": m}

        return {"success": False, "error": "Model bulunamadı"}

    except Exception as e:
        return {"success": False, "error": str(e)}


@router.post("/{model_id}/predict")
async def predict(model_id: str, request: PredictionRequest):
    """ML modeli ile tahmin yap"""
    try:
        from src.api.ml_prediction import MLPredictionAPI

        api = MLPredictionAPI()

        features = {
            "source_ip": request.source_ip,
            "destination_ip": request.destination_ip,
            "port": request.port,
            "severity": request.severity,
            "blocked": request.blocked,
            "status": request.status,
        }

        result = api.predict(features, model_id=model_id)

        return result

    except Exception as e:
        import traceback

        traceback.print_exc()
        return {"success": False, "error": str(e)}


@router.post("/predict")
async def predict_auto(request: PredictionRequest):
    """En iyi model ile otomatik tahmin"""
    try:
        from src.api.ml_prediction import MLPredictionAPI

        api = MLPredictionAPI()

        features = {
            "source_ip": request.source_ip,
            "destination_ip": request.destination_ip,
            "port": request.port,
            "severity": request.severity,
            "blocked": request.blocked,
            "status": request.status,
        }

        result = api.predict(features)

        return result

    except Exception as e:
        import traceback

        traceback.print_exc()
        return {"success": False, "error": str(e)}


@router.get("/compare/all")
async def compare_models():
    """Tüm modelleri karşılaştır"""
    try:
        registry = load_registry()

        comparison = []
        for m in registry.get("models", []):
            metrics = m.get("metrics", {})
            comparison.append(
                {
                    "name": m.get("name", m.get("model_name", "")),
                    "accuracy": round(metrics.get("accuracy", 0) * 100, 2),
                    "f1_score": round(metrics.get("f1_score", 0) * 100, 2),
                    "precision": round(metrics.get("precision", 0) * 100, 2),
                    "recall": round(metrics.get("recall", 0) * 100, 2),
                    "status": m.get("status", "unknown"),
                }
            )

        # Accuracy'ye göre sırala
        comparison.sort(key=lambda x: x["accuracy"], reverse=True)

        return {"success": True, "data": comparison}

    except Exception as e:
        return {"success": False, "error": str(e), "data": []}


def save_registry(registry):
    """Registry'yi kaydet"""
    with open(registry_path, "w", encoding="utf-8") as f:
        json.dump(registry, f, indent=2, ensure_ascii=False)


@router.delete("/{model_id}")
async def delete_model(model_id: str):
    """Modeli sil"""
    try:
        registry = load_registry()

        # Model bul ve sil
        models = registry.get("models", [])
        original_count = len(models)

        registry["models"] = [
            m
            for m in models
            if m.get("id") != model_id and m.get("model_id") != model_id
        ]

        if len(registry["models"]) == original_count:
            return {"success": False, "error": "Model bulunamadı"}

        # Registry'yi kaydet
        save_registry(registry)

        # Model klasörünü de sil (opsiyonel)
        import shutil

        for m in models:
            if m.get("id") == model_id or m.get("model_id") == model_id:
                model_path = m.get("path", "")
                if model_path and os.path.exists(os.path.dirname(model_path)):
                    try:
                        shutil.rmtree(os.path.dirname(model_path))
                    except:
                        pass
                break

        return {"success": True, "message": "Model silindi"}

    except Exception as e:
        import traceback

        traceback.print_exc()
        return {"success": False, "error": str(e)}


@router.post("/{model_id}/deploy")
async def deploy_model(model_id: str):
    """Modeli deploy et"""
    try:
        registry = load_registry()

        # Tüm modelleri undeployed yap, hedef modeli deployed yap
        found = False
        for m in registry.get("models", []):
            if m.get("id") == model_id or m.get("model_id") == model_id:
                m["status"] = "deployed"
                found = True
            else:
                if m.get("status") == "deployed":
                    m["status"] = "trained"

        if not found:
            return {"success": False, "error": "Model bulunamadı"}

        save_registry(registry)

        return {"success": True, "message": "Model deploy edildi"}

    except Exception as e:
        return {"success": False, "error": str(e)}


@router.post("/{model_id}/archive")
async def archive_model(model_id: str):
    """Modeli arşivle"""
    try:
        registry = load_registry()

        for m in registry.get("models", []):
            if m.get("id") == model_id or m.get("model_id") == model_id:
                m["status"] = "archived"
                save_registry(registry)
                return {"success": True, "message": "Model arşivlendi"}

        return {"success": False, "error": "Model bulunamadı"}

    except Exception as e:
        return {"success": False, "error": str(e)}
