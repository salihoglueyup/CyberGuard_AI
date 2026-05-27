"""
Prediction API Routes - CyberGuard AI
Gerçek ML Model Entegrasyonu - Batch ve Tekli Tahmin

Dosya Yolu: app/api/routes/prediction.py
"""

import csv
import io
import json
import logging
import os
from datetime import datetime
from typing import Any

import numpy as np
from fastapi import APIRouter, Depends, File, UploadFile
from pydantic import BaseModel

from app.api.routes.auth import require_auth

from app.paths import PROJECT_ROOT

project_root = str(PROJECT_ROOT)

from src.utils.database import DatabaseManager

logger = logging.getLogger(__name__)
router = APIRouter(dependencies=[Depends(require_auth)])
db = DatabaseManager()

# Thresholds
BINARY_THRESHOLD = 0.5
CONFIDENCE_HIGH = 0.8
CONFIDENCE_MEDIUM = 0.5
RISK_HIGH = 0.6
RISK_MEDIUM = 0.3
HIGH_TRAFFIC_BYTES = 10000
DOS_TRAFFIC_BYTES = 5000
RISK_TRAFFIC_WEIGHT = 0.2
RISK_DOS_WEIGHT = 0.3
RISK_FLAG_WEIGHT = 0.3
RISK_IP_WEIGHT = 0.2

# Model paths
models_dir = os.path.join(project_root, "model_artifacts")
registry_path = os.path.join(models_dir, "model_registry.json")

# Model cache
MAX_LOADED_MODELS = 10
_loaded_models = {}


def _cache_model(model_id, entry):
    """Cache a model with LRU eviction"""
    if model_id not in _loaded_models and len(_loaded_models) >= MAX_LOADED_MODELS:
        oldest = next(iter(_loaded_models))
        del _loaded_models[oldest]
    _loaded_models[model_id] = entry


def load_registry():
    """Model registry yükle"""
    if os.path.exists(registry_path):
        with open(registry_path, encoding="utf-8") as f:
            return json.load(f)
    return {"models": []}


def get_deployed_model():
    """Deploy edilmiş modeli getir"""
    registry = load_registry()
    for model in registry.get("models", []):
        if model.get("status") == "deployed":
            return model
    # Deploy edilen yoksa en iyi accuracy'ye sahip olanı döndür
    models = registry.get("models", [])
    if models:
        return max(models, key=lambda x: x.get("metrics", {}).get("accuracy", 0))
    return None


def load_ml_model(model_info: dict):
    """Gerçek ML modelini yükle"""
    if not model_info:
        return None

    model_id = model_info.get("id")

    # Cache kontrolü
    if model_id in _loaded_models:
        return _loaded_models[model_id]

    try:
        framework = model_info.get("framework", "tensorflow")
        model_path = model_info.get("path", "")

        if not os.path.exists(model_path):
            # Model klasöründe ara
            potential_paths = [
                os.path.join(models_dir, model_info.get("name", ""), "model.h5"),
                os.path.join(models_dir, model_info.get("name", ""), "model.keras"),
                os.path.join(models_dir, model_info.get("name", ""), "model.pkl"),
                os.path.join(models_dir, model_info.get("name", ""), "best_model.h5"),
            ]
            for path in potential_paths:
                if os.path.exists(path):
                    model_path = path
                    break

        if not os.path.exists(model_path):
            return None

        # Framework'e göre yükle
        if framework == "tensorflow":
            try:
                import tensorflow as tf

                model = tf.keras.models.load_model(model_path)
                _cache_model(model_id, {
                    "model": model,
                    "framework": "tensorflow",
                    "info": model_info,
                })
                return _loaded_models[model_id]
            except Exception as e:
                print(f"TensorFlow model yüklenemedi: {e}")

        elif framework == "sklearn":
            try:
                import joblib

                model = joblib.load(model_path)
                _cache_model(model_id, {
                    "model": model,
                    "framework": "sklearn",
                    "info": model_info,
                })
                return _loaded_models[model_id]
            except Exception as e:
                print(f"Sklearn model yüklenemedi: {e}")

        elif framework == "pytorch":
            try:
                import torch

                model = torch.load(model_path, weights_only=True)
                model.eval()
                _cache_model(model_id, {
                    "model": model,
                    "framework": "pytorch",
                    "info": model_info,
                })
                return _loaded_models[model_id]
            except Exception as e:
                print(f"PyTorch model yüklenemedi: {e}")

    except Exception as e:
        print(f"Model yüklenirken hata: {e}")

    return None


def predict_with_model(loaded_model: dict, features: np.ndarray) -> dict:
    """Model ile tahmin yap"""
    try:
        framework = loaded_model.get("framework")
        model = loaded_model.get("model")

        if framework == "tensorflow":
            prediction = model.predict(features, verbose=0)
            if len(prediction.shape) > 1 and prediction.shape[1] > 1:
                # Multi-class
                pred_class = int(np.argmax(prediction[0]))
                confidence = float(np.max(prediction[0]))
            else:
                # Binary
                pred_class = int(prediction[0][0] > BINARY_THRESHOLD)
                confidence = (
                    float(prediction[0][0])
                    if pred_class == 1
                    else float(1 - prediction[0][0])
                )

        elif framework == "sklearn":
            pred_class = int(model.predict(features)[0])
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(features)[0]
                confidence = float(max(proba))
            else:
                confidence = CONFIDENCE_HIGH

        elif framework == "pytorch":
            import torch

            with torch.no_grad():
                features_tensor = torch.FloatTensor(features)
                output = model(features_tensor)
                pred_class = int(torch.argmax(output).item())
                confidence = float(torch.softmax(output, dim=1).max().item())

        else:
            return {"prediction": "unknown", "probability": BINARY_THRESHOLD}

        # Sonuç yorumlama
        attack_types = ["normal", "DoS", "Probe", "R2L", "U2R", "Malware", "DDoS"]

        if pred_class < len(attack_types):
            prediction_label = attack_types[pred_class]
        else:
            prediction_label = "attack" if pred_class > 0 else "normal"

        return {
            "prediction": prediction_label,
            "probability": round(confidence, 4),
            "is_attack": pred_class > 0,
            "attack_type": prediction_label if pred_class > 0 else None,
            "severity": (
                "high" if confidence > CONFIDENCE_HIGH else "medium" if confidence > CONFIDENCE_MEDIUM else "low"
            ),
            "confidence": round(confidence, 4),
        }

    except Exception as e:
        return {"prediction": "error", "probability": 0.0, "error": str(e)}


def prepare_features(data: dict) -> np.ndarray:
    """Veriyi model için hazırla"""
    # Standart özellik sırası (NSL-KDD benzeri)
    feature_keys = [
        "duration",
        "protocol_type",
        "service",
        "flag",
        "src_bytes",
        "dst_bytes",
        "land",
        "wrong_fragment",
        "urgent",
        "hot",
        "num_failed_logins",
        "logged_in",
        "num_compromised",
        "root_shell",
        "su_attempted",
        "num_root",
        "num_file_creations",
        "num_shells",
        "num_access_files",
        "num_outbound_cmds",
        "is_host_login",
        "is_guest_login",
        "count",
        "srv_count",
        "serror_rate",
        "srv_serror_rate",
        "rerror_rate",
        "srv_rerror_rate",
        "same_srv_rate",
        "diff_srv_rate",
        "srv_diff_host_rate",
        "dst_host_count",
        "dst_host_srv_count",
        "dst_host_same_srv_rate",
        "dst_host_diff_srv_rate",
        "dst_host_same_src_port_rate",
        "dst_host_srv_diff_host_rate",
        "dst_host_serror_rate",
        "dst_host_srv_serror_rate",
        "dst_host_rerror_rate",
        "dst_host_srv_rerror_rate",
    ]

    features = []
    for key in feature_keys:
        value = data.get(key, 0)
        if isinstance(value, str):
            # Kategorik değer - basit encoding
            value = hash(value) % 1000 / 1000.0
        features.append(float(value) if value else 0.0)

    return np.array([features])


def fallback_prediction(data: dict) -> dict:
    """Model yoksa basit heuristic tahmin"""
    # Basit kurallar
    risk_score = 0.0

    src_bytes = float(data.get("src_bytes", data.get("packet_size", 0)))
    dst_bytes = float(data.get("dst_bytes", 0))
    duration = float(data.get("duration", 0))

    # Yüksek trafik
    if src_bytes > HIGH_TRAFFIC_BYTES:
        risk_score += RISK_TRAFFIC_WEIGHT
    if dst_bytes > HIGH_TRAFFIC_BYTES:
        risk_score += RISK_TRAFFIC_WEIGHT

    # Kısa süre, yüksek trafik (potansiyel DoS)
    if duration < 1 and src_bytes > DOS_TRAFFIC_BYTES:
        risk_score += RISK_DOS_WEIGHT

    # Port tarama belirtileri
    if data.get("flag") in ["REJ", "S0", "RSTR"]:
        risk_score += RISK_FLAG_WEIGHT

    # IP adresi kontrolü
    source_ip = data.get("source_ip", "")
    if source_ip.startswith(("185.", "194.", "45.", "91.")):
        risk_score += RISK_IP_WEIGHT

    risk_score = min(1.0, risk_score)

    if risk_score > RISK_HIGH:
        return {
            "prediction": "attack",
            "probability": round(risk_score, 4),
            "is_attack": True,
            "attack_type": "Unknown",
            "severity": "high",
            "confidence": round(risk_score, 4),
            "source": "heuristic",
        }
    elif risk_score > RISK_MEDIUM:
        return {
            "prediction": "suspicious",
            "probability": round(risk_score, 4),
            "is_attack": False,
            "attack_type": None,
            "severity": "medium",
            "confidence": round(risk_score, 4),
            "source": "heuristic",
        }
    else:
        return {
            "prediction": "normal",
            "probability": round(1 - risk_score, 4),
            "is_attack": False,
            "attack_type": None,
            "severity": "low",
            "confidence": round(1 - risk_score, 4),
            "source": "heuristic",
        }


class PredictionInput(BaseModel):
    source_ip: str
    destination_ip: str | None = "10.0.0.1"
    source_port: int | None = 0
    destination_port: int | None = 80
    protocol: str | None = "TCP"
    packet_size: int | None = 1024
    duration: float | None = 0
    src_bytes: int | None = 0
    dst_bytes: int | None = 0
    model_id: str | None = None


class BatchPredictionInput(BaseModel):
    data: list[dict[str, Any]]
    model_id: str | None = None


@router.post("/single")
async def predict_single(input_data: PredictionInput):
    """Tekli tahmin - Gerçek model veya heuristic"""
    try:
        data = input_data.model_dump()

        # Model yükle
        model_info = get_deployed_model()
        loaded_model = load_ml_model(model_info) if model_info else None

        if loaded_model:
            # Gerçek model tahmini
            features = prepare_features(data)
            prediction = predict_with_model(loaded_model, features)
            prediction["source"] = "ml_model"
            prediction["model_name"] = model_info.get("name", "Unknown")
        else:
            # Fallback
            prediction = fallback_prediction(data)

        result = {
            "id": int(datetime.now().timestamp()),
            "timestamp": datetime.now().isoformat(),
            "input": data,
            "prediction": prediction,
        }

        return {"success": True, "data": result}
    except Exception as e:
        return {"success": False, "error": str(e)}


@router.post("/batch")
async def predict_batch(input_data: BatchPredictionInput):
    """Toplu tahmin"""
    try:
        # Model yükle
        model_info = get_deployed_model()
        loaded_model = load_ml_model(model_info) if model_info else None

        results = []
        for item in input_data.data[:1000]:  # Max 1000 kayıt
            if loaded_model:
                features = prepare_features(item)
                prediction = predict_with_model(loaded_model, features)
                prediction["source"] = "ml_model"
            else:
                prediction = fallback_prediction(item)

            results.append({"input": item, "prediction": prediction})

        # Summary
        summary = {
            "total": len(results),
            "normal": len(
                [r for r in results if r["prediction"]["prediction"] == "normal"]
            ),
            "suspicious": len(
                [r for r in results if r["prediction"]["prediction"] == "suspicious"]
            ),
            "attack": len(
                [r for r in results if r["prediction"]["prediction"] == "attack"]
            ),
        }

        batch_result = {
            "id": int(datetime.now().timestamp()),
            "timestamp": datetime.now().isoformat(),
            "model_name": model_info.get("name") if model_info else "heuristic",
            "total_items": len(results),
            "predictions": results,
            "summary": summary,
        }

        return {"success": True, "data": batch_result}
    except Exception as e:
        return {"success": False, "error": str(e)}


@router.post("/batch-file")
async def predict_batch_from_file(file: UploadFile = File(...)):
    """CSV/JSON dosyasından toplu tahmin"""
    try:
        content = await file.read()
        filename = file.filename.lower()

        data = []

        if filename.endswith(".json"):
            data = json.loads(content.decode("utf-8"))
            if isinstance(data, dict):
                data = data.get("data", [data])

        elif filename.endswith(".csv"):
            csv_reader = csv.DictReader(io.StringIO(content.decode("utf-8")))
            data = list(csv_reader)

        else:
            return {
                "success": False,
                "error": "Desteklenmeyen format. CSV veya JSON kullanın.",
            }

        if not data:
            return {"success": False, "error": "Dosyada veri bulunamadı"}

        # Batch tahmin yap
        batch_input = BatchPredictionInput(data=data[:1000])
        return await predict_batch(batch_input)

    except Exception as e:
        return {"success": False, "error": str(e)}


@router.get("/models")
async def get_available_models():
    """Kullanılabilir modelleri listele"""
    try:
        registry = load_registry()
        models = []

        for model in registry.get("models", []):
            models.append(
                {
                    "id": model.get("id"),
                    "name": model.get("name"),
                    "framework": model.get("framework"),
                    "status": model.get("status"),
                    "accuracy": model.get("metrics", {}).get("accuracy", 0),
                    "created_at": model.get("created_at"),
                }
            )

        return {"success": True, "data": models, "deployed": get_deployed_model()}
    except Exception as e:
        return {"success": False, "error": str(e)}


@router.get("/stats")
async def get_prediction_stats():
    """Tahmin istatistikleri"""
    try:
        model_info = get_deployed_model()

        return {
            "success": True,
            "data": {
                "active_model": model_info.get("name") if model_info else None,
                "model_accuracy": (
                    model_info.get("metrics", {}).get("accuracy", 0)
                    if model_info
                    else 0
                ),
                "total_models": len(load_registry().get("models", [])),
                "ml_available": model_info is not None,
            },
        }
    except Exception as e:
        return {"success": False, "error": str(e)}
