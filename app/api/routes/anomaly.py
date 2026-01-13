"""
Anomaly Detection API - REAL DATA VERSION
ML-based anomaly detection with real model integration
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Optional
from datetime import datetime
import os
import json
import glob
import numpy as np

try:
    import tensorflow as tf

    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

router = APIRouter()

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
models_dir = os.path.join(project_root, "models")
data_dir = os.path.join(project_root, "data")
anomalies_file = os.path.join(data_dir, "anomalies.json")

ANOMALIES = []
MODEL_CACHE = {}


class AnomalyDetectRequest(BaseModel):
    features: List[float]
    threshold: float = 0.7


def load_anomalies():
    global ANOMALIES
    if os.path.exists(anomalies_file):
        try:
            with open(anomalies_file, "r") as f:
                ANOMALIES = json.load(f)
        except:
            pass


def save_anomalies():
    os.makedirs(os.path.dirname(anomalies_file), exist_ok=True)
    with open(anomalies_file, "w") as f:
        json.dump(ANOMALIES[-500:], f, indent=2, default=str)


def load_model():
    if "anomaly" in MODEL_CACHE:
        return MODEL_CACHE["anomaly"]
    if not TF_AVAILABLE:
        return None

    for pattern in ["best_cicids*.keras", "deep_ssa*.keras"]:
        for path in glob.glob(os.path.join(models_dir, pattern)):
            try:
                model = tf.keras.models.load_model(path)
                MODEL_CACHE["anomaly"] = model
                return model
            except:
                pass
    return None


def statistical_anomaly_score(features: List[float]) -> float:
    arr = np.array(features)
    z_scores = np.abs((arr - np.mean(arr)) / (np.std(arr) + 1e-10))
    return min(1.0, float(np.mean(z_scores[z_scores > 2])) / 3)


load_anomalies()


@router.get("/status")
async def get_status():
    model = load_model()
    return {
        "success": True,
        "data": {
            "status": "active",
            "model_loaded": model is not None,
            "tensorflow_available": TF_AVAILABLE,
            "anomalies_detected": len(ANOMALIES),
        },
    }


@router.post("/detect")
async def detect_anomaly(request: AnomalyDetectRequest):
    model = load_model()

    if model is not None:
        try:
            x = np.array(request.features).reshape(1, -1)
            pred = model.predict(x, verbose=0)
            # Assume benign is class 0
            score = 1 - float(pred[0][0]) if pred.shape[-1] > 1 else float(pred[0][0])
            method = "neural_network"
        except:
            score = statistical_anomaly_score(request.features)
            method = "statistical"
    else:
        score = statistical_anomaly_score(request.features)
        method = "statistical"

    is_anomaly = score >= request.threshold

    result = {
        "score": score,
        "is_anomaly": is_anomaly,
        "threshold": request.threshold,
        "method": method,
        "timestamp": datetime.now().isoformat(),
    }

    if is_anomaly:
        anomaly_record = {
            "id": f"ANOM-{datetime.now().strftime('%Y%m%d%H%M%S')}",
            **result,
            "features_hash": hash(tuple(request.features[:10])),
        }
        ANOMALIES.append(anomaly_record)
        save_anomalies()
        result["anomaly_id"] = anomaly_record["id"]

    return {"success": True, "data": result}


@router.get("/anomalies")
async def get_anomalies(limit: int = 100):
    return {
        "success": True,
        "data": {"anomalies": ANOMALIES[-limit:][::-1], "total": len(ANOMALIES)},
    }


@router.get("/anomaly/{anomaly_id}")
async def get_anomaly(anomaly_id: str):
    for a in ANOMALIES:
        if a.get("id") == anomaly_id:
            return {"success": True, "data": a}
    raise HTTPException(status_code=404)


@router.get("/stats")
async def get_stats():
    scores = [a.get("score", 0) for a in ANOMALIES]
    return {
        "success": True,
        "data": {
            "total": len(ANOMALIES),
            "avg_score": float(np.mean(scores)) if scores else 0,
            "max_score": float(np.max(scores)) if scores else 0,
            "model_loaded": load_model() is not None,
        },
    }
