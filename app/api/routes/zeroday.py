"""
Zero-Day Detection API - REAL DATA VERSION
ML-based zero-day detection with model integration
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Optional
from datetime import datetime, timedelta
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
detections_file = os.path.join(data_dir, "zeroday_detections.json")

DETECTIONS = []
MODEL_CACHE = {}


class AnalysisRequest(BaseModel):
    features: List[float]
    source: str = "network"
    context: Optional[Dict] = None


def load_detections():
    global DETECTIONS
    if os.path.exists(detections_file):
        try:
            with open(detections_file, "r") as f:
                DETECTIONS = json.load(f)
        except:
            pass


def save_detections():
    os.makedirs(os.path.dirname(detections_file), exist_ok=True)
    with open(detections_file, "w") as f:
        json.dump(DETECTIONS[-500:], f, indent=2, default=str)


def load_model():
    if "anomaly" in MODEL_CACHE:
        return MODEL_CACHE["anomaly"]
    if not TF_AVAILABLE:
        return None
    for pattern in ["best_cicids*.keras", "deep_ssa*.keras"]:
        models = glob.glob(os.path.join(models_dir, pattern))
        if models:
            try:
                model = tf.keras.models.load_model(models[0])
                MODEL_CACHE["anomaly"] = model
                return model
            except:
                pass
    return None


def calc_anomaly_score(features, model):
    if model is None:
        arr = np.array(features)
        z = np.abs((arr - np.mean(arr)) / (np.std(arr) + 1e-10))
        return {
            "score": min(1.0, float(np.mean(z[z > 2])) / 5),
            "method": "statistical",
        }
    try:
        x = np.array(features).reshape(1, -1)
        pred = model.predict(x, verbose=0)
        score = 1 - pred[0][0] if pred.shape[-1] > 1 else float(pred[0][0])
        return {"score": float(score), "method": "neural_network"}
    except:
        return {"score": 0.5, "method": "error"}


load_detections()


@router.get("/status")
async def get_status():
    model = load_model()
    return {
        "success": True,
        "data": {
            "status": "active",
            "model_loaded": model is not None,
            "detections": len(DETECTIONS),
        },
    }


@router.post("/analyze")
async def analyze(request: AnalysisRequest):
    model = load_model()
    result = calc_anomaly_score(request.features, model)
    threat = (
        "critical"
        if result["score"] > 0.9
        else (
            "high"
            if result["score"] > 0.7
            else "medium" if result["score"] > 0.5 else "low"
        )
    )

    if result["score"] > 0.5:
        det = {
            "id": f"ZD-{datetime.now().strftime('%Y%m%d%H%M%S')}",
            "timestamp": datetime.now().isoformat(),
            "score": result["score"],
            "threat_level": threat,
            "resolved": False,
        }
        DETECTIONS.append(det)
        save_detections()

    return {"success": True, "data": {"anomaly_score": result, "threat_level": threat}}


@router.get("/detections")
async def get_detections(limit: int = 50):
    return {
        "success": True,
        "data": {"detections": DETECTIONS[-limit:], "total": len(DETECTIONS)},
    }


@router.put("/detection/{det_id}/resolve")
async def resolve(det_id: str):
    for d in DETECTIONS:
        if d.get("id") == det_id:
            d["resolved"] = True
            save_detections()
            return {"success": True}
    raise HTTPException(status_code=404)


@router.get("/stats")
async def stats():
    by_level = {"critical": 0, "high": 0, "medium": 0, "low": 0}
    for d in DETECTIONS:
        tl = d.get("threat_level", "low")
        if tl in by_level:
            by_level[tl] += 1
    return {
        "success": True,
        "data": {"total": len(DETECTIONS), "by_threat_level": by_level},
    }
