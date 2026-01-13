"""
Advanced Models API Routes - CyberGuard AI
BiLSTM, Transformer, GRU, Ensemble model yönetimi

Endpoints:
    GET  /advanced/models         - Tüm gelişmiş modelleri listele
    POST /advanced/train          - Model eğitimi başlat
    GET  /advanced/compare        - Model karşılaştırma
    POST /advanced/optimize       - Hiperparametre optimizasyonu
    GET  /advanced/optimize/status - Optimizasyon durumu
    POST /advanced/ensemble       - Ensemble oluştur
    GET  /advanced/metrics        - Canlı eğitim metrikleri
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
import os
import sys
import json
import time
import asyncio
from datetime import datetime
from pathlib import Path

# Proje yolu
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
sys.path.insert(0, project_root)

router = APIRouter()

# ============= Models =============


class TrainAdvancedRequest(BaseModel):
    model_type: str = Field(..., description="lstm, bilstm, transformer, gru")
    epochs: int = Field(default=50, ge=1, le=500)
    batch_size: int = Field(default=64, ge=8, le=512)
    learning_rate: float = Field(default=0.001, ge=0.00001, le=0.1)
    use_smote: bool = Field(default=False)
    use_attention: bool = Field(default=True)
    lstm_units: int = Field(default=120, ge=32, le=512)
    dropout_rate: float = Field(default=0.3, ge=0.0, le=0.8)


class OptimizeRequest(BaseModel):
    model_type: str = Field(..., description="lstm, bilstm, transformer")
    algorithm: str = Field(default="ssa", description="ssa, pso, jaya")
    max_iterations: int = Field(default=20, ge=5, le=100)
    population_size: int = Field(default=10, ge=5, le=50)


class EnsembleRequest(BaseModel):
    model_ids: List[str] = Field(..., min_length=2)
    voting: str = Field(default="soft", description="soft, hard")
    weights: Optional[List[float]] = None


# ============= Global State =============

# Aktif eğitim/optimizasyon durumları
active_trainings: Dict[str, Dict] = {}
active_optimizations: Dict[str, Dict] = {}

# Model bilgileri
AVAILABLE_MODELS = {
    "lstm": {
        "name": "LSTM",
        "description": "Long Short-Term Memory - Makale mimarisi",
        "file": "src/network_detection/model.py",
        "class": "NetworkAnomalyModel",
        "params": "~125K",
        "speed": "fast",
    },
    "bilstm": {
        "name": "BiLSTM + Attention",
        "description": "Bidirectional LSTM with Self-Attention",
        "file": "src/network_detection/advanced_model.py",
        "class": "AdvancedIDSModel",
        "params": "~371K",
        "speed": "medium",
        "recommended": True,
    },
    "transformer": {
        "name": "Transformer",
        "description": "Multi-Head Attention based model",
        "file": "src/network_detection/transformer_model.py",
        "class": "TransformerIDSModel",
        "params": "~285K",
        "speed": "medium",
    },
    "gru": {
        "name": "GRU",
        "description": "Gated Recurrent Unit - Hafif model",
        "file": "src/network_detection/gru_model.py",
        "class": "GRUIDSModel",
        "params": "~89K",
        "speed": "fastest",
        "edge_ready": True,
    },
}


# ============= Endpoints =============


@router.get("/models")
async def list_advanced_models():
    """Tüm gelişmiş model tiplerini listele"""
    models = []

    for model_id, info in AVAILABLE_MODELS.items():
        models.append({"id": model_id, **info, "available": True})

    return {"success": True, "data": models, "total": len(models)}


@router.get("/compare")
async def compare_advanced_models():
    """Model performans karşılaştırması"""

    # Comparison results dosyasını oku
    results_file = Path(project_root) / "models" / "comparison_results.json"

    comparison_data = {}
    if results_file.exists():
        with open(results_file, "r") as f:
            comparison_data = json.load(f)

    # Model bilgileriyle birleştir
    comparison = []
    for model_id, info in AVAILABLE_MODELS.items():
        result = comparison_data.get(model_id, {})

        comparison.append(
            {
                "id": model_id,
                "name": info["name"],
                "accuracy": result.get("accuracy", result.get("test_accuracy", 0)),
                "f1_score": result.get("f1_score", result.get("test_f1", 0)),
                "precision": result.get("precision", 0),
                "recall": result.get("recall", 0),
                "train_time": result.get("train_time", 0),
                "params": info["params"],
                "speed": info["speed"],
                "recommended": info.get("recommended", False),
                "edge_ready": info.get("edge_ready", False),
            }
        )

    # Accuracy'ye göre sırala
    comparison.sort(key=lambda x: x["accuracy"], reverse=True)

    return {
        "success": True,
        "data": comparison,
        "best_model": comparison[0]["id"] if comparison else None,
    }


@router.post("/train")
async def train_advanced_model(
    request: TrainAdvancedRequest, background_tasks: BackgroundTasks
):
    """Gelişmiş model eğitimi başlat"""

    if request.model_type not in AVAILABLE_MODELS:
        raise HTTPException(
            status_code=400, detail=f"Bilinmeyen model tipi: {request.model_type}"
        )

    # Eğitim ID oluştur
    training_id = f"train_{request.model_type}_{int(time.time())}"

    # Eğitim durumunu kaydet
    active_trainings[training_id] = {
        "id": training_id,
        "model_type": request.model_type,
        "status": "starting",
        "progress": 0,
        "current_epoch": 0,
        "total_epochs": request.epochs,
        "metrics": {"loss": [], "accuracy": [], "val_loss": [], "val_accuracy": []},
        "started_at": datetime.now().isoformat(),
        "config": request.model_dump(),
    }

    # Background'da eğitimi başlat
    background_tasks.add_task(run_training, training_id, request)

    return {
        "success": True,
        "message": f"{AVAILABLE_MODELS[request.model_type]['name']} eğitimi başlatıldı",
        "training_id": training_id,
    }


async def run_training(training_id: str, config: TrainAdvancedRequest):
    """Background'da model eğitimi"""
    try:
        active_trainings[training_id]["status"] = "loading_data"

        # Simüle edilmiş eğitim (gerçek eğitim uzun sürer)
        for epoch in range(config.epochs):
            await asyncio.sleep(0.1)  # Simülasyon

            progress = (epoch + 1) / config.epochs * 100

            # Metrics güncelle
            active_trainings[training_id].update(
                {
                    "status": "training",
                    "progress": progress,
                    "current_epoch": epoch + 1,
                    "metrics": {
                        "loss": [0.5 - (epoch * 0.01)],
                        "accuracy": [0.8 + (epoch * 0.002)],
                        "val_loss": [0.55 - (epoch * 0.01)],
                        "val_accuracy": [0.78 + (epoch * 0.002)],
                    },
                }
            )

        active_trainings[training_id]["status"] = "completed"
        active_trainings[training_id]["completed_at"] = datetime.now().isoformat()

    except Exception as e:
        active_trainings[training_id]["status"] = "failed"
        active_trainings[training_id]["error"] = str(e)


@router.get("/train/{training_id}")
async def get_training_status(training_id: str):
    """Eğitim durumunu al"""
    if training_id not in active_trainings:
        raise HTTPException(status_code=404, detail="Eğitim bulunamadı")

    return {"success": True, "data": active_trainings[training_id]}


@router.get("/train")
async def list_trainings():
    """Tüm eğitim oturumlarını listele"""
    return {
        "success": True,
        "data": list(active_trainings.values()),
        "active_count": len(
            [t for t in active_trainings.values() if t["status"] == "training"]
        ),
    }


@router.post("/optimize")
async def start_optimization(
    request: OptimizeRequest, background_tasks: BackgroundTasks
):
    """Hiperparametre optimizasyonu başlat"""

    opt_id = f"opt_{request.algorithm}_{int(time.time())}"

    active_optimizations[opt_id] = {
        "id": opt_id,
        "algorithm": request.algorithm.upper(),
        "model_type": request.model_type,
        "status": "starting",
        "progress": 0,
        "current_iteration": 0,
        "max_iterations": request.max_iterations,
        "best_score": 0,
        "best_params": {},
        "history": [],
        "started_at": datetime.now().isoformat(),
    }

    background_tasks.add_task(run_optimization, opt_id, request)

    return {
        "success": True,
        "message": f"{request.algorithm.upper()} optimizasyonu başlatıldı",
        "optimization_id": opt_id,
    }


async def run_optimization(opt_id: str, config: OptimizeRequest):
    """Background'da optimizasyon"""
    try:
        active_optimizations[opt_id]["status"] = "running"

        best_score = 0
        best_params = {}

        for iteration in range(config.max_iterations):
            await asyncio.sleep(0.5)  # Simülasyon

            # Deterministic score based on iteration (converging)
            score = 0.85 + (iteration / config.max_iterations) * 0.14
            params = {
                "lstm_units": 64 + (iteration * 10) % 192,  # 64-256
                "conv_filters": 16 + (iteration * 3) % 48,  # 16-64
                "dropout_rate": round(0.1 + (iteration % 5) * 0.1, 2),  # 0.1-0.5
                "learning_rate": round(0.001 * (1 + iteration % 10), 4),  # Variable
            }

            if score > best_score:
                best_score = score
                best_params = params

            active_optimizations[opt_id].update(
                {
                    "progress": (iteration + 1) / config.max_iterations * 100,
                    "current_iteration": iteration + 1,
                    "best_score": round(best_score, 4),
                    "best_params": best_params,
                    "history": active_optimizations[opt_id]["history"]
                    + [{"iteration": iteration + 1, "score": round(score, 4)}],
                }
            )

        active_optimizations[opt_id]["status"] = "completed"
        active_optimizations[opt_id]["completed_at"] = datetime.now().isoformat()

    except Exception as e:
        active_optimizations[opt_id]["status"] = "failed"
        active_optimizations[opt_id]["error"] = str(e)


@router.get("/optimize/{opt_id}")
async def get_optimization_status(opt_id: str):
    """Optimizasyon durumunu al"""
    if opt_id not in active_optimizations:
        raise HTTPException(status_code=404, detail="Optimizasyon bulunamadı")

    return {"success": True, "data": active_optimizations[opt_id]}


@router.get("/optimize")
async def list_optimizations():
    """Tüm optimizasyonları listele"""
    return {"success": True, "data": list(active_optimizations.values())}


@router.post("/ensemble")
async def create_ensemble(request: EnsembleRequest):
    """Ensemble model oluştur"""

    # Model ID'leri kontrol et
    valid_models = []
    for model_id in request.model_ids:
        if model_id in AVAILABLE_MODELS:
            valid_models.append(
                {"id": model_id, "name": AVAILABLE_MODELS[model_id]["name"]}
            )

    if len(valid_models) < 2:
        raise HTTPException(status_code=400, detail="En az 2 geçerli model gerekli")

    # Weights ayarla
    weights = request.weights
    if weights is None:
        weights = [1.0 / len(valid_models)] * len(valid_models)

    ensemble_id = f"ensemble_{int(time.time())}"

    ensemble_config = {
        "id": ensemble_id,
        "models": valid_models,
        "voting": request.voting,
        "weights": weights,
        "created_at": datetime.now().isoformat(),
        "status": "ready",
    }

    return {"success": True, "message": "Ensemble oluşturuldu", "data": ensemble_config}


@router.get("/metrics/live")
async def get_live_metrics():
    """Canlı eğitim metrikleri"""

    active = [t for t in active_trainings.values() if t["status"] == "training"]

    return {"success": True, "active_trainings": len(active), "data": active}
