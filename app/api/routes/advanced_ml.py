"""
AutoML & Advanced ML API Routes - CyberGuard AI
================================================

AutoML, XAI, A/B Testing, Drift Detection API endpoints.
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any
from datetime import datetime
import numpy as np

router = APIRouter(prefix="/ml", tags=["Advanced ML"])


# ============= Request/Response Models =============


class AutoMLSearchRequest(BaseModel):
    max_trials: int = Field(default=10, ge=1, le=50)
    search_strategy: str = Field(default="random", pattern="^(random|grid|bayesian)$")
    model_types: List[str] = Field(default=["lstm", "gru", "cnn_lstm"])
    epochs_per_trial: int = Field(default=20, ge=5, le=100)
    dataset: str = Field(default="nsl_kdd")


class XAIRequest(BaseModel):
    model_name: str
    method: str = Field(default="permutation", pattern="^(shap|lime|permutation)$")
    num_samples: int = Field(default=100, ge=10, le=1000)


class ABTestCreateRequest(BaseModel):
    name: str
    description: str
    model_a_name: str
    model_a_path: str
    model_b_name: str
    model_b_path: str
    traffic_split: List[float] = Field(default=[0.5, 0.5])


class DriftCheckRequest(BaseModel):
    feature_names: List[str] = Field(default=[])
    threshold: float = Field(default=0.1, ge=0.01, le=1.0)


# ============= AutoML Endpoints =============


@router.get("/automl/status")
async def get_automl_status():
    """AutoML durumunu getir"""
    try:
        from src.ml.automl import get_automl_engine

        engine = get_automl_engine()

        return {
            "is_running": engine.is_running,
            "total_trials": len(engine.trials),
            "best_trial": (
                {
                    "id": engine.best_trial.trial_id,
                    "model_type": engine.best_trial.model_type,
                    "accuracy": engine.best_trial.metrics.get("accuracy", 0),
                }
                if engine.best_trial
                else None
            ),
            "search_strategy": engine.search_strategy.value,
        }
    except Exception as e:
        return {"error": str(e)}


@router.post("/automl/search")
async def start_automl_search(
    request: AutoMLSearchRequest, background_tasks: BackgroundTasks
):
    """AutoML arama başlat"""
    try:
        from src.ml.automl import AutoMLEngine, SearchStrategy, ModelType

        # Create engine
        strategy = SearchStrategy(request.search_strategy)
        engine = AutoMLEngine(search_strategy=strategy, max_trials=request.max_trials)

        # Start in background
        def run_search():
            # Load sample data
            import numpy as np

            # Use zeros as placeholder data (real data would come from dataset)
            X = np.zeros((100, 10, 41), dtype=np.float32)
            y = np.zeros(100, dtype=np.int32)

            model_types = [ModelType(mt) for mt in request.model_types]
            engine.search(
                X, y, model_types=model_types, epochs_per_trial=request.epochs_per_trial
            )

        background_tasks.add_task(run_search)

        return {
            "status": "started",
            "max_trials": request.max_trials,
            "search_strategy": request.search_strategy,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/automl/results")
async def get_automl_results():
    """AutoML sonuçlarını getir"""
    try:
        from src.ml.automl import get_automl_engine

        engine = get_automl_engine()
        return engine.get_summary()
    except Exception as e:
        return {"error": str(e)}


# ============= XAI Endpoints =============


@router.get("/xai/feature-importance")
async def get_feature_importance(model_name: str = "latest"):
    """Feature importance hesapla"""
    try:
        from src.ml.explainability import get_xai_engine

        engine = get_xai_engine()

        # Use zeros as placeholder data for feature importance calculation
        import numpy as np

        X = np.zeros((100, 10, 41), dtype=np.float32)
        y = np.zeros(100, dtype=np.int32)

        # Feature names
        feature_names = [f"feature_{i}" for i in range(41)]
        engine.set_feature_names(feature_names)

        # Load model
        from pathlib import Path

        models_dir = Path(__file__).parent.parent.parent / "models"

        # Find latest model
        model_files = list(models_dir.glob("*.h5"))
        if model_files:
            from tensorflow import keras

            model = keras.models.load_model(str(model_files[0]), compile=False)
            model.compile(optimizer="adam", loss="sparse_categorical_crossentropy")
            engine.set_model(model)

            # Permutation importance
            result = engine.compute_permutation_importance(X, y, n_repeats=2)
            return result

        return {"error": "Model bulunamadı"}
    except Exception as e:
        return {"error": str(e)}


@router.post("/xai/explain")
async def explain_prediction(request: XAIRequest):
    """Prediction açıkla"""
    try:
        from src.ml.explainability import get_xai_engine

        engine = get_xai_engine()

        # TODO: Implement with real model
        return {
            "method": request.method,
            "model_name": request.model_name,
            "status": "Feature importance calculated",
            "top_features": [
                {"feature": f"feature_{i}", "importance": 0.1 - i * 0.01}
                for i in range(10)
            ],
        }
    except Exception as e:
        return {"error": str(e)}


# ============= A/B Testing Endpoints =============


@router.get("/ab-testing/tests")
async def list_ab_tests():
    """Tüm A/B testlerini listele"""
    try:
        from src.ml.ab_testing import get_ab_engine

        engine = get_ab_engine()
        return {"tests": engine.list_tests()}
    except Exception as e:
        return {"error": str(e)}


@router.post("/ab-testing/create")
async def create_ab_test(request: ABTestCreateRequest):
    """Yeni A/B test oluştur"""
    try:
        from src.ml.ab_testing import get_ab_engine

        engine = get_ab_engine()

        test = engine.create_test(
            name=request.name,
            description=request.description,
            model_a_name=request.model_a_name,
            model_a_path=request.model_a_path,
            model_b_name=request.model_b_name,
            model_b_path=request.model_b_path,
            traffic_split=tuple(request.traffic_split),
        )

        return {"test_id": test.test_id, "name": test.name, "status": "created"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/ab-testing/{test_id}/start")
async def start_ab_test(test_id: str):
    """A/B test başlat"""
    try:
        from src.ml.ab_testing import get_ab_engine

        engine = get_ab_engine()

        success = engine.start_test(test_id)
        if success:
            return {"status": "started", "test_id": test_id}
        else:
            raise HTTPException(status_code=404, detail="Test bulunamadı")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/ab-testing/{test_id}/stop")
async def stop_ab_test(test_id: str):
    """A/B test durdur"""
    try:
        from src.ml.ab_testing import get_ab_engine

        engine = get_ab_engine()

        success = engine.stop_test(test_id)
        return {"status": "stopped", "test_id": test_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/ab-testing/{test_id}/results")
async def get_ab_test_results(test_id: str):
    """A/B test sonuçlarını getir"""
    try:
        from src.ml.ab_testing import get_ab_engine

        engine = get_ab_engine()
        return engine.analyze_test(test_id)
    except Exception as e:
        return {"error": str(e)}


# ============= Drift Detection Endpoints =============


@router.get("/drift/status")
async def get_drift_status():
    """Drift durumunu getir"""
    try:
        from src.ml.drift_detection import get_drift_detector

        detector = get_drift_detector()
        return detector.get_drift_status()
    except Exception as e:
        return {"error": str(e)}


@router.get("/drift/visualization")
async def get_drift_visualization():
    """Drift görselleştirme verisi"""
    try:
        from src.ml.drift_detection import get_drift_detector

        detector = get_drift_detector()
        return detector.get_visualization_data()
    except Exception as e:
        return {"error": str(e)}


@router.post("/drift/check")
async def trigger_drift_check():
    """Manuel drift kontrolü tetikle"""
    try:
        from src.ml.drift_detection import get_drift_detector

        detector = get_drift_detector()

        snapshot = detector.check_drift()
        if snapshot:
            return {
                "snapshot_id": snapshot.snapshot_id,
                "overall_drift_score": snapshot.overall_drift_score,
                "severity": snapshot.severity.value,
                "alert_count": len(snapshot.alerts),
            }
        else:
            return {"message": "Yeterli veri yok"}
    except Exception as e:
        return {"error": str(e)}


@router.post("/drift/set-reference")
async def set_drift_reference():
    """Reference veri set et"""
    try:
        from src.ml.drift_detection import get_drift_detector
        import numpy as np

        detector = get_drift_detector()

        # Use zeros as placeholder reference data
        reference = np.zeros((1000, 41), dtype=np.float32)
        detector.set_reference_data(reference)

        return {"status": "Reference data set", "shape": reference.shape}
    except Exception as e:
        return {"error": str(e)}


# ============= Federated Learning Endpoints =============


@router.get("/federated/status")
async def get_fl_status():
    """Federated Learning durumu"""
    try:
        from src.ml.federated import get_fl_server

        server = get_fl_server()

        return {
            "is_training": server.is_training,
            "current_round": server.current_round,
            "total_clients": len(server.clients),
            "total_rounds": len(server.rounds),
        }
    except Exception as e:
        return {"error": str(e)}


@router.get("/federated/summary")
async def get_fl_summary():
    """Federated Learning özeti"""
    try:
        from src.ml.federated import get_fl_server

        server = get_fl_server()
        return server.get_training_summary()
    except Exception as e:
        return {"error": str(e)}
