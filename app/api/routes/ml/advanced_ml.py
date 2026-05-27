"""
AutoML & Advanced ML API Routes - CyberGuard AI
================================================

AutoML, XAI, A/B Testing, Drift Detection API endpoints.
"""


from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from pydantic import BaseModel, Field

from app.api.routes.auth import require_auth

router = APIRouter(prefix="/ml", tags=["Advanced ML"])


# ============= Request/Response Models =============


class AutoMLSearchRequest(BaseModel):
    max_trials: int = Field(default=10, ge=1, le=50)
    search_strategy: str = Field(default="random", pattern="^(random|grid|bayesian)$")
    model_types: list[str] = Field(default=["lstm", "gru", "cnn_lstm"])
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
    traffic_split: list[float] = Field(default=[0.5, 0.5])


class DriftCheckRequest(BaseModel):
    feature_names: list[str] = Field(default=[])
    threshold: float = Field(default=0.1, ge=0.01, le=1.0)


# ============= AutoML Endpoints =============


@router.get("/automl/status")
async def get_automl_status(user: dict = Depends(require_auth)):
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
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/automl/search")
async def start_automl_search(
    request: AutoMLSearchRequest, background_tasks: BackgroundTasks
):
    """AutoML arama başlat"""
    try:
        from src.ml.automl import AutoMLEngine, ModelType, SearchStrategy

        # Create engine
        strategy = SearchStrategy(request.search_strategy)
        engine = AutoMLEngine(search_strategy=strategy, max_trials=request.max_trials)

        # Start in background
        def run_search():
            try:
                # Load sample data
                import numpy as np

                # Use zeros as placeholder data (real data would come from dataset)
                X = np.zeros((100, 10, 41), dtype=np.float32)
                y = np.zeros(100, dtype=np.int32)

                model_types = [ModelType(mt) for mt in request.model_types]
                engine.search(
                    X, y, model_types=model_types, epochs_per_trial=request.epochs_per_trial
                )
            except Exception:
                import logging
                logging.getLogger(__name__).exception("AutoML search background task failed")

        background_tasks.add_task(run_search)

        return {
            "status": "started",
            "max_trials": request.max_trials,
            "search_strategy": request.search_strategy,
        }
    except Exception:
        raise HTTPException(status_code=500, detail="AutoML search failed")


@router.get("/automl/results")
async def get_automl_results(user: dict = Depends(require_auth)):
    """AutoML sonuçlarını getir"""
    try:
        from src.ml.automl import get_automl_engine

        engine = get_automl_engine()
        return engine.get_summary()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============= XAI Endpoints =============


@router.get("/xai/feature-importance")
async def get_feature_importance(model_name: str = "latest", user: dict = Depends(require_auth)):
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
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/xai/explain")
async def explain_prediction(request: XAIRequest, user: dict = Depends(require_auth)):
    """Prediction açıkla"""
    try:
        from pathlib import Path

        import numpy as np
        from tensorflow import keras

        from src.ml.explainability import get_xai_engine

        engine = get_xai_engine()

        # model_artifacts/ klasöründe .keras modellerini ara
        models_dir = Path(__file__).parent.parent.parent.parent / "model_artifacts"
        model_name = request.model_name or "latest"

        if model_name == "latest":
            model_files = sorted(models_dir.glob("*.keras"), key=lambda p: p.stat().st_mtime, reverse=True)
        else:
            model_files = list(models_dir.glob(f"*{model_name}*.keras"))

        if not model_files:
            raise HTTPException(status_code=404, detail=f"Model bulunamadı: {model_name}")

        model = keras.models.load_model(str(model_files[0]), compile=False)
        model.compile(optimizer="adam", loss="sparse_categorical_crossentropy")
        engine.set_model(model)

        # CICIDS2017 feature adları (41 özellik)
        feature_names = [
            "Destination Port", "Flow Duration", "Total Fwd Packets",
            "Total Backward Packets", "Total Length of Fwd Packets",
            "Total Length of Bwd Packets", "Fwd Packet Length Max",
            "Fwd Packet Length Min", "Fwd Packet Length Mean",
            "Fwd Packet Length Std", "Bwd Packet Length Max",
            "Bwd Packet Length Min", "Bwd Packet Length Mean",
            "Bwd Packet Length Std", "Flow Bytes/s", "Flow Packets/s",
            "Flow IAT Mean", "Flow IAT Std", "Flow IAT Max", "Flow IAT Min",
            "Fwd IAT Total", "Fwd IAT Mean", "Fwd IAT Std", "Fwd IAT Max",
            "Fwd IAT Min", "Bwd IAT Total", "Bwd IAT Mean", "Bwd IAT Std",
            "Bwd IAT Max", "Bwd IAT Min", "Fwd PSH Flags", "Bwd PSH Flags",
            "Fwd URG Flags", "Bwd URG Flags", "Fwd Header Length",
            "Bwd Header Length", "Fwd Packets/s", "Bwd Packets/s",
            "Min Packet Length", "Max Packet Length", "Packet Length Mean",
        ]
        engine.set_feature_names(feature_names)

        # Temsil verisi ile permutation importance
        n_features = model.input_shape[-1] if hasattr(model, "input_shape") else 41
        X = np.random.rand(50, 10, n_features).astype(np.float32)
        y = np.zeros(50, dtype=np.int32)

        result = engine.compute_permutation_importance(X, y, n_repeats=3)
        result["model_file"] = model_files[0].name
        result["method"] = getattr(request, "method", "permutation_importance")
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============= A/B Testing Endpoints =============


@router.get("/ab-testing/tests")
async def list_ab_tests(user: dict = Depends(require_auth)):
    """Tüm A/B testlerini listele"""
    try:
        from src.ml.ab_testing import get_ab_engine

        engine = get_ab_engine()
        return {"tests": engine.list_tests()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/ab-testing/create")
async def create_ab_test(request: ABTestCreateRequest, user: dict = Depends(require_auth)):
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
async def start_ab_test(test_id: str, user: dict = Depends(require_auth)):
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
async def stop_ab_test(test_id: str, user: dict = Depends(require_auth)):
    """A/B test durdur"""
    try:
        from src.ml.ab_testing import get_ab_engine

        engine = get_ab_engine()

        success = engine.stop_test(test_id)
        return {"status": "stopped", "test_id": test_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/ab-testing/{test_id}/results")
async def get_ab_test_results(test_id: str, user: dict = Depends(require_auth)):
    """A/B test sonuçlarını getir"""
    try:
        from src.ml.ab_testing import get_ab_engine

        engine = get_ab_engine()
        return engine.analyze_test(test_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============= Drift Detection Endpoints =============


@router.get("/drift/status")
async def get_drift_status(user: dict = Depends(require_auth)):
    """Drift durumunu getir"""
    try:
        from src.ml.drift_detection import get_drift_detector

        detector = get_drift_detector()
        return detector.get_drift_status()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/drift/visualization")
async def get_drift_visualization(user: dict = Depends(require_auth)):
    """Drift görselleştirme verisi"""
    try:
        from src.ml.drift_detection import get_drift_detector

        detector = get_drift_detector()
        return detector.get_visualization_data()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/drift/check")
async def trigger_drift_check(user: dict = Depends(require_auth)):
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
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/drift/set-reference")
async def set_drift_reference(user: dict = Depends(require_auth)):
    """Reference veri set et"""
    try:
        import numpy as np

        from src.ml.drift_detection import get_drift_detector

        detector = get_drift_detector()

        # Use zeros as placeholder reference data
        reference = np.zeros((1000, 41), dtype=np.float32)
        detector.set_reference_data(reference)

        return {"status": "Reference data set", "shape": reference.shape}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============= Federated Learning Endpoints =============


@router.get("/federated/status")
async def get_fl_status(user: dict = Depends(require_auth)):
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
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/federated/summary")
async def get_fl_summary(user: dict = Depends(require_auth)):
    """Federated Learning özeti"""
    try:
        from src.ml.federated import get_fl_server

        server = get_fl_server()
        return server.get_training_summary()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
