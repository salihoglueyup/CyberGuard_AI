"""
Attack-Specific Training API
Saldırı bazlı model eğitimi için API endpoint'leri

Endpoints:
    POST /api/training/attack-specific - Saldırı bazlı eğitim başlat
    GET /api/training/datasets - Mevcut dataset listesi
    GET /api/training/attack-types - Saldırı tipi listesi
    POST /api/training/realtime-ids/start - Real-time IDS başlat
    GET /api/training/realtime-ids/status - IDS durumu
    GET /api/training/realtime-ids/alerts - Son alert'ler
"""

from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime
import json
import uuid
import asyncio
from pathlib import Path

router = APIRouter(prefix="/training", tags=["attack-specific-training"])

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "raw"
MODELS_DIR = PROJECT_ROOT / "models"

# Training sessions
training_sessions: Dict[str, Dict] = {}

# Real-time IDS instance
realtime_ids_instance = None


# ============= Request/Response Models =============


class AttackSpecificTrainingRequest(BaseModel):
    """Saldırı bazlı eğitim isteği"""

    dataset: str = Field(..., description="Dataset: nsl_kdd, bot_iot, cicids2017")
    attack_types: List[str] = Field(
        default=["all"],
        description="Saldırı tipleri: all, dos, ddos, probe, r2l, u2r, botnet, mirai, gafgyt",
    )
    model_type: str = Field(
        default="ssa_lstmids",
        description="Model tipi: ssa_lstmids, bilstm, transformer, gru",
    )
    epochs: int = Field(default=100, ge=10, le=500)
    batch_size: int = Field(default=120, ge=16, le=512)
    use_smote: bool = Field(default=True, description="SMOTE veri dengeleme")
    use_ssa_bayesian: bool = Field(
        default=False, description="SSA+Bayesian hibrit optimizasyon"
    )
    patience: int = Field(default=20, ge=5, le=50)
    max_samples: int = Field(default=100000, ge=1000)
    model_name: Optional[str] = Field(
        default=None, description="Kaydedilecek model ismi"
    )


class DatasetInfo(BaseModel):
    """Dataset bilgisi"""

    id: str
    name: str
    path: str
    size_mb: float
    samples: int
    features: int
    attack_types: List[str]
    status: str


class AttackTypeInfo(BaseModel):
    """Saldırı tipi bilgisi"""

    id: str
    name: str
    description: str
    datasets: List[str]
    severity: str


class RealTimeIDSRequest(BaseModel):
    """Real-time IDS başlatma isteği"""

    model_path: str = Field(..., description="Model dosya yolu")
    threshold: float = Field(default=0.5, ge=0, le=1)
    window_size: int = Field(default=10, ge=1, le=100)


# ============= Dataset Info =============

DATASETS = {
    "nsl_kdd": {
        "name": "NSL-KDD",
        "path": "nsl_kdd",
        "attack_types": ["normal", "dos", "probe", "r2l", "u2r"],
        "features": 41,
        "description": "Klasik network intrusion detection benchmark",
    },
    "bot_iot": {
        "name": "BoT-IoT",
        "path": "bot_iot",
        "attack_types": ["benign", "mirai", "gafgyt", "ddos", "dos", "reconnaissance"],
        "features": 115,
        "description": "IoT botnet saldırıları",
    },
    "cicids2017": {
        "name": "CICIDS2017",
        "path": "cicids2017",
        "attack_types": [
            "benign",
            "dos",
            "ddos",
            "bruteforce",
            "botnet",
            "portscan",
            "webattack",
            "infiltration",
        ],
        "features": 84,
        "description": "Modern network traffic dataset",
    },
}

ATTACK_TYPES = {
    "all": {
        "name": "Tüm Saldırılar",
        "description": "Tüm saldırı tiplerini dahil et",
        "severity": "mixed",
        "datasets": ["nsl_kdd", "bot_iot", "cicids2017"],
    },
    "dos": {
        "name": "DoS",
        "description": "Denial of Service saldırıları",
        "severity": "high",
        "datasets": ["nsl_kdd", "cicids2017"],
    },
    "ddos": {
        "name": "DDoS",
        "description": "Distributed DoS saldırıları",
        "severity": "critical",
        "datasets": ["bot_iot", "cicids2017"],
    },
    "probe": {
        "name": "Probe/Scan",
        "description": "Port tarama ve keşif",
        "severity": "medium",
        "datasets": ["nsl_kdd"],
    },
    "r2l": {
        "name": "R2L",
        "description": "Remote to Local saldırıları",
        "severity": "high",
        "datasets": ["nsl_kdd"],
    },
    "u2r": {
        "name": "U2R",
        "description": "User to Root privilege escalation",
        "severity": "critical",
        "datasets": ["nsl_kdd"],
    },
    "botnet": {
        "name": "Botnet",
        "description": "Botnet aktiviteleri",
        "severity": "critical",
        "datasets": ["cicids2017"],
    },
    "mirai": {
        "name": "Mirai",
        "description": "Mirai IoT botnet",
        "severity": "critical",
        "datasets": ["bot_iot"],
    },
    "gafgyt": {
        "name": "Gafgyt",
        "description": "Gafgyt IoT botnet",
        "severity": "critical",
        "datasets": ["bot_iot"],
    },
    "bruteforce": {
        "name": "Brute Force",
        "description": "Brute force saldırıları",
        "severity": "medium",
        "datasets": ["cicids2017"],
    },
    "webattack": {
        "name": "Web Attack",
        "description": "Web tabanlı saldırılar",
        "severity": "high",
        "datasets": ["cicids2017"],
    },
}


# ============= Endpoints =============


@router.get("/datasets")
async def get_datasets() -> Dict[str, Any]:
    """Mevcut dataset listesi"""
    result = []

    for dataset_id, info in DATASETS.items():
        path = DATA_DIR / info["path"]

        if path.exists():
            # Boyut hesapla
            size_bytes = sum(f.stat().st_size for f in path.glob("**/*") if f.is_file())
            size_mb = size_bytes / (1024 * 1024)

            # Dosya sayısı
            file_count = (
                len(list(path.glob("*.csv")))
                + len(list(path.glob("*.txt")))
                + len(list(path.glob("*.parquet")))
            )

            status = "ready"
        else:
            size_mb = 0
            file_count = 0
            status = "not_found"

        result.append(
            {
                "id": dataset_id,
                "name": info["name"],
                "path": str(path),
                "size_mb": round(size_mb, 2),
                "files": file_count,
                "features": info["features"],
                "attack_types": info["attack_types"],
                "description": info["description"],
                "status": status,
            }
        )

    return {"success": True, "data": result}


@router.get("/attack-types")
async def get_attack_types() -> Dict[str, Any]:
    """Saldırı tipi listesi"""
    result = []

    for attack_id, info in ATTACK_TYPES.items():
        result.append(
            {
                "id": attack_id,
                "name": info["name"],
                "description": info["description"],
                "severity": info["severity"],
                "datasets": info["datasets"],
            }
        )

    return {"success": True, "data": result}


@router.post("/attack-specific")
async def start_attack_specific_training(
    request: AttackSpecificTrainingRequest, background_tasks: BackgroundTasks
) -> Dict[str, Any]:
    """Saldırı bazlı model eğitimi başlat"""

    # Dataset kontrol
    if request.dataset not in DATASETS:
        raise HTTPException(
            status_code=400, detail=f"Geçersiz dataset: {request.dataset}"
        )

    dataset_path = DATA_DIR / DATASETS[request.dataset]["path"]
    if not dataset_path.exists():
        raise HTTPException(
            status_code=404, detail=f"Dataset bulunamadı: {dataset_path}"
        )

    # Session oluştur
    session_id = str(uuid.uuid4())[:8]

    training_sessions[session_id] = {
        "id": session_id,
        "status": "initializing",
        "progress": 0,
        "dataset": request.dataset,
        "attack_types": request.attack_types,
        "model_type": request.model_type,
        "epochs": request.epochs,
        "use_smote": request.use_smote,
        "use_ssa_bayesian": request.use_ssa_bayesian,
        "started_at": datetime.now().isoformat(),
        "metrics": {},
        "error": None,
    }

    # Background task
    background_tasks.add_task(run_attack_specific_training, session_id, request)

    return {
        "success": True,
        "message": "Eğitim başlatıldı",
        "session_id": session_id,
        "data": training_sessions[session_id],
    }


async def run_attack_specific_training(
    session_id: str, request: AttackSpecificTrainingRequest
):
    """Background eğitim görevi"""
    import sys

    sys.path.insert(0, str(PROJECT_ROOT))

    try:
        training_sessions[session_id]["status"] = "loading_data"
        training_sessions[session_id]["progress"] = 5

        # Dataset yükle
        from scripts.train_ssa_lstmids import load_nsl_kdd, load_bot_iot, prepare_data
        import numpy as np

        dataset_path = DATA_DIR / DATASETS[request.dataset]["path"]

        if request.dataset == "nsl_kdd":
            X, y, class_names = load_nsl_kdd(dataset_path, request.max_samples)
        elif request.dataset == "bot_iot":
            X, y, class_names = load_bot_iot(dataset_path, request.max_samples)
        else:
            raise ValueError(f"Unsupported dataset: {request.dataset}")

        if X is None:
            raise ValueError("Dataset yüklenemedi")

        # Saldırı tipi filtrele
        if "all" not in request.attack_types:
            # Attack type filtering logic here
            pass

        training_sessions[session_id]["progress"] = 15
        training_sessions[session_id]["status"] = "preprocessing"

        # Veri hazırla
        X_train, X_test, y_train, y_test, scaler = prepare_data(X, y)

        # SMOTE
        if request.use_smote:
            training_sessions[session_id]["status"] = "applying_smote"
            try:
                from imblearn.over_sampling import SMOTE

                smote = SMOTE(random_state=42)
                X_train_flat = X_train.reshape(X_train.shape[0], -1)
                X_train_flat, y_train = smote.fit_resample(X_train_flat, y_train)
                X_train = X_train_flat.reshape(-1, X_train.shape[1], X_train.shape[2])
            except:
                pass

        training_sessions[session_id]["progress"] = 25

        # SSA+Bayesian optimizasyon
        if request.use_ssa_bayesian:
            training_sessions[session_id]["status"] = "optimizing_hyperparameters"
            from src.network_detection.optimizers.hybrid_optimizer import (
                HybridOptimizer,
            )

            # Optimizasyon kodu...

        training_sessions[session_id]["status"] = "training"
        training_sessions[session_id]["progress"] = 30

        # Model oluştur ve eğit
        from src.network_detection.ssa_lstmids import SSA_LSTMIDS

        num_classes = len(np.unique(y))
        model = SSA_LSTMIDS(
            input_shape=X_train.shape[1:],
            num_classes=num_classes,
            use_paper_params=True,
        )
        model.build()

        # Custom callback for progress
        class ProgressCallback:
            def __init__(self, session_id, total_epochs):
                self.session_id = session_id
                self.total_epochs = total_epochs

            def on_epoch_end(self, epoch, logs=None):
                progress = 30 + int((epoch / self.total_epochs) * 60)
                training_sessions[self.session_id]["progress"] = progress
                training_sessions[self.session_id]["metrics"]["current_epoch"] = epoch
                if logs:
                    training_sessions[self.session_id]["metrics"]["accuracy"] = (
                        logs.get("accuracy", 0)
                    )
                    training_sessions[self.session_id]["metrics"]["val_accuracy"] = (
                        logs.get("val_accuracy", 0)
                    )

        # Eğit
        results = model.train(
            X_train,
            y_train,
            X_val=X_test,
            y_val=y_test,
            epochs=request.epochs,
            batch_size=request.batch_size,
        )

        training_sessions[session_id]["progress"] = 90
        training_sessions[session_id]["status"] = "evaluating"

        # Değerlendir
        eval_results = model.evaluate(X_test, y_test)

        # Model kaydet
        model_name = request.model_name or f"ssa_lstmids_{request.dataset}_{session_id}"
        save_path = MODELS_DIR / f"{model_name}.h5"
        model.save(str(save_path))

        training_sessions[session_id]["status"] = "completed"
        training_sessions[session_id]["progress"] = 100
        training_sessions[session_id]["metrics"] = {
            **results,
            **eval_results,
            "model_path": str(save_path),
        }
        training_sessions[session_id]["completed_at"] = datetime.now().isoformat()

    except Exception as e:
        training_sessions[session_id]["status"] = "failed"
        training_sessions[session_id]["error"] = str(e)


@router.get("/attack-specific/{session_id}")
async def get_training_status(session_id: str) -> Dict[str, Any]:
    """Eğitim durumunu sorgula"""
    if session_id not in training_sessions:
        raise HTTPException(status_code=404, detail="Session bulunamadı")

    return {"success": True, "data": training_sessions[session_id]}


@router.get("/attack-specific")
async def list_training_sessions() -> Dict[str, Any]:
    """Tüm eğitim session'larını listele"""
    return {"success": True, "data": list(training_sessions.values())}


# ============= Real-time IDS Endpoints =============


@router.post("/realtime-ids/start")
async def start_realtime_ids(request: RealTimeIDSRequest) -> Dict[str, Any]:
    """Real-time IDS başlat"""
    global realtime_ids_instance

    model_path = Path(request.model_path)
    if not model_path.exists():
        # Models klasöründe ara
        model_path = MODELS_DIR / request.model_path
        if not model_path.exists():
            raise HTTPException(
                status_code=404, detail=f"Model bulunamadı: {request.model_path}"
            )

    try:
        from src.network_detection.realtime_ids import RealTimeIDS

        realtime_ids_instance = RealTimeIDS(
            model_path=str(model_path),
            threshold=request.threshold,
            window_size=request.window_size,
            verbose=True,
        )
        realtime_ids_instance.start()

        return {
            "success": True,
            "message": "Real-time IDS başlatıldı",
            "data": {
                "model_path": str(model_path),
                "threshold": request.threshold,
                "window_size": request.window_size,
                "status": "running",
            },
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/realtime-ids/stop")
async def stop_realtime_ids() -> Dict[str, Any]:
    """Real-time IDS durdur"""
    global realtime_ids_instance

    if realtime_ids_instance is None:
        raise HTTPException(status_code=400, detail="IDS çalışmıyor")

    realtime_ids_instance.stop()
    realtime_ids_instance = None

    return {"success": True, "message": "Real-time IDS durduruldu"}


@router.get("/realtime-ids/status")
async def get_realtime_ids_status() -> Dict[str, Any]:
    """Real-time IDS durumu"""
    global realtime_ids_instance

    if realtime_ids_instance is None:
        return {"success": True, "data": {"status": "stopped", "metrics": None}}

    metrics = realtime_ids_instance.get_metrics()
    drift_detected, drift_score = realtime_ids_instance.check_drift()

    return {
        "success": True,
        "data": {
            "status": "running" if realtime_ids_instance.is_running else "stopped",
            "metrics": metrics,
            "drift_detected": drift_detected,
            "drift_score": drift_score,
        },
    }


@router.get("/realtime-ids/alerts")
async def get_realtime_ids_alerts(limit: int = 20) -> Dict[str, Any]:
    """Son alert'leri getir"""
    global realtime_ids_instance

    if realtime_ids_instance is None:
        return {"success": True, "data": []}

    alerts = realtime_ids_instance.get_recent_alerts(limit)
    return {"success": True, "data": alerts}


@router.get("/models")
async def list_trained_models() -> Dict[str, Any]:
    """Eğitilmiş modelleri listele"""
    models = []

    for model_file in MODELS_DIR.glob("*.h5"):
        stat = model_file.stat()

        # Params dosyası var mı?
        params_file = model_file.with_suffix(".h5").with_name(
            model_file.stem + "_params.json"
        )
        params = {}
        if params_file.exists():
            try:
                with open(params_file) as f:
                    params = json.load(f)
            except:
                pass

        models.append(
            {
                "name": model_file.stem,
                "path": str(model_file),
                "size_mb": round(stat.st_size / (1024 * 1024), 2),
                "created_at": datetime.fromtimestamp(stat.st_ctime).isoformat(),
                "params": params,
            }
        )

    return {"success": True, "data": models}
