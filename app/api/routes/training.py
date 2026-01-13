"""
Training API Routes - CyberGuard AI
Model eğitimi endpoint'leri

Dosya Yolu: app/api/routes/training.py
"""

from fastapi import APIRouter, BackgroundTasks
from pydantic import BaseModel
from typing import Optional, Dict, List
import sys
import os

# Path düzeltmesi - app artık ana dizinde
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
sys.path.insert(0, project_root)

router = APIRouter()

# Training API singleton
_training_api = None


def get_training_api():
    global _training_api
    if _training_api is None:
        try:
            from src.api.training_api import get_training_api as get_api

            _training_api = get_api()
        except Exception as e:
            print(f"Training API init error: {e}")
            return None
    return _training_api


class TrainingConfig(BaseModel):
    model_name: str = "CyberDefender"
    description: str = "Deep Learning model for threat detection"
    db_path: str = "src/database/cyberguard.db"
    data_limit: int = 100000  # Daha fazla veri
    epochs: int = 150  # Daha uzun eğitim (önceki: 50)
    batch_size: int = 64  # Daha büyük batch (önceki: 32)
    hidden_layers: List[int] = [512, 256, 128, 64]  # Daha derin ağ
    dropout_rate: float = 0.3
    learning_rate: float = 0.0005  # Daha düşük learning rate
    test_size: float = 0.2
    val_size: float = 0.1


@router.post("/start")
async def start_training(config: TrainingConfig, background_tasks: BackgroundTasks):
    """Model eğitimi başlat"""
    try:
        api = get_training_api()
        if api is None:
            return {"success": False, "error": "Training API başlatılamadı"}

        training_config = {
            "db_path": os.path.join(project_root, config.db_path),
            "data_limit": config.data_limit,
            "epochs": config.epochs,
            "batch_size": config.batch_size,
            "hidden_layers": config.hidden_layers,
            "dropout_rate": config.dropout_rate,
            "learning_rate": config.learning_rate,
            "test_size": config.test_size,
            "val_size": config.val_size,
        }

        session_id = api.start_training(
            model_name=config.model_name,
            description=config.description,
            config=training_config,
        )

        return {
            "success": True,
            "session_id": session_id,
            "message": f"Training başlatıldı: {config.model_name}",
        }

    except Exception as e:
        import traceback

        traceback.print_exc()
        return {"success": False, "error": str(e)}


@router.get("/status/{session_id}")
async def get_training_status(session_id: str):
    """Training durumunu al"""
    try:
        api = get_training_api()
        if api is None:
            return {"success": False, "error": "Training API başlatılamadı"}

        status = api.get_session_status(session_id)
        if status:
            return {"success": True, "data": status}
        return {"success": False, "error": "Session bulunamadı"}

    except Exception as e:
        return {"success": False, "error": str(e)}


@router.get("/sessions")
async def list_sessions():
    """Tüm training session'larını listele"""
    try:
        api = get_training_api()
        if api is None:
            return {"success": False, "error": "Training API başlatılamadı", "data": []}

        sessions = api.list_sessions()
        return {"success": True, "data": sessions}

    except Exception as e:
        return {"success": False, "error": str(e), "data": []}


@router.post("/stop/{session_id}")
async def stop_training(session_id: str):
    """Training'i durdur"""
    try:
        api = get_training_api()
        if api is None:
            return {"success": False, "error": "Training API başlatılamadı"}

        api.stop_training(session_id)
        return {"success": True, "message": "Durdurma isteği gönderildi"}

    except Exception as e:
        return {"success": False, "error": str(e)}


@router.delete("/sessions/old")
async def clear_old_sessions(keep_last: int = 10):
    """Eski session'ları temizle"""
    try:
        api = get_training_api()
        if api is None:
            return {"success": False, "error": "Training API başlatılamadı"}

        api.clear_old_sessions(keep_last)
        return {"success": True, "message": f"Son {keep_last} session tutuldu"}

    except Exception as e:
        return {"success": False, "error": str(e)}
