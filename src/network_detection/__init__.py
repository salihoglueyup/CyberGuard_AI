"""
Network Detection Module - CyberGuard AI
=========================================

Ağ saldırı tespiti için ML modülleri.

Yapı:
    - models/     : IDS modelleri (LSTM, Attention, Transformer)
    - data/       : Veri işleme, augmentation, feature selection
    - training/   : Model eğitimi ve değerlendirme
    - inference/  : Real-time IDS
    - optimizers/ : Meta-heuristic optimizasyon
    - preprocessing/ : Veri ön işleme
"""

# Models
# Data
from .data import (
    balance_dataset,
    select_features,
)

# Inference
from .inference import (
    RealtimeIDS,
    get_ids,
)
from .models import (
    build_attention_lstm,
    build_bilstm_attention,
    build_cnn_bilstm_attention,
    build_ssa_lstmids,
    build_transformer_ids,
)

# Training
from .training import (
    evaluate_model,
    train_model,
)

__all__ = [
    # Models
    "build_ssa_lstmids",
    "build_attention_lstm",
    "build_bilstm_attention",
    "build_cnn_bilstm_attention",
    "build_transformer_ids",
    # Data
    "balance_dataset",
    "select_features",
    # Training
    "train_model",
    "evaluate_model",
    # Inference
    "RealtimeIDS",
    "get_ids",
]
