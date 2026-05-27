"""
Network Detection Models - CyberGuard AI
=========================================

Tüm IDS modelleri.
"""

# Base model
# LSTM variants
from .advanced_model import (
    AdvancedIDSModel,
    SelfAttention,
    build_bilstm_attention,
    build_lstm_model,
    create_bilstm_attention_model,
)
from .base import NetworkAnomalyModel

# GRU
from .gru_model import GRUIDSModel

# Paper model (conditional import)
try:
    from .ssa_lstmids import SSA_LSTMIDS, optimize_with_ssa

    SSALSTMIDS = SSA_LSTMIDS  # Alias for backward compatibility
    build_ssa_lstmids = None
except ImportError:
    SSA_LSTMIDS = None
    SSALSTMIDS = None
    build_ssa_lstmids = None
    optimize_with_ssa = None

# Attention models (conditional import)
try:
    from .attention import (
        AttentionLSTM,
        BiLSTM_Attention,
        CNN_BiLSTM_Attention,
    )
except ImportError:
    AttentionLSTM = None
    BiLSTM_Attention = None
    CNN_BiLSTM_Attention = None

# Transformer models (conditional import)
try:
    from .transformer_ids import (
        CNNTransformerIDS,
        InformerIDS,
        TransformerEncoderBlock,
        TransformerIDS,
    )
except ImportError:
    TransformerEncoderBlock = None
    TransformerIDS = None
    CNNTransformerIDS = None
    InformerIDS = None

__all__ = [
    # Base
    "NetworkAnomalyModel",
    # LSTM
    "build_lstm_model",
    "build_bilstm_attention",
    "AdvancedIDSModel",
    "create_bilstm_attention_model",
    # GRU
    "GRUIDSModel",
    # Paper
    "SSA_LSTMIDS",
    "SSALSTMIDS",
    "build_ssa_lstmids",
    "optimize_with_ssa",
    # Attention
    "AttentionLSTM",
    "BiLSTM_Attention",
    "CNN_BiLSTM_Attention",
    "SelfAttention",
    # Transformer
    "TransformerEncoderBlock",
    "TransformerIDS",
    "CNNTransformerIDS",
    "InformerIDS",
]
