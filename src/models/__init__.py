"""
CyberGuard AI - Models Package
Machine Learning modelleri
"""

# Import edilebilir sınıflar
try:
    from .model_manager import ModelManager
except ImportError:
    ModelManager = None

# CyberAttackModel opsiyonel - dosya mevcut değilse atla
try:
    from .random_forest_model import CyberAttackModel
except ImportError:
    CyberAttackModel = None

__all__ = [
    'CyberAttackModel',  # Virgül eklendi
    'ModelManager',
]