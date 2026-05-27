"""
Network Detection Training - CyberGuard AI
==========================================

Model eğitimi ve değerlendirme.
"""

from .evaluator import ModelEvaluator, evaluate_model
from .online_learning import OnlineLearner
from .trainer import Trainer, train_model

__all__ = [
    "train_model",
    "Trainer",
    "ModelEvaluator",
    "evaluate_model",
    "OnlineLearner",
]
