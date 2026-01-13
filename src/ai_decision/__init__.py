"""
AI Decision Layer - CyberGuard AI
==================================

Intelligent IDS için AI karar katmanı.

Bileşenler:
    - Zero-Day Detector (VAE + IDS hybrid)
    - Explainer (SHAP/Attention)
    - Meta-Classifier (Model selector)
    - RL Threshold (Adaptive thresholds)
    - LLM Reporter (Natural language reports)
"""


# Lazy imports to avoid circular dependencies
def get_zero_day_detector():
    from .zero_day_detector import ZeroDayDetector

    return ZeroDayDetector


def get_hybrid_pipeline():
    from .zero_day_detector import HybridIDSPipeline

    return HybridIDSPipeline


def get_attack_explainer():
    from .explainer import AttackExplainer

    return AttackExplainer


def get_meta_selector():
    from .meta_classifier import MetaModelSelector

    return MetaModelSelector


def get_rl_agent():
    from .rl_threshold import RLThresholdAgent

    return RLThresholdAgent


def get_llm_reporter():
    from .llm_reporter import LLMReporter

    return LLMReporter


def get_decision_engine():
    from .decision_engine import AIDecisionEngine

    return AIDecisionEngine


# Direct imports for convenience (will work after all modules are loaded)
try:
    from .zero_day_detector import ZeroDayDetector, HybridIDSPipeline
    from .explainer import AttackExplainer
    from .meta_classifier import MetaModelSelector
    from .rl_threshold import RLThresholdAgent
    from .llm_reporter import LLMReporter
    from .decision_engine import AIDecisionEngine
except ImportError as e:
    import logging

    logging.warning(f"AI Decision Layer partial import: {e}")

    # Set placeholders
    ZeroDayDetector = None
    HybridIDSPipeline = None
    AttackExplainer = None
    MetaModelSelector = None
    RLThresholdAgent = None
    LLMReporter = None
    AIDecisionEngine = None


__all__ = [
    "ZeroDayDetector",
    "HybridIDSPipeline",
    "AttackExplainer",
    "MetaModelSelector",
    "RLThresholdAgent",
    "LLMReporter",
    "AIDecisionEngine",
    # Factory functions
    "get_zero_day_detector",
    "get_hybrid_pipeline",
    "get_attack_explainer",
    "get_meta_selector",
    "get_rl_agent",
    "get_llm_reporter",
    "get_decision_engine",
]
