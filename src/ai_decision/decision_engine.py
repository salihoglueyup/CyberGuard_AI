"""
AI Decision Engine - CyberGuard AI
====================================

Tüm AI karar bileşenlerini birleştiren ana orchestrator.

Components:
    - ZeroDayDetector: VAE-based unsupervised detection
    - AttackExplainer: SHAP/Attention explanations
    - MetaModelSelector: Dynamic model selection
    - RLThresholdAgent: Adaptive thresholds
    - LLMReporter: Natural language reports

Flow:
    Traffic → VAE Filter → Model Select → Predict →
    RL Decision → Explain → Report
"""

import os
import sys
import numpy as np
from typing import Dict, List, Optional, Union, Tuple
from datetime import datetime
import logging
import json

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
sys.path.insert(0, PROJECT_ROOT)

try:
    import tensorflow as tf
    from tensorflow import keras

    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

# Local imports
from .zero_day_detector import ZeroDayDetector, HybridIDSPipeline
from .explainer import AttackExplainer
from .meta_classifier import MetaModelSelector
from .rl_threshold import RLThresholdAgent
from .llm_reporter import LLMReporter

logger = logging.getLogger("AIDecisionEngine")


class AIDecisionEngine:
    """
    AI-Driven Intelligent IDS - Main Decision Engine

    Projeyi Model-centric IDS'den AI-driven Intelligent IDS'e dönüştürür.

    Features:
        - Unsupervised zero-day detection (VAE)
        - Explainability (SHAP/Attention)
        - Adaptive decision making (RL)
        - Meta-learning model selection
        - Natural language reporting (LLM)

    Academic Reference:
        "A hybrid unsupervised–supervised intrusion detection framework
        was designed to identify both known and zero-day attacks."
    """

    VERSION = "1.0.0"

    def __init__(
        self,
        input_dim: int = 78,
        models: Dict[str, keras.Model] = None,
        use_rl: bool = True,
        use_meta_selector: bool = True,
        sensitivity: int = 3,
    ):
        """
        Args:
            input_dim: Feature dimension
            models: {model_name: keras_model} dict
            use_rl: Enable RL threshold optimization
            use_meta_selector: Enable meta-learning model selection
            sensitivity: Zero-day detection sensitivity (1-5)
        """
        logger.info("=" * 60)
        logger.info("🧠 Initializing AI Decision Engine")
        logger.info("=" * 60)

        self.input_dim = input_dim
        self.models = models or {}
        self.use_rl = use_rl
        self.use_meta_selector = use_meta_selector

        # Components
        self.zero_day_detector = ZeroDayDetector(
            input_dim=input_dim, sensitivity=sensitivity
        )

        self.explainer = AttackExplainer()

        self.meta_selector = (
            MetaModelSelector(
                models=models, use_learned_selector=False  # Start with heuristics
            )
            if use_meta_selector
            else None
        )

        self.rl_agent = RLThresholdAgent() if use_rl else None

        self.reporter = LLMReporter()

        # State
        self.is_initialized = False
        self.decision_history = []
        self.stats = {
            "total_decisions": 0,
            "zero_day_detections": 0,
            "alerts_generated": 0,
            "ignored": 0,
        }

        logger.info(f"✅ Components initialized:")
        logger.info(f"   └─ ZeroDayDetector: sensitivity={sensitivity}")
        logger.info(f"   └─ AttackExplainer: SHAP/Attention")
        logger.info(f"   └─ MetaSelector: {use_meta_selector}")
        logger.info(f"   └─ RLThreshold: {use_rl}")
        logger.info(f"   └─ LLMReporter: enabled")

    def initialize(
        self,
        X_normal: np.ndarray = None,
        epochs: int = 30,
    ) -> Dict:
        """
        Engine'i initialize et (VAE train, RL pretrain)

        Args:
            X_normal: Normal trafik verileri (VAE için)
            epochs: Training epochs

        Returns:
            Initialization stats
        """
        logger.info("🔧 Initializing AI Decision Engine...")

        results = {}

        # 1. Train VAE
        if X_normal is not None:
            logger.info("📊 Training Zero-Day Detector...")
            self.zero_day_detector.build()
            vae_result = self.zero_day_detector.fit(X_normal, epochs=epochs)
            results["vae_training"] = vae_result

        # 2. Pretrain RL agent
        if self.rl_agent:
            logger.info("🎮 Pretraining RL Agent...")
            rl_result = self.rl_agent.train(episodes=50, steps_per_episode=50)
            results["rl_training"] = rl_result

        self.is_initialized = True
        logger.info("✅ AI Decision Engine initialized!")

        return results

    def decide(
        self,
        X: np.ndarray,
        generate_report: bool = True,
        source_info: str = "Unknown",
        target_info: str = "Unknown",
    ) -> Dict:
        """
        Ana karar fonksiyonu - Full AI pipeline

        Args:
            X: Traffic data (batch, features)
            generate_report: LLM raporu oluştur
            source_info: Kaynak bilgisi
            target_info: Hedef bilgisi

        Returns:
            Comprehensive decision dict
        """
        start_time = datetime.now()

        if len(X.shape) == 1:
            X = X.reshape(1, -1)

        # === STEP 1: Zero-Day Detection (VAE) ===
        if self.zero_day_detector.is_trained:
            zd_result = self.zero_day_detector.detect(X)
            is_zero_day = bool(zd_result["is_zero_day"][0])
            anomaly_score = float(zd_result["anomaly_scores"][0])
        else:
            is_zero_day = False
            anomaly_score = 0.0

        # === STEP 2: Model Selection (Meta-learning) ===
        if self.meta_selector and not is_zero_day:
            selection = self.meta_selector.predict_with_selection(X)
            selected_model = selection.get("selected_model")
            model_confidence = selection.get("confidence", 0.5)
            predicted_class = selection.get("predicted_class", 0)
            model_scores = selection.get("model_scores", {})
        else:
            selected_model = None
            model_confidence = 0.0
            predicted_class = 0
            model_scores = {}

        # === STEP 3: Attack Type Determination ===
        attack_labels = [
            "Normal",
            "DoS",
            "Probe",
            "R2L",
            "U2R",
            "DDoS",
            "PortScan",
            "Bot",
            "Infiltration",
        ]

        if is_zero_day:
            attack_type = "ZERO_DAY"
            confidence = anomaly_score
        else:
            attack_type = attack_labels[min(predicted_class, len(attack_labels) - 1)]
            confidence = model_confidence

        # === STEP 4: RL Threshold Decision ===
        if self.rl_agent:
            rl_decision = self.rl_agent.get_threshold_recommendation(
                model_confidence, anomaly_score
            )
            should_alert = rl_decision["should_alert"]
            rl_action = rl_decision["action"]
        else:
            # Simple threshold
            should_alert = confidence > 0.7 or is_zero_day
            rl_action = "ALERT" if should_alert else "IGNORE"

        # === STEP 5: Explanation (XAI) ===
        explanation = self.explainer.explain_attack(X, attack_type, top_n=5)

        # === STEP 6: Report Generation ===
        if generate_report and should_alert:
            if is_zero_day:
                report = self.reporter.generate_zero_day_report(
                    anomaly_score=anomaly_score,
                    raw_error=float(zd_result.get("raw_errors", [0])[0]),
                    threshold=float(zd_result.get("threshold", 0)),
                    anomalous_features=[
                        f["name"] for f in explanation.get("top_features", [])[:3]
                    ],
                )
            else:
                report = self.reporter.generate_attack_report(
                    attack_type=attack_type,
                    confidence=confidence,
                    explanation=explanation,
                    source_info=source_info,
                    target_info=target_info,
                )
        else:
            report = None

        # === Compile Result ===
        latency_ms = (datetime.now() - start_time).total_seconds() * 1000

        result = {
            # Main decision
            "should_alert": should_alert,
            "action": rl_action,
            "attack_type": attack_type,
            "confidence": confidence,
            # Zero-day
            "is_zero_day": is_zero_day,
            "anomaly_score": anomaly_score,
            # Model selection
            "selected_model": selected_model,
            "model_scores": model_scores,
            # Explanation
            "explanation": explanation,
            "top_features": explanation.get("top_features", [])[:3],
            # Report
            "report": report,
            "quick_alert": (
                self.reporter.generate_quick_alert(attack_type, confidence, target_info)
                if should_alert
                else None
            ),
            # Meta
            "timestamp": datetime.now().isoformat(),
            "latency_ms": latency_ms,
            "engine_version": self.VERSION,
        }

        # Update stats
        self._update_stats(result)

        return result

    def decide_batch(
        self,
        X: np.ndarray,
        generate_reports: bool = False,
    ) -> List[Dict]:
        """Batch decision"""
        results = []
        for i in range(len(X)):
            result = self.decide(
                X[i : i + 1],
                generate_report=generate_reports,
            )
            results.append(result)
        return results

    def _update_stats(self, result: Dict):
        """Update internal stats"""
        self.stats["total_decisions"] += 1

        if result["is_zero_day"]:
            self.stats["zero_day_detections"] += 1

        if result["should_alert"]:
            self.stats["alerts_generated"] += 1
        else:
            self.stats["ignored"] += 1

        # Keep last 1000 decisions
        self.decision_history.append(
            {
                "attack_type": result["attack_type"],
                "confidence": result["confidence"],
                "action": result["action"],
                "timestamp": result["timestamp"],
            }
        )

        if len(self.decision_history) > 1000:
            self.decision_history = self.decision_history[-1000:]

    def register_model(self, name: str, model: keras.Model):
        """Model kaydet"""
        self.models[name] = model
        if self.meta_selector:
            self.meta_selector.register_model(name, model)
        logger.info(f"✅ Model registered: {name}")

    def get_stats(self) -> Dict:
        """Engine istatistikleri"""
        return {
            "version": self.VERSION,
            "is_initialized": self.is_initialized,
            "stats": self.stats,
            "components": {
                "zero_day": {
                    "trained": self.zero_day_detector.is_trained,
                    "threshold": self.zero_day_detector.threshold,
                },
                "meta_selector": {
                    "enabled": self.meta_selector is not None,
                    "models": list(self.models.keys()),
                },
                "rl_agent": {
                    "enabled": self.rl_agent is not None,
                    "total_steps": self.rl_agent.total_steps if self.rl_agent else 0,
                },
                "reporter": {
                    "total_reports": len(self.reporter.report_history),
                },
            },
            "recent_decisions": self.decision_history[-10:],
        }

    def save(self, path: str):
        """Engine'i kaydet"""
        os.makedirs(path, exist_ok=True)

        # Components
        if self.zero_day_detector.is_trained:
            self.zero_day_detector.save(os.path.join(path, "zero_day"))

        if self.meta_selector:
            self.meta_selector.save(os.path.join(path, "meta_selector"))

        if self.rl_agent:
            self.rl_agent.save(os.path.join(path, "rl_agent"))

        # Metadata
        with open(os.path.join(path, "engine_metadata.json"), "w") as f:
            json.dump(
                {
                    "version": self.VERSION,
                    "input_dim": self.input_dim,
                    "use_rl": self.use_rl,
                    "use_meta_selector": self.use_meta_selector,
                    "stats": self.stats,
                    "saved_at": datetime.now().isoformat(),
                },
                f,
                indent=2,
            )

        logger.info(f"✅ Engine saved to {path}")

    @classmethod
    def load(cls, path: str) -> "AIDecisionEngine":
        """Engine'i yükle"""
        with open(os.path.join(path, "engine_metadata.json"), "r") as f:
            metadata = json.load(f)

        engine = cls(
            input_dim=metadata["input_dim"],
            use_rl=metadata["use_rl"],
            use_meta_selector=metadata["use_meta_selector"],
        )

        # Load components
        zd_path = os.path.join(path, "zero_day")
        if os.path.exists(zd_path):
            engine.zero_day_detector = ZeroDayDetector.load(zd_path)

        meta_path = os.path.join(path, "meta_selector")
        if os.path.exists(meta_path) and engine.meta_selector:
            engine.meta_selector = MetaModelSelector.load(meta_path)

        rl_path = os.path.join(path, "rl_agent")
        if os.path.exists(rl_path) and engine.rl_agent:
            engine.rl_agent = RLThresholdAgent.load(rl_path)

        engine.stats = metadata.get("stats", engine.stats)
        engine.is_initialized = True

        logger.info(f"✅ Engine loaded from {path}")
        return engine


# ============= Factory Functions =============


def create_ai_engine(
    input_dim: int = 78,
    sensitivity: int = 3,
    pretrain: bool = False,
) -> AIDecisionEngine:
    """
    AI Decision Engine factory

    Args:
        input_dim: Feature dimension
        sensitivity: Zero-day sensitivity (1-5)
        pretrain: Pretrain components
    """
    engine = AIDecisionEngine(
        input_dim=input_dim,
        sensitivity=sensitivity,
    )

    if pretrain:
        # Generate dummy normal data for pretraining
        X_normal = np.random.randn(500, input_dim).astype(np.float32)
        engine.initialize(X_normal, epochs=10)

    return engine


# ============= Test =============

if __name__ == "__main__":
    print("=" * 60)
    print("🧪 AI DECISION ENGINE TEST")
    print("=" * 60)

    # Create engine
    engine = create_ai_engine(input_dim=78, sensitivity=3, pretrain=True)

    # Test decision
    print("\n📊 Test Decisions:")

    # Normal traffic
    X_normal = np.random.randn(5, 78).astype(np.float32)

    # Anomalous traffic
    X_anomaly = np.random.randn(5, 78).astype(np.float32) * 3 + 5

    print("\n1️⃣ Normal Traffic:")
    for i in range(2):
        result = engine.decide(X_normal[i])
        print(
            f"   Sample {i}: {result['attack_type']} (conf: {result['confidence']:.2f}) → {result['action']}"
        )

    print("\n2️⃣ Anomalous Traffic:")
    for i in range(2):
        result = engine.decide(X_anomaly[i])
        print(
            f"   Sample {i}: {result['attack_type']} (conf: {result['confidence']:.2f}) → {result['action']}"
        )
        if result["quick_alert"]:
            print(f"      {result['quick_alert'].strip()}")

    # Stats
    print("\n📈 Engine Stats:")
    stats = engine.get_stats()
    print(f"   Total decisions: {stats['stats']['total_decisions']}")
    print(f"   Zero-day detections: {stats['stats']['zero_day_detections']}")
    print(f"   Alerts: {stats['stats']['alerts_generated']}")
    print(f"   Ignored: {stats['stats']['ignored']}")

    print("\n✅ Test completed!")
