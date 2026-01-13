"""
Meta-Model Selector - CyberGuard AI
=====================================

Traffic context'e göre en uygun modeli seçen meta-learning sistemi.

Özellikler:
    - Traffic feature extraction
    - MLP meta-classifier
    - Dynamic model selection
    - Performance tracking

Flow:
    Traffic → Context Features → Meta-Classifier → Best Model Selection
"""

import os
import sys
import numpy as np
from typing import Dict, List, Optional, Tuple, Callable, Union
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
    from tensorflow.keras import layers, Model

    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

logger = logging.getLogger("MetaClassifier")


class TrafficContextExtractor:
    """
    Traffic verisinden context feature'ları çıkarır

    Context features:
    - Statistical features (mean, std, skew)
    - Temporal features (burst, periodicity)
    - Size features (packet sizes, flow lengths)
    """

    def __init__(self):
        self.feature_names = [
            "mean_packet_size",
            "std_packet_size",
            "max_packet_size",
            "mean_flow_duration",
            "std_flow_duration",
            "packet_rate",
            "byte_rate",
            "unique_ports_ratio",
            "tcp_ratio",
            "udp_ratio",
            "inbound_ratio",
            "outbound_ratio",
            "burst_score",
            "periodicity_score",
            "entropy_score",
            "imbalance_score",
        ]

    def extract(self, X: np.ndarray) -> np.ndarray:
        """
        Batch traffic'ten context features çıkar

        Args:
            X: Traffic data (batch, features) veya (batch, seq, features)

        Returns:
            Context features (batch, context_dim)
        """
        if len(X.shape) == 3:
            X = X.reshape(X.shape[0], -1)  # Flatten sequence

        n_samples = X.shape[0]
        context = np.zeros((n_samples, len(self.feature_names)), dtype=np.float32)

        for i in range(n_samples):
            sample = X[i]

            # Statistical features
            context[i, 0] = np.mean(sample)
            context[i, 1] = np.std(sample)
            context[i, 2] = np.max(sample)

            # Temporal features (simulated)
            context[i, 3] = np.mean(sample[: len(sample) // 2])
            context[i, 4] = np.std(sample[: len(sample) // 2])

            # Rate features
            context[i, 5] = np.sum(np.abs(sample)) / len(sample)
            context[i, 6] = np.sum(sample > 0) / len(sample)

            # Distribution features
            context[i, 7] = len(np.unique(sample)) / len(sample)
            context[i, 8] = np.mean(sample > np.mean(sample))
            context[i, 9] = np.mean(sample < np.mean(sample))

            # Direction features
            context[i, 10] = np.sum(sample > 0) / len(sample)
            context[i, 11] = np.sum(sample < 0) / len(sample)

            # Complexity features
            context[i, 12] = np.max(np.abs(np.diff(sample)))  # Burst
            context[i, 13] = np.std(np.diff(sample))  # Periodicity

            # Information features
            hist, _ = np.histogram(sample, bins=20)
            hist = hist / (np.sum(hist) + 1e-8)
            context[i, 14] = -np.sum(hist * np.log(hist + 1e-8))  # Entropy
            context[i, 15] = np.abs(np.mean(sample) - np.median(sample))  # Imbalance

        return context


class MetaModelSelector:
    """
    Meta-learning based model selector

    Traffic context'e göre en uygun IDS modelini seçer.

    Models:
        - GRU: IoT/Edge traffic (hafif)
        - LSTM: Standard traffic
        - BiLSTM: Complex patterns
        - Transformer: Long sequences
        - Ensemble: High-risk situations
    """

    MODEL_TYPES = ["gru", "lstm", "bilstm", "transformer", "ensemble"]

    # Model selection heuristics
    MODEL_CHARACTERISTICS = {
        "gru": {
            "best_for": ["iot", "edge", "realtime", "simple"],
            "params": "~89K",
            "speed": 5,
            "accuracy": 3,
        },
        "lstm": {
            "best_for": ["standard", "temporal", "moderate"],
            "params": "~125K",
            "speed": 4,
            "accuracy": 4,
        },
        "bilstm": {
            "best_for": ["bidirectional", "complex", "patterns"],
            "params": "~250K",
            "speed": 3,
            "accuracy": 4,
        },
        "transformer": {
            "best_for": ["long_sequence", "attention", "parallel"],
            "params": "~285K",
            "speed": 4,
            "accuracy": 5,
        },
        "ensemble": {
            "best_for": ["critical", "high_risk", "robustness"],
            "params": "~750K",
            "speed": 2,
            "accuracy": 5,
        },
    }

    def __init__(
        self,
        models: Dict[str, keras.Model] = None,
        use_learned_selector: bool = True,
    ):
        """
        Args:
            models: {model_name: keras_model} dict
            use_learned_selector: Learn from performance or use heuristics
        """
        self.models = models or {}
        self.use_learned = use_learned_selector

        self.context_extractor = TrafficContextExtractor()
        self.selector_model: Optional[keras.Model] = None

        # Performance tracking
        self.performance_history = {name: [] for name in self.MODEL_TYPES}

        logger.info(f"🎯 MetaModelSelector initialized")
        logger.info(f"   Models: {list(self.models.keys())}")
        logger.info(f"   Learned selector: {use_learned_selector}")

    def register_model(self, name: str, model: keras.Model):
        """Model kaydet"""
        if name not in self.MODEL_TYPES:
            logger.warning(f"Unknown model type: {name}")
        self.models[name] = model
        logger.info(f"✅ Registered model: {name}")

    def build_selector(self, context_dim: int = 16) -> keras.Model:
        """Meta-classifier modelini oluştur"""
        inputs = layers.Input(shape=(context_dim,), name="context_input")

        x = layers.Dense(64, activation="relu")(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.3)(x)

        x = layers.Dense(32, activation="relu")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.2)(x)

        # Output: probability for each model
        outputs = layers.Dense(
            len(self.MODEL_TYPES), activation="softmax", name="model_probs"
        )(x)

        self.selector_model = keras.Model(inputs, outputs, name="meta_selector")
        self.selector_model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )

        logger.info(
            f"✅ Selector model built! Params: {self.selector_model.count_params():,}"
        )
        return self.selector_model

    def train_selector(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        epochs: int = 50,
        batch_size: int = 32,
    ) -> Dict:
        """
        Meta-classifier'ı eğit

        Args:
            X_train: Traffic samples
            y_train: Best model indices (0-4)

        Returns:
            Training history
        """
        # Extract context
        context = self.context_extractor.extract(X_train)

        if self.selector_model is None:
            self.build_selector(context.shape[1])

        history = self.selector_model.fit(
            context,
            y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.2,
            callbacks=[
                keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)
            ],
            verbose=1,
        )

        return {
            "accuracy": history.history["accuracy"][-1],
            "val_accuracy": history.history.get("val_accuracy", [0])[-1],
        }

    def select_model(
        self,
        X: np.ndarray,
        return_all: bool = False,
    ) -> Union[str, Dict]:
        """
        Traffic için en uygun modeli seç

        Args:
            X: Traffic sample(s)
            return_all: Tüm model skorlarını döndür

        Returns:
            Model name or {model: score} dict
        """
        if self.use_learned and self.selector_model is not None:
            return self._learned_selection(X, return_all)
        else:
            return self._heuristic_selection(X, return_all)

    def _learned_selection(self, X: np.ndarray, return_all: bool) -> Union[str, Dict]:
        """Öğrenilmiş selector ile seçim"""
        context = self.context_extractor.extract(X)
        probs = self.selector_model.predict(context, verbose=0)

        # Average over batch
        avg_probs = np.mean(probs, axis=0)

        if return_all:
            return {
                name: float(prob) for name, prob in zip(self.MODEL_TYPES, avg_probs)
            }

        best_idx = np.argmax(avg_probs)
        return self.MODEL_TYPES[best_idx]

    def _heuristic_selection(self, X: np.ndarray, return_all: bool) -> Union[str, Dict]:
        """Heuristic-based model selection"""
        context = self.context_extractor.extract(X)
        avg_context = np.mean(context, axis=0)

        scores = {}

        # Scoring based on context features
        entropy = avg_context[14]  # Feature 14
        burst = avg_context[12]  # Feature 12
        complexity = avg_context[15]  # Feature 15

        # GRU: Low complexity, fast
        scores["gru"] = 0.5 + 0.3 * (1 - complexity) + 0.2 * (1 - burst)

        # LSTM: Moderate complexity
        scores["lstm"] = 0.5 + 0.3 * complexity * (1 - burst)

        # BiLSTM: High complexity
        scores["bilstm"] = 0.5 + 0.3 * complexity + 0.2 * burst

        # Transformer: High entropy (diverse patterns)
        scores["transformer"] = 0.5 + 0.4 * entropy

        # Ensemble: Very high complexity/risk
        scores["ensemble"] = 0.3 + 0.5 * complexity + 0.2 * burst

        # Normalize
        total = sum(scores.values())
        scores = {k: v / total for k, v in scores.items()}

        if return_all:
            return scores

        return max(scores, key=scores.get)

    def predict_with_selection(self, X: np.ndarray) -> Dict:
        """
        Model seç ve tahmin yap

        Returns:
            {
                "selected_model": str,
                "model_scores": dict,
                "prediction": array,
                "confidence": float
            }
        """
        # Select model
        model_scores = self.select_model(X, return_all=True)
        selected_model = max(model_scores, key=model_scores.get)

        # Check if model exists
        if selected_model not in self.models:
            # Fallback to any available model
            available = [m for m in self.models.keys()]
            if available:
                selected_model = available[0]
            else:
                return {
                    "selected_model": None,
                    "model_scores": model_scores,
                    "prediction": None,
                    "error": "No models registered",
                }

        # Make prediction
        model = self.models[selected_model]

        # Reshape if needed
        if len(model.input_shape) == 3 and len(X.shape) == 2:
            X = X.reshape(X.shape[0], 1, X.shape[1])

        prediction = model.predict(X, verbose=0)
        confidence = float(np.max(prediction))

        return {
            "selected_model": selected_model,
            "model_scores": model_scores,
            "prediction": prediction,
            "predicted_class": int(
                np.argmax(prediction, axis=1)[0]
                if len(prediction.shape) > 1
                else np.argmax(prediction)
            ),
            "confidence": confidence,
        }

    def update_performance(self, model_name: str, accuracy: float, latency: float):
        """Model performansını güncelle (online learning için)"""
        self.performance_history[model_name].append(
            {
                "accuracy": accuracy,
                "latency": latency,
                "timestamp": datetime.now().isoformat(),
            }
        )

        # Keep last 1000 entries
        if len(self.performance_history[model_name]) > 1000:
            self.performance_history[model_name] = self.performance_history[model_name][
                -1000:
            ]

    def get_stats(self) -> Dict:
        """Selector istatistikleri"""
        stats = {
            "registered_models": list(self.models.keys()),
            "selector_trained": self.selector_model is not None,
            "use_learned": self.use_learned,
            "model_characteristics": self.MODEL_CHARACTERISTICS,
        }

        # Performance summary
        for name in self.MODEL_TYPES:
            history = self.performance_history[name]
            if history:
                stats[f"{name}_avg_accuracy"] = np.mean(
                    [h["accuracy"] for h in history]
                )
                stats[f"{name}_avg_latency"] = np.mean([h["latency"] for h in history])

        return stats

    def save(self, path: str):
        """Selector'ı kaydet"""
        os.makedirs(path, exist_ok=True)

        if self.selector_model:
            self.selector_model.save(os.path.join(path, "selector_model.h5"))

        with open(os.path.join(path, "metadata.json"), "w") as f:
            json.dump(
                {
                    "model_types": self.MODEL_TYPES,
                    "registered_models": list(self.models.keys()),
                    "use_learned": self.use_learned,
                    "saved_at": datetime.now().isoformat(),
                },
                f,
                indent=2,
            )

        logger.info(f"✅ Selector saved to {path}")

    @classmethod
    def load(cls, path: str) -> "MetaModelSelector":
        """Selector'ı yükle"""
        with open(os.path.join(path, "metadata.json"), "r") as f:
            metadata = json.load(f)

        selector = cls(use_learned_selector=metadata["use_learned"])

        selector_path = os.path.join(path, "selector_model.h5")
        if os.path.exists(selector_path):
            selector.selector_model = keras.models.load_model(selector_path)

        logger.info(f"✅ Selector loaded from {path}")
        return selector


# ============= Factory Functions =============


def create_meta_selector(
    models: Dict[str, keras.Model] = None,
    use_learned: bool = False,
) -> MetaModelSelector:
    """Create meta selector with optional models"""
    return MetaModelSelector(models=models, use_learned_selector=use_learned)


# ============= Test =============

if __name__ == "__main__":
    print("🧪 Meta-Model Selector Test\n")

    # Simulated traffic
    np.random.seed(42)
    X_simple = np.random.randn(50, 78).astype(np.float32)  # Simple traffic
    X_complex = np.random.randn(50, 78).astype(np.float32) * 3  # Complex traffic

    # Create selector
    selector = MetaModelSelector(use_learned_selector=False)

    # Test context extraction
    context = selector.context_extractor.extract(X_simple)
    print(f"Context shape: {context.shape}")
    print(f"Context features: {selector.context_extractor.feature_names}")

    # Test heuristic selection
    simple_model = selector.select_model(X_simple)
    complex_model = selector.select_model(X_complex)

    print(f"\nSimple traffic → {simple_model}")
    print(f"Complex traffic → {complex_model}")

    # Test all scores
    all_scores = selector.select_model(X_simple, return_all=True)
    print(f"\nAll model scores for simple traffic:")
    for name, score in sorted(all_scores.items(), key=lambda x: x[1], reverse=True):
        print(f"  {name}: {score*100:.1f}%")

    print("\n✅ Test completed!")
