"""
Explainable AI (XAI) Module - CyberGuard AI
============================================

SHAP ve LIME ile model açıklanabilirliği.

Özellikler:
    - SHAP values hesaplama
    - LIME explanations
    - Feature importance
    - Per-prediction açıklamalar
"""

import os
import sys
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import logging

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logger = logging.getLogger("XAI")


class ExplainabilityEngine:
    """
    Explainable AI Engine - SHAP ve LIME entegrasyonu
    """

    def __init__(self, model=None, feature_names: List[str] = None):
        self.model = model
        self.feature_names = feature_names or []
        self.shap_values = None
        self.lime_explainer = None

    def set_model(self, model):
        """Model set et"""
        self.model = model

    def set_feature_names(self, names: List[str]):
        """Feature isimlerini set et"""
        self.feature_names = names

    # ============= SHAP =============

    def compute_shap_values(
        self,
        X: np.ndarray,
        background_samples: int = 100,
        method: str = "deep",  # deep, kernel, tree
    ) -> Dict:
        """
        SHAP değerlerini hesapla

        Args:
            X: Açıklanacak veriler
            background_samples: Background veri sayısı
            method: SHAP method (deep, kernel, tree)
        """
        try:
            import shap
        except ImportError:
            return {"error": "SHAP paketi yüklü değil. pip install shap"}

        if self.model is None:
            return {"error": "Model set edilmedi"}

        print(f"🔍 SHAP değerleri hesaplanıyor (method={method})...")

        try:
            # Background data
            if len(X) > background_samples:
                bg_indices = np.random.choice(len(X), background_samples, replace=False)
                background = X[bg_indices]
            else:
                background = X

            if method == "deep":
                # Deep SHAP for neural networks
                explainer = shap.DeepExplainer(self.model, background)
                shap_values = explainer.shap_values(X[:100])  # İlk 100 sample
            elif method == "kernel":
                # Kernel SHAP (model-agnostic)
                def predict_fn(x):
                    return self.model.predict(x, verbose=0)

                explainer = shap.KernelExplainer(predict_fn, background[:50])
                shap_values = explainer.shap_values(X[:20])  # Daha az sample (yavaş)
            else:
                return {"error": f"Bilinmeyen method: {method}"}

            self.shap_values = shap_values

            # Feature importance hesapla
            if isinstance(shap_values, list):
                # Multi-class
                importance = np.mean(
                    [np.abs(sv).mean(axis=0) for sv in shap_values], axis=0
                )
            else:
                importance = np.abs(shap_values).mean(axis=0)

            # Flatten if needed
            if len(importance.shape) > 1:
                importance = importance.mean(axis=0)

            # Feature importance dict
            feature_importance = {}
            for i, imp in enumerate(
                importance[: len(self.feature_names)]
                if self.feature_names
                else importance
            ):
                name = (
                    self.feature_names[i]
                    if i < len(self.feature_names)
                    else f"feature_{i}"
                )
                feature_importance[name] = float(imp)

            # Sort by importance
            sorted_importance = dict(
                sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
            )

            print(f"✅ SHAP hesaplandı - {len(sorted_importance)} feature")

            return {
                "method": "shap",
                "feature_importance": sorted_importance,
                "top_features": list(sorted_importance.keys())[:10],
                "samples_analyzed": len(X[:100]),
            }

        except Exception as e:
            logger.error(f"SHAP error: {e}")
            return {"error": str(e)}

    # ============= LIME =============

    def compute_lime_explanation(
        self, instance: np.ndarray, num_features: int = 10, num_samples: int = 1000
    ) -> Dict:
        """
        Tek bir instance için LIME açıklaması

        Args:
            instance: Açıklanacak tek sample
            num_features: Gösterilecek feature sayısı
            num_samples: LIME için sample sayısı
        """
        try:
            from lime import lime_tabular
        except ImportError:
            return {"error": "LIME paketi yüklü değil. pip install lime"}

        if self.model is None:
            return {"error": "Model set edilmedi"}

        try:
            # Instance'ı düzleştir
            if len(instance.shape) > 1:
                flat_instance = instance.flatten()
            else:
                flat_instance = instance

            # LIME explainer
            if self.lime_explainer is None:
                # Dummy training data for explainer
                dummy_data = np.random.randn(100, len(flat_instance))

                feature_names = (
                    self.feature_names
                    if self.feature_names
                    else [f"feature_{i}" for i in range(len(flat_instance))]
                )

                self.lime_explainer = lime_tabular.LimeTabularExplainer(
                    training_data=dummy_data,
                    feature_names=feature_names[: len(flat_instance)],
                    mode="classification",
                )

            def predict_fn(x):
                # Reshape for model
                original_shape = instance.shape
                if len(original_shape) > 1:
                    x_reshaped = x.reshape(-1, *original_shape)
                else:
                    x_reshaped = x.reshape(-1, 1, len(flat_instance))
                return self.model.predict(x_reshaped, verbose=0)

            # Explanation
            explanation = self.lime_explainer.explain_instance(
                flat_instance,
                predict_fn,
                num_features=num_features,
                num_samples=num_samples,
            )

            # Extract results
            feature_weights = {}
            for feature, weight in explanation.as_list():
                feature_weights[feature] = float(weight)

            return {
                "method": "lime",
                "feature_weights": feature_weights,
                "prediction_proba": (
                    explanation.predict_proba.tolist()
                    if hasattr(explanation, "predict_proba")
                    else None
                ),
                "num_features": num_features,
            }

        except Exception as e:
            logger.error(f"LIME error: {e}")
            return {"error": str(e)}

    # ============= Feature Importance =============

    def compute_permutation_importance(
        self, X: np.ndarray, y: np.ndarray, n_repeats: int = 5
    ) -> Dict:
        """
        Permutation feature importance hesapla
        """
        if self.model is None:
            return {"error": "Model set edilmedi"}

        print("🔄 Permutation importance hesaplanıyor...")

        try:
            from sklearn.metrics import accuracy_score

            # Base score
            y_pred = np.argmax(self.model.predict(X, verbose=0), axis=1)
            base_score = accuracy_score(y, y_pred)

            importance_scores = {}
            n_features = X.shape[-1]

            for i in range(n_features):
                scores = []
                for _ in range(n_repeats):
                    X_permuted = X.copy()
                    # Permute feature i
                    if len(X.shape) == 3:
                        X_permuted[:, :, i] = np.random.permutation(
                            X_permuted[:, :, i].flatten()
                        ).reshape(X_permuted[:, :, i].shape)
                    else:
                        X_permuted[:, i] = np.random.permutation(X_permuted[:, i])

                    y_pred_perm = np.argmax(
                        self.model.predict(X_permuted, verbose=0), axis=1
                    )
                    perm_score = accuracy_score(y, y_pred_perm)
                    scores.append(base_score - perm_score)

                feature_name = (
                    self.feature_names[i]
                    if i < len(self.feature_names)
                    else f"feature_{i}"
                )
                importance_scores[feature_name] = {
                    "mean": float(np.mean(scores)),
                    "std": float(np.std(scores)),
                }

            # Sort
            sorted_importance = dict(
                sorted(
                    importance_scores.items(), key=lambda x: x[1]["mean"], reverse=True
                )
            )

            print(f"✅ Permutation importance tamamlandı")

            return {
                "method": "permutation",
                "base_score": base_score,
                "feature_importance": sorted_importance,
                "top_features": list(sorted_importance.keys())[:10],
            }

        except Exception as e:
            logger.error(f"Permutation importance error: {e}")
            return {"error": str(e)}

    # ============= Visualization Data =============

    def get_visualization_data(self, top_n: int = 15) -> Dict:
        """
        Görselleştirme için veri hazırla
        """
        if self.shap_values is None:
            return {"error": "Önce SHAP değerlerini hesaplayın"}

        try:
            if isinstance(self.shap_values, list):
                importance = np.mean(
                    [np.abs(sv).mean(axis=0) for sv in self.shap_values], axis=0
                )
            else:
                importance = np.abs(self.shap_values).mean(axis=0)

            if len(importance.shape) > 1:
                importance = importance.mean(axis=0)

            # Top features
            feature_data = []
            for i, imp in enumerate(importance[: min(len(importance), top_n)]):
                name = (
                    self.feature_names[i]
                    if i < len(self.feature_names)
                    else f"feature_{i}"
                )
                feature_data.append(
                    {"feature": name, "importance": float(imp), "index": i}
                )

            # Sort
            feature_data.sort(key=lambda x: x["importance"], reverse=True)

            return {
                "bar_chart_data": feature_data[:top_n],
                "total_features": len(importance),
            }

        except Exception as e:
            return {"error": str(e)}


# ============= Utility Functions =============


def explain_prediction(
    model, instance: np.ndarray, feature_names: List[str] = None, method: str = "shap"
) -> Dict:
    """
    Tek bir prediction için açıklama
    """
    engine = ExplainabilityEngine(model, feature_names)

    if method == "shap":
        # SHAP için background data gerekli
        return engine.compute_shap_values(instance.reshape(1, *instance.shape))
    elif method == "lime":
        return engine.compute_lime_explanation(instance)
    else:
        return {"error": f"Bilinmeyen method: {method}"}


def get_feature_importance(
    model,
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str] = None,
    method: str = "permutation",
) -> Dict:
    """
    Feature importance hesapla
    """
    engine = ExplainabilityEngine(model, feature_names)

    if method == "permutation":
        return engine.compute_permutation_importance(X, y)
    elif method == "shap":
        return engine.compute_shap_values(X)
    else:
        return {"error": f"Bilinmeyen method: {method}"}


# Singleton
_xai_instance: Optional[ExplainabilityEngine] = None


def get_xai_engine() -> ExplainabilityEngine:
    """Global XAI engine"""
    global _xai_instance
    if _xai_instance is None:
        _xai_instance = ExplainabilityEngine()
    return _xai_instance


# Test
if __name__ == "__main__":
    print("🧪 XAI Module Test\n")

    # Dummy model and data
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers

    # Simple model
    model = keras.Sequential(
        [
            layers.Input(shape=(10, 5)),
            layers.LSTM(32),
            layers.Dense(2, activation="softmax"),
        ]
    )
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy")

    # Dummy data
    X = np.random.randn(100, 10, 5).astype(np.float32)
    y = np.random.randint(0, 2, 100)

    # Train briefly
    model.fit(X, y, epochs=2, verbose=0)

    # Test XAI
    engine = ExplainabilityEngine(
        model=model, feature_names=[f"feat_{i}" for i in range(5)]
    )

    # Permutation importance
    print("📊 Permutation Importance:")
    result = engine.compute_permutation_importance(X[:50], y[:50], n_repeats=2)
    print(json.dumps(result, indent=2))
