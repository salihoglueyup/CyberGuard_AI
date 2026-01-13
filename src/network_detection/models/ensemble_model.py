"""
Ensemble IDS Model
CyberGuard AI için çoklu model birleştirme

Stratejiler:
    - Voting (hard/soft)
    - Stacking
    - Weighted ensemble

Avantajlar:
    - Tek modelden daha güvenilir
    - Farklı saldırı tiplerine farklı modeller
    - False positive azaltma
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from collections import Counter


class EnsembleIDSModel:
    """
    Ensemble IDS - Çoklu model birleştirme

    LSTM, BiLSTM, GRU, Transformer modellerini birleştirir.
    """

    def __init__(
        self,
        models: List[Any] = None,
        model_names: List[str] = None,
        voting: str = "soft",  # "hard" or "soft"
        weights: List[float] = None,
    ):
        self.models = models or []
        self.model_names = model_names or []
        self.voting = voting
        self.weights = weights

        print(f"🎯 Ensemble IDS başlatılıyor...")
        print(f"   Voting: {voting}")
        print(f"   Models: {len(self.models)}")

    def add_model(self, model, name: str, weight: float = 1.0):
        """Ensemble'a model ekle"""
        self.models.append(model)
        self.model_names.append(name)

        if self.weights is None:
            self.weights = [1.0] * len(self.models)
        else:
            self.weights.append(weight)

        print(f"   ➕ {name} eklendi (weight: {weight})")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Ensemble prediction"""
        if len(self.models) == 0:
            raise ValueError("Ensemble boş! Model ekleyin.")

        if self.voting == "soft":
            return self._soft_voting(X)
        else:
            return self._hard_voting(X)

    def _soft_voting(self, X: np.ndarray) -> np.ndarray:
        """Soft voting - olasılık ortalaması"""
        predictions = []

        for i, model in enumerate(self.models):
            try:
                proba = model.predict(X, verbose=0)
                predictions.append(proba * self.weights[i])
            except Exception as e:
                print(f"⚠️ {self.model_names[i]} prediction failed: {e}")

        if len(predictions) == 0:
            raise ValueError("Hiçbir model prediction yapamadı!")

        # Weighted average
        avg_proba = np.sum(predictions, axis=0) / np.sum(self.weights)
        return np.argmax(avg_proba, axis=1)

    def _hard_voting(self, X: np.ndarray) -> np.ndarray:
        """Hard voting - çoğunluk oylaması"""
        all_predictions = []

        for i, model in enumerate(self.models):
            try:
                proba = model.predict(X, verbose=0)
                pred = np.argmax(proba, axis=1)
                all_predictions.append(pred)
            except Exception as e:
                print(f"⚠️ {self.model_names[i]} prediction failed: {e}")

        if len(all_predictions) == 0:
            raise ValueError("Hiçbir model prediction yapamadı!")

        # Majority voting
        all_predictions = np.array(all_predictions).T
        final_predictions = []

        for sample_preds in all_predictions:
            counter = Counter(sample_preds)
            final_predictions.append(counter.most_common(1)[0][0])

        return np.array(final_predictions)

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """Ensemble değerlendirmesi"""
        from sklearn.metrics import (
            accuracy_score,
            precision_score,
            recall_score,
            f1_score,
        )

        y_pred = self.predict(X)

        return {
            "accuracy": accuracy_score(y, y_pred),
            "precision": precision_score(
                y, y_pred, average="weighted", zero_division=0
            ),
            "recall": recall_score(y, y_pred, average="weighted", zero_division=0),
            "f1_score": f1_score(y, y_pred, average="weighted", zero_division=0),
            "n_models": len(self.models),
        }

    def get_individual_accuracies(
        self, X: np.ndarray, y: np.ndarray
    ) -> Dict[str, float]:
        """Her modelin ayrı accuracy'si"""
        from sklearn.metrics import accuracy_score

        accuracies = {}
        for i, model in enumerate(self.models):
            try:
                proba = model.predict(X, verbose=0)
                pred = np.argmax(proba, axis=1)
                acc = accuracy_score(y, pred)
                accuracies[self.model_names[i]] = acc
            except:
                accuracies[self.model_names[i]] = 0.0

        return accuracies


class StackingEnsemble:
    """
    Stacking Ensemble

    Base modellerin prediction'larını meta-model'e besler.
    Daha güçlü ama daha yavaş.
    """

    def __init__(self, base_models: List[Any] = None, meta_model: Any = None):
        self.base_models = base_models or []
        self.meta_model = meta_model

        print("📚 Stacking Ensemble başlatılıyor...")

    def fit(self, X, y, X_val=None, y_val=None):
        """Base modelleri eğit ve meta features oluştur"""
        print("🏋️ Base modeller eğitiliyor...")

        # Base model predictions (meta features)
        meta_features = []

        for model in self.base_models:
            proba = model.predict(X, verbose=0)
            meta_features.append(proba)

        # Stack meta features
        X_meta = np.hstack(meta_features)

        # Train meta model
        if self.meta_model is not None:
            print("🏋️ Meta model eğitiliyor...")
            self.meta_model.fit(X_meta, y)

        return self

    def predict(self, X) -> np.ndarray:
        """Stacking prediction"""
        meta_features = []

        for model in self.base_models:
            proba = model.predict(X, verbose=0)
            meta_features.append(proba)

        X_meta = np.hstack(meta_features)

        if self.meta_model is not None:
            return self.meta_model.predict(X_meta)
        else:
            # Meta model yoksa soft voting
            return np.argmax(np.mean(meta_features, axis=0), axis=1)


# Helper function
def create_ensemble_from_paths(
    model_paths: List[str], names: List[str] = None
) -> EnsembleIDSModel:
    """Kayıtlı modellerden ensemble oluştur"""
    from tensorflow import keras

    ensemble = EnsembleIDSModel(voting="soft")

    for i, path in enumerate(model_paths):
        name = names[i] if names else f"model_{i}"
        try:
            model = keras.models.load_model(path)
            ensemble.add_model(model, name)
        except Exception as e:
            print(f"⚠️ {path} yüklenemedi: {e}")

    return ensemble


# Test
if __name__ == "__main__":
    print("🧪 Ensemble Test\n")

    # Mock predictions
    class MockModel:
        def __init__(self, bias=0):
            self.bias = bias

        def predict(self, X, verbose=0):
            n = len(X)
            proba = np.random.rand(n, 5)
            proba[:, self.bias] += 0.5
            return proba / proba.sum(axis=1, keepdims=True)

    ensemble = EnsembleIDSModel(voting="soft")
    ensemble.add_model(MockModel(0), "model_a", weight=1.0)
    ensemble.add_model(MockModel(1), "model_b", weight=1.5)
    ensemble.add_model(MockModel(2), "model_c", weight=0.8)

    X_test = np.random.rand(100, 10)
    y_test = np.random.randint(0, 5, 100)

    results = ensemble.evaluate(X_test, y_test)
    print(f"Ensemble accuracy: {results['accuracy']*100:.2f}%")

    print("\n✅ Test tamamlandı!")
