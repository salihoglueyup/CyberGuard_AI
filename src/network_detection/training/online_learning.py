"""
Online Learning IDS
CyberGuard AI için streaming ve real-time öğrenme

Özellikler:
    - Incremental learning (mini-batch)
    - Concept drift detection
    - Adaptive threshold
    - Real-time prediction

Ref: Makaledeki "Online IDS" önerisi
"""

import numpy as np
from typing import Dict, Optional, Tuple, List, Deque
from collections import deque
import time

try:
    import tensorflow as tf
    from tensorflow import keras

    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False


class OnlineLearningIDS:
    """
    Online Learning IDS

    Streaming veri ile sürekli öğrenen model.
    Concept drift'e adapte olabilen.
    """

    def __init__(
        self,
        base_model,
        buffer_size: int = 1000,
        update_interval: int = 100,
        learning_rate: float = 0.0001,
        drift_threshold: float = 0.1,
    ):
        """
        Args:
            base_model: Pre-trained base model
            buffer_size: Replay buffer boyutu
            update_interval: Kaç örnekte bir güncelleme
            learning_rate: Online learning rate (düşük!)
            drift_threshold: Concept drift algılama eşiği
        """
        self.base_model = base_model
        self.buffer_size = buffer_size
        self.update_interval = update_interval
        self.learning_rate = learning_rate
        self.drift_threshold = drift_threshold

        # Replay buffer
        self.X_buffer: Deque = deque(maxlen=buffer_size)
        self.y_buffer: Deque = deque(maxlen=buffer_size)

        # Stats
        self.samples_seen = 0
        self.updates_done = 0
        self.recent_accuracy: Deque = deque(maxlen=100)
        self.drift_detected = False

        # Original optimizer learning rate'i düşür
        if hasattr(self.base_model, "optimizer"):
            keras.backend.set_value(
                self.base_model.optimizer.learning_rate, self.learning_rate
            )

        print("🔄 Online Learning IDS başlatıldı")
        print(f"   Buffer: {buffer_size}, Update interval: {update_interval}")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Real-time prediction"""
        proba = self.base_model.predict(X, verbose=0)
        return np.argmax(proba, axis=1)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Probability prediction"""
        return self.base_model.predict(X, verbose=0)

    def partial_fit(self, X: np.ndarray, y: np.ndarray):
        """
        Incremental update with new data

        Args:
            X: Yeni örnekler
            y: Yeni etiketler
        """
        # Buffer'a ekle
        for i in range(len(X)):
            self.X_buffer.append(X[i])
            self.y_buffer.append(y[i])

        self.samples_seen += len(X)

        # Accuracy tracking
        y_pred = self.predict(X)
        batch_accuracy = np.mean(y_pred == y)
        self.recent_accuracy.append(batch_accuracy)

        # Concept drift kontrolü
        self._check_drift()

        # Update interval'da model güncelle
        if self.samples_seen % self.update_interval == 0:
            self._update_model()

    def _update_model(self):
        """Model'i buffer'daki veri ile güncelle"""
        if len(self.X_buffer) < 32:
            return

        X_train = np.array(self.X_buffer)
        y_train = np.array(self.y_buffer)

        # Mini-batch update
        self.base_model.fit(
            X_train, y_train, epochs=1, batch_size=min(32, len(X_train)), verbose=0
        )

        self.updates_done += 1

    def _check_drift(self):
        """Concept drift algılama"""
        if len(self.recent_accuracy) < 50:
            return

        recent = list(self.recent_accuracy)
        first_half = np.mean(recent[: len(recent) // 2])
        second_half = np.mean(recent[len(recent) // 2 :])

        drift = abs(first_half - second_half)

        if drift > self.drift_threshold:
            if not self.drift_detected:
                print(f"⚠️ Concept drift algılandı! (Δ={drift:.3f})")
                self.drift_detected = True
        else:
            self.drift_detected = False

    def get_stats(self) -> Dict:
        """İstatistikleri döndür"""
        return {
            "samples_seen": self.samples_seen,
            "updates_done": self.updates_done,
            "buffer_size": len(self.X_buffer),
            "recent_accuracy": (
                np.mean(self.recent_accuracy) if self.recent_accuracy else 0
            ),
            "drift_detected": self.drift_detected,
        }


class StreamingPredictor:
    """
    Streaming data için prediction pipeline

    Network packet stream'i işler.
    """

    def __init__(self, model, batch_size: int = 32, threshold: float = 0.5):
        self.model = model
        self.batch_size = batch_size
        self.threshold = threshold

        self.pending_batch: List = []
        self.predictions: List = []
        self.latencies: List = []

    def add_sample(self, sample: np.ndarray) -> Optional[int]:
        """
        Tek örnek ekle, batch dolunca predict et

        Returns:
            None (batch dolmadı) veya prediction
        """
        self.pending_batch.append(sample)

        if len(self.pending_batch) >= self.batch_size:
            return self._process_batch()

        return None

    def _process_batch(self) -> int:
        """Batch'i işle"""
        start = time.time()

        X = np.array(self.pending_batch)
        proba = self.model.predict(X, verbose=0)
        preds = np.argmax(proba, axis=1)

        latency = (time.time() - start) * 1000  # ms
        self.latencies.append(latency)

        self.predictions.extend(preds.tolist())
        self.pending_batch = []

        return preds[-1]  # Son prediction

    def flush(self) -> List[int]:
        """Pending örnekleri işle"""
        if len(self.pending_batch) > 0:
            X = np.array(self.pending_batch)
            proba = self.model.predict(X, verbose=0)
            preds = np.argmax(proba, axis=1)
            self.predictions.extend(preds.tolist())
            self.pending_batch = []

        return self.predictions

    def get_stats(self) -> Dict:
        return {
            "total_predictions": len(self.predictions),
            "avg_latency_ms": np.mean(self.latencies) if self.latencies else 0,
            "pending": len(self.pending_batch),
        }


# Test
if __name__ == "__main__":
    print("🧪 Online Learning Test\n")

    # Mock model
    class MockModel:
        def predict(self, X, verbose=0):
            n = len(X)
            proba = np.random.rand(n, 5)
            return proba / proba.sum(axis=1, keepdims=True)

        def fit(self, X, y, epochs=1, batch_size=32, verbose=0):
            pass

    model = MockModel()
    online_ids = OnlineLearningIDS(model, buffer_size=500, update_interval=50)

    # Simulate streaming
    for i in range(200):
        X = np.random.rand(10, 1, 78)
        y = np.random.randint(0, 5, 10)
        online_ids.partial_fit(X, y)

    print(f"Stats: {online_ids.get_stats()}")
    print("\n✅ Test tamamlandı!")
