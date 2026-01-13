"""
Network Anomaly Model - CyberGuard AI
LSTM + Dense Network tabanlı ağ anomali tespiti

Dosya Yolu: src/network_detection/model.py
"""

import os
import json
import pickle
import numpy as np
from typing import Dict, List, Optional
from datetime import datetime

try:
    from sklearn.ensemble import IsolationForest, RandomForestClassifier
    from sklearn.preprocessing import StandardScaler

    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import tensorflow as tf
    from tensorflow import keras

    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False


class NetworkAnomalyModel:
    """
    Ağ anomali tespit modeli

    Modeller:
    - Isolation Forest (unsupervised)
    - Random Forest Classifier (supervised)
    - LSTM Neural Network (sequential)

    Saldırı türleri:
    - DDoS, Port Scan, SQL Injection, XSS, Brute Force
    """

    ATTACK_TYPES = [
        "Normal",
        "DDoS",
        "SQL Injection",
        "XSS",
        "Port Scan",
        "Brute Force",
    ]

    def __init__(self, model_type: str = "random_forest", use_lstm: bool = False):
        """
        Args:
            model_type: 'random_forest' veya 'isolation_forest'
            use_lstm: LSTM neural network kullan
        """
        self.model_type = model_type
        self.use_lstm = use_lstm and TF_AVAILABLE

        self.scaler = StandardScaler() if SKLEARN_AVAILABLE else None
        self.classifier = None
        self.lstm_model = None
        self.is_trained = False

        # Model türüne göre başlat
        if SKLEARN_AVAILABLE:
            if model_type == "random_forest":
                self.classifier = RandomForestClassifier(
                    n_estimators=100, max_depth=10, random_state=42, n_jobs=-1
                )
            elif model_type == "isolation_forest":
                self.classifier = IsolationForest(
                    n_estimators=100, contamination=0.1, random_state=42
                )

        print(f"🌐 Network Anomaly Model başlatıldı")
        print(f"   Model tipi: {model_type}")
        print(f"   LSTM: {'Evet' if self.use_lstm else 'Hayır'}")

    def build_lstm(self, input_shape: tuple) -> None:
        """
        Optimized LSTM modeli oluştur

        Mimari (Scientific Reports 2025):
        Conv1D(30) → MaxPool → LSTM(120) → Dense(512) → Output

        Ref: "An optimized LSTM-based deep learning model for anomaly network intrusion detection"
        """
        if not TF_AVAILABLE:
            print("❌ TensorFlow bulunamadı, LSTM kullanılamaz")
            return

        print("🔧 Optimized LSTM-IDS modeli oluşturuluyor...")

        self.lstm_model = keras.Sequential(
            [
                # 1. Conv1D Layer - Pattern Extraction
                keras.layers.Conv1D(
                    filters=30,
                    kernel_size=5,
                    padding="same",
                    activation="relu",
                    input_shape=input_shape,
                    name="conv1d_pattern_extraction",
                ),
                # 2. MaxPooling - Dimensionality Reduction
                keras.layers.MaxPooling1D(pool_size=2, name="maxpool_reduction"),
                # 3. LSTM Layer - Temporal Learning
                keras.layers.LSTM(units=120, dropout=0.2, name="lstm_temporal"),
                # 4. Dense Layer - Feature Transformation
                keras.layers.Dense(
                    units=512, activation="sigmoid", name="dense_transform"
                ),
                # 5. Output Layer - Classification
                keras.layers.Dense(
                    units=len(self.ATTACK_TYPES),
                    activation="softmax",
                    name="output_classification",
                ),
            ]
        )

        # Compile with Adam optimizer
        self.lstm_model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )

        # Model summary
        print("✅ Optimized LSTM-IDS modeli oluşturuldu!")
        print(f"   📊 Conv1D: 30 filters, kernel=5")
        print(f"   📊 LSTM: 120 units, dropout=0.2")
        print(f"   📊 Dense: 512 units, sigmoid")
        print(f"   📊 Output: {len(self.ATTACK_TYPES)} classes")

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int = 50,
        batch_size: int = 32,
        validation_split: float = 0.2,
    ) -> Dict:
        """
        Modeli eğit

        Args:
            X: Özellik matrisi
            y: Etiketler (0=normal, 1-5=saldırı türleri)
            epochs: LSTM epochs
            batch_size: Batch size
            validation_split: Validation oranı

        Returns:
            Eğitim sonuçları
        """
        if not SKLEARN_AVAILABLE:
            raise RuntimeError("scikit-learn yüklü değil!")

        print(f"\n🎯 Eğitim başlıyor...")
        print(f"   Samples: {len(X)}")
        print(f"   Features: {X.shape[1]}")
        print(f"   Classes: {len(np.unique(y))}")

        results = {}

        # Scaler fit
        X_scaled = self.scaler.fit_transform(X)

        # Classifier eğit
        if self.model_type == "isolation_forest":
            # Unsupervised - sadece normal verileri kullan
            X_normal = X_scaled[y == 0]
            self.classifier.fit(X_normal)
            results["model"] = "isolation_forest"
        else:
            # Supervised
            self.classifier.fit(X_scaled, y)
            train_score = self.classifier.score(X_scaled, y)
            results["train_accuracy"] = train_score
            print(f"   Classifier accuracy: {train_score:.4f}")

        # LSTM eğit
        if self.use_lstm:
            print(f"\n🧠 Optimized LSTM-IDS eğitiliyor...")
            print(f"   Epochs: {epochs}, Batch: {batch_size}")

            # LSTM için 3D shape gerekli (samples, timesteps, features)
            X_lstm = X_scaled.reshape(X_scaled.shape[0], 1, X_scaled.shape[1])

            self.build_lstm((1, X_scaled.shape[1]))

            # Early Stopping callback (Makale: patience=6, monitor='loss')
            early_stopping = keras.callbacks.EarlyStopping(
                monitor="loss", patience=6, verbose=1, restore_best_weights=True
            )

            history = self.lstm_model.fit(
                X_lstm,
                y,
                validation_split=validation_split,
                epochs=epochs,
                batch_size=batch_size,
                callbacks=[early_stopping],
                verbose=1,  # Eğitim sürecini göster
            )

            results["lstm_accuracy"] = history.history["accuracy"][-1]
            results["lstm_val_accuracy"] = history.history["val_accuracy"][-1]
            results["epochs_trained"] = len(history.history["accuracy"])

        self.is_trained = True
        results["trained_at"] = datetime.now().isoformat()

        print(f"\n✅ Eğitim tamamlandı!")

        return results

    def predict(self, X: np.ndarray, return_proba: bool = False) -> np.ndarray:
        """Tahmin yap"""
        if not self.is_trained:
            raise RuntimeError("Model henüz eğitilmedi!")

        X_scaled = self.scaler.transform(X)

        if self.model_type == "isolation_forest":
            # -1 = anomaly, 1 = normal
            pred = self.classifier.predict(X_scaled)
            return (pred == -1).astype(int)  # 1 = anomaly
        else:
            if return_proba:
                return self.classifier.predict_proba(X_scaled)
            return self.classifier.predict(X_scaled)

    def predict_single(self, features: List[float]) -> Dict:
        """Tek örnek için tahmin"""
        X = np.array([features])

        if self.model_type == "isolation_forest":
            is_anomaly = self.predict(X)[0]
            return {
                "is_anomaly": bool(is_anomaly),
                "prediction": "Anomaly" if is_anomaly else "Normal",
                "attack_type": None,
            }
        else:
            pred = self.predict(X)[0]
            proba = self.predict(X, return_proba=True)[0]

            return {
                "prediction_id": int(pred),
                "prediction": self.ATTACK_TYPES[pred],
                "is_attack": pred > 0,
                "confidence": float(max(proba)),
                "probabilities": {
                    self.ATTACK_TYPES[i]: float(p) for i, p in enumerate(proba)
                },
            }

    def save(self, model_dir: str) -> str:
        """Modeli kaydet"""
        os.makedirs(model_dir, exist_ok=True)

        # Classifier
        with open(os.path.join(model_dir, "classifier.pkl"), "wb") as f:
            pickle.dump(self.classifier, f)

        # Scaler
        with open(os.path.join(model_dir, "scaler.pkl"), "wb") as f:
            pickle.dump(self.scaler, f)

        # LSTM
        if self.use_lstm and self.lstm_model:
            self.lstm_model.save(os.path.join(model_dir, "lstm_model.h5"))

        # Metadata
        metadata = {
            "model_type": self.model_type,
            "use_lstm": self.use_lstm,
            "attack_types": self.ATTACK_TYPES,
            "saved_at": datetime.now().isoformat(),
        }

        with open(os.path.join(model_dir, "metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)

        print(f"✅ Model kaydedildi: {model_dir}")
        return model_dir

    @classmethod
    def load(cls, model_dir: str) -> "NetworkAnomalyModel":
        """Modeli yükle"""
        with open(os.path.join(model_dir, "metadata.json"), "r") as f:
            metadata = json.load(f)

        instance = cls(model_type=metadata["model_type"], use_lstm=metadata["use_lstm"])

        with open(os.path.join(model_dir, "classifier.pkl"), "rb") as f:
            instance.classifier = pickle.load(f)

        with open(os.path.join(model_dir, "scaler.pkl"), "rb") as f:
            instance.scaler = pickle.load(f)

        lstm_path = os.path.join(model_dir, "lstm_model.h5")
        if os.path.exists(lstm_path) and TF_AVAILABLE:
            instance.lstm_model = keras.models.load_model(lstm_path)

        instance.is_trained = True

        print(f"✅ Model yüklendi: {model_dir}")
        return instance


# Test
if __name__ == "__main__":
    if SKLEARN_AVAILABLE:
        print("🧪 Network Anomaly Model Test\n")

        np.random.seed(42)
        X = np.random.rand(200, 11)
        y = np.random.randint(0, 6, 200)

        model = NetworkAnomalyModel(model_type="random_forest")
        results = model.train(X, y, epochs=10)

        pred = model.predict_single(np.random.rand(11).tolist())
        print(f"\n📊 Tahmin: {pred['prediction']}")
        print(f"   Güven: {pred['confidence']:.2%}")
    else:
        print("❌ Test için scikit-learn gerekli!")
