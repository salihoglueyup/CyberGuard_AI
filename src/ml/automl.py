"""
AutoML Engine - CyberGuard AI
=============================

Otomatik model seçimi ve hyperparameter tuning.

Özellikler:
    - Model karşılaştırma (LSTM, GRU, Transformer, BiLSTM)
    - Grid/Random/Bayesian search
    - Cross-validation
    - Best model seçimi
"""

import os
import sys
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Callable
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum
import logging

# Suppress TensorFlow warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logger = logging.getLogger("AutoML")


class SearchStrategy(Enum):
    GRID = "grid"
    RANDOM = "random"
    BAYESIAN = "bayesian"


class ModelType(Enum):
    LSTM = "lstm"
    GRU = "gru"
    BILSTM = "bilstm"
    TRANSFORMER = "transformer"
    CNN_LSTM = "cnn_lstm"
    SSA_LSTMIDS = "ssa_lstmids"


@dataclass
class HyperparameterSpace:
    """Hyperparameter arama uzayı"""

    lstm_units: List[int] = field(default_factory=lambda: [64, 128, 256])
    dense_units: List[int] = field(default_factory=lambda: [256, 512])
    dropout_rate: List[float] = field(default_factory=lambda: [0.2, 0.3, 0.5])
    learning_rate: List[float] = field(default_factory=lambda: [0.001, 0.0001])
    batch_size: List[int] = field(default_factory=lambda: [32, 64, 128])
    conv_filters: List[int] = field(default_factory=lambda: [16, 32, 64])
    kernel_size: List[int] = field(default_factory=lambda: [3, 5, 7])


@dataclass
class TrialResult:
    """Bir deneme sonucu"""

    trial_id: str
    model_type: str
    hyperparameters: Dict
    metrics: Dict
    training_time: float
    timestamp: str


class AutoMLEngine:
    """
    AutoML Engine - Otomatik model seçimi ve hyperparameter optimization
    """

    def __init__(
        self,
        search_strategy: SearchStrategy = SearchStrategy.RANDOM,
        max_trials: int = 10,
        cv_folds: int = 3,
        metric: str = "accuracy",
    ):
        self.search_strategy = search_strategy
        self.max_trials = max_trials
        self.cv_folds = cv_folds
        self.metric = metric

        self.trials: List[TrialResult] = []
        self.best_trial: Optional[TrialResult] = None
        self.is_running = False

        # Sonuçlar dizini
        self.results_dir = PROJECT_ROOT / "models" / "automl_results"
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def _build_model(
        self, model_type: ModelType, input_shape: Tuple, num_classes: int, params: Dict
    ):
        """Model oluştur"""
        import tensorflow as tf
        from tensorflow import keras
        from tensorflow.keras import layers

        if model_type == ModelType.LSTM:
            model = keras.Sequential(
                [
                    layers.Input(shape=input_shape),
                    layers.LSTM(
                        params.get("lstm_units", 128),
                        dropout=params.get("dropout_rate", 0.2),
                    ),
                    layers.Dense(params.get("dense_units", 256), activation="relu"),
                    layers.Dropout(params.get("dropout_rate", 0.2)),
                    layers.Dense(num_classes, activation="softmax"),
                ]
            )

        elif model_type == ModelType.GRU:
            model = keras.Sequential(
                [
                    layers.Input(shape=input_shape),
                    layers.GRU(
                        params.get("lstm_units", 128),
                        dropout=params.get("dropout_rate", 0.2),
                    ),
                    layers.Dense(params.get("dense_units", 256), activation="relu"),
                    layers.Dropout(params.get("dropout_rate", 0.2)),
                    layers.Dense(num_classes, activation="softmax"),
                ]
            )

        elif model_type == ModelType.BILSTM:
            model = keras.Sequential(
                [
                    layers.Input(shape=input_shape),
                    layers.Bidirectional(
                        layers.LSTM(
                            params.get("lstm_units", 64),
                            dropout=params.get("dropout_rate", 0.2),
                        )
                    ),
                    layers.Dense(params.get("dense_units", 256), activation="relu"),
                    layers.Dropout(params.get("dropout_rate", 0.2)),
                    layers.Dense(num_classes, activation="softmax"),
                ]
            )

        elif model_type == ModelType.CNN_LSTM:
            model = keras.Sequential(
                [
                    layers.Input(shape=input_shape),
                    layers.Conv1D(
                        params.get("conv_filters", 32),
                        params.get("kernel_size", 3),
                        activation="relu",
                        padding="same",
                    ),
                    layers.MaxPooling1D(2),
                    layers.LSTM(
                        params.get("lstm_units", 64),
                        dropout=params.get("dropout_rate", 0.2),
                    ),
                    layers.Dense(params.get("dense_units", 256), activation="relu"),
                    layers.Dropout(params.get("dropout_rate", 0.2)),
                    layers.Dense(num_classes, activation="softmax"),
                ]
            )

        elif model_type == ModelType.TRANSFORMER:
            # Simplified transformer
            inputs = layers.Input(shape=input_shape)
            x = layers.MultiHeadAttention(num_heads=4, key_dim=32)(inputs, inputs)
            x = layers.GlobalAveragePooling1D()(x)
            x = layers.Dense(params.get("dense_units", 256), activation="relu")(x)
            x = layers.Dropout(params.get("dropout_rate", 0.2))(x)
            outputs = layers.Dense(num_classes, activation="softmax")(x)
            model = keras.Model(inputs, outputs)

        else:  # SSA_LSTMIDS (paper model)
            model = keras.Sequential(
                [
                    layers.Input(shape=input_shape),
                    layers.Conv1D(30, 5, activation="relu", padding="same"),
                    layers.MaxPooling1D(2),
                    layers.LSTM(120, dropout=0.2),
                    layers.Dense(512, activation="relu"),
                    layers.Dropout(0.2),
                    layers.Dense(num_classes, activation="softmax"),
                ]
            )

        optimizer = keras.optimizers.Adam(
            learning_rate=params.get("learning_rate", 0.001)
        )
        model.compile(
            optimizer=optimizer,
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )

        return model

    def _sample_hyperparameters(self, space: HyperparameterSpace) -> Dict:
        """Hyperparameter sample et"""
        if self.search_strategy == SearchStrategy.RANDOM:
            return {
                "lstm_units": np.random.choice(space.lstm_units),
                "dense_units": np.random.choice(space.dense_units),
                "dropout_rate": np.random.choice(space.dropout_rate),
                "learning_rate": np.random.choice(space.learning_rate),
                "batch_size": np.random.choice(space.batch_size),
                "conv_filters": np.random.choice(space.conv_filters),
                "kernel_size": np.random.choice(space.kernel_size),
            }
        elif self.search_strategy == SearchStrategy.GRID:
            # Grid için sıralı seçim (simplified)
            idx = len(self.trials) % len(space.lstm_units)
            return {
                "lstm_units": space.lstm_units[idx % len(space.lstm_units)],
                "dense_units": space.dense_units[idx % len(space.dense_units)],
                "dropout_rate": space.dropout_rate[idx % len(space.dropout_rate)],
                "learning_rate": space.learning_rate[idx % len(space.learning_rate)],
                "batch_size": space.batch_size[idx % len(space.batch_size)],
                "conv_filters": space.conv_filters[idx % len(space.conv_filters)],
                "kernel_size": space.kernel_size[idx % len(space.kernel_size)],
            }
        else:  # Bayesian - simplified
            # Gerçek Bayesian için gaussian process kullanılır
            return self._sample_hyperparameters_bayesian(space)

    def _sample_hyperparameters_bayesian(self, space: HyperparameterSpace) -> Dict:
        """Bayesian hyperparameter sampling (simplified)"""
        # Önceki sonuçlara göre ağırlıklı sampling
        if len(self.trials) < 3:
            return {
                "lstm_units": np.random.choice(space.lstm_units),
                "dense_units": np.random.choice(space.dense_units),
                "dropout_rate": np.random.choice(space.dropout_rate),
                "learning_rate": np.random.choice(space.learning_rate),
                "batch_size": np.random.choice(space.batch_size),
                "conv_filters": np.random.choice(space.conv_filters),
                "kernel_size": np.random.choice(space.kernel_size),
            }

        # En iyi trial'ın parametrelerine yakın sample et
        best = max(self.trials, key=lambda t: t.metrics.get(self.metric, 0))
        best_params = best.hyperparameters

        def nearby_choice(values, current):
            if current in values:
                idx = values.index(current)
                # %70 olasılıkla yakın değer, %30 random
                if np.random.random() < 0.7:
                    nearby_idx = max(
                        0, min(len(values) - 1, idx + np.random.choice([-1, 0, 1]))
                    )
                    return values[nearby_idx]
            return np.random.choice(values)

        return {
            "lstm_units": nearby_choice(
                space.lstm_units, best_params.get("lstm_units")
            ),
            "dense_units": nearby_choice(
                space.dense_units, best_params.get("dense_units")
            ),
            "dropout_rate": nearby_choice(
                space.dropout_rate, best_params.get("dropout_rate")
            ),
            "learning_rate": nearby_choice(
                space.learning_rate, best_params.get("learning_rate")
            ),
            "batch_size": nearby_choice(
                space.batch_size, best_params.get("batch_size")
            ),
            "conv_filters": nearby_choice(
                space.conv_filters, best_params.get("conv_filters")
            ),
            "kernel_size": nearby_choice(
                space.kernel_size, best_params.get("kernel_size")
            ),
        }

    def run_trial(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        model_type: ModelType,
        params: Dict,
        epochs: int = 20,
    ) -> TrialResult:
        """Tek bir trial çalıştır"""
        import time
        from tensorflow import keras

        trial_id = f"trial_{len(self.trials)+1:03d}"
        start_time = time.time()

        try:
            # Model oluştur
            input_shape = (X_train.shape[1], X_train.shape[2])
            num_classes = len(np.unique(y_train))

            model = self._build_model(model_type, input_shape, num_classes, params)

            # Eğit
            callbacks = [
                keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
            ]

            model.fit(
                X_train,
                y_train,
                batch_size=int(params.get("batch_size", 32)),
                epochs=epochs,
                validation_data=(X_val, y_val),
                callbacks=callbacks,
                verbose=0,
            )

            # Değerlendir
            y_pred = np.argmax(model.predict(X_val, verbose=0), axis=1)

            from sklearn.metrics import (
                accuracy_score,
                precision_score,
                recall_score,
                f1_score,
            )

            metrics = {
                "accuracy": float(accuracy_score(y_val, y_pred)),
                "precision": float(
                    precision_score(y_val, y_pred, average="weighted", zero_division=0)
                ),
                "recall": float(
                    recall_score(y_val, y_pred, average="weighted", zero_division=0)
                ),
                "f1_score": float(
                    f1_score(y_val, y_pred, average="weighted", zero_division=0)
                ),
            }

        except Exception as e:
            logger.error(f"Trial failed: {e}")
            metrics = {
                "accuracy": 0,
                "precision": 0,
                "recall": 0,
                "f1_score": 0,
                "error": str(e),
            }

        training_time = time.time() - start_time

        result = TrialResult(
            trial_id=trial_id,
            model_type=model_type.value,
            hyperparameters=params,
            metrics=metrics,
            training_time=training_time,
            timestamp=datetime.now().isoformat(),
        )

        self.trials.append(result)

        # Best update
        if self.best_trial is None or metrics.get(
            self.metric, 0
        ) > self.best_trial.metrics.get(self.metric, 0):
            self.best_trial = result

        return result

    def search(
        self,
        X: np.ndarray,
        y: np.ndarray,
        model_types: List[ModelType] = None,
        space: HyperparameterSpace = None,
        epochs_per_trial: int = 20,
        callback: Callable[[TrialResult], None] = None,
    ) -> Dict:
        """
        AutoML arama çalıştır

        Args:
            X: Training data (samples, timesteps, features)
            y: Labels
            model_types: Test edilecek model tipleri
            space: Hyperparameter arama uzayı
            epochs_per_trial: Her trial için epoch sayısı
            callback: Her trial sonrası çağrılacak fonksiyon
        """
        from sklearn.model_selection import train_test_split

        if model_types is None:
            model_types = [
                ModelType.LSTM,
                ModelType.GRU,
                ModelType.CNN_LSTM,
                ModelType.BILSTM,
            ]

        if space is None:
            space = HyperparameterSpace()

        self.is_running = True
        self.trials = []
        self.best_trial = None

        # Train/Val split
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        print(f"\n{'='*60}")
        print(f"🤖 AutoML Search Başlıyor")
        print(f"{'='*60}")
        print(f"   Strategy: {self.search_strategy.value}")
        print(f"   Max Trials: {self.max_trials}")
        print(f"   Model Types: {[m.value for m in model_types]}")
        print(f"   Train Size: {len(X_train)}, Val Size: {len(X_val)}")

        trials_per_model = max(1, self.max_trials // len(model_types))

        for model_type in model_types:
            print(f"\n🔄 Testing {model_type.value}...")

            for i in range(trials_per_model):
                if not self.is_running:
                    break

                params = self._sample_hyperparameters(space)

                result = self.run_trial(
                    X_train, y_train, X_val, y_val, model_type, params, epochs_per_trial
                )

                print(
                    f"   Trial {result.trial_id}: {self.metric}={result.metrics.get(self.metric, 0)*100:.2f}%"
                )

                if callback:
                    callback(result)

        self.is_running = False

        # Sonuçları kaydet
        self._save_results()

        return self.get_summary()

    def get_summary(self) -> Dict:
        """Arama özeti"""
        if not self.trials:
            return {"status": "no_trials"}

        return {
            "total_trials": len(self.trials),
            "best_trial": (
                {
                    "id": self.best_trial.trial_id,
                    "model_type": self.best_trial.model_type,
                    "metrics": self.best_trial.metrics,
                    "hyperparameters": self.best_trial.hyperparameters,
                }
                if self.best_trial
                else None
            ),
            "all_trials": [
                {
                    "id": t.trial_id,
                    "model_type": t.model_type,
                    "accuracy": t.metrics.get("accuracy", 0),
                    "training_time": t.training_time,
                }
                for t in sorted(
                    self.trials,
                    key=lambda x: x.metrics.get("accuracy", 0),
                    reverse=True,
                )
            ],
            "search_strategy": self.search_strategy.value,
        }

    def _save_results(self):
        """Sonuçları kaydet"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = self.results_dir / f"automl_results_{timestamp}.json"

        data = {
            "summary": self.get_summary(),
            "trials": [
                {
                    "trial_id": t.trial_id,
                    "model_type": t.model_type,
                    "hyperparameters": t.hyperparameters,
                    "metrics": t.metrics,
                    "training_time": t.training_time,
                    "timestamp": t.timestamp,
                }
                for t in self.trials
            ],
        }

        with open(results_file, "w") as f:
            json.dump(data, f, indent=2)

        print(f"\n💾 Sonuçlar kaydedildi: {results_file}")

    def stop(self):
        """Aramayı durdur"""
        self.is_running = False


# Singleton instance
_automl_instance: Optional[AutoMLEngine] = None


def get_automl_engine() -> AutoMLEngine:
    """Global AutoML engine"""
    global _automl_instance
    if _automl_instance is None:
        _automl_instance = AutoMLEngine()
    return _automl_instance


# Test
if __name__ == "__main__":
    print("🧪 AutoML Engine Test\n")

    # Dummy data
    np.random.seed(42)
    X = np.random.randn(1000, 10, 41).astype(np.float32)
    y = np.random.randint(0, 2, 1000)

    engine = AutoMLEngine(
        search_strategy=SearchStrategy.RANDOM, max_trials=4, metric="accuracy"
    )

    results = engine.search(
        X, y, model_types=[ModelType.LSTM, ModelType.GRU], epochs_per_trial=5
    )

    print("\n📊 Sonuçlar:")
    print(json.dumps(results, indent=2))
