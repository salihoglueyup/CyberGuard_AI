"""
SSA-LSTMIDS Model - Scientific Reports 2025
Makaledeki tam mimari: "An optimized LSTM-based deep learning model for anomaly network intrusion detection"

Mimari:
    Input → Conv1D(30, kernel=5) → MaxPooling → LSTM(120) → Dense(512) → Dropout(0.2) → Output

SSA ile optimize edilen parametreler:
    - Conv1D filters: 30
    - LSTM units: 120
    - Dense units: 512
    - Dropout: 0.2
    - Batch size: 120
    - Epochs: 300
    - Early stopping patience: 6
"""

import os
import sys
import json
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Tuple
from datetime import datetime

# Proje yolu
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, Model
    from tensorflow.keras.callbacks import (
        EarlyStopping,
        ReduceLROnPlateau,
        ModelCheckpoint,
    )

    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    print("❌ TensorFlow gerekli!")


class SSA_LSTMIDS:
    """
    SSA-LSTMIDS Model - Makaledeki tam implementasyon

    Scientific Reports (2025) 15:1554
    "An optimized LSTM-based deep learning model for anomaly network intrusion detection"

    Özellikler:
        - Conv1D + LSTM hibrit mimari
        - SSA (Salp Swarm Algorithm) ile optimize edilmiş parametreler
        - Makale parametreleri varsayılan
    """

    # Makaledeki optimum parametreler (SSA ile bulunmuş)
    PAPER_PARAMS = {
        "conv_filters": 30,
        "kernel_size": 5,
        "lstm_units": 120,
        "dense_units": 512,
        "dropout_rate": 0.2,
        "batch_size": 120,
        "epochs": 300,
        "early_stopping_patience": 6,
        "learning_rate": 0.001,
    }

    def __init__(
        self,
        input_shape: Tuple[int, int],
        num_classes: int,
        use_paper_params: bool = True,
        **kwargs,
    ):
        if not TF_AVAILABLE:
            raise ImportError("TensorFlow gerekli!")

        self.input_shape = input_shape
        self.num_classes = num_classes

        # Makale parametreleri veya custom
        if use_paper_params:
            self.params = self.PAPER_PARAMS.copy()
            print("📄 Makale parametreleri kullanılıyor (SSA ile optimize edilmiş)")
        else:
            self.params = {
                "conv_filters": kwargs.get("conv_filters", 30),
                "kernel_size": kwargs.get("kernel_size", 5),
                "lstm_units": kwargs.get("lstm_units", 120),
                "dense_units": kwargs.get("dense_units", 512),
                "dropout_rate": kwargs.get("dropout_rate", 0.2),
                "batch_size": kwargs.get("batch_size", 120),
                "epochs": kwargs.get("epochs", 300),
                "early_stopping_patience": kwargs.get("early_stopping_patience", 6),
                "learning_rate": kwargs.get("learning_rate", 0.001),
            }

        # Override any custom params
        for key in kwargs:
            if key in self.params:
                self.params[key] = kwargs[key]

        self.model: Optional[Model] = None
        self.history = None

        print("\n" + "=" * 60)
        print("🧠 SSA-LSTMIDS Model - Scientific Reports 2025")
        print("=" * 60)
        print(f"   Input Shape: {input_shape}")
        print(f"   Classes: {num_classes}")
        print(f"   Conv Filters: {self.params['conv_filters']}")
        print(f"   LSTM Units: {self.params['lstm_units']}")
        print(f"   Dense Units: {self.params['dense_units']}")
        print(f"   Dropout: {self.params['dropout_rate']}")

    def build(self) -> Model:
        """
        Makaledeki mimariyi oluştur

        Input → Conv1D(30) → MaxPool → LSTM(120) → Dense(512) → Dropout → Output
        """
        print("\n🔨 Model mimarisi oluşturuluyor...")

        inputs = layers.Input(shape=self.input_shape, name="input")

        # Conv1D Layer - Feature extraction
        x = layers.Conv1D(
            filters=self.params["conv_filters"],
            kernel_size=self.params["kernel_size"],
            padding="same",
            activation="relu",
            name="conv1d",
        )(inputs)

        # MaxPooling
        x = layers.MaxPooling1D(pool_size=2, name="maxpool")(x)

        # LSTM Layer - Temporal pattern learning
        x = layers.LSTM(
            units=self.params["lstm_units"],
            activation="tanh",
            recurrent_activation="sigmoid",
            return_sequences=False,
            name="lstm",
        )(x)

        # Dense Layer
        x = layers.Dense(
            units=self.params["dense_units"], activation="relu", name="dense"
        )(x)

        # Dropout
        x = layers.Dropout(rate=self.params["dropout_rate"], name="dropout")(x)

        # Output Layer
        outputs = layers.Dense(
            units=self.num_classes, activation="softmax", name="output"
        )(x)

        self.model = Model(inputs=inputs, outputs=outputs, name="SSA_LSTMIDS")

        # Compile
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.params["learning_rate"]),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )

        print("✅ Model oluşturuldu!")
        print(f"   📊 Toplam parametre: {self.model.count_params():,}")

        return self.model

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        epochs: Optional[int] = None,
        batch_size: Optional[int] = None,
        save_path: Optional[str] = None,
        verbose: int = 1,
    ) -> Dict:
        """
        Modeli eğit

        Args:
            X_train: Eğitim verisi (samples, timesteps, features)
            y_train: Eğitim etiketleri
            X_val: Validation verisi
            y_val: Validation etiketleri
            epochs: Epoch sayısı (varsayılan: 300)
            batch_size: Batch boyutu (varsayılan: 120)
            save_path: Model kayıt yolu

        Returns:
            Eğitim sonuçları
        """
        if self.model is None:
            self.build()

        epochs = epochs or self.params["epochs"]
        batch_size = batch_size or self.params["batch_size"]

        print(f"\n🏋️ Eğitim başlıyor...")
        print(f"   Epochs: {epochs}")
        print(f"   Batch Size: {batch_size}")
        print(f"   Train samples: {len(X_train):,}")

        # Callbacks
        callbacks = [
            EarlyStopping(
                monitor="val_loss" if X_val is not None else "loss",
                patience=self.params["early_stopping_patience"],
                restore_best_weights=True,
                verbose=1,
            ),
            ReduceLROnPlateau(
                monitor="val_loss" if X_val is not None else "loss",
                factor=0.5,
                patience=3,
                min_lr=1e-6,
                verbose=1,
            ),
        ]

        if save_path:
            callbacks.append(
                ModelCheckpoint(
                    filepath=save_path,
                    monitor="val_loss" if X_val is not None else "loss",
                    save_best_only=True,
                    verbose=1,
                )
            )

        # Validation data
        validation_data = (X_val, y_val) if X_val is not None else None

        # Train
        self.history = self.model.fit(
            X_train,
            y_train,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=verbose,
        )

        # Results
        results = {
            "final_loss": float(self.history.history["loss"][-1]),
            "final_accuracy": float(self.history.history["accuracy"][-1]),
            "epochs_trained": len(self.history.history["loss"]),
            "params": self.params,
        }

        if X_val is not None:
            results["final_val_loss"] = float(self.history.history["val_loss"][-1])
            results["final_val_accuracy"] = float(
                self.history.history["val_accuracy"][-1]
            )

        print(f"\n✅ Eğitim tamamlandı!")
        print(f"   Epoch: {results['epochs_trained']}")
        print(f"   Accuracy: {results['final_accuracy']*100:.2f}%")
        if "final_val_accuracy" in results:
            print(f"   Val Accuracy: {results['final_val_accuracy']*100:.2f}%")

        return results

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict:
        """Model değerlendirmesi"""
        from sklearn.metrics import (
            accuracy_score,
            precision_score,
            recall_score,
            f1_score,
        )

        y_pred_proba = self.model.predict(X_test, verbose=0)
        y_pred = np.argmax(y_pred_proba, axis=1)

        return {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(
                y_test, y_pred, average="weighted", zero_division=0
            ),
            "recall": recall_score(y_test, y_pred, average="weighted", zero_division=0),
            "f1_score": f1_score(y_test, y_pred, average="weighted", zero_division=0),
        }

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Tahmin"""
        return np.argmax(self.model.predict(X, verbose=0), axis=1)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Olasılık tahmini"""
        return self.model.predict(X, verbose=0)

    def save(self, path: str):
        """Modeli kaydet"""
        self.model.save(path)

        # Parametreleri de kaydet
        params_path = path.replace(".h5", "_params.json").replace(
            ".keras", "_params.json"
        )
        with open(params_path, "w") as f:
            json.dump(self.params, f, indent=2)

        print(f"💾 Model kaydedildi: {path}")

    @classmethod
    def load(cls, path: str) -> "SSA_LSTMIDS":
        """Modeli yükle"""
        model = keras.models.load_model(path)

        # Parametreleri yükle
        params_path = path.replace(".h5", "_params.json").replace(
            ".keras", "_params.json"
        )
        if os.path.exists(params_path):
            with open(params_path, "r") as f:
                params = json.load(f)
        else:
            params = cls.PAPER_PARAMS

        instance = cls(
            input_shape=model.input_shape[1:],
            num_classes=model.output_shape[-1],
            use_paper_params=False,
            **params,
        )
        instance.model = model

        return instance

    def summary(self):
        """Model özeti"""
        if self.model:
            self.model.summary()


def optimize_with_ssa(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    num_classes: int,
    max_iterations: int = 20,
    population_size: int = 10,
) -> Tuple[Dict, float]:
    """
    SSA ile hiperparametre optimizasyonu

    Returns:
        (best_params, best_score)
    """
    from src.network_detection.optimizers.ssa import SSAOptimizer

    print("\n🦠 SSA Hiperparametre Optimizasyonu")
    print("=" * 50)

    def objective(params: Dict) -> float:
        model = SSA_LSTMIDS(
            input_shape=X_train.shape[1:],
            num_classes=num_classes,
            use_paper_params=False,
            conv_filters=params.get("conv_filters", 30),
            lstm_units=params.get("lstm_units", 120),
            dropout_rate=params.get("dropout_rate", 0.2),
            epochs=20,  # Optimizasyon için kısa eğitim
        )
        model.build()
        model.train(X_train, y_train, X_val, y_val, epochs=20, verbose=0)

        results = model.evaluate(X_val, y_val)

        del model
        keras.backend.clear_session()

        return results["f1_score"]

    search_space = {
        "conv_filters": (16, 64, "int"),
        "lstm_units": (64, 256, "int"),
        "dropout_rate": (0.1, 0.5, "float"),
    }

    optimizer = SSAOptimizer(
        objective_function=objective,
        search_space=search_space,
        population_size=population_size,
        max_iterations=max_iterations,
        minimize=False,
        verbose=True,
    )

    best_params, best_score = optimizer.optimize()

    return best_params, best_score


# Test
if __name__ == "__main__":
    print("🧪 SSA-LSTMIDS Test\n")

    # Test verisi
    X = np.random.rand(1000, 10, 41).astype(np.float32)
    y = np.random.randint(0, 5, 1000)

    # Split
    X_train, X_val = X[:800], X[800:]
    y_train, y_val = y[:800], y[800:]

    # Model
    model = SSA_LSTMIDS(input_shape=(10, 41), num_classes=5, use_paper_params=True)
    model.build()
    model.summary()

    # Training (kısa test)
    results = model.train(X_train, y_train, X_val, y_val, epochs=5)

    # Evaluate
    eval_results = model.evaluate(X_val, y_val)
    print(f"\n📊 Test Sonuçları:")
    for key, value in eval_results.items():
        print(f"   {key}: {value*100:.2f}%")

    print("\n✅ Test tamamlandı!")
