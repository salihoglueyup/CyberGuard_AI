"""
TensorFlow Deep Learning Model - CyberGuard AI
Daha yüksek doğruluk için neural network kullanır
"""

import os

# Windows console encoding fix — use env var instead of overriding sys.stdout
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

import json
import os
import warnings
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import (
    confusion_matrix,
)
from tensorflow import keras
from tensorflow.keras import callbacks, layers, regularizers
from tensorflow.keras.models import Sequential, load_model

warnings.filterwarnings("ignore")

# GPU ayarları
physical_devices = tf.config.list_physical_devices("GPU")
if physical_devices:
    print(f"🎮 GPU bulundu: {len(physical_devices)} adet")
    for gpu in physical_devices:
        tf.config.experimental.set_memory_growth(gpu, True)
else:
    print("💻 CPU modunda çalışıyor")


class CyberThreatNeuralNetwork:
    """
    Gelişmiş Deep Learning modeli - Siber tehdit tespiti

    Özellikler:
    - Multi-layer neural network
    - Dropout ve BatchNormalization
    - Learning rate scheduling
    - Early stopping
    - Model checkpointing
    - TensorBoard logging
    """

    def __init__(
        self,
        input_dim: int = 8,
        hidden_layers: list[int] | None = None,
        dropout_rate: float = 0.3,
        l2_reg: float = 0.001,
        learning_rate: float = 0.001,
        activation: str = "relu",
        output_activation: str = "softmax",
    ):
        """
        Args:
            input_dim: Girdi özellik sayısı
            hidden_layers: Gizli katman nöron sayıları
            dropout_rate: Dropout oranı (overfitting önleme)
            l2_reg: L2 regularization katsayısı
            learning_rate: Öğrenme hızı
            activation: Aktivasyon fonksiyonu
            output_activation: Çıktı aktivasyonu
        """
        self.input_dim = input_dim
        self.hidden_layers = hidden_layers if hidden_layers is not None else [256, 128, 64, 32]
        self.dropout_rate = dropout_rate
        self.l2_reg = l2_reg
        self.learning_rate = learning_rate
        self.activation = activation
        self.output_activation = output_activation

        self.model = None
        self.history = None
        self.num_classes = None
        self.class_names = None

        print("🧠 TensorFlow Neural Network başlatıldı")
        print(f"   Mimari: {hidden_layers}")
        print(f"   Dropout: {dropout_rate}")
        print(f"   L2 Reg: {l2_reg}")
        print(f"   Learning Rate: {learning_rate}")

    def build_model(self, num_classes: int) -> keras.Model:
        """
        Neural network modelini oluştur

        Args:
            num_classes: Sınıf sayısı

        Returns:
            Keras model
        """
        self.num_classes = num_classes

        model = Sequential(name="CyberThreat_DNN")

        # Input layer
        model.add(layers.Input(shape=(self.input_dim,), name="input_features"))

        # Hidden layers ile Batch Normalization ve Dropout
        for i, units in enumerate(self.hidden_layers):
            # Dense layer
            model.add(
                layers.Dense(
                    units=units,
                    activation=self.activation,
                    kernel_regularizer=regularizers.l2(self.l2_reg),
                    name=f"dense_{i + 1}",
                )
            )

            # Batch Normalization (training stability için)
            model.add(layers.BatchNormalization(name=f"batch_norm_{i + 1}"))

            # Dropout (overfitting önleme)
            model.add(layers.Dropout(self.dropout_rate, name=f"dropout_{i + 1}"))

        # Output layer
        model.add(
            layers.Dense(num_classes, activation=self.output_activation, name="output")
        )

        # Model compile
        optimizer = keras.optimizers.Adam(learning_rate=self.learning_rate)

        model.compile(
            optimizer=optimizer,
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],  # Sadece accuracy - diğerleri problem çıkarıyor
        )

        self.model = model

        print("\n" + "=" * 70)
        print("🏗️  MODEL MİMARİSİ")
        print("=" * 70)
        model.summary()
        print("=" * 70 + "\n")

        return model

    def build_lstm_model(
        self,
        input_shape: tuple[int, int] = (1, 78),
        num_classes: int = 15,
        conv_filters: int = 30,
        lstm_units: int = 120,
        dense_units: int = 512,
    ) -> keras.Model:
        """
        Optimized LSTM-IDS Model (Makaleye Uygun)

        Mimari: Conv1D → MaxPool → LSTM → Dense → Output

        Ref: "An optimized LSTM-based deep learning model for anomaly network intrusion detection"
        Scientific Reports (2025) 15:1554

        Args:
            input_shape: (timesteps, features)
            num_classes: Sınıf sayısı
            conv_filters: Conv1D filter sayısı (default: 30)
            lstm_units: LSTM unit sayısı (default: 120)
            dense_units: Dense layer unit sayısı (default: 512)

        Returns:
            Compiled Keras model
        """
        self.num_classes = num_classes

        print("🔧 Optimized LSTM-IDS modeli oluşturuluyor...")

        model = Sequential(name="Optimized_LSTM_IDS")

        # 1. Conv1D Layer - Pattern Extraction
        model.add(
            layers.Conv1D(
                filters=conv_filters,
                kernel_size=5,
                padding="same",
                activation="relu",
                input_shape=input_shape,
                name="conv1d_pattern_extraction",
            )
        )

        # 2. MaxPooling - Dimensionality Reduction
        model.add(layers.MaxPooling1D(pool_size=2, name="maxpool_reduction"))

        # 3. LSTM Layer - Temporal Learning
        model.add(layers.LSTM(units=lstm_units, dropout=0.2, name="lstm_temporal"))

        # 4. Dense Layer - Feature Transformation
        model.add(
            layers.Dense(
                units=dense_units, activation="sigmoid", name="dense_transform"
            )
        )

        # 5. Output Layer - Classification
        model.add(
            layers.Dense(
                units=num_classes, activation="softmax", name="output_classification"
            )
        )

        # Compile with Adam optimizer
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )

        self.model = model

        print("\n" + "=" * 70)
        print("🏗️  OPTIMIZED LSTM-IDS MİMARİSİ")
        print("=" * 70)
        print(f"   📊 Conv1D: {conv_filters} filters, kernel=5, ReLU")
        print("   📊 MaxPooling: pool_size=2")
        print(f"   📊 LSTM: {lstm_units} units, dropout=0.2")
        print(f"   📊 Dense: {dense_units} units, sigmoid")
        print(f"   📊 Output: {num_classes} classes, softmax")
        print("=" * 70)
        model.summary()
        print("=" * 70 + "\n")

        return model

    def get_callbacks(
        self,
        model_path: str = "models/best_model.h5",
        tensorboard_dir: str = "logs/tensorboard",
        patience: int = 15,
    ) -> list[callbacks.Callback]:
        """
        Training callbacks oluştur

        Args:
            model_path: En iyi modelin kaydedileceği yer
            tensorboard_dir: TensorBoard log dizini
            patience: Early stopping için sabır

        Returns:
            Callback listesi
        """
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        os.makedirs(tensorboard_dir, exist_ok=True)

        callback_list = [
            # Early Stopping - Validation loss artmayı durdurduğunda dur
            callbacks.EarlyStopping(
                monitor="val_loss",
                patience=patience,
                restore_best_weights=True,
                verbose=1,
            ),
            # Model Checkpoint - En iyi modeli kaydet
            callbacks.ModelCheckpoint(
                filepath=model_path,
                monitor="val_accuracy",
                save_best_only=True,
                mode="max",
                verbose=1,
            ),
            # Learning Rate Reduction - Plateau'da öğrenme hızını azalt
            callbacks.ReduceLROnPlateau(
                monitor="val_loss", factor=0.5, patience=5, min_lr=1e-7, verbose=1
            ),
            # CSV Logger - Metrikleri kaydet
            callbacks.CSVLogger(
                filename="logs/training/training_log.csv", separator=",", append=False
            ),
        ]

        return callback_list

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray = None,
        y_val: np.ndarray = None,
        epochs: int = 100,
        batch_size: int = 32,
        class_names: list[str] = None,
        verbose: int = 1,
    ) -> dict:
        """
        Modeli eğit (Thread-safe version)

        Args:
            X_train: Eğitim verileri
            y_train: Eğitim etiketleri
            X_val: Validasyon verileri
            y_val: Validasyon etiketleri
            epochs: Epoch sayısı
            batch_size: Batch boyutu
            class_names: Sınıf isimleri
            verbose: Loglama seviyesi

        Returns:
            Training history
        """
        self.class_names = class_names

        # Model henüz oluşturulmadıysa oluştur
        if self.model is None:
            num_classes = len(np.unique(y_train))
            self.build_model(num_classes)

        print("\n" + "=" * 70)
        print("🎯 MODEL EĞİTİMİ BAŞLIYOR")
        print("=" * 70)
        print(f"📊 Eğitim verileri: {X_train.shape}")
        print(f"📊 Eğitim etiketleri: {y_train.shape}")
        if X_val is not None:
            print(f"📊 Validasyon verileri: {X_val.shape}")
        print(f"⚙️  Epochs: {epochs}")
        print(f"⚙️  Batch size: {batch_size}")
        print("=" * 70 + "\n")

        # CRITICAL FIX: TensorFlow session'ı temizle
        import gc

        tf.keras.backend.clear_session()
        gc.collect()

        # Basit callbacks - threading sorunlarını önlemek için
        callbacks_list = []

        # Early stopping
        early_stop = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=15, restore_best_weights=True, verbose=1
        )
        callbacks_list.append(early_stop)

        # Reduce LR
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=5, min_lr=1e-7, verbose=1
        )
        callbacks_list.append(reduce_lr)

        # Eğitim
        start_time = datetime.now()

        try:
            # Model fit - threading safe
            self.history = self.model.fit(
                X_train,
                y_train,
                validation_data=(X_val, y_val) if X_val is not None else None,
                epochs=epochs,
                batch_size=batch_size,
                callbacks=callbacks_list,
                verbose=verbose,
                shuffle=True,
            )

        except Exception as e:
            print(f"\n⚠️ Training hatası: {e}")
            print("🔄 Yeniden deneniyor (basit mod)...")

            # Fallback: Callback'siz dene
            self.history = self.model.fit(
                X_train,
                y_train,
                validation_data=(X_val, y_val) if X_val is not None else None,
                epochs=epochs,
                batch_size=batch_size,
                callbacks=[],  # Callback'siz
                verbose=0,  # Sessiz mod
                shuffle=True,
            )

        end_time = datetime.now()
        training_time = (end_time - start_time).total_seconds()

        print("\n" + "=" * 70)
        print("✅ EĞİTİM TAMAMLANDI!")
        print("=" * 70)
        print(f"⏱️  Süre: {training_time:.2f} saniye ({training_time / 60:.2f} dakika)")
        print(
            f"📈 Final Train Accuracy: {self.history.history['accuracy'][-1] * 100:.2f}%"
        )
        if X_val is not None:
            print(
                f"📈 Final Val Accuracy: {self.history.history['val_accuracy'][-1] * 100:.2f}%"
            )
        print("=" * 70 + "\n")

        return self.history.history

    def predict(self, X: np.ndarray, return_proba: bool = False) -> np.ndarray:
        """
        Tahmin yap

        Args:
            X: Girdi verileri
            return_proba: Olasılıkları döndür

        Returns:
            Tahminler veya olasılıklar
        """
        predictions = self.model.predict(X, verbose=0)

        if return_proba:
            return predictions

        return np.argmax(predictions, axis=1)

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> dict:
        """
        Modeli değerlendir

        Args:
            X: Test verileri
            y: Test etiketleri

        Returns:
            Dict: Değerlendirme metrikleri (loss, accuracy)
        """
        if self.model is None:
            raise ValueError("Model henüz eğitilmemiş!")

        results = self.model.evaluate(X, y, verbose=0)

        # Keras model.evaluate [loss, accuracy, ...] döner
        metrics = {"loss": float(results[0]), "accuracy": float(results[1])}

        print("📊 Evaluation Results:")
        print(f"   Loss:     {metrics['loss']:.4f}")
        print(f"   Accuracy: {metrics['accuracy']*100:.2f}%")

        return metrics

    def plot_training_history(
        self,
        save_path: str = "models/training_history.png",
        figsize: tuple[int, int] = (15, 10),
    ):
        """
        Eğitim geçmişini görselleştir

        Args:
            save_path: Kaydedilecek dosya yolu
            figsize: Figure boyutu
        """
        if self.history is None:
            print("⚠️  Henüz eğitim yapılmamış!")
            return

        history = self.history.history

        fig, axes = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle("Model Training History", fontsize=16, fontweight="bold")

        # Accuracy
        axes[0, 0].plot(history["accuracy"], label="Train", linewidth=2)
        if "val_accuracy" in history:
            axes[0, 0].plot(history["val_accuracy"], label="Validation", linewidth=2)
        axes[0, 0].set_title("Model Accuracy", fontweight="bold")
        axes[0, 0].set_xlabel("Epoch")
        axes[0, 0].set_ylabel("Accuracy")
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # Loss
        axes[0, 1].plot(history["loss"], label="Train", linewidth=2)
        if "val_loss" in history:
            axes[0, 1].plot(history["val_loss"], label="Validation", linewidth=2)
        axes[0, 1].set_title("Model Loss", fontweight="bold")
        axes[0, 1].set_xlabel("Epoch")
        axes[0, 1].set_ylabel("Loss")
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # Precision
        if "precision" in history:
            axes[1, 0].plot(history["precision"], label="Train", linewidth=2)
            if "val_precision" in history:
                axes[1, 0].plot(
                    history["val_precision"], label="Validation", linewidth=2
                )
            axes[1, 0].set_title("Model Precision", fontweight="bold")
            axes[1, 0].set_xlabel("Epoch")
            axes[1, 0].set_ylabel("Precision")
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)

        # Recall
        if "recall" in history:
            axes[1, 1].plot(history["recall"], label="Train", linewidth=2)
            if "val_recall" in history:
                axes[1, 1].plot(history["val_recall"], label="Validation", linewidth=2)
            axes[1, 1].set_title("Model Recall", fontweight="bold")
            axes[1, 1].set_xlabel("Epoch")
            axes[1, 1].set_ylabel("Recall")
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"✅ Training history kaydedildi: {save_path}")
        plt.close()

    def plot_confusion_matrix(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        save_path: str = "models/confusion_matrix.png",
        figsize: tuple[int, int] = (10, 8),
    ):
        """
        Confusion matrix görselleştir

        Args:
            y_true: Gerçek etiketler
            y_pred: Tahmin edilen etiketler
            save_path: Kaydedilecek dosya yolu
            figsize: Figure boyutu
        """
        cm = confusion_matrix(y_true, y_pred)

        plt.figure(figsize=figsize)
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=self.class_names if self.class_names else range(len(cm)),
            yticklabels=self.class_names if self.class_names else range(len(cm)),
            cbar_kws={"label": "Count"},
        )
        plt.title("Confusion Matrix", fontsize=14, fontweight="bold")
        plt.xlabel("Predicted Label", fontweight="bold")
        plt.ylabel("True Label", fontweight="bold")
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"✅ Confusion matrix kaydedildi: {save_path}")
        plt.close()

    def save(
        self, model_path: str = "models/tensorflow_model.h5", save_format: str = "h5"
    ):
        """
        Modeli kaydet

        Args:
            model_path: Model dosya yolu
            save_format: Kayıt formatı ('h5' veya 'tf')
        """
        os.makedirs(os.path.dirname(model_path), exist_ok=True)

        # Model kaydet
        self.model.save(model_path, save_format=save_format)

        # Metadata kaydet
        metadata = {
            "input_dim": self.input_dim,
            "hidden_layers": self.hidden_layers,
            "dropout_rate": self.dropout_rate,
            "l2_reg": self.l2_reg,
            "learning_rate": self.learning_rate,
            "activation": self.activation,
            "output_activation": self.output_activation,
            "num_classes": self.num_classes,
            "class_names": self.class_names,
            "saved_at": datetime.now().isoformat(),
        }

        metadata_path = model_path.replace(".h5", "_metadata.json")
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=4)

        print(f"✅ Model kaydedildi: {model_path}")
        print(f"✅ Metadata kaydedildi: {metadata_path}")

    @classmethod
    def load(
        cls, model_path: str = "models/tensorflow_model.h5"
    ) -> "CyberThreatNeuralNetwork":
        """
        Modeli yükle

        Args:
            model_path: Model dosya yolu

        Returns:
            CyberThreatNeuralNetwork instance
        """
        # Metadata yükle
        metadata_path = model_path.replace(".h5", "_metadata.json")
        with open(metadata_path) as f:
            metadata = json.load(f)

        # Instance oluştur
        instance = cls(
            input_dim=metadata["input_dim"],
            hidden_layers=metadata["hidden_layers"],
            dropout_rate=metadata["dropout_rate"],
            l2_reg=metadata["l2_reg"],
            learning_rate=metadata["learning_rate"],
            activation=metadata["activation"],
            output_activation=metadata["output_activation"],
        )

        # Modeli yükle
        instance.model = load_model(model_path)
        instance.num_classes = metadata["num_classes"]
        instance.class_names = metadata["class_names"]

        print(f"✅ Model yüklendi: {model_path}")
        print(f"✅ Metadata yüklendi: {metadata_path}")

        return instance


# Örnek kullanım
if __name__ == "__main__":
    print("🧪 TensorFlow Model Test")
    print("=" * 70)

    # Örnek veri
    X_train = np.random.rand(1000, 8)
    y_train = np.random.randint(0, 5, 1000)
    X_test = np.random.rand(200, 8)
    y_test = np.random.randint(0, 5, 200)

    class_names = ["DDoS", "SQL Injection", "XSS", "Port Scan", "Brute Force"]

    # Model oluştur
    model = CyberThreatNeuralNetwork(
        input_dim=8,
        hidden_layers=[256, 128, 64, 32],
        dropout_rate=0.3,
        learning_rate=0.001,
    )

    # Eğit
    model.train(
        X_train,
        y_train,
        X_val=X_test,
        y_val=y_test,
        epochs=50,
        batch_size=32,
        class_names=class_names,
    )

    # Değerlendir
    metrics = model.evaluate(X_test, y_test)

    # Görselleştir
    model.plot_training_history()
    y_pred = model.predict(X_test)
    model.plot_confusion_matrix(y_test, y_pred)

    # Kaydet
    model.save("models/test_model.h5")

    print("\n✅ Test tamamlandı!")
