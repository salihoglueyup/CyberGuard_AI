"""
Advanced IDS Model - BiLSTM + Attention
CyberGuard AI için gelişmiş anomali tespiti modeli

Mimari:
    Conv1D → BiLSTM → Self-Attention → Dense → Output

Referans:
    "An optimized LSTM-based deep learning model for anomaly network intrusion detection"
    Scientific Reports (2025) 15:1554

Geliştirmeler:
    - Bidirectional LSTM (hem ileri hem geri öğrenme)
    - Self-Attention mekanizması (önemli özelliklere odaklanma)
    - Dropout ve BatchNormalization (regularization)
"""

import os
import sys
import numpy as np
from typing import Dict, List, Optional, Tuple
from pathlib import Path

# TensorFlow import
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, Model
    from tensorflow.keras.callbacks import (
        EarlyStopping,
        ModelCheckpoint,
        ReduceLROnPlateau,
    )

    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    print("⚠️ TensorFlow bulunamadı!")


class SelfAttention(layers.Layer):
    """
    Self-Attention Layer

    Girdi dizisindeki önemli elementlere ağırlık verir.
    IDS'de önemli trafik özelliklerini vurgular.
    """

    def __init__(self, units: int = 128, **kwargs):
        super(SelfAttention, self).__init__(**kwargs)
        self.units = units
        self.W_q = layers.Dense(units, name="query")
        self.W_k = layers.Dense(units, name="key")
        self.W_v = layers.Dense(units, name="value")

    def call(self, inputs):
        # Query, Key, Value
        Q = self.W_q(inputs)  # (batch, seq, units)
        K = self.W_k(inputs)
        V = self.W_v(inputs)

        # Attention scores
        scores = tf.matmul(Q, K, transpose_b=True)  # (batch, seq, seq)
        scores = scores / tf.math.sqrt(tf.cast(self.units, tf.float32))

        # Softmax
        attention_weights = tf.nn.softmax(scores, axis=-1)

        # Weighted sum
        output = tf.matmul(attention_weights, V)  # (batch, seq, units)

        return output, attention_weights

    def get_config(self):
        config = super().get_config()
        config.update({"units": self.units})
        return config


class AdvancedIDSModel:
    """
    Gelişmiş IDS Modeli - BiLSTM + Attention

    Özellikler:
    - Bidirectional LSTM: Her iki yönden temporal pattern öğrenme
    - Self-Attention: Önemli özelliklere odaklanma
    - Conv1D: Yerel pattern extraction
    - Multiple layers: Derin hiyerarşik özellik öğrenme
    """

    ATTACK_TYPES = [
        "Normal",
        "DoS",
        "Probe",
        "R2L",
        "U2R",  # NSL-KDD
        "DDoS",
        "PortScan",
        "Bot",
        "Infiltration",  # CICIDS2017
        "Brute_Force",
        "XSS",
        "SQL_Injection",
        "Heartbleed",
        "Web_Attack",
        "Other",
    ]

    def __init__(
        self,
        input_shape: Tuple[int, int] = (10, 78),  # (sequence_length, features)
        num_classes: int = 15,
        lstm_units: int = 120,
        attention_units: int = 64,
        conv_filters: int = 30,
        dropout_rate: float = 0.3,
        use_attention: bool = True,
        use_bidirectional: bool = True,
        model_name: str = "BiLSTM_Attention_IDS",
    ):
        """
        Model parametreleri

        Args:
            input_shape: (sequence_length, features)
            num_classes: Sınıf sayısı
            lstm_units: LSTM hücre sayısı
            attention_units: Attention boyutu
            conv_filters: Conv1D filtre sayısı
            dropout_rate: Dropout oranı
            use_attention: Attention kullanılsın mı
            use_bidirectional: Bidirectional LSTM kullanılsın mı
        """
        if not TF_AVAILABLE:
            raise ImportError("TensorFlow gerekli! pip install tensorflow")

        self.input_shape = input_shape
        self.num_classes = num_classes
        self.lstm_units = lstm_units
        self.attention_units = attention_units
        self.conv_filters = conv_filters
        self.dropout_rate = dropout_rate
        self.use_attention = use_attention
        self.use_bidirectional = use_bidirectional
        self.model_name = model_name

        self.model: Optional[Model] = None
        self.history = None

        print(f"🧠 {model_name} başlatılıyor...")
        print(f"   📊 Input shape: {input_shape}")
        print(f"   🔢 Classes: {num_classes}")
        print(f"   🔄 BiLSTM: {'✅' if use_bidirectional else '❌'}")
        print(f"   👁️ Attention: {'✅' if use_attention else '❌'}")

    def build(self) -> Model:
        """
        Model mimarisini oluştur

        Mimari:
            Input → Conv1D → MaxPool → BiLSTM → Attention → Dense → Output
        """
        print("\n🔧 Model mimarisi oluşturuluyor...")

        # Input layer
        inputs = layers.Input(shape=self.input_shape, name="input")

        # 1. Conv1D - Pattern Extraction
        x = layers.Conv1D(
            filters=self.conv_filters,
            kernel_size=5,
            padding="same",
            activation="relu",
            name="conv1d_feature",
        )(inputs)
        x = layers.BatchNormalization(name="bn_conv")(x)
        x = layers.MaxPooling1D(pool_size=2, name="maxpool")(x)
        x = layers.Dropout(self.dropout_rate, name="dropout_conv")(x)

        # 2. LSTM Layer(s)
        if self.use_bidirectional:
            # Bidirectional LSTM
            x = layers.Bidirectional(
                layers.LSTM(
                    units=self.lstm_units,
                    return_sequences=self.use_attention,  # Attention için sequence gerekli
                    dropout=0.2,
                    recurrent_dropout=0.1,
                    name="lstm",
                ),
                name="bilstm",
            )(x)
        else:
            # Standard LSTM
            x = layers.LSTM(
                units=self.lstm_units,
                return_sequences=self.use_attention,
                dropout=0.2,
                name="lstm",
            )(x)

        # 3. Self-Attention (optional)
        if self.use_attention:
            attention_output, attention_weights = SelfAttention(
                units=self.attention_units, name="self_attention"
            )(x)
            x = layers.GlobalAveragePooling1D(name="gap")(attention_output)
            x = layers.Dropout(self.dropout_rate, name="dropout_attention")(x)

        # 4. Dense layers
        x = layers.Dense(512, activation="relu", name="dense_1")(x)
        x = layers.BatchNormalization(name="bn_dense")(x)
        x = layers.Dropout(self.dropout_rate, name="dropout_dense")(x)

        x = layers.Dense(256, activation="relu", name="dense_2")(x)
        x = layers.Dropout(0.2, name="dropout_dense_2")(x)

        # 5. Output layer
        outputs = layers.Dense(self.num_classes, activation="softmax", name="output")(x)

        # Model oluştur
        self.model = Model(inputs=inputs, outputs=outputs, name=self.model_name)

        # Compile
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )

        print("\n✅ Model oluşturuldu!")
        self._print_architecture()

        return self.model

    def _print_architecture(self):
        """Model mimarisini yazdır"""
        print("\n📐 Model Mimarisi:")
        print("=" * 50)
        print(f"   Conv1D: {self.conv_filters} filters, kernel=5")
        if self.use_bidirectional:
            print(f"   BiLSTM: {self.lstm_units * 2} units (2x{self.lstm_units})")
        else:
            print(f"   LSTM: {self.lstm_units} units")
        if self.use_attention:
            print(f"   Attention: {self.attention_units} units")
        print(f"   Dense: 512 → 256 → {self.num_classes}")
        print(f"   Dropout: {self.dropout_rate}")
        print("=" * 50)

        # Parametre sayısı
        total_params = self.model.count_params()
        print(f"   📊 Toplam parametre: {total_params:,}")

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        epochs: int = 100,
        batch_size: int = 64,
        patience: int = 10,
        model_save_path: Optional[str] = None,
    ) -> Dict:
        """
        Modeli eğit

        Args:
            X_train: Eğitim verisi
            y_train: Eğitim etiketleri
            X_val: Validation verisi (optional)
            y_val: Validation etiketleri (optional)
            epochs: Epoch sayısı
            batch_size: Batch boyutu
            patience: Early stopping patience
            model_save_path: Model kayıt yolu

        Returns:
            Eğitim metrikleri
        """
        if self.model is None:
            self.build()

        print(f"\n🏋️ Eğitim başlıyor...")
        print(f"   📊 Train: {X_train.shape}")
        print(f"   📊 Epochs: {epochs}, Batch: {batch_size}")

        # Callbacks
        callbacks = [
            EarlyStopping(
                monitor="val_loss" if X_val is not None else "loss",
                patience=patience,
                restore_best_weights=True,
                verbose=1,
            ),
            ReduceLROnPlateau(
                monitor="val_loss" if X_val is not None else "loss",
                factor=0.5,
                patience=5,
                min_lr=1e-6,
                verbose=1,
            ),
        ]

        if model_save_path:
            callbacks.append(
                ModelCheckpoint(
                    model_save_path,
                    monitor="val_accuracy" if X_val is not None else "accuracy",
                    save_best_only=True,
                    verbose=1,
                )
            )

        # Validation data
        validation_data = (X_val, y_val) if X_val is not None else None

        # Eğitim
        self.history = self.model.fit(
            X_train,
            y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=validation_data,
            callbacks=callbacks,
            verbose=1,
        )

        # Sonuçlar
        results = {
            "final_accuracy": float(self.history.history["accuracy"][-1]),
            "final_loss": float(self.history.history["loss"][-1]),
            "epochs_trained": len(self.history.history["accuracy"]),
        }

        if X_val is not None:
            results["final_val_accuracy"] = float(
                self.history.history["val_accuracy"][-1]
            )
            results["final_val_loss"] = float(self.history.history["val_loss"][-1])

        print(f"\n✅ Eğitim tamamlandı!")
        print(f"   📊 Accuracy: {results['final_accuracy']*100:.2f}%")
        if "final_val_accuracy" in results:
            print(f"   📊 Val Accuracy: {results['final_val_accuracy']*100:.2f}%")

        return results

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict:
        """Model değerlendirmesi"""
        if self.model is None:
            raise ValueError("Model henüz oluşturulmadı!")

        print("\n📈 Model değerlendiriliyor...")

        loss, accuracy = self.model.evaluate(X_test, y_test, verbose=0)

        # Predictions
        y_pred = self.model.predict(X_test, verbose=0)
        y_pred_classes = np.argmax(y_pred, axis=1)

        # Metrics
        from sklearn.metrics import (
            precision_score,
            recall_score,
            f1_score,
            confusion_matrix,
        )

        precision = precision_score(
            y_test, y_pred_classes, average="weighted", zero_division=0
        )
        recall = recall_score(
            y_test, y_pred_classes, average="weighted", zero_division=0
        )
        f1 = f1_score(y_test, y_pred_classes, average="weighted", zero_division=0)

        results = {
            "accuracy": float(accuracy),
            "loss": float(loss),
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1),
        }

        print(f"\n📊 Değerlendirme Sonuçları:")
        print(f"   Accuracy:  {accuracy*100:.2f}%")
        print(f"   Precision: {precision*100:.2f}%")
        print(f"   Recall:    {recall*100:.2f}%")
        print(f"   F1-Score:  {f1*100:.2f}%")

        return results

    def save(self, path: str):
        """Modeli kaydet"""
        if self.model is None:
            raise ValueError("Model henüz oluşturulmadı!")

        self.model.save(path)
        print(f"✅ Model kaydedildi: {path}")

    def load(self, path: str):
        """Modeli yükle"""
        self.model = keras.models.load_model(
            path, custom_objects={"SelfAttention": SelfAttention}
        )
        print(f"✅ Model yüklendi: {path}")

    def summary(self):
        """Model özeti"""
        if self.model:
            self.model.summary()


# Factory function
def create_bilstm_attention_model(
    input_shape: Tuple[int, int] = (10, 78), num_classes: int = 15, **kwargs
) -> AdvancedIDSModel:
    """
    BiLSTM + Attention modeli oluştur

    Args:
        input_shape: (sequence_length, features)
        num_classes: Sınıf sayısı
        **kwargs: Ek parametreler

    Returns:
        AdvancedIDSModel instance
    """
    model = AdvancedIDSModel(
        input_shape=input_shape,
        num_classes=num_classes,
        use_bidirectional=True,
        use_attention=True,
        **kwargs,
    )
    model.build()
    return model


def build_lstm_model(
    input_shape: Tuple[int, int], num_classes: int, lstm_units: int = 128
) -> Model:
    """
    Basit LSTM modeli oluştur (Keras Model döner)

    Args:
        input_shape: (sequence_length, features)
        num_classes: Sınıf sayısı
        lstm_units: LSTM hücre sayısı

    Returns:
        Compiled Keras Model
    """
    inputs = layers.Input(shape=input_shape, name="input")

    # Conv1D
    x = layers.Conv1D(32, kernel_size=3, padding="same", activation="relu")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(pool_size=2)(x)
    x = layers.Dropout(0.2)(x)

    # LSTM
    x = layers.LSTM(lstm_units, return_sequences=False, dropout=0.2)(x)

    # Dense
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(128, activation="relu")(x)

    # Output
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = Model(inputs=inputs, outputs=outputs, name="LSTM_IDS")

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    print(f"✅ LSTM modeli oluşturuldu! Parametre: {model.count_params():,}")
    return model


def build_bilstm_attention(
    input_shape: Tuple[int, int], num_classes: int, lstm_units: int = 128
) -> Model:
    """
    BiLSTM + Attention modeli oluştur (Keras Model döner)

    Args:
        input_shape: (sequence_length, features)
        num_classes: Sınıf sayısı
        lstm_units: LSTM hücre sayısı

    Returns:
        Compiled Keras Model
    """
    inputs = layers.Input(shape=input_shape, name="input")

    # Conv1D
    x = layers.Conv1D(32, kernel_size=3, padding="same", activation="relu")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(pool_size=2)(x)
    x = layers.Dropout(0.2)(x)

    # BiLSTM
    x = layers.Bidirectional(
        layers.LSTM(lstm_units, return_sequences=True, dropout=0.2)
    )(x)

    # Self-Attention
    attention_output, _ = SelfAttention(units=64)(x)
    x = layers.GlobalAveragePooling1D()(attention_output)
    x = layers.Dropout(0.3)(x)

    # Dense
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(128, activation="relu")(x)

    # Output
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = Model(inputs=inputs, outputs=outputs, name="BiLSTM_Attention_IDS")

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    print(
        f"✅ BiLSTM+Attention modeli oluşturuldu! Parametre: {model.count_params():,}"
    )
    return model


# Test
if __name__ == "__main__":
    print("🧪 Advanced IDS Model Test\n")

    # Test verisi
    X_dummy = np.random.rand(100, 10, 78).astype(np.float32)
    y_dummy = np.random.randint(0, 5, 100)

    # Model oluştur
    model = AdvancedIDSModel(
        input_shape=(10, 78),
        num_classes=5,
        lstm_units=64,
        use_bidirectional=True,
        use_attention=True,
    )

    model.build()
    model.summary()

    # Eğitim testi
    print("\n🏋️ Test eğitimi başlıyor...")
    results = model.train(X_dummy, y_dummy, epochs=3, batch_size=32)

    print("\n✅ Test tamamlandı!")
    print(f"   Results: {results}")
