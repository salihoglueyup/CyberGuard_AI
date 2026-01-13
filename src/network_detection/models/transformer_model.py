"""
Transformer IDS Model
CyberGuard AI için Transformer tabanlı anomali tespiti

Mimari:
    Input → Positional Encoding → Multi-Head Attention → FFN → Output

Avantajlar:
    - LSTM'den daha iyi uzun bağımlılık
    - Paralel işlem → hızlı eğitim
    - Self-attention ile önemli özelliklere odaklanma

Referans:
    "Attention Is All You Need" - Vaswani et al. (2017)
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from pathlib import Path

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


class PositionalEncoding(layers.Layer):
    """
    Positional Encoding Layer

    Transformer için pozisyon bilgisi ekler.
    Sinüzoidal encoding kullanır.
    """

    def __init__(self, max_len: int = 100, d_model: int = 128, **kwargs):
        super(PositionalEncoding, self).__init__(**kwargs)
        self.max_len = max_len
        self.d_model = d_model

        # Pozisyon encoding'i hesapla
        position = np.arange(max_len)[:, np.newaxis]
        div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))

        pe = np.zeros((max_len, d_model))
        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)

        self.pe = tf.constant(pe, dtype=tf.float32)

    def call(self, x):
        seq_len = tf.shape(x)[1]
        return x + self.pe[:seq_len, :]

    def get_config(self):
        config = super().get_config()
        config.update({"max_len": self.max_len, "d_model": self.d_model})
        return config


class TransformerBlock(layers.Layer):
    """
    Transformer Encoder Block

    Multi-Head Attention + Feed-Forward Network
    """

    def __init__(
        self,
        d_model: int = 128,
        num_heads: int = 8,
        ff_dim: int = 256,
        dropout_rate: float = 0.1,
        **kwargs,
    ):
        super(TransformerBlock, self).__init__(**kwargs)
        self.d_model = d_model
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.dropout_rate = dropout_rate

        # Multi-Head Attention
        self.attention = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=d_model // num_heads, dropout=dropout_rate
        )

        # Feed-Forward Network
        self.ffn = keras.Sequential(
            [layers.Dense(ff_dim, activation="relu"), layers.Dense(d_model)]
        )

        # Layer Normalization
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)

        # Dropout
        self.dropout1 = layers.Dropout(dropout_rate)
        self.dropout2 = layers.Dropout(dropout_rate)

    def call(self, x, training=False):
        # Multi-Head Attention
        attn_output = self.attention(x, x, training=training)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)

        # Feed-Forward Network
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)

        return out2

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "d_model": self.d_model,
                "num_heads": self.num_heads,
                "ff_dim": self.ff_dim,
                "dropout_rate": self.dropout_rate,
            }
        )
        return config


class TransformerIDSModel:
    """
    Transformer tabanlı IDS Modeli

    Mimari:
        Input → Linear Projection → Positional Encoding →
        Transformer Blocks → Global Average Pooling → Dense → Output
    """

    ATTACK_TYPES = [
        "Normal",
        "DoS",
        "Probe",
        "R2L",
        "U2R",
        "DDoS",
        "PortScan",
        "Bot",
        "Infiltration",
        "Brute_Force",
        "XSS",
        "SQL_Injection",
        "Heartbleed",
        "Web_Attack",
        "Other",
    ]

    def __init__(
        self,
        input_shape: Tuple[int, int] = (10, 78),
        num_classes: int = 15,
        d_model: int = 128,
        num_heads: int = 8,
        ff_dim: int = 256,
        num_transformer_blocks: int = 2,
        dropout_rate: float = 0.1,
        model_name: str = "Transformer_IDS",
    ):
        """
        Transformer IDS başlat

        Args:
            input_shape: (sequence_length, features)
            num_classes: Sınıf sayısı
            d_model: Transformer model boyutu
            num_heads: Attention head sayısı
            ff_dim: Feed-forward boyutu
            num_transformer_blocks: Transformer block sayısı
            dropout_rate: Dropout oranı
        """
        if not TF_AVAILABLE:
            raise ImportError("TensorFlow gerekli!")

        self.input_shape = input_shape
        self.num_classes = num_classes
        self.d_model = d_model
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.num_transformer_blocks = num_transformer_blocks
        self.dropout_rate = dropout_rate
        self.model_name = model_name

        self.model: Optional[Model] = None
        self.history = None

        print(f"🤖 {model_name} başlatılıyor...")
        print(f"   📊 Input shape: {input_shape}")
        print(f"   🔢 d_model: {d_model}, heads: {num_heads}")
        print(f"   📦 Transformer blocks: {num_transformer_blocks}")

    def build(self) -> Model:
        """Model mimarisini oluştur"""
        print("\n🔧 Transformer modeli oluşturuluyor...")

        # Input
        inputs = layers.Input(shape=self.input_shape, name="input")

        # Linear projection to d_model
        x = layers.Dense(self.d_model, name="projection")(inputs)

        # Positional Encoding
        x = PositionalEncoding(
            max_len=self.input_shape[0],
            d_model=self.d_model,
            name="positional_encoding",
        )(x)

        # Transformer Blocks
        for i in range(self.num_transformer_blocks):
            x = TransformerBlock(
                d_model=self.d_model,
                num_heads=self.num_heads,
                ff_dim=self.ff_dim,
                dropout_rate=self.dropout_rate,
                name=f"transformer_block_{i}",
            )(x)

        # Global Average Pooling
        x = layers.GlobalAveragePooling1D(name="gap")(x)

        # Dense layers
        x = layers.Dense(256, activation="relu", name="dense_1")(x)
        x = layers.Dropout(self.dropout_rate, name="dropout_1")(x)
        x = layers.Dense(128, activation="relu", name="dense_2")(x)
        x = layers.Dropout(self.dropout_rate / 2, name="dropout_2")(x)

        # Output
        outputs = layers.Dense(self.num_classes, activation="softmax", name="output")(x)

        self.model = Model(inputs=inputs, outputs=outputs, name=self.model_name)

        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.0001),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )

        print("\n✅ Transformer modeli oluşturuldu!")
        self._print_architecture()

        return self.model

    def _print_architecture(self):
        """Model mimarisini yazdır"""
        print("\n📐 Model Mimarisi:")
        print("=" * 50)
        print(f"   Projection: {self.input_shape[-1]} → {self.d_model}")
        print(f"   Positional Encoding: {self.input_shape[0]} positions")
        print(f"   Transformer Blocks: {self.num_transformer_blocks}")
        print(f"      - Heads: {self.num_heads}")
        print(f"      - FFN dim: {self.ff_dim}")
        print(f"   Dense: 256 → 128 → {self.num_classes}")
        print("=" * 50)

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
        """Modeli eğit"""
        if self.model is None:
            self.build()

        print(f"\n🏋️ Eğitim başlıyor...")
        print(f"   📊 Train: {X_train.shape}")

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
                min_lr=1e-7,
                verbose=1,
            ),
        ]

        if model_save_path:
            callbacks.append(
                ModelCheckpoint(model_save_path, save_best_only=True, verbose=1)
            )

        validation_data = (X_val, y_val) if X_val is not None else None

        self.history = self.model.fit(
            X_train,
            y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=validation_data,
            callbacks=callbacks,
            verbose=1,
        )

        results = {
            "final_accuracy": float(self.history.history["accuracy"][-1]),
            "final_loss": float(self.history.history["loss"][-1]),
            "epochs_trained": len(self.history.history["accuracy"]),
        }

        if X_val is not None:
            results["final_val_accuracy"] = float(
                self.history.history["val_accuracy"][-1]
            )

        print(f"\n✅ Eğitim tamamlandı!")
        print(f"   📊 Accuracy: {results['final_accuracy']*100:.2f}%")

        return results

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict:
        """Model değerlendirmesi"""
        if self.model is None:
            raise ValueError("Model oluşturulmadı!")

        from sklearn.metrics import precision_score, recall_score, f1_score

        loss, accuracy = self.model.evaluate(X_test, y_test, verbose=0)
        y_pred = np.argmax(self.model.predict(X_test, verbose=0), axis=1)

        precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
        recall = recall_score(y_test, y_pred, average="weighted", zero_division=0)
        f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

        return {
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1),
            "loss": float(loss),
        }

    def save(self, path: str):
        if self.model:
            self.model.save(path)
            print(f"✅ Model kaydedildi: {path}")

    def load(self, path: str):
        self.model = keras.models.load_model(
            path,
            custom_objects={
                "PositionalEncoding": PositionalEncoding,
                "TransformerBlock": TransformerBlock,
            },
        )
        print(f"✅ Model yüklendi: {path}")


# Test
if __name__ == "__main__":
    print("🧪 Transformer IDS Model Test\n")

    X_dummy = np.random.rand(100, 10, 78).astype(np.float32)
    y_dummy = np.random.randint(0, 5, 100)

    model = TransformerIDSModel(
        input_shape=(10, 78),
        num_classes=5,
        d_model=64,
        num_heads=4,
        num_transformer_blocks=2,
    )

    model.build()
    print("\n✅ Transformer model test başarılı!")
