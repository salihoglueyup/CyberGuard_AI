"""
Transformer IDS - CyberGuard AI
===============================

Transformer tabanlı Intrusion Detection System.

Özellikler:
    - Positional Encoding
    - Multi-Head Self-Attention
    - Transformer Encoder
    - Time-Series Transformer
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from typing import Tuple, Optional, List
import logging

logger = logging.getLogger("TransformerIDS")


class PositionalEncoding(layers.Layer):
    """
    Positional Encoding Layer

    Paket sıralaması bilgisini encode eder.
    """

    def __init__(self, max_len: int = 5000, d_model: int = 64, **kwargs):
        super().__init__(**kwargs)
        self.max_len = max_len
        self.d_model = d_model

    def build(self, input_shape):
        # Pre-compute positional encoding
        position = np.arange(self.max_len)[:, np.newaxis]
        div_term = np.exp(
            np.arange(0, self.d_model, 2) * (-np.log(10000.0) / self.d_model)
        )

        pe = np.zeros((self.max_len, self.d_model))
        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)

        self.pe = tf.constant(pe[np.newaxis, :, :], dtype=tf.float32)
        super().build(input_shape)

    def call(self, x):
        seq_len = tf.shape(x)[1]
        return x + self.pe[:, :seq_len, : self.d_model]

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "max_len": self.max_len,
                "d_model": self.d_model,
            }
        )
        return config


class TransformerEncoderBlock(layers.Layer):
    """
    Transformer Encoder Block

    Multi-Head Attention + Feed Forward + Residual + LayerNorm
    """

    def __init__(
        self,
        d_model: int = 64,
        num_heads: int = 4,
        ff_dim: int = 256,
        dropout_rate: float = 0.1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.dropout_rate = dropout_rate

        # Multi-Head Attention
        self.mha = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=d_model // num_heads,
            dropout=dropout_rate,
        )

        # Feed Forward
        self.ffn = keras.Sequential(
            [
                layers.Dense(ff_dim, activation="gelu"),
                layers.Dropout(dropout_rate),
                layers.Dense(d_model),
                layers.Dropout(dropout_rate),
            ]
        )

        # Layer Normalization
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = layers.Dropout(dropout_rate)
        self.dropout2 = layers.Dropout(dropout_rate)

    def call(self, x, training=False, mask=None):
        # Multi-Head Attention + Residual
        attn_output = self.mha(x, x, attention_mask=mask, training=training)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)

        # Feed Forward + Residual
        ffn_output = self.ffn(out1, training=training)
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


class TransformerClassifier(layers.Layer):
    """
    Transformer Classifier Head

    Global pooling + MLP classifier
    """

    def __init__(
        self,
        num_classes: int,
        mlp_units: List[int] = [256, 128],
        dropout_rate: float = 0.2,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_classes = num_classes

        self.pooling = layers.GlobalAveragePooling1D()

        self.mlp = keras.Sequential()
        for units in mlp_units:
            self.mlp.add(layers.Dense(units, activation="gelu"))
            self.mlp.add(layers.Dropout(dropout_rate))

        self.output_layer = layers.Dense(num_classes, activation="softmax")

    def call(self, x, training=False):
        x = self.pooling(x)
        x = self.mlp(x, training=training)
        return self.output_layer(x)


# ============= Transformer Models =============


def build_transformer_ids(
    input_shape: Tuple[int, int],
    num_classes: int,
    d_model: int = 64,
    num_heads: int = 4,
    num_layers: int = 3,
    ff_dim: int = 256,
    dropout: float = 0.1,
) -> Model:
    """
    Transformer-based IDS Model

    Args:
        input_shape: (seq_len, features)
        num_classes: Sınıf sayısı
        d_model: Transformer dimension
        num_heads: Attention head sayısı
        num_layers: Encoder block sayısı
        ff_dim: Feed forward dimension
        dropout: Dropout rate
    """
    inputs = layers.Input(shape=input_shape)

    # Project to d_model dimension
    x = layers.Dense(d_model)(inputs)

    # Positional Encoding
    x = PositionalEncoding(max_len=input_shape[0], d_model=d_model)(x)
    x = layers.Dropout(dropout)(x)

    # Transformer Encoder blocks
    for i in range(num_layers):
        x = TransformerEncoderBlock(
            d_model=d_model,
            num_heads=num_heads,
            ff_dim=ff_dim,
            dropout_rate=dropout,
            name=f"encoder_{i}",
        )(x)

    # Classification head
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(256, activation="gelu")(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Dense(128, activation="gelu")(x)
    x = layers.Dropout(dropout)(x)

    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = Model(inputs, outputs, name="Transformer_IDS")
    return model


def build_hybrid_cnn_transformer(
    input_shape: Tuple[int, int],
    num_classes: int,
    conv_filters: int = 32,
    d_model: int = 64,
    num_heads: int = 4,
    num_layers: int = 2,
    dropout: float = 0.1,
) -> Model:
    """
    Hybrid CNN + Transformer Model

    CNN ile local feature extraction + Transformer ile global attention
    """
    inputs = layers.Input(shape=input_shape)

    # CNN block - local feature extraction
    x = layers.Conv1D(conv_filters, 3, activation="relu", padding="same")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Conv1D(conv_filters * 2, 3, activation="relu", padding="same")(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Dropout(dropout)(x)

    # Project to d_model
    x = layers.Dense(d_model)(x)

    # Positional Encoding
    x = PositionalEncoding(max_len=input_shape[0] // 2, d_model=d_model)(x)

    # Transformer Encoder blocks
    for i in range(num_layers):
        x = TransformerEncoderBlock(
            d_model=d_model,
            num_heads=num_heads,
            ff_dim=d_model * 4,
            dropout_rate=dropout,
        )(x)

    # Classification
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(256, activation="gelu")(x)
    x = layers.Dropout(dropout)(x)

    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = Model(inputs, outputs, name="CNN_Transformer")
    return model


def build_informer_ids(
    input_shape: Tuple[int, int],
    num_classes: int,
    d_model: int = 64,
    num_heads: int = 4,
    num_layers: int = 2,
    dropout: float = 0.1,
) -> Model:
    """
    Informer-inspired IDS Model

    Daha verimli attention mekanizması.
    Uzun sequence'lar için optimize.
    """
    inputs = layers.Input(shape=input_shape)

    # Feature embedding
    x = layers.Dense(d_model)(inputs)
    x = PositionalEncoding(max_len=input_shape[0], d_model=d_model)(x)

    # ProbSparse Attention simulation (simplified)
    # Use regular attention with pooling
    for i in range(num_layers):
        # Distilling - reduce sequence length
        attn_out = TransformerEncoderBlock(
            d_model=d_model,
            num_heads=num_heads,
            ff_dim=d_model * 2,
            dropout_rate=dropout,
        )(x)

        # Max pooling to reduce sequence (distilling layer)
        if i < num_layers - 1:
            attn_out = layers.MaxPooling1D(2, padding="same")(attn_out)

        x = attn_out

    # Classification
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(128, activation="gelu")(x)
    x = layers.Dropout(dropout)(x)

    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = Model(inputs, outputs, name="Informer_IDS")
    return model


# ============= Model Factory =============


def create_transformer_model(
    model_type: str, input_shape: Tuple[int, int], num_classes: int, **kwargs
) -> Model:
    """
    Transformer model factory

    Args:
        model_type: "transformer", "cnn_transformer", "informer"
        input_shape: (seq_len, features)
        num_classes: Output classes
    """
    models = {
        "transformer": build_transformer_ids,
        "cnn_transformer": build_hybrid_cnn_transformer,
        "informer": build_informer_ids,
    }

    if model_type not in models:
        raise ValueError(
            f"Unknown model type: {model_type}. Available: {list(models.keys())}"
        )

    return models[model_type](input_shape, num_classes, **kwargs)


# ============= Training Utilities =============


def create_transformer_scheduler(
    d_model: int = 64,
    warmup_steps: int = 4000,
) -> keras.optimizers.schedules.LearningRateSchedule:
    """
    Transformer learning rate schedule

    Warmup + decay
    """

    class TransformerSchedule(keras.optimizers.schedules.LearningRateSchedule):
        def __init__(self, d_model, warmup_steps):
            super().__init__()
            self.d_model = tf.cast(d_model, tf.float32)
            self.warmup_steps = warmup_steps

        def __call__(self, step):
            step = tf.cast(step, tf.float32)
            arg1 = tf.math.rsqrt(step)
            arg2 = step * (self.warmup_steps**-1.5)
            return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)

        def get_config(self):
            return {
                "d_model": int(self.d_model.numpy()),
                "warmup_steps": self.warmup_steps,
            }

    return TransformerSchedule(d_model, warmup_steps)


# Test
if __name__ == "__main__":
    print("🧪 Transformer IDS Test\n")

    # Test input
    batch_size = 4
    seq_len = 20
    features = 41
    num_classes = 5

    x = np.random.randn(batch_size, seq_len, features).astype(np.float32)

    # Test Positional Encoding
    print("📊 Positional Encoding Test:")
    pe = PositionalEncoding(max_len=100, d_model=64)
    x_proj = layers.Dense(64)(x)
    output = pe(x_proj)
    print(f"   Input: {x_proj.shape} → Output: {output.shape}")

    # Test Transformer Encoder
    print("\n📊 Transformer Encoder Test:")
    encoder = TransformerEncoderBlock(d_model=64, num_heads=4)
    output = encoder(x_proj)
    print(f"   Input: {x_proj.shape} → Output: {output.shape}")

    # Test models
    print("\n🤖 Transformer IDS Model:")
    model1 = build_transformer_ids((seq_len, features), num_classes)
    print(f"   Parameters: {model1.count_params():,}")

    print("\n🤖 CNN + Transformer Model:")
    model2 = build_hybrid_cnn_transformer((seq_len, features), num_classes)
    print(f"   Parameters: {model2.count_params():,}")

    print("\n🤖 Informer IDS Model:")
    model3 = build_informer_ids((seq_len, features), num_classes)
    print(f"   Parameters: {model3.count_params():,}")

    # Test prediction
    y_pred = model1.predict(x, verbose=0)
    print(f"\n✅ Prediction shape: {y_pred.shape}")
