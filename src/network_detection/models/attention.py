"""
Attention Mechanisms - CyberGuard AI
====================================

Self-Attention ve Multi-Head Attention layerları.

Özellikler:
    - Self-Attention
    - Multi-Head Attention
    - Attention visualisation
    - LSTM/BiLSTM ile entegrasyon
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from typing import Tuple, Optional, List
import logging

logger = logging.getLogger("Attention")


class SelfAttention(layers.Layer):
    """
    Self-Attention Layer

    Önemli özelliklere yüksek ağırlık verir.
    IDS için: kritik trafik pattern'lerini vurgular.
    """

    def __init__(self, units: int = 64, return_attention: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.return_attention = return_attention

        # Attention weights
        self.W_query = layers.Dense(units, use_bias=False)
        self.W_key = layers.Dense(units, use_bias=False)
        self.W_value = layers.Dense(units, use_bias=False)

    def call(self, inputs, mask=None):
        """
        Args:
            inputs: (batch, seq_len, features)

        Returns:
            output: (batch, seq_len, features)
            attention_weights: (batch, seq_len, seq_len) if return_attention
        """
        # Query, Key, Value
        Q = self.W_query(inputs)  # (batch, seq, units)
        K = self.W_key(inputs)
        V = self.W_value(inputs)

        # Attention scores
        d_k = tf.cast(tf.shape(K)[-1], tf.float32)
        scores = tf.matmul(Q, K, transpose_b=True) / tf.math.sqrt(d_k)

        # Masking (optional)
        if mask is not None:
            scores += mask * -1e9

        # Softmax
        attention_weights = tf.nn.softmax(scores, axis=-1)

        # Weighted sum
        output = tf.matmul(attention_weights, V)

        if self.return_attention:
            return output, attention_weights
        return output

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "units": self.units,
                "return_attention": self.return_attention,
            }
        )
        return config


class MultiHeadAttention(layers.Layer):
    """
    Multi-Head Self-Attention Layer

    Birden fazla attention head ile farklı pattern'leri öğrenir.
    """

    def __init__(
        self,
        num_heads: int = 8,
        d_model: int = 64,
        dropout_rate: float = 0.1,
        return_attention: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_heads = num_heads
        self.d_model = d_model
        self.return_attention = return_attention

        assert d_model % num_heads == 0
        self.depth = d_model // num_heads

        self.W_query = layers.Dense(d_model)
        self.W_key = layers.Dense(d_model)
        self.W_value = layers.Dense(d_model)

        self.dense = layers.Dense(d_model)
        self.dropout = layers.Dropout(dropout_rate)
        self.layernorm = layers.LayerNormalization(epsilon=1e-6)

    def split_heads(self, x, batch_size):
        """Split last dimension into (num_heads, depth)"""
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])  # (batch, heads, seq, depth)

    def call(self, inputs, training=False, mask=None):
        batch_size = tf.shape(inputs)[0]

        # Linear projections
        Q = self.W_query(inputs)
        K = self.W_key(inputs)
        V = self.W_value(inputs)

        # Split heads
        Q = self.split_heads(Q, batch_size)
        K = self.split_heads(K, batch_size)
        V = self.split_heads(V, batch_size)

        # Scaled dot-product attention
        d_k = tf.cast(self.depth, tf.float32)
        scores = tf.matmul(Q, K, transpose_b=True) / tf.math.sqrt(d_k)

        if mask is not None:
            scores += mask * -1e9

        attention_weights = tf.nn.softmax(scores, axis=-1)
        attention_weights = self.dropout(attention_weights, training=training)

        # Apply attention
        output = tf.matmul(attention_weights, V)

        # Concat heads
        output = tf.transpose(output, perm=[0, 2, 1, 3])
        output = tf.reshape(output, (batch_size, -1, self.d_model))

        # Final linear + residual
        output = self.dense(output)
        output = self.dropout(output, training=training)
        output = self.layernorm(inputs + output)  # Residual connection

        if self.return_attention:
            return output, attention_weights
        return output

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "num_heads": self.num_heads,
                "d_model": self.d_model,
                "return_attention": self.return_attention,
            }
        )
        return config


class AttentionPooling(layers.Layer):
    """
    Attention-based Pooling Layer

    Sequence'i tek bir vektöre sıkıştırır.
    Global average pooling yerine kullanılabilir.
    """

    def __init__(self, units: int = 64, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.attention = layers.Dense(1, activation="tanh")

    def call(self, inputs):
        # inputs: (batch, seq_len, features)

        # Attention scores
        scores = self.attention(inputs)  # (batch, seq_len, 1)
        weights = tf.nn.softmax(scores, axis=1)

        # Weighted sum
        output = tf.reduce_sum(inputs * weights, axis=1)  # (batch, features)

        return output

    def get_config(self):
        config = super().get_config()
        config.update({"units": self.units})
        return config


# ============= Attention-based Models =============


def build_attention_lstm(
    input_shape: Tuple[int, int],
    num_classes: int,
    lstm_units: int = 128,
    attention_units: int = 64,
    num_heads: int = 4,
    dropout: float = 0.2,
) -> Model:
    """
    LSTM + Self-Attention Model

    Args:
        input_shape: (seq_len, features)
        num_classes: Sınıf sayısı
        lstm_units: LSTM unit sayısı
        attention_units: Attention dimension
        num_heads: Multi-head sayısı
        dropout: Dropout rate
    """
    inputs = layers.Input(shape=input_shape)

    # LSTM
    x = layers.LSTM(lstm_units, return_sequences=True, dropout=dropout)(inputs)

    # Self-Attention
    x = SelfAttention(units=attention_units)(x)

    # Pooling
    x = AttentionPooling()(x)

    # Dense
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(dropout)(x)

    # Output
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = Model(inputs, outputs, name="AttentionLSTM")
    return model


def build_bilstm_attention(
    input_shape: Tuple[int, int],
    num_classes: int,
    lstm_units: int = 128,
    attention_units: int = 64,
    dropout: float = 0.2,
) -> Model:
    """
    BiLSTM + Multi-Head Attention Model

    State-of-the-art IDS mimarisi.
    """
    inputs = layers.Input(shape=input_shape)

    # BiLSTM
    x = layers.Bidirectional(
        layers.LSTM(lstm_units, return_sequences=True, dropout=dropout)
    )(inputs)

    # Multi-Head Attention
    x = MultiHeadAttention(
        num_heads=4,
        d_model=lstm_units * 2,  # BiLSTM output is doubled
        dropout_rate=dropout,
    )(x)

    # Pooling
    x = layers.GlobalAveragePooling1D()(x)

    # Dense
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(dropout)(x)

    # Output
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = Model(inputs, outputs, name="BiLSTM_Attention")
    return model


def build_cnn_bilstm_attention(
    input_shape: Tuple[int, int],
    num_classes: int,
    conv_filters: int = 32,
    lstm_units: int = 128,
    num_heads: int = 4,
    dropout: float = 0.2,
) -> Model:
    """
    CNN + BiLSTM + Multi-Head Attention

    En güçlü mimari: Uzamsal + Zamansal + Attention
    """
    inputs = layers.Input(shape=input_shape)

    # CNN block
    x = layers.Conv1D(conv_filters, 3, activation="relu", padding="same")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Conv1D(conv_filters * 2, 3, activation="relu", padding="same")(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Dropout(dropout)(x)

    # BiLSTM
    x = layers.Bidirectional(
        layers.LSTM(lstm_units, return_sequences=True, dropout=dropout)
    )(x)

    # Multi-Head Attention
    x = MultiHeadAttention(
        num_heads=num_heads,
        d_model=lstm_units * 2,
        dropout_rate=dropout,
    )(x)

    # Attention Pooling
    x = AttentionPooling()(x)

    # Dense
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Dense(128, activation="relu")(x)

    # Output
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = Model(inputs, outputs, name="CNN_BiLSTM_Attention")
    return model


# ============= Attention Visualization =============


def get_attention_weights(
    model: Model, inputs: np.ndarray, layer_name: str = None
) -> Optional[np.ndarray]:
    """
    Model'den attention weight'lerini çıkar

    Args:
        model: Keras model
        inputs: Input data
        layer_name: Attention layer adı (None = otomatik bul)

    Returns:
        attention_weights: (batch, heads, seq, seq) veya (batch, seq, seq)
    """
    # Find attention layer
    attention_layer = None
    for layer in model.layers:
        if isinstance(layer, (SelfAttention, MultiHeadAttention)):
            if layer_name is None or layer.name == layer_name:
                attention_layer = layer
                break

    if attention_layer is None:
        logger.warning("Attention layer bulunamadı")
        return None

    # Create model to output attention weights
    # (Requires layer to have return_attention=True)
    try:
        attention_output = attention_layer.output
        if isinstance(attention_output, tuple):
            weights = attention_output[1]
        else:
            logger.warning("Layer return_attention=False")
            return None

        attention_model = Model(inputs=model.input, outputs=weights)

        return attention_model.predict(inputs, verbose=0)

    except Exception as e:
        logger.error(f"Attention weights çıkarılamadı: {e}")
        return None


def visualize_attention(
    attention_weights: np.ndarray,
    feature_names: List[str] = None,
    sample_idx: int = 0,
) -> dict:
    """
    Attention weights görselleştirme verisi

    Returns:
        dict: Plotly için veri
    """
    if attention_weights is None:
        return {"error": "No attention weights"}

    weights = attention_weights[sample_idx]

    # Multi-head ise ortalama al
    if len(weights.shape) == 3:  # (heads, seq, seq)
        weights = np.mean(weights, axis=0)

    seq_len = weights.shape[0]
    labels = feature_names or [f"pos_{i}" for i in range(seq_len)]

    return {
        "heatmap": weights.tolist(),
        "x_labels": labels[:seq_len],
        "y_labels": labels[:seq_len],
        "title": "Attention Weights",
    }


# Test
if __name__ == "__main__":
    print("🧪 Attention Layers Test\n")

    # Test input
    batch_size = 4
    seq_len = 10
    features = 41
    num_classes = 5

    x = np.random.randn(batch_size, seq_len, features).astype(np.float32)

    # Test Self-Attention
    print("📊 Self-Attention Test:")
    attention = SelfAttention(units=64)
    output = attention(x)
    print(f"   Input: {x.shape} → Output: {output.shape}")

    # Test Multi-Head Attention
    print("\n📊 Multi-Head Attention Test:")
    mha = MultiHeadAttention(num_heads=4, d_model=64)
    output = mha(x)
    print(f"   Input: {x.shape} → Output: {output.shape}")

    # Test models
    print("\n🤖 Attention LSTM Model:")
    model1 = build_attention_lstm((seq_len, features), num_classes)
    model1.summary()

    print("\n🤖 BiLSTM + Attention Model:")
    model2 = build_bilstm_attention((seq_len, features), num_classes)
    print(f"   Parameters: {model2.count_params():,}")

    print("\n🤖 CNN + BiLSTM + Attention Model:")
    model3 = build_cnn_bilstm_attention((seq_len, features), num_classes)
    print(f"   Parameters: {model3.count_params():,}")

    # Test prediction
    y_pred = model3.predict(x, verbose=0)
    print(f"\n✅ Prediction shape: {y_pred.shape}")
