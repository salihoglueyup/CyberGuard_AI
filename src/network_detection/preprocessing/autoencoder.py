"""
Autoencoder Feature Embedding
CyberGuard AI için feature extraction ve dimensionality reduction

Avantajlar:
    - Gürültülü verileri temizler
    - Feature boyutunu azaltır
    - Latent representation öğrenir
    - Anomali tespiti için kullanılabilir

Mimari:
    Input → Encoder → Latent Space → Decoder → Reconstruction
"""

import numpy as np
from typing import Dict, Optional, Tuple

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, Model

    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False


class FeatureAutoencoder:
    """
    Autoencoder tabanlı feature embedding

    IDS verilerini temizler ve boyut azaltır.
    Latent space anomali tespiti için kullanılabilir.
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 32,
        hidden_layers: list = [128, 64],
        dropout_rate: float = 0.2,
        activation: str = "relu",
    ):
        if not TF_AVAILABLE:
            raise ImportError("TensorFlow gerekli!")

        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_layers = hidden_layers
        self.dropout_rate = dropout_rate
        self.activation = activation

        self.autoencoder: Optional[Model] = None
        self.encoder: Optional[Model] = None
        self.decoder: Optional[Model] = None

        print(f"🔮 Autoencoder başlatılıyor...")
        print(f"   Input: {input_dim} → Latent: {latent_dim}")

    def build(self) -> Tuple[Model, Model, Model]:
        """Autoencoder mimarisini oluştur"""

        # Encoder
        encoder_inputs = layers.Input(shape=(self.input_dim,), name="encoder_input")
        x = encoder_inputs

        for i, units in enumerate(self.hidden_layers):
            x = layers.Dense(
                units, activation=self.activation, name=f"encoder_dense_{i}"
            )(x)
            x = layers.BatchNormalization()(x)
            x = layers.Dropout(self.dropout_rate)(x)

        latent = layers.Dense(
            self.latent_dim, activation=self.activation, name="latent"
        )(x)
        self.encoder = Model(encoder_inputs, latent, name="encoder")

        # Decoder
        decoder_inputs = layers.Input(shape=(self.latent_dim,), name="decoder_input")
        y = decoder_inputs

        for i, units in enumerate(reversed(self.hidden_layers)):
            y = layers.Dense(
                units, activation=self.activation, name=f"decoder_dense_{i}"
            )(y)
            y = layers.BatchNormalization()(y)
            y = layers.Dropout(self.dropout_rate)(y)

        decoder_outputs = layers.Dense(
            self.input_dim, activation="linear", name="decoder_output"
        )(y)
        self.decoder = Model(decoder_inputs, decoder_outputs, name="decoder")

        # Full Autoencoder
        autoencoder_outputs = self.decoder(self.encoder(encoder_inputs))
        self.autoencoder = Model(
            encoder_inputs, autoencoder_outputs, name="autoencoder"
        )

        self.autoencoder.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001), loss="mse"
        )

        print(f"✅ Autoencoder oluşturuldu!")
        print(f"   Encoder params: {self.encoder.count_params():,}")

        return self.autoencoder, self.encoder, self.decoder

    def fit(self, X, epochs=50, batch_size=64, validation_split=0.1):
        """Autoencoder'ı eğit"""
        if self.autoencoder is None:
            self.build()

        print(f"\n🏋️ Autoencoder eğitiliyor...")

        history = self.autoencoder.fit(
            X,
            X,  # Input = Output (reconstruction)
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            verbose=1,
            callbacks=[
                keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)
            ],
        )

        return history

    def encode(self, X) -> np.ndarray:
        """Veriyi latent space'e encode et"""
        if self.encoder is None:
            raise ValueError("Model henüz eğitilmedi!")
        return self.encoder.predict(X, verbose=0)

    def decode(self, Z) -> np.ndarray:
        """Latent space'den decode et"""
        if self.decoder is None:
            raise ValueError("Model henüz eğitilmedi!")
        return self.decoder.predict(Z, verbose=0)

    def get_reconstruction_error(self, X) -> np.ndarray:
        """Reconstruction error hesapla (anomali tespiti için)"""
        X_reconstructed = self.autoencoder.predict(X, verbose=0)
        errors = np.mean(np.square(X - X_reconstructed), axis=1)
        return errors

    def detect_anomalies(self, X, threshold: float = None) -> np.ndarray:
        """Anomali tespiti"""
        errors = self.get_reconstruction_error(X)

        if threshold is None:
            threshold = np.percentile(errors, 95)  # Top 5% anomali

        return errors > threshold


class VariationalAutoencoder(FeatureAutoencoder):
    """
    Variational Autoencoder (VAE)

    Daha güçlü latent representation.
    Generative özellik.
    """

    def __init__(self, input_dim, latent_dim=32, **kwargs):
        super().__init__(input_dim, latent_dim, **kwargs)
        self.z_mean = None
        self.z_log_var = None

    def _sampling(self, args):
        """Reparameterization trick"""
        z_mean, z_log_var = args
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.random.normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

    def build(self):
        # Encoder
        encoder_inputs = layers.Input(shape=(self.input_dim,))
        x = encoder_inputs

        for units in self.hidden_layers:
            x = layers.Dense(units, activation="relu")(x)
            x = layers.BatchNormalization()(x)

        z_mean = layers.Dense(self.latent_dim, name="z_mean")(x)
        z_log_var = layers.Dense(self.latent_dim, name="z_log_var")(x)

        z = layers.Lambda(self._sampling, name="z")([z_mean, z_log_var])

        self.encoder = Model(encoder_inputs, [z_mean, z_log_var, z], name="vae_encoder")

        # Decoder
        latent_inputs = layers.Input(shape=(self.latent_dim,))
        y = latent_inputs

        for units in reversed(self.hidden_layers):
            y = layers.Dense(units, activation="relu")(y)
            y = layers.BatchNormalization()(y)

        decoder_outputs = layers.Dense(self.input_dim, activation="linear")(y)
        self.decoder = Model(latent_inputs, decoder_outputs, name="vae_decoder")

        # VAE
        outputs = self.decoder(self.encoder(encoder_inputs)[2])
        self.autoencoder = Model(encoder_inputs, outputs, name="vae")

        # VAE Loss
        reconstruction_loss = tf.reduce_mean(tf.square(encoder_inputs - outputs))
        kl_loss = -0.5 * tf.reduce_mean(
            1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
        )
        vae_loss = reconstruction_loss + 0.001 * kl_loss

        self.autoencoder.add_loss(vae_loss)
        self.autoencoder.compile(optimizer="adam")

        print(f"✅ VAE oluşturuldu!")
        return self.autoencoder, self.encoder, self.decoder


# Test
if __name__ == "__main__":
    print("🧪 Autoencoder Test\n")

    X = np.random.rand(1000, 78).astype(np.float32)

    # Standard Autoencoder
    ae = FeatureAutoencoder(input_dim=78, latent_dim=16)
    ae.build()
    ae.fit(X, epochs=5, batch_size=32)

    X_encoded = ae.encode(X)
    print(f"Encoded shape: {X_encoded.shape}")

    # Anomaly detection
    anomalies = ae.detect_anomalies(X)
    print(f"Anomalies: {np.sum(anomalies)}")

    print("\n✅ Test tamamlandı!")
