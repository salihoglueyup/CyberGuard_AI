"""
Zero-Day Detector - CyberGuard AI
==================================

VAE (Variational Autoencoder) + IDS Hybrid System

Unsupervised zero-day saldırı tespiti için iki aşamalı pipeline:
1. VAE reconstruction error → Anomali skoru
2. Yüksek anomali → Zero-day candidate
3. Normal trafik → IDS modeline gönder

Referans:
    "A hybrid unsupervised–supervised intrusion detection framework
    was designed to identify both known and zero-day attacks."
"""

import os
import sys
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from datetime import datetime
import logging

# Project imports
PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
sys.path.insert(0, PROJECT_ROOT)

try:
    import tensorflow as tf
    from tensorflow import keras

    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

logger = logging.getLogger("ZeroDayDetector")


class ZeroDayDetector:
    """
    VAE-based Zero-Day Attack Detector

    Normal trafik profili öğrenir, anormal trafikleri zero-day olarak işaretler.

    Attributes:
        vae_model: Variational Autoencoder model
        threshold: Anomali eşik değeri (otomatik veya manuel)
        sensitivity: Hassasiyet (1=düşük, 5=yüksek)
    """

    def __init__(
        self,
        input_dim: int = 78,
        latent_dim: int = 32,
        hidden_layers: List[int] = [128, 64],
        threshold: float = None,
        sensitivity: int = 3,
    ):
        """
        Args:
            input_dim: Feature sayısı
            latent_dim: VAE latent space boyutu
            hidden_layers: Encoder/Decoder hidden layers
            threshold: Manuel anomali eşiği (None=otomatik)
            sensitivity: 1-5 arası hassasiyet
        """
        if not TF_AVAILABLE:
            raise ImportError("TensorFlow gerekli!")

        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_layers = hidden_layers
        self.threshold = threshold
        self.sensitivity = sensitivity

        # Percentile mapping (sensitivity → percentile)
        self._sensitivity_map = {
            1: 99,  # Çok düşük hassasiyet (sadece %1 anomali)
            2: 97,
            3: 95,  # Varsayılan
            4: 90,
            5: 85,  # Yüksek hassasiyet (%15 anomali)
        }

        self.encoder: Optional[keras.Model] = None
        self.decoder: Optional[keras.Model] = None
        self.vae: Optional[keras.Model] = None

        self.is_trained = False
        self.training_errors = None

        logger.info(f"🔮 ZeroDayDetector initialized")
        logger.info(f"   Input dim: {input_dim}, Latent dim: {latent_dim}")
        logger.info(f"   Sensitivity: {sensitivity}/5")

    def build(self) -> keras.Model:
        """VAE modelini oluştur"""
        from tensorflow.keras import layers, Model

        # === ENCODER ===
        encoder_inputs = layers.Input(shape=(self.input_dim,), name="encoder_input")
        x = encoder_inputs

        for i, units in enumerate(self.hidden_layers):
            x = layers.Dense(units, activation="relu", name=f"enc_dense_{i}")(x)
            x = layers.BatchNormalization(name=f"enc_bn_{i}")(x)
            x = layers.Dropout(0.2, name=f"enc_drop_{i}")(x)

        # Latent space (mean and log variance)
        z_mean = layers.Dense(self.latent_dim, name="z_mean")(x)
        z_log_var = layers.Dense(self.latent_dim, name="z_log_var")(x)

        # Reparameterization trick
        def sampling(args):
            z_mean, z_log_var = args
            batch = tf.shape(z_mean)[0]
            dim = tf.shape(z_mean)[1]
            epsilon = tf.random.normal(shape=(batch, dim))
            return z_mean + tf.exp(0.5 * z_log_var) * epsilon

        z = layers.Lambda(sampling, name="z")([z_mean, z_log_var])

        self.encoder = Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")

        # === DECODER ===
        decoder_inputs = layers.Input(shape=(self.latent_dim,), name="decoder_input")
        y = decoder_inputs

        for i, units in enumerate(reversed(self.hidden_layers)):
            y = layers.Dense(units, activation="relu", name=f"dec_dense_{i}")(y)
            y = layers.BatchNormalization(name=f"dec_bn_{i}")(y)

        decoder_outputs = layers.Dense(
            self.input_dim, activation="linear", name="output"
        )(y)
        self.decoder = Model(decoder_inputs, decoder_outputs, name="decoder")

        # === VAE ===
        outputs = self.decoder(self.encoder(encoder_inputs)[2])
        self.vae = Model(encoder_inputs, outputs, name="vae")

        # VAE Loss
        reconstruction_loss = tf.reduce_mean(
            tf.square(encoder_inputs - outputs), axis=1
        )
        kl_loss = -0.5 * tf.reduce_sum(
            1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=1
        )
        vae_loss = tf.reduce_mean(reconstruction_loss + 0.001 * kl_loss)

        self.vae.add_loss(vae_loss)
        self.vae.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001))

        logger.info(f"✅ VAE built! Parameters: {self.vae.count_params():,}")
        return self.vae

    def fit(
        self,
        X_normal: np.ndarray,
        epochs: int = 50,
        batch_size: int = 64,
        validation_split: float = 0.1,
        verbose: int = 1,
    ) -> Dict:
        """
        Normal trafik ile VAE'yi eğit

        Args:
            X_normal: Sadece NORMAL trafik verileri
            epochs: Epoch sayısı
            batch_size: Batch boyutu

        Returns:
            Training history
        """
        if self.vae is None:
            self.build()

        logger.info(f"🏋️ Training VAE on {len(X_normal)} normal samples...")

        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor="loss", patience=5, restore_best_weights=True
            ),
            keras.callbacks.ReduceLROnPlateau(monitor="loss", factor=0.5, patience=3),
        ]

        history = self.vae.fit(
            X_normal,
            X_normal,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=callbacks,
            verbose=verbose,
        )

        # Threshold hesapla
        self.training_errors = self.get_reconstruction_error(X_normal)

        if self.threshold is None:
            percentile = self._sensitivity_map.get(self.sensitivity, 95)
            self.threshold = np.percentile(self.training_errors, percentile)

        self.is_trained = True

        logger.info(f"✅ VAE trained!")
        logger.info(f"   Threshold: {self.threshold:.4f}")
        logger.info(f"   Mean error: {np.mean(self.training_errors):.4f}")

        return {
            "loss": history.history["loss"][-1],
            "threshold": self.threshold,
            "mean_error": float(np.mean(self.training_errors)),
            "std_error": float(np.std(self.training_errors)),
        }

    def get_reconstruction_error(self, X: np.ndarray) -> np.ndarray:
        """Reconstruction error hesapla"""
        if self.vae is None:
            raise ValueError("Model not built!")

        X_reconstructed = self.vae.predict(X, verbose=0)
        errors = np.mean(np.square(X - X_reconstructed), axis=1)
        return errors

    def detect(self, X: np.ndarray) -> Dict:
        """
        Zero-day anomali tespiti

        Args:
            X: Test verileri

        Returns:
            {
                "is_zero_day": bool array,
                "anomaly_scores": float array,
                "threshold": float
            }
        """
        if not self.is_trained:
            raise ValueError("Model not trained!")

        errors = self.get_reconstruction_error(X)
        is_zero_day = errors > self.threshold

        # Normalize scores (0-1)
        max_error = max(np.max(errors), self.threshold * 2)
        normalized_scores = np.clip(errors / max_error, 0, 1)

        return {
            "is_zero_day": is_zero_day,
            "anomaly_scores": normalized_scores,
            "raw_errors": errors,
            "threshold": self.threshold,
            "num_anomalies": int(np.sum(is_zero_day)),
            "anomaly_rate": float(np.mean(is_zero_day)),
        }

    def save(self, path: str):
        """Modeli kaydet"""
        os.makedirs(path, exist_ok=True)

        self.vae.save(os.path.join(path, "vae_model.h5"))

        import json

        metadata = {
            "input_dim": self.input_dim,
            "latent_dim": self.latent_dim,
            "hidden_layers": self.hidden_layers,
            "threshold": float(self.threshold) if self.threshold else None,
            "sensitivity": self.sensitivity,
            "saved_at": datetime.now().isoformat(),
        }

        with open(os.path.join(path, "metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"✅ Model saved to {path}")

    @classmethod
    def load(cls, path: str) -> "ZeroDayDetector":
        """Modeli yükle"""
        import json

        with open(os.path.join(path, "metadata.json"), "r") as f:
            metadata = json.load(f)

        detector = cls(
            input_dim=metadata["input_dim"],
            latent_dim=metadata["latent_dim"],
            hidden_layers=metadata["hidden_layers"],
            threshold=metadata["threshold"],
            sensitivity=metadata["sensitivity"],
        )

        detector.vae = keras.models.load_model(
            os.path.join(path, "vae_model.h5"), compile=False
        )
        detector.is_trained = True

        logger.info(f"✅ Model loaded from {path}")
        return detector


class HybridIDSPipeline:
    """
    Hybrid Unsupervised-Supervised IDS Pipeline

    İki aşamalı tespit:
    1. ZeroDayDetector (VAE) → Anomali mi?
    2. IDS Model → Saldırı türü nedir?

    Flow:
        Traffic → VAE → High error? → "ZERO_DAY"
                      → Low error  → IDS → Attack type
    """

    def __init__(
        self,
        zero_day_detector: ZeroDayDetector,
        ids_model=None,
        ids_model_path: str = None,
    ):
        """
        Args:
            zero_day_detector: Eğitilmiş ZeroDayDetector
            ids_model: Keras IDS modeli (veya path)
            ids_model_path: IDS model dosya yolu
        """
        self.zero_day = zero_day_detector
        self.ids_model = ids_model

        if ids_model_path and ids_model is None:
            self.ids_model = keras.models.load_model(ids_model_path)

        self.attack_labels = [
            "Normal",
            "DoS",
            "Probe",
            "R2L",
            "U2R",  # NSL-KDD
            "DDoS",
            "PortScan",
            "Bot",
            "Infiltration",  # CICIDS2017
            "BruteForce",
            "WebAttack",
            "Unknown",
        ]

        logger.info("🔥 HybridIDSPipeline initialized")

    def predict(self, X: np.ndarray) -> List[Dict]:
        """
        Hybrid prediction

        Returns:
            List of {
                "prediction": str,
                "is_zero_day": bool,
                "confidence": float,
                "anomaly_score": float,
                "attack_type": str or None
            }
        """
        results = []

        # Step 1: Zero-day detection
        zd_result = self.zero_day.detect(X)

        # Step 2: IDS prediction for non-zero-day
        ids_predictions = None
        if self.ids_model is not None:
            # Reshape if needed
            if len(X.shape) == 2 and len(self.ids_model.input_shape) == 3:
                X_ids = X.reshape(X.shape[0], 1, X.shape[1])
            else:
                X_ids = X

            ids_predictions = self.ids_model.predict(X_ids, verbose=0)

        # Combine results
        for i in range(len(X)):
            is_zero_day = zd_result["is_zero_day"][i]
            anomaly_score = zd_result["anomaly_scores"][i]

            if is_zero_day:
                # Zero-day attack
                result = {
                    "prediction": "ZERO_DAY",
                    "is_zero_day": True,
                    "confidence": float(anomaly_score),
                    "anomaly_score": float(anomaly_score),
                    "attack_type": "Unknown (Zero-Day)",
                    "explanation": "High reconstruction error indicates novel attack pattern",
                }
            else:
                # Known attack or normal
                if ids_predictions is not None:
                    pred_idx = np.argmax(ids_predictions[i])
                    confidence = float(ids_predictions[i][pred_idx])
                    attack_type = (
                        self.attack_labels[pred_idx]
                        if pred_idx < len(self.attack_labels)
                        else "Unknown"
                    )
                else:
                    pred_idx = 0
                    confidence = 1.0 - anomaly_score
                    attack_type = "Normal" if anomaly_score < 0.3 else "Suspicious"

                result = {
                    "prediction": attack_type,
                    "is_zero_day": False,
                    "confidence": confidence,
                    "anomaly_score": float(anomaly_score),
                    "attack_type": attack_type,
                    "explanation": f"Known traffic pattern classified as {attack_type}",
                }

            results.append(result)

        return results

    def predict_single(self, x: np.ndarray) -> Dict:
        """Tek örnek için prediction"""
        x = x.reshape(1, -1) if len(x.shape) == 1 else x
        return self.predict(x)[0]

    def get_stats(self) -> Dict:
        """Pipeline istatistikleri"""
        return {
            "zero_day_threshold": self.zero_day.threshold,
            "zero_day_sensitivity": self.zero_day.sensitivity,
            "ids_model_loaded": self.ids_model is not None,
            "attack_labels": self.attack_labels,
        }


# ============= Factory Functions =============


def create_hybrid_pipeline(
    input_dim: int = 78,
    ids_model_path: str = None,
    sensitivity: int = 3,
) -> HybridIDSPipeline:
    """
    Hazır hybrid pipeline oluştur

    Args:
        input_dim: Feature sayısı
        ids_model_path: IDS model path (optional)
        sensitivity: Zero-day hassasiyeti (1-5)
    """
    zd = ZeroDayDetector(input_dim=input_dim, sensitivity=sensitivity)
    return HybridIDSPipeline(zd, ids_model_path=ids_model_path)


# ============= β-VAE (Advanced) =============


class BetaVAE(ZeroDayDetector):
    """
    β-VAE for Zero-Day Detection

    β-VAE disentangled representations öğrenir, bu da
    daha iyi genelleme ve daha anlamlı latent space sağlar.

    β > 1: Daha fazla disentanglement (daha iyi zero-day detection)
    β = 1: Standart VAE

    Reference:
        "Understanding disentangling in β-VAE" (Burgess et al., 2018)

    Avantajlar:
        - Daha robust anomaly detection
        - Daha interpretable latent space
        - Daha az overfitting
    """

    VERSION = "2.0.0"

    def __init__(
        self,
        input_dim: int = 78,
        latent_dim: int = 32,
        hidden_layers: List[int] = [128, 64],
        beta: float = 4.0,
        threshold: float = None,
        sensitivity: int = 3,
    ):
        """
        Args:
            beta: KL divergence weight (β > 1 = more disentanglement)
        """
        super().__init__(
            input_dim=input_dim,
            latent_dim=latent_dim,
            hidden_layers=hidden_layers,
            threshold=threshold,
            sensitivity=sensitivity,
        )
        self.beta = beta
        logger.info(f"🔮 BetaVAE initialized with β={beta}")

    def build(self) -> keras.Model:
        """Build β-VAE with weighted KL loss"""
        from tensorflow.keras import layers, Model

        # === ENCODER ===
        encoder_inputs = layers.Input(shape=(self.input_dim,), name="encoder_input")
        x = encoder_inputs

        for i, units in enumerate(self.hidden_layers):
            x = layers.Dense(units, activation="relu", name=f"enc_dense_{i}")(x)
            x = layers.BatchNormalization(name=f"enc_bn_{i}")(x)
            x = layers.Dropout(0.2, name=f"enc_drop_{i}")(x)

        # Latent space (mean and log variance)
        z_mean = layers.Dense(self.latent_dim, name="z_mean")(x)
        z_log_var = layers.Dense(self.latent_dim, name="z_log_var")(x)

        # Reparameterization trick
        def sampling(args):
            z_mean, z_log_var = args
            batch = tf.shape(z_mean)[0]
            dim = tf.shape(z_mean)[1]
            epsilon = tf.random.normal(shape=(batch, dim))
            return z_mean + tf.exp(0.5 * z_log_var) * epsilon

        z = layers.Lambda(sampling, name="z")([z_mean, z_log_var])

        self.encoder = Model(
            encoder_inputs, [z_mean, z_log_var, z], name="beta_encoder"
        )

        # === DECODER ===
        decoder_inputs = layers.Input(shape=(self.latent_dim,), name="decoder_input")
        y = decoder_inputs

        for i, units in enumerate(reversed(self.hidden_layers)):
            y = layers.Dense(units, activation="relu", name=f"dec_dense_{i}")(y)
            y = layers.BatchNormalization(name=f"dec_bn_{i}")(y)

        decoder_outputs = layers.Dense(
            self.input_dim, activation="linear", name="output"
        )(y)
        self.decoder = Model(decoder_inputs, decoder_outputs, name="beta_decoder")

        # === β-VAE ===
        outputs = self.decoder(self.encoder(encoder_inputs)[2])
        self.vae = Model(encoder_inputs, outputs, name="beta_vae")

        # β-VAE Loss: recon + β * KL
        reconstruction_loss = tf.reduce_mean(
            tf.square(encoder_inputs - outputs), axis=1
        )
        kl_loss = -0.5 * tf.reduce_sum(
            1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=1
        )

        # β weighting for KL divergence
        beta_vae_loss = tf.reduce_mean(
            reconstruction_loss + self.beta * 0.001 * kl_loss
        )

        self.vae.add_loss(beta_vae_loss)
        self.vae.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001))

        logger.info(
            f"✅ β-VAE built! β={self.beta}, Parameters: {self.vae.count_params():,}"
        )
        return self.vae

    def get_disentanglement_score(self, X: np.ndarray) -> Dict:
        """
        Latent space disentanglement score

        Higher score = more disentangled = better generalization
        """
        if not self.is_trained:
            raise ValueError("Model not trained!")

        z_mean, z_log_var, z = self.encoder.predict(X, verbose=0)

        # Variance of each latent dimension
        latent_vars = np.var(z, axis=0)

        # Disentanglement heuristic: variance should be spread across dimensions
        normalized_vars = latent_vars / np.sum(latent_vars)
        entropy = -np.sum(normalized_vars * np.log(normalized_vars + 1e-10))
        max_entropy = np.log(self.latent_dim)
        disentanglement_score = entropy / max_entropy

        return {
            "disentanglement_score": float(disentanglement_score),
            "latent_variances": latent_vars.tolist(),
            "active_dimensions": int(np.sum(latent_vars > 0.01)),
            "entropy": float(entropy),
        }


def create_beta_vae_detector(
    input_dim: int = 78,
    beta: float = 4.0,
    sensitivity: int = 3,
) -> BetaVAE:
    """
    Factory function for β-VAE detector

    Recommended beta values:
        - β=4.0: Balanced disentanglement (default)
        - β=10.0: Strong disentanglement (for complex data)
        - β=2.0: Mild disentanglement (for simple data)
    """
    return BetaVAE(
        input_dim=input_dim,
        beta=beta,
        sensitivity=sensitivity,
    )


# ============= Test =============

if __name__ == "__main__":
    print("🧪 Zero-Day Detector Test\n")

    # Simulated data
    np.random.seed(42)
    X_normal = np.random.randn(1000, 78).astype(np.float32)
    X_anomaly = np.random.randn(100, 78).astype(np.float32) * 3 + 5  # Farklı dağılım

    # Train
    detector = ZeroDayDetector(input_dim=78, sensitivity=3)
    detector.build()
    result = detector.fit(X_normal, epochs=10, verbose=1)

    print(f"\nTraining result: {result}")

    # Test
    zd_result = detector.detect(X_anomaly)
    print(f"\nAnomaly detection:")
    print(f"  Detected: {zd_result['num_anomalies']}/100")
    print(f"  Rate: {zd_result['anomaly_rate']*100:.1f}%")

    # β-VAE Test
    print("\n🔥 β-VAE Test")
    beta_detector = BetaVAE(input_dim=78, beta=4.0, sensitivity=3)
    beta_detector.build()
    beta_result = beta_detector.fit(X_normal, epochs=10, verbose=0)
    print(f"β-VAE Training: {beta_result}")

    ds = beta_detector.get_disentanglement_score(X_normal[:100])
    print(f"Disentanglement: {ds['disentanglement_score']:.3f}")

    # Hybrid pipeline
    print("\n🔥 Hybrid Pipeline Test")
    pipeline = HybridIDSPipeline(detector)
    predictions = pipeline.predict(X_anomaly[:5])

    for i, pred in enumerate(predictions):
        print(
            f"  Sample {i}: {pred['prediction']} (score: {pred['anomaly_score']:.3f})"
        )

    print("\n✅ Test completed!")
