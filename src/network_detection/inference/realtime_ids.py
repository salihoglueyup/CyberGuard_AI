"""
Real-time IDS (Intrusion Detection System)
Canlı ağ trafiği izleme ve saldırı tespiti

Özellikler:
    - Streaming veri işleme
    - Real-time prediction
    - Alert yönetimi
    - Concept drift detection
    - Model hot-reload

Kullanım:
    ids = RealTimeIDS(model_path="models/ssa_lstmids.h5")
    ids.start()

    # Veri akışı simülasyonu
    for packet in network_stream:
        alert = ids.process(packet)
        if alert:
            print(f"🚨 Alert: {alert}")
"""

import os
import queue
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path

import numpy as np

# Proje yolu

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

try:
    import tensorflow as tf
    from tensorflow import keras

    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False


class AlertSeverity(Enum):
    """Alert önem derecesi"""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AttackType(Enum):
    """Saldırı tipleri"""

    NORMAL = 0
    DOS = 1
    PROBE = 2
    R2L = 3
    U2R = 4
    DDOS = 5
    BOTNET = 6
    OTHER = 99


@dataclass
class Alert:
    """Güvenlik alertı"""

    id: str
    timestamp: datetime
    attack_type: AttackType
    severity: AlertSeverity
    confidence: float
    source_ip: str | None = None
    dest_ip: str | None = None
    description: str = ""
    raw_features: np.ndarray | None = None

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "timestamp": self.timestamp.isoformat(),
            "attack_type": self.attack_type.name,
            "severity": self.severity.value,
            "confidence": self.confidence,
            "source_ip": self.source_ip,
            "dest_ip": self.dest_ip,
            "description": self.description,
        }


@dataclass
class IDSMetrics:
    """IDS performans metrikleri"""

    total_packets: int = 0
    normal_packets: int = 0
    attack_packets: int = 0
    alerts_generated: int = 0
    processing_time_avg: float = 0.0
    packets_per_second: float = 0.0

    def to_dict(self) -> dict:
        return {
            "total_packets": self.total_packets,
            "normal_packets": self.normal_packets,
            "attack_packets": self.attack_packets,
            "alerts_generated": self.alerts_generated,
            "processing_time_avg_ms": self.processing_time_avg * 1000,
            "packets_per_second": self.packets_per_second,
        }


class ConceptDriftDetector:
    """
    Concept Drift Detection
    Model performansındaki değişiklikleri tespit eder
    """

    def __init__(self, window_size: int = 1000, threshold: float = 0.1):
        self.window_size = window_size
        self.threshold = threshold
        self.predictions = deque(maxlen=window_size)
        self.ground_truth = deque(maxlen=window_size)
        self.error_rates = deque(maxlen=100)

    def update(self, prediction: int, true_label: int | None = None):
        """Prediction ekle"""
        self.predictions.append(prediction)
        if true_label is not None:
            self.ground_truth.append(true_label)

    def check_drift(self) -> tuple[bool, float]:
        """
        Drift var mı kontrol et

        Returns:
            (drift_detected, drift_score)
        """
        if len(self.predictions) < self.window_size // 2:
            return False, 0.0

        # Prediction dağılımındaki değişimi kontrol et
        predictions = list(self.predictions)
        half = len(predictions) // 2

        first_half = predictions[:half]
        second_half = predictions[half:]

        # Sınıf dağılımlarını karşılaştır
        first_dist = np.bincount(first_half, minlength=10) / len(first_half)
        second_dist = np.bincount(second_half, minlength=10) / len(second_half)

        # KL divergence basit approximation
        drift_score = np.sum(np.abs(first_dist - second_dist))
        drift_detected = drift_score > self.threshold

        return drift_detected, drift_score


class RealTimeIDS:
    """
    Real-time Intrusion Detection System

    Canlı ağ trafiği izleme ve saldırı tespiti

    Args:
        model_path: Eğitilmiş model dosyası
        threshold: Saldırı tespit eşiği (0-1)
        buffer_size: İşlenecek paket buffer boyutu
        alert_callback: Alert oluşturulduğunda çağrılacak fonksiyon
    """

    # Saldırı tipi severity mapping
    SEVERITY_MAP = {
        AttackType.NORMAL: AlertSeverity.LOW,
        AttackType.PROBE: AlertSeverity.MEDIUM,
        AttackType.DOS: AlertSeverity.HIGH,
        AttackType.DDOS: AlertSeverity.CRITICAL,
        AttackType.R2L: AlertSeverity.HIGH,
        AttackType.U2R: AlertSeverity.CRITICAL,
        AttackType.BOTNET: AlertSeverity.CRITICAL,
    }

    def __init__(
        self,
        model_path: str | None = None,
        threshold: float = 0.5,
        buffer_size: int = 100,
        window_size: int = 10,
        alert_callback: Callable[[Alert], None] | None = None,
        verbose: bool = True,
    ):
        self.model_path = model_path
        self.threshold = threshold
        self.buffer_size = buffer_size
        self.window_size = window_size
        self.alert_callback = alert_callback
        self.verbose = verbose

        self.model: keras.Model | None = None
        self.scaler = None
        self.is_running = False

        # Buffers
        self.packet_buffer = deque(maxlen=buffer_size)
        self.alert_queue = queue.Queue()
        self.feature_window = deque(maxlen=window_size)

        # Metrics
        self.metrics = IDSMetrics()
        self.start_time: datetime | None = None

        # Drift detection
        self.drift_detector = ConceptDriftDetector()

        # Alerts
        self.recent_alerts: list[Alert] = []
        self.alert_counter = 0

        # Threading
        self._processing_thread: threading.Thread | None = None
        self._lock = threading.Lock()

        # Model yükle
        if model_path and Path(model_path).exists():
            self.load_model(model_path)

    def load_model(self, path: str):
        """Model yükle"""
        if not TF_AVAILABLE:
            raise ImportError("TensorFlow gerekli!")

        self.model = keras.models.load_model(path, compile=False)
        self.model.compile(optimizer="adam", loss="sparse_categorical_crossentropy")

        if self.verbose:
            print(f"✅ Model yüklendi: {path}")
            print(f"   Input shape: {self.model.input_shape}")
            print(f"   Output classes: {self.model.output_shape[-1]}")

    def start(self):
        """IDS'i başlat"""
        if self.model is None:
            raise ValueError("Model yüklenmedi!")

        self.is_running = True
        self.start_time = datetime.now()

        # Processing thread
        self._processing_thread = threading.Thread(
            target=self._processing_loop, daemon=True
        )
        self._processing_thread.start()

        if self.verbose:
            print("🚀 Real-time IDS başlatıldı")

    def stop(self):
        """IDS'i durdur"""
        self.is_running = False
        if self._processing_thread:
            self._processing_thread.join(timeout=2.0)

        if self.verbose:
            print("⏹️ Real-time IDS durduruldu")

    def process(
        self, features: np.ndarray, metadata: dict | None = None
    ) -> Alert | None:
        """
        Tek bir paket/flow işle

        Args:
            features: Feature vektörü (1D array)
            metadata: Ek bilgiler (source_ip, dest_ip, vb.)

        Returns:
            Alert varsa döndür, yoksa None
        """
        start_time = time.time()

        # Feature window'a ekle
        self.feature_window.append(features)

        if len(self.feature_window) < self.window_size:
            return None

        # Sequence oluştur
        X = np.array(list(self.feature_window)).reshape(1, self.window_size, -1)

        # Prediction
        prediction_proba = self.model.predict(X, verbose=0)[0]
        prediction = np.argmax(prediction_proba)
        confidence = float(prediction_proba[prediction])

        # Metrics güncelle
        with self._lock:
            self.metrics.total_packets += 1
            proc_time = time.time() - start_time
            self.metrics.processing_time_avg = (
                self.metrics.processing_time_avg * 0.99 + proc_time * 0.01
            )

        # Drift detection
        self.drift_detector.update(prediction)

        # Alert oluştur (saldırı tespit edilirse)
        alert = None
        if prediction != 0 and confidence >= self.threshold:
            attack_type = AttackType(min(prediction, 6))
            severity = self.SEVERITY_MAP.get(attack_type, AlertSeverity.MEDIUM)

            self.alert_counter += 1
            alert = Alert(
                id=f"ALERT-{self.alert_counter:06d}",
                timestamp=datetime.now(),
                attack_type=attack_type,
                severity=severity,
                confidence=confidence,
                source_ip=metadata.get("source_ip") if metadata else None,
                dest_ip=metadata.get("dest_ip") if metadata else None,
                description=f"{attack_type.name} attack detected with {confidence*100:.1f}% confidence",
                raw_features=features,
            )

            with self._lock:
                self.metrics.attack_packets += 1
                self.metrics.alerts_generated += 1
                self.recent_alerts.append(alert)
                if len(self.recent_alerts) > 100:
                    self.recent_alerts.pop(0)

            # Callback
            if self.alert_callback:
                self.alert_callback(alert)

            if self.verbose:
                print(f"🚨 {alert.severity.value.upper()}: {alert.description}")
        else:
            with self._lock:
                self.metrics.normal_packets += 1

        return alert

    def process_batch(self, features_batch: np.ndarray) -> list[Alert]:
        """Batch işleme"""
        alerts = []
        for features in features_batch:
            alert = self.process(features)
            if alert:
                alerts.append(alert)
        return alerts

    def _processing_loop(self):
        """Background processing loop"""
        while self.is_running:
            try:
                if self.packet_buffer:
                    item = self.packet_buffer.popleft()
                    if isinstance(item, tuple) and len(item) == 2:
                        features, metadata = item
                        self.process(features, metadata)
                    else:
                        self.process(item)
                else:
                    time.sleep(0.001)  # 1ms wait
            except Exception as e:
                if self.verbose:
                    print(f"⚠️ Processing error: {e}")

    def add_to_buffer(self, features: np.ndarray, metadata: dict | None = None):
        """Packet buffer'a ekle"""
        self.packet_buffer.append((features, metadata))

    def get_metrics(self) -> dict:
        """Güncel metrikleri döndür"""
        with self._lock:
            if self.start_time:
                elapsed = (datetime.now() - self.start_time).total_seconds()
                if elapsed > 0:
                    self.metrics.packets_per_second = (
                        self.metrics.total_packets / elapsed
                    )
            return self.metrics.to_dict()

    def get_recent_alerts(self, limit: int = 10) -> list[dict]:
        """Son alert'leri döndür"""
        with self._lock:
            return [a.to_dict() for a in self.recent_alerts[-limit:]]

    def check_drift(self) -> tuple[bool, float]:
        """Concept drift kontrol et"""
        return self.drift_detector.check_drift()

    def reload_model(self, new_model_path: str):
        """Model hot-reload"""
        if self.verbose:
            print(f"♻️ Model reload: {new_model_path}")

        # Geçici olarak durdur
        was_running = self.is_running
        if was_running:
            self.is_running = False
            time.sleep(0.1)

        # Yeni model yükle
        self.load_model(new_model_path)

        # Tekrar başlat
        if was_running:
            self.is_running = True


class NetworkTrafficSimulator:
    """
    Ağ trafiği simülatörü (test için)
    """

    def __init__(self, num_features: int = 41, attack_ratio: float = 0.3):
        self.num_features = num_features
        self.attack_ratio = attack_ratio

    def generate(self, n_samples: int = 100) -> tuple:
        """Random trafik verisi üret"""
        # Normal trafik
        n_normal = int(n_samples * (1 - self.attack_ratio))
        X_normal = np.random.randn(n_normal, self.num_features) * 0.5
        y_normal = np.zeros(n_normal, dtype=int)

        # Saldırı trafiği
        n_attack = n_samples - n_normal
        X_attack = np.random.randn(n_attack, self.num_features) * 2 + 1
        y_attack = np.random.randint(1, 5, n_attack)

        # Birleştir ve shuffle
        X = np.vstack([X_normal, X_attack])
        y = np.concatenate([y_normal, y_attack])

        indices = np.random.permutation(len(X))
        return X[indices], y[indices]

    def stream(self, interval: float = 0.1):
        """Streaming generator"""
        while True:
            X, y = self.generate(1)
            metadata = {
                "source_ip": f"192.168.1.{np.random.randint(1, 255)}",
                "dest_ip": f"10.0.0.{np.random.randint(1, 255)}",
                "timestamp": datetime.now().isoformat(),
            }
            yield X[0], y[0], metadata
            time.sleep(interval)


# Test
if __name__ == "__main__":
    print("🧪 Real-time IDS Test\n")

    # Simülatör
    simulator = NetworkTrafficSimulator(num_features=41)

    # Dummy model için test
    print("⚠️ Gerçek test için eğitilmiş model gerekli")
    print("   Örnek kullanım:")
    print("   ids = RealTimeIDS(model_path='models/ssa_lstmids_nsl_kdd.h5')")
    print("   ids.start()")
    print("")

    # Simüle edilmiş veri
    X, y = simulator.generate(100)
    print(f"📊 Simüle edilmiş veri: {X.shape}")
    print(f"   Normal: {np.sum(y == 0)}")
    print(f"   Attack: {np.sum(y != 0)}")
