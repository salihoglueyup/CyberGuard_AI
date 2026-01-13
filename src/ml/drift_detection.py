"""
Drift Detection Module - CyberGuard AI
=======================================

Concept ve Data drift tespiti ve görselleştirme.

Özellikler:
    - PSI (Population Stability Index)
    - KS Test (Kolmogorov-Smirnov)
    - Drift alertleri
    - Timeline görselleştirme
"""

import os
import sys
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import logging
from collections import deque

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logger = logging.getLogger("DriftDetection")


class DriftType(Enum):
    DATA = "data"
    CONCEPT = "concept"
    PREDICTION = "prediction"


class DriftSeverity(Enum):
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class DriftAlert:
    """Drift alert"""

    alert_id: str
    drift_type: DriftType
    severity: DriftSeverity
    feature: str
    metric_name: str
    metric_value: float
    threshold: float
    timestamp: str
    message: str


@dataclass
class DriftSnapshot:
    """Belirli bir zamandaki drift durumu"""

    snapshot_id: str
    timestamp: str
    psi_scores: Dict[str, float]
    ks_scores: Dict[str, float]
    overall_drift_score: float
    severity: DriftSeverity
    alerts: List[DriftAlert]


class DriftDetector:
    """
    Drift Detection Engine
    """

    # Thresholds
    PSI_THRESHOLDS = {
        DriftSeverity.NONE: 0.1,
        DriftSeverity.LOW: 0.15,
        DriftSeverity.MEDIUM: 0.25,
        DriftSeverity.HIGH: 0.4,
        DriftSeverity.CRITICAL: float("inf"),
    }

    KS_THRESHOLDS = {
        DriftSeverity.NONE: 0.05,
        DriftSeverity.LOW: 0.1,
        DriftSeverity.MEDIUM: 0.2,
        DriftSeverity.HIGH: 0.3,
        DriftSeverity.CRITICAL: float("inf"),
    }

    def __init__(
        self,
        reference_data: np.ndarray = None,
        feature_names: List[str] = None,
        window_size: int = 1000,
        check_interval: int = 100,
    ):
        self.reference_data = reference_data
        self.feature_names = feature_names or []
        self.window_size = window_size
        self.check_interval = check_interval

        self.current_window = deque(maxlen=window_size)
        self.snapshots: List[DriftSnapshot] = []
        self.alerts: List[DriftAlert] = []
        self.samples_since_check = 0

        # Metrics history
        self.psi_history: List[Dict] = []
        self.ks_history: List[Dict] = []
        self.accuracy_history: List[Dict] = []

    def set_reference_data(self, data: np.ndarray):
        """Reference (baseline) verisini set et"""
        self.reference_data = data
        print(f"✅ Reference data set edildi: {data.shape}")

    def add_sample(
        self, sample: np.ndarray, prediction: int = None, ground_truth: int = None
    ):
        """Yeni sample ekle"""
        self.current_window.append(
            {
                "features": sample,
                "prediction": prediction,
                "ground_truth": ground_truth,
                "timestamp": datetime.now().isoformat(),
            }
        )

        self.samples_since_check += 1

        # Periyodik kontrol
        if self.samples_since_check >= self.check_interval:
            self.check_drift()
            self.samples_since_check = 0

    def add_batch(
        self,
        samples: np.ndarray,
        predictions: np.ndarray = None,
        ground_truths: np.ndarray = None,
    ):
        """Batch sample ekle"""
        for i in range(len(samples)):
            pred = predictions[i] if predictions is not None else None
            gt = ground_truths[i] if ground_truths is not None else None
            self.add_sample(samples[i], pred, gt)

    def calculate_psi(
        self, reference: np.ndarray, current: np.ndarray, bins: int = 10
    ) -> float:
        """
        Population Stability Index (PSI) hesapla

        PSI < 0.1: Anlamlı değişim yok
        PSI 0.1-0.25: Orta düzey değişim
        PSI > 0.25: Önemli değişim
        """
        # Bin edges from reference
        try:
            _, bin_edges = np.histogram(reference, bins=bins)

            # Histograms
            ref_hist, _ = np.histogram(reference, bins=bin_edges)
            cur_hist, _ = np.histogram(current, bins=bin_edges)

            # Normalize
            ref_pct = ref_hist / len(reference)
            cur_pct = cur_hist / len(current)

            # Avoid division by zero
            ref_pct = np.where(ref_pct == 0, 0.0001, ref_pct)
            cur_pct = np.where(cur_pct == 0, 0.0001, cur_pct)

            # PSI
            psi = np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct))

            return float(psi)
        except Exception as e:
            logger.error(f"PSI calculation error: {e}")
            return 0.0

    def calculate_ks_statistic(
        self, reference: np.ndarray, current: np.ndarray
    ) -> Tuple[float, float]:
        """
        Kolmogorov-Smirnov test istatistiği

        Returns:
            (ks_statistic, p_value)
        """
        try:
            from scipy import stats

            ks_stat, p_value = stats.ks_2samp(reference, current)
            return float(ks_stat), float(p_value)
        except ImportError:
            # Fallback - basit KS
            ref_sorted = np.sort(reference)
            cur_sorted = np.sort(current)

            # CDF
            ref_cdf = np.arange(1, len(ref_sorted) + 1) / len(ref_sorted)
            cur_cdf = np.arange(1, len(cur_sorted) + 1) / len(cur_sorted)

            # Interpolate
            all_values = np.union1d(ref_sorted, cur_sorted)
            ref_interp = np.interp(all_values, ref_sorted, ref_cdf, left=0, right=1)
            cur_interp = np.interp(all_values, cur_sorted, cur_cdf, left=0, right=1)

            ks_stat = np.max(np.abs(ref_interp - cur_interp))
            return float(ks_stat), None

    def check_drift(self) -> DriftSnapshot:
        """Drift kontrolü yap"""
        if self.reference_data is None or len(self.current_window) < 100:
            return None

        # Current window data
        current_data = np.array([s["features"] for s in self.current_window])

        # Flatten if needed
        if len(current_data.shape) > 2:
            current_data = current_data.reshape(len(current_data), -1)
        if len(self.reference_data.shape) > 2:
            ref_data = self.reference_data.reshape(len(self.reference_data), -1)
        else:
            ref_data = self.reference_data

        n_features = min(current_data.shape[-1], ref_data.shape[-1])

        psi_scores = {}
        ks_scores = {}
        alerts = []

        for i in range(min(n_features, 20)):  # Max 20 feature
            feature_name = (
                self.feature_names[i] if i < len(self.feature_names) else f"feature_{i}"
            )

            ref_feature = ref_data[:, i]
            cur_feature = current_data[:, i]

            # PSI
            psi = self.calculate_psi(ref_feature, cur_feature)
            psi_scores[feature_name] = psi

            # KS
            ks_stat, _ = self.calculate_ks_statistic(ref_feature, cur_feature)
            ks_scores[feature_name] = ks_stat

            # Check for alerts
            severity = self._get_severity(psi, self.PSI_THRESHOLDS)
            if severity != DriftSeverity.NONE:
                alert = DriftAlert(
                    alert_id=f"alert_{len(self.alerts)+1:04d}",
                    drift_type=DriftType.DATA,
                    severity=severity,
                    feature=feature_name,
                    metric_name="PSI",
                    metric_value=psi,
                    threshold=self.PSI_THRESHOLDS[severity],
                    timestamp=datetime.now().isoformat(),
                    message=f"Feature '{feature_name}' drift detected (PSI={psi:.3f})",
                )
                alerts.append(alert)
                self.alerts.append(alert)

        # Overall drift score
        overall_psi = np.mean(list(psi_scores.values()))
        overall_severity = self._get_severity(overall_psi, self.PSI_THRESHOLDS)

        # Create snapshot
        snapshot = DriftSnapshot(
            snapshot_id=f"snapshot_{len(self.snapshots)+1:04d}",
            timestamp=datetime.now().isoformat(),
            psi_scores=psi_scores,
            ks_scores=ks_scores,
            overall_drift_score=overall_psi,
            severity=overall_severity,
            alerts=alerts,
        )

        self.snapshots.append(snapshot)

        # Update history
        self.psi_history.append(
            {
                "timestamp": snapshot.timestamp,
                "overall": overall_psi,
                **{f"psi_{k}": v for k, v in list(psi_scores.items())[:5]},
            }
        )

        if alerts:
            print(
                f"⚠️ Drift detected! Severity: {overall_severity.value}, Alerts: {len(alerts)}"
            )

        return snapshot

    def _get_severity(
        self, value: float, thresholds: Dict[DriftSeverity, float]
    ) -> DriftSeverity:
        """Severity belirle"""
        for severity in [
            DriftSeverity.NONE,
            DriftSeverity.LOW,
            DriftSeverity.MEDIUM,
            DriftSeverity.HIGH,
        ]:
            if value < thresholds[severity]:
                return severity
        return DriftSeverity.CRITICAL

    def get_drift_status(self) -> Dict:
        """Güncel drift durumu"""
        if not self.snapshots:
            return {"status": "no_data", "message": "Henüz drift kontrolü yapılmadı"}

        latest = self.snapshots[-1]

        return {
            "status": "active",
            "latest_check": latest.timestamp,
            "overall_drift_score": latest.overall_drift_score,
            "severity": latest.severity.value,
            "total_alerts": len(self.alerts),
            "recent_alerts": [
                {
                    "id": a.alert_id,
                    "severity": a.severity.value,
                    "feature": a.feature,
                    "value": a.metric_value,
                    "timestamp": a.timestamp,
                }
                for a in self.alerts[-5:]
            ],
            "top_drifting_features": sorted(
                latest.psi_scores.items(), key=lambda x: x[1], reverse=True
            )[:5],
        }

    def get_visualization_data(self) -> Dict:
        """Görselleştirme için veri"""
        if not self.snapshots:
            return {"error": "Veri yok"}

        # Timeline data
        timeline = [
            {
                "timestamp": s.timestamp,
                "drift_score": s.overall_drift_score,
                "severity": s.severity.value,
                "alert_count": len(s.alerts),
            }
            for s in self.snapshots[-50:]  # Son 50 snapshot
        ]

        # Feature drift heatmap data
        latest = self.snapshots[-1]
        heatmap_data = [
            {"feature": k, "psi": v, "ks": latest.ks_scores.get(k, 0)}
            for k, v in latest.psi_scores.items()
        ]

        # Severity distribution
        severity_counts = {}
        for s in self.snapshots:
            sev = s.severity.value
            severity_counts[sev] = severity_counts.get(sev, 0) + 1

        return {
            "timeline": timeline,
            "feature_drift": heatmap_data,
            "severity_distribution": severity_counts,
            "psi_history": self.psi_history[-100:],
        }

    def reset(self):
        """Detektörü sıfırla"""
        self.current_window.clear()
        self.snapshots.clear()
        self.alerts.clear()
        self.psi_history.clear()
        self.ks_history.clear()
        self.samples_since_check = 0


# Singleton
_drift_detector: Optional[DriftDetector] = None


def get_drift_detector() -> DriftDetector:
    """Global drift detector"""
    global _drift_detector
    if _drift_detector is None:
        _drift_detector = DriftDetector()
    return _drift_detector


# Test
if __name__ == "__main__":
    print("🧪 Drift Detection Test\n")

    # Reference data
    np.random.seed(42)
    reference = np.random.randn(1000, 10)

    # Detector
    detector = DriftDetector(
        reference_data=reference,
        feature_names=[f"feat_{i}" for i in range(10)],
        window_size=500,
        check_interval=100,
    )

    # Simulate drift
    print("📊 Normal veri ekleniyor...")
    normal_data = np.random.randn(200, 10)
    detector.add_batch(normal_data)

    print("\n📊 Drifted veri ekleniyor (mean shifted)...")
    drifted_data = np.random.randn(200, 10) + 2  # Mean shift
    detector.add_batch(drifted_data)

    # Status
    print("\n📊 Drift Status:")
    status = detector.get_drift_status()
    print(json.dumps(status, indent=2, default=str))
