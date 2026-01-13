"""
Model A/B Testing Framework - CyberGuard AI
============================================

Modelleri gerçek trafikte karşılaştırma.

Özellikler:
    - Traffic splitting
    - Statistical significance testing
    - Performans karşılaştırma
"""

import os
import sys
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum
import logging
from collections import defaultdict

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logger = logging.getLogger("ABTesting")


class TestStatus(Enum):
    DRAFT = "draft"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"


@dataclass
class ModelVariant:
    """A/B test model varyantı"""

    variant_id: str
    model_name: str
    model_path: str
    traffic_weight: float = 0.5
    predictions: List[Dict] = field(default_factory=list)
    metrics: Dict = field(default_factory=dict)


@dataclass
class ABTest:
    """Bir A/B test instance"""

    test_id: str
    name: str
    description: str
    variant_a: ModelVariant
    variant_b: ModelVariant
    status: TestStatus = TestStatus.DRAFT
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    winner: Optional[str] = None


class ABTestingEngine:
    """
    Model A/B Testing Engine
    """

    def __init__(self):
        self.tests: Dict[str, ABTest] = {}
        self.active_test_id: Optional[str] = None
        self.loaded_models: Dict[str, Any] = {}

        # Results directory
        self.results_dir = PROJECT_ROOT / "models" / "ab_test_results"
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def create_test(
        self,
        name: str,
        description: str,
        model_a_name: str,
        model_a_path: str,
        model_b_name: str,
        model_b_path: str,
        traffic_split: Tuple[float, float] = (0.5, 0.5),
    ) -> ABTest:
        """
        Yeni A/B test oluştur
        """
        test_id = f"ab_test_{len(self.tests)+1:03d}_{datetime.now().strftime('%Y%m%d')}"

        variant_a = ModelVariant(
            variant_id="variant_a",
            model_name=model_a_name,
            model_path=model_a_path,
            traffic_weight=traffic_split[0],
        )

        variant_b = ModelVariant(
            variant_id="variant_b",
            model_name=model_b_name,
            model_path=model_b_path,
            traffic_weight=traffic_split[1],
        )

        test = ABTest(
            test_id=test_id,
            name=name,
            description=description,
            variant_a=variant_a,
            variant_b=variant_b,
        )

        self.tests[test_id] = test

        print(f"✅ A/B Test oluşturuldu: {test_id}")
        print(f"   Variant A: {model_a_name} ({traffic_split[0]*100:.0f}%)")
        print(f"   Variant B: {model_b_name} ({traffic_split[1]*100:.0f}%)")

        return test

    def start_test(self, test_id: str) -> bool:
        """Test başlat"""
        if test_id not in self.tests:
            return False

        test = self.tests[test_id]

        # Modelleri yükle
        self._load_models(test)

        test.status = TestStatus.RUNNING
        test.started_at = datetime.now().isoformat()
        self.active_test_id = test_id

        print(f"🚀 A/B Test başlatıldı: {test_id}")
        return True

    def stop_test(self, test_id: str) -> bool:
        """Test durdur"""
        if test_id not in self.tests:
            return False

        test = self.tests[test_id]
        test.status = TestStatus.COMPLETED
        test.completed_at = datetime.now().isoformat()

        # Winner belirle
        results = self.analyze_test(test_id)
        if results.get("winner"):
            test.winner = results["winner"]

        if self.active_test_id == test_id:
            self.active_test_id = None

        # Sonuçları kaydet
        self._save_results(test_id)

        print(f"🏁 A/B Test tamamlandı: {test_id}")
        return True

    def _load_models(self, test: ABTest):
        """Modelleri yükle"""
        from tensorflow import keras

        for variant in [test.variant_a, test.variant_b]:
            if variant.model_path not in self.loaded_models:
                try:
                    if os.path.exists(variant.model_path):
                        model = keras.models.load_model(
                            variant.model_path, compile=False
                        )
                        model.compile(
                            optimizer="adam", loss="sparse_categorical_crossentropy"
                        )
                        self.loaded_models[variant.model_path] = model
                        print(f"   ✅ Yüklendi: {variant.model_name}")
                except Exception as e:
                    logger.error(f"Model yükleme hatası: {e}")

    def predict(
        self, test_id: str, features: np.ndarray, ground_truth: Optional[int] = None
    ) -> Dict:
        """
        A/B test ile prediction yap

        Traffic weight'e göre A veya B modeli seçilir
        """
        if test_id not in self.tests:
            return {"error": "Test bulunamadı"}

        test = self.tests[test_id]

        if test.status != TestStatus.RUNNING:
            return {"error": "Test çalışmıyor"}

        # Traffic splitting
        rand = np.random.random()

        if rand < test.variant_a.traffic_weight:
            variant = test.variant_a
        else:
            variant = test.variant_b

        # Prediction
        model = self.loaded_models.get(variant.model_path)
        if model is None:
            return {"error": f"Model yüklenmemiş: {variant.model_name}"}

        try:
            # Ensure correct shape
            if len(features.shape) == 2:
                features = features.reshape(1, *features.shape)
            elif len(features.shape) == 1:
                features = features.reshape(1, 1, -1)

            proba = model.predict(features, verbose=0)
            prediction = int(np.argmax(proba, axis=-1)[0])
            confidence = float(proba[0][prediction])

            # Record
            record = {
                "timestamp": datetime.now().isoformat(),
                "prediction": prediction,
                "confidence": confidence,
                "ground_truth": ground_truth,
                "correct": (
                    prediction == ground_truth if ground_truth is not None else None
                ),
            }

            variant.predictions.append(record)

            return {
                "variant": variant.variant_id,
                "model_name": variant.model_name,
                "prediction": prediction,
                "confidence": confidence,
            }

        except Exception as e:
            return {"error": str(e)}

    def analyze_test(self, test_id: str) -> Dict:
        """
        A/B test sonuçlarını analiz et
        """
        if test_id not in self.tests:
            return {"error": "Test bulunamadı"}

        test = self.tests[test_id]

        def calculate_metrics(predictions: List[Dict]) -> Dict:
            if not predictions:
                return {"count": 0}

            correct = [p for p in predictions if p.get("correct") is True]
            total_with_gt = [
                p for p in predictions if p.get("ground_truth") is not None
            ]

            return {
                "count": len(predictions),
                "accuracy": (
                    len(correct) / len(total_with_gt) if total_with_gt else None
                ),
                "avg_confidence": np.mean([p["confidence"] for p in predictions]),
            }

        metrics_a = calculate_metrics(test.variant_a.predictions)
        metrics_b = calculate_metrics(test.variant_b.predictions)

        # Statistical significance test
        significance = self._calculate_significance(
            test.variant_a.predictions, test.variant_b.predictions
        )

        # Winner
        winner = None
        if metrics_a.get("accuracy") and metrics_b.get("accuracy"):
            if significance.get("significant"):
                if metrics_a["accuracy"] > metrics_b["accuracy"]:
                    winner = "variant_a"
                else:
                    winner = "variant_b"

        return {
            "test_id": test_id,
            "test_name": test.name,
            "status": test.status.value,
            "variant_a": {
                "model_name": test.variant_a.model_name,
                "metrics": metrics_a,
            },
            "variant_b": {
                "model_name": test.variant_b.model_name,
                "metrics": metrics_b,
            },
            "statistical_significance": significance,
            "winner": winner,
            "recommendation": self._get_recommendation(
                metrics_a, metrics_b, significance
            ),
        }

    def _calculate_significance(
        self, predictions_a: List[Dict], predictions_b: List[Dict]
    ) -> Dict:
        """Statistical significance hesapla"""

        # Minimum sample
        if len(predictions_a) < 30 or len(predictions_b) < 30:
            return {
                "significant": False,
                "p_value": None,
                "reason": "Yeterli sample yok (min 30 gerekli)",
            }

        # Correct rates
        correct_a = [
            1 if p.get("correct") else 0
            for p in predictions_a
            if p.get("ground_truth") is not None
        ]
        correct_b = [
            1 if p.get("correct") else 0
            for p in predictions_b
            if p.get("ground_truth") is not None
        ]

        if not correct_a or not correct_b:
            return {
                "significant": False,
                "p_value": None,
                "reason": "Ground truth verisi yok",
            }

        try:
            from scipy import stats

            # Two-proportion z-test
            n_a, n_b = len(correct_a), len(correct_b)
            p_a, p_b = np.mean(correct_a), np.mean(correct_b)
            p_pool = (sum(correct_a) + sum(correct_b)) / (n_a + n_b)

            if p_pool == 0 or p_pool == 1:
                return {"significant": False, "p_value": 1.0}

            se = np.sqrt(p_pool * (1 - p_pool) * (1 / n_a + 1 / n_b))
            z_score = (p_a - p_b) / se if se > 0 else 0
            p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))

            return {
                "significant": p_value < 0.05,
                "p_value": float(p_value),
                "z_score": float(z_score),
                "confidence_level": "95%",
            }

        except ImportError:
            # Fallback without scipy
            return {"significant": False, "reason": "scipy gerekli"}

    def _get_recommendation(
        self, metrics_a: Dict, metrics_b: Dict, significance: Dict
    ) -> str:
        """Öneri oluştur"""
        if not significance.get("significant"):
            return "Henüz istatistiksel olarak anlamlı bir fark yok. Daha fazla veri toplayın."

        acc_a = metrics_a.get("accuracy", 0) or 0
        acc_b = metrics_b.get("accuracy", 0) or 0

        diff = abs(acc_a - acc_b) * 100

        if acc_a > acc_b:
            return f"Variant A (accuracy: {acc_a*100:.1f}%) öneriliyor. +{diff:.1f}% daha iyi."
        else:
            return f"Variant B (accuracy: {acc_b*100:.1f}%) öneriliyor. +{diff:.1f}% daha iyi."

    def get_test_status(self, test_id: str) -> Dict:
        """Test durumunu getir"""
        if test_id not in self.tests:
            return {"error": "Test bulunamadı"}

        test = self.tests[test_id]

        return {
            "test_id": test_id,
            "name": test.name,
            "status": test.status.value,
            "variant_a_count": len(test.variant_a.predictions),
            "variant_b_count": len(test.variant_b.predictions),
            "started_at": test.started_at,
            "winner": test.winner,
        }

    def list_tests(self) -> List[Dict]:
        """Tüm testleri listele"""
        return [
            {
                "test_id": t.test_id,
                "name": t.name,
                "status": t.status.value,
                "variant_a": t.variant_a.model_name,
                "variant_b": t.variant_b.model_name,
                "winner": t.winner,
            }
            for t in self.tests.values()
        ]

    def _save_results(self, test_id: str):
        """Sonuçları kaydet"""
        test = self.tests.get(test_id)
        if not test:
            return

        results = self.analyze_test(test_id)
        results_file = self.results_dir / f"{test_id}_results.json"

        with open(results_file, "w") as f:
            json.dump(results, f, indent=2, default=str)

        print(f"💾 Sonuçlar kaydedildi: {results_file}")


# Singleton
_ab_engine: Optional[ABTestingEngine] = None


def get_ab_engine() -> ABTestingEngine:
    """Global A/B testing engine"""
    global _ab_engine
    if _ab_engine is None:
        _ab_engine = ABTestingEngine()
    return _ab_engine


# Test
if __name__ == "__main__":
    print("🧪 A/B Testing Engine Test\n")

    engine = ABTestingEngine()

    # Test oluştur
    test = engine.create_test(
        name="LSTM vs GRU",
        description="LSTM ve GRU performans karşılaştırması",
        model_a_name="LSTM Model",
        model_a_path="models/lstm_model.h5",
        model_b_name="GRU Model",
        model_b_path="models/gru_model.h5",
        traffic_split=(0.5, 0.5),
    )

    print(f"\n📋 Test listesi:")
    print(json.dumps(engine.list_tests(), indent=2))
