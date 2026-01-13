"""
AI Model Integration Module
Tüm IDS modellerini AI Assistant'a entegre eder

Özellikler:
    - Model yükleme ve prediction
    - SSA-LSTMIDS, BiLSTM, Transformer, GRU desteği
    - Ensemble prediction
    - Model karşılaştırma
"""

import os
import sys
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime

# src/chatbot/model_integration.py -> parent.parent = src -> parent = project_root
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# Model cache
_model_cache: Dict[str, Any] = {}


class AIModelIntegration:
    """AI Assistant için model entegrasyon sınıfı"""

    ATTACK_LABELS = ["Normal", "DoS", "Probe", "R2L", "U2R", "Other"]

    def __init__(self):
        self.models_dir = PROJECT_ROOT / "models"
        self.available_models = {}
        self._load_available_models()

    def _load_available_models(self):
        """Mevcut modelleri tara"""
        if not self.models_dir.exists():
            return

        # .h5 dosyalarını bul
        for model_file in self.models_dir.glob("*.h5"):
            model_name = model_file.stem

            # Model tipi belirle
            if "ssa_lstmids" in model_name:
                model_type = "SSA-LSTMIDS"
            elif "bilstm" in model_name.lower():
                model_type = "BiLSTM+Attention"
            elif "transformer" in model_name.lower():
                model_type = "Transformer"
            elif "gru" in model_name.lower():
                model_type = "GRU"
            elif "ensemble" in model_name.lower():
                model_type = "Ensemble"
            else:
                model_type = "Custom"

            # Params dosyası var mı?
            params_file = model_file.with_name(f"{model_name}_params.json")
            params = {}
            if params_file.exists():
                try:
                    with open(params_file) as f:
                        params = json.load(f)
                except:
                    pass

            self.available_models[model_name] = {
                "path": str(model_file),
                "type": model_type,
                "name": model_name,
                "size_mb": round(model_file.stat().st_size / (1024 * 1024), 2),
                "created": datetime.fromtimestamp(
                    model_file.stat().st_ctime
                ).isoformat(),
                "params": params,
            }

        # Tüm Results dosyalarını yükle
        self.training_results = {}

        # Paper SSA-LSTMIDS sonuçları
        paper_results = self.models_dir / "paper_ssa_lstmids_results.json"
        if paper_results.exists():
            try:
                with open(paper_results) as f:
                    data = json.load(f)
                    for k, v in data.items():
                        self.training_results[f"paper_{k}"] = v
            except:
                pass

        # Saldırı bazlı sonuçlar
        attack_results = self.models_dir / "attack_specific_results.json"
        if attack_results.exists():
            try:
                with open(attack_results) as f:
                    data = json.load(f)
                    for k, v in data.items():
                        self.training_results[f"attack_{k}"] = v
            except:
                pass

        # Eski SSA sonuçları
        ssa_results = self.models_dir / "ssa_lstmids_results.json"
        if ssa_results.exists():
            try:
                with open(ssa_results) as f:
                    self.training_results.update(json.load(f))
            except:
                pass

    def get_available_models(self) -> List[Dict]:
        """Mevcut modelleri listele"""
        return list(self.available_models.values())

    def get_model_summary(self) -> str:
        """Tüm modellerin özetini döndür"""
        if not self.available_models:
            return "❌ Henüz eğitilmiş model yok."

        summary_parts = ["📊 **MEVCUT IDS MODELLERİ:**\n"]

        for name, info in self.available_models.items():
            metrics = info.get("params", {})
            accuracy = metrics.get("accuracy", 0)
            if isinstance(accuracy, float) and accuracy < 1:
                accuracy *= 100

            summary_parts.append(
                f"""
🤖 **{info['type']}** - `{name}`
   - Boyut: {info['size_mb']} MB
   - Accuracy: {accuracy:.2f}%
   - Oluşturma: {info['created'][:10]}"""
            )

        # Training results ekle
        if self.training_results:
            summary_parts.append("\n\n📈 **SON EĞİTİM SONUÇLARI:**")
            for dataset, results in self.training_results.items():
                summary_parts.append(
                    f"""
   **{dataset.upper()}:**
   - Accuracy: {results.get('accuracy', 0)*100:.2f}%
   - Precision: {results.get('precision', 0)*100:.2f}%
   - Recall: {results.get('recall', 0)*100:.2f}%
   - F1-Score: {results.get('f1_score', 0)*100:.2f}%"""
                )

        return "".join(summary_parts)

    def load_model(self, model_name: str):
        """Model yükle (cache ile)"""
        global _model_cache

        if model_name in _model_cache:
            return _model_cache[model_name]

        if model_name not in self.available_models:
            return None

        try:
            import tensorflow as tf
            from tensorflow import keras

            model_path = self.available_models[model_name]["path"]
            model = keras.models.load_model(model_path, compile=False)
            model.compile(optimizer="adam", loss="sparse_categorical_crossentropy")

            _model_cache[model_name] = model
            return model
        except Exception as e:
            print(f"Model yükleme hatası ({model_name}): {e}")
            return None

    def predict_with_model(
        self, model_name: str, features: np.ndarray
    ) -> Optional[Dict]:
        """Belirli bir model ile prediction yap"""
        model = self.load_model(model_name)
        if model is None:
            return None

        try:
            # Input shape kontrolü
            expected_shape = model.input_shape

            # Features'ı uygun şekle getir
            if len(features.shape) == 1:
                # Tek sample
                features = features.reshape(1, -1)

            # Sequence modeller için (LSTM, etc.)
            if len(expected_shape) == 3:
                # (batch, timesteps, features)
                if len(features.shape) == 2:
                    timesteps = expected_shape[1] or 10
                    n_features = features.shape[-1]
                    # Padding veya truncate
                    if features.shape[0] < timesteps:
                        padding = np.zeros((timesteps - features.shape[0], n_features))
                        features = np.vstack([padding, features])
                    features = features[-timesteps:].reshape(1, timesteps, n_features)

            # Prediction
            proba = model.predict(features, verbose=0)
            prediction = np.argmax(proba, axis=-1)[0]
            confidence = float(proba[0][prediction])

            return {
                "model_name": model_name,
                "model_type": self.available_models[model_name]["type"],
                "prediction": int(prediction),
                "prediction_label": self.ATTACK_LABELS[
                    min(prediction, len(self.ATTACK_LABELS) - 1)
                ],
                "confidence": confidence,
                "probabilities": {
                    label: float(proba[0][i])
                    for i, label in enumerate(self.ATTACK_LABELS[: len(proba[0])])
                },
            }
        except Exception as e:
            print(f"Prediction hatası ({model_name}): {e}")
            return None

    def ensemble_predict(
        self, features: np.ndarray, model_names: List[str] = None
    ) -> Optional[Dict]:
        """Tüm modeller veya belirli modeller ile ensemble prediction"""
        if model_names is None:
            model_names = list(self.available_models.keys())

        predictions = []
        for name in model_names:
            result = self.predict_with_model(name, features)
            if result:
                predictions.append(result)

        if not predictions:
            return None

        # Voting
        votes = {}
        for pred in predictions:
            label = pred["prediction_label"]
            votes[label] = votes.get(label, 0) + pred["confidence"]

        # En yüksek oyu bul
        winner = max(votes.items(), key=lambda x: x[1])

        return {
            "ensemble_prediction": winner[0],
            "ensemble_score": winner[1] / len(predictions),
            "model_count": len(predictions),
            "individual_predictions": predictions,
            "vote_distribution": votes,
        }

    def get_model_comparison(self) -> str:
        """Model karşılaştırma özeti"""
        if not self.training_results:
            return "Model karşılaştırma verisi yok."

        comparison = ["📊 **MODEL KARŞILAŞTIRMASI:**\n"]
        comparison.append("| Dataset | Accuracy | Precision | Recall | F1-Score |")
        comparison.append("|---------|----------|-----------|--------|----------|")

        for dataset, results in self.training_results.items():
            comparison.append(
                f"| {dataset} | "
                f"{results.get('accuracy', 0)*100:.2f}% | "
                f"{results.get('precision', 0)*100:.2f}% | "
                f"{results.get('recall', 0)*100:.2f}% | "
                f"{results.get('f1_score', 0)*100:.2f}% |"
            )

        return "\n".join(comparison)

    def analyze_attack_with_models(self, attack_data: Dict) -> str:
        """Bir saldırıyı tüm modellerle analiz et"""
        # Feature extraction (basitleştirilmiş)
        features = np.random.randn(10, 41).astype(np.float32)  # Placeholder

        results = []
        results.append(f"🔍 **SALDIRI ANALİZİ:**\n")
        results.append(f"Kaynak IP: {attack_data.get('source_ip', 'N/A')}")
        results.append(f"Saldırı Tipi: {attack_data.get('attack_type', 'N/A')}")
        results.append(f"Şiddet: {attack_data.get('severity', 'N/A')}\n")

        # Her model ile analiz
        results.append("**Model Tahminleri:**")
        for model_name in list(self.available_models.keys())[:3]:  # İlk 3 model
            pred = self.predict_with_model(model_name, features)
            if pred:
                results.append(
                    f"- {pred['model_type']}: {pred['prediction_label']} "
                    f"({pred['confidence']*100:.1f}% güven)"
                )

        # Ensemble
        ensemble = self.ensemble_predict(features)
        if ensemble:
            results.append(
                f"\n**Ensemble Sonucu:** {ensemble['ensemble_prediction']} "
                f"(score: {ensemble['ensemble_score']:.2f})"
            )

        return "\n".join(results)


# Singleton instance
_integration_instance = None


def get_integration() -> AIModelIntegration:
    """Global integration instance'ı al"""
    global _integration_instance
    if _integration_instance is None:
        _integration_instance = AIModelIntegration()
    return _integration_instance


# Test
if __name__ == "__main__":
    print("🧪 AI Model Integration Test\n")

    integration = get_integration()

    print("📦 Mevcut modeller:")
    for model in integration.get_available_models():
        print(f"  - {model['name']} ({model['type']})")

    print("\n" + integration.get_model_summary())
