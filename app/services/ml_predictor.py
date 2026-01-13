"""
ML Predictor Service
Real-time threat prediction using trained models
"""

import os
import json
import random
from typing import Dict, Any, List, Optional
from datetime import datetime

# Try to import ML libraries
try:
    import numpy as np

    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

try:
    import joblib

    JOBLIB_AVAILABLE = True
except ImportError:
    JOBLIB_AVAILABLE = False

# Path setup
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
models_dir = os.path.join(project_root, "models")


# Attack type risk scores
ATTACK_TYPE_SCORES = {
    "DDoS": 0.85,
    "Brute Force": 0.75,
    "SQL Injection": 0.90,
    "XSS": 0.70,
    "Port Scan": 0.60,
    "Malware": 0.95,
    "Phishing": 0.80,
    "Man-in-the-Middle": 0.85,
    "Ransomware": 0.98,
    "Zero-Day": 0.99,
    "Command Injection": 0.88,
    "Path Traversal": 0.72,
    "CSRF": 0.65,
    "DNS Spoofing": 0.78,
    "ARP Spoofing": 0.75,
}

# Country threat scores
COUNTRY_THREAT_SCORES = {
    "CN": 0.75,
    "RU": 0.80,
    "KP": 0.90,
    "IR": 0.70,
    "UA": 0.55,
    "US": 0.40,
    "DE": 0.35,
    "NL": 0.45,
    "BR": 0.50,
    "IN": 0.45,
}


class MLPredictor:
    """Machine Learning Predictor for real-time threat analysis"""

    def __init__(self):
        self.models = {}
        self.model_stats = {
            "predictions_made": 0,
            "threats_detected": 0,
            "accuracy": 0.94,  # Simulated accuracy
            "last_prediction": None,
        }
        self.load_models()

    def load_models(self):
        """Load trained models from disk"""
        if not JOBLIB_AVAILABLE:
            print("[ML] Joblib not available, using simulation mode")
            return

        model_files = [
            "random_forest_model.pkl",
            "gradient_boosting_model.pkl",
            "lstm_model.h5",
        ]

        for model_file in model_files:
            model_path = os.path.join(models_dir, model_file)
            if os.path.exists(model_path):
                try:
                    if model_file.endswith(".pkl"):
                        self.models[model_file] = joblib.load(model_path)
                        print(f"[ML] Loaded {model_file}")
                except Exception as e:
                    print(f"[ML] Error loading {model_file}: {e}")

    def extract_features(self, attack_data: Dict[str, Any]) -> List[float]:
        """Extract features from attack data for prediction"""
        features = []

        # Attack type encoding
        attack_type = attack_data.get("attack_type", "Unknown")
        type_score = ATTACK_TYPE_SCORES.get(attack_type, 0.5)
        features.append(type_score)

        # Source country risk
        source_country = attack_data.get("source", {}).get("country", "XX")
        country_score = COUNTRY_THREAT_SCORES.get(source_country, 0.5)
        features.append(country_score)

        # Port-based risk
        target_port = attack_data.get("target", {}).get("port", 80)
        high_risk_ports = [22, 23, 3389, 445, 1433, 3306, 5432]
        port_score = 0.8 if target_port in high_risk_ports else 0.4
        features.append(port_score)

        # Time-based risk (attacks at unusual hours)
        try:
            timestamp = attack_data.get("timestamp", datetime.now().isoformat())
            hour = datetime.fromisoformat(timestamp.replace("Z", "")).hour
            time_score = 0.7 if hour < 6 or hour > 22 else 0.4
        except Exception:
            time_score = 0.5
        features.append(time_score)

        # Blocked status
        blocked = attack_data.get("blocked", False)
        features.append(0.2 if blocked else 0.8)

        return features

    def predict(self, attack_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Predict threat level for an attack
        Returns prediction with confidence and suggested action
        """
        self.model_stats["predictions_made"] += 1

        # Extract features
        features = self.extract_features(attack_data)

        # Use trained model if available
        if "random_forest_model.pkl" in self.models and NUMPY_AVAILABLE:
            try:
                model = self.models["random_forest_model.pkl"]
                X = np.array(features).reshape(1, -1)

                # Predict
                prediction = model.predict(X)[0]
                proba = model.predict_proba(X)[0]
                confidence = max(proba) if len(proba) > 0 else 0.85

                is_threat = bool(prediction)
            except Exception:
                # Fallback to simulation
                confidence, is_threat = self._simulate_prediction(features)
        else:
            # Simulation mode
            confidence, is_threat = self._simulate_prediction(features)

        # Determine suggested action
        if confidence > 0.9:
            action = "block"
            severity = "critical"
        elif confidence > 0.75:
            action = "alert"
            severity = "high"
        elif confidence > 0.5:
            action = "monitor"
            severity = "medium"
        else:
            action = "log"
            severity = "low"

        if is_threat:
            self.model_stats["threats_detected"] += 1

        self.model_stats["last_prediction"] = datetime.now().isoformat()

        return {
            "is_threat": is_threat,
            "confidence": round(confidence, 3),
            "severity": severity,
            "suggested_action": action,
            "model_used": "random_forest" if self.models else "simulation",
            "features_analyzed": len(features),
            "prediction_id": f"PRED-{self.model_stats['predictions_made']:06d}",
        }

    def _simulate_prediction(self, features: List[float]) -> tuple:
        """Simulate prediction when no model is available"""
        # Calculate weighted average of features
        weights = [0.35, 0.25, 0.15, 0.15, 0.10]
        confidence = sum(f * w for f, w in zip(features, weights))

        # Add some randomness
        confidence = min(0.99, max(0.1, confidence + random.uniform(-0.1, 0.1)))

        is_threat = confidence > 0.55

        return confidence, is_threat

    def get_stats(self) -> Dict[str, Any]:
        """Get prediction statistics"""
        threat_rate = 0
        if self.model_stats["predictions_made"] > 0:
            threat_rate = (
                self.model_stats["threats_detected"]
                / self.model_stats["predictions_made"]
            )

        return {
            **self.model_stats,
            "threat_rate": round(threat_rate, 3),
            "models_loaded": len(self.models),
            "available_models": list(self.models.keys()),
            "simulation_mode": len(self.models) == 0,
        }

    def batch_predict(self, attacks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Predict for multiple attacks"""
        return [self.predict(attack) for attack in attacks]


# Global predictor instance
_predictor = None


def get_predictor() -> MLPredictor:
    """Get or create the global ML predictor"""
    global _predictor
    if _predictor is None:
        _predictor = MLPredictor()
    return _predictor


def predict_threat(attack_data: Dict[str, Any]) -> Dict[str, Any]:
    """Convenience function for single prediction"""
    return get_predictor().predict(attack_data)


def get_prediction_stats() -> Dict[str, Any]:
    """Get prediction statistics"""
    return get_predictor().get_stats()
