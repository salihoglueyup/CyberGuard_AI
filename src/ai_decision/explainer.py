"""
Attack Explainer - CyberGuard AI
=================================

XAI (Explainable AI) entegrasyonu.
SHAP ve Attention-based açıklamalar.

Özellikler:
    - SHAP feature importance
    - Attention heatmaps
    - Attack → Feature mapping
    - Human-readable explanations
"""

import os
import sys
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from datetime import datetime
import logging

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
sys.path.insert(0, PROJECT_ROOT)

logger = logging.getLogger("AttackExplainer")


# Attack type → Feature mapping (domain knowledge)
ATTACK_FEATURE_PATTERNS = {
    "DDoS": {
        "key_features": [
            "packet_rate",
            "flow_duration",
            "total_fwd_packets",
            "total_bwd_packets",
        ],
        "description": "High packet rate with abnormal flow patterns",
        "indicators": ["Packet rate ↑", "Flow duration ↑", "Bandwidth saturation"],
    },
    "DoS": {
        "key_features": ["packet_rate", "syn_flag_count", "flow_bytes_per_sec"],
        "description": "Single source overwhelming target with requests",
        "indicators": ["SYN flags ↑", "Connection attempts ↑"],
    },
    "Probe": {
        "key_features": ["dst_port_entropy", "unique_ports", "scan_rate"],
        "description": "Port scanning and reconnaissance activity",
        "indicators": ["Port entropy ↑", "Unique ports ↑", "Short connections"],
    },
    "PortScan": {
        "key_features": ["dst_port_entropy", "unique_ports", "flow_duration"],
        "description": "Sequential port scanning detected",
        "indicators": ["Port variance ↑", "Quick connects/disconnects"],
    },
    "BruteForce": {
        "key_features": ["failed_login_count", "same_dest_rate", "retry_count"],
        "description": "Multiple authentication attempts",
        "indicators": ["Failed logins ↑", "Same destination ↑"],
    },
    "WebAttack": {
        "key_features": ["http_error_rate", "payload_size", "request_rate"],
        "description": "HTTP-based attack (SQL injection, XSS, etc.)",
        "indicators": ["Error rate ↑", "Unusual payloads"],
    },
    "Bot": {
        "key_features": ["pattern_regularity", "c2_communication", "periodic_activity"],
        "description": "Automated bot behavior detected",
        "indicators": ["Regular patterns", "C2 beacons"],
    },
    "Infiltration": {
        "key_features": [
            "data_exfiltration",
            "internal_spread",
            "privilege_escalation",
        ],
        "description": "Internal network compromise",
        "indicators": ["Data transfer ↑", "Lateral movement"],
    },
    "ZERO_DAY": {
        "key_features": ["reconstruction_error", "anomaly_score", "distribution_shift"],
        "description": "Novel attack pattern not seen in training",
        "indicators": ["High VAE error", "Unknown signature"],
    },
}

# Standard IDS feature names (CICIDS2017)
CICIDS_FEATURES = [
    "flow_duration",
    "total_fwd_packets",
    "total_bwd_packets",
    "total_fwd_bytes",
    "total_bwd_bytes",
    "fwd_packet_len_max",
    "fwd_packet_len_min",
    "fwd_packet_len_mean",
    "fwd_packet_len_std",
    "bwd_packet_len_max",
    "bwd_packet_len_min",
    "bwd_packet_len_mean",
    "bwd_packet_len_std",
    "flow_bytes_per_sec",
    "flow_packets_per_sec",
    "flow_iat_mean",
    "flow_iat_std",
    "flow_iat_max",
    "flow_iat_min",
    "fwd_iat_total",
    "fwd_iat_mean",
    "fwd_iat_std",
    "fwd_iat_max",
    "fwd_iat_min",
    "bwd_iat_total",
    "bwd_iat_mean",
    "bwd_iat_std",
    "bwd_iat_max",
    "bwd_iat_min",
    "fwd_psh_flags",
    "bwd_psh_flags",
    "fwd_urg_flags",
    "bwd_urg_flags",
    "fwd_header_len",
    "bwd_header_len",
    "fwd_packets_per_sec",
    "bwd_packets_per_sec",
    "min_packet_len",
    "max_packet_len",
    "packet_len_mean",
    "packet_len_std",
    "packet_len_var",
    "fin_flag_cnt",
    "syn_flag_cnt",
    "rst_flag_cnt",
    "psh_flag_cnt",
    "ack_flag_cnt",
    "urg_flag_cnt",
    "cwe_flag_cnt",
    "ece_flag_cnt",
    "down_up_ratio",
    "avg_packet_size",
    "avg_fwd_segment_size",
    "avg_bwd_segment_size",
    "fwd_header_len_2",
    "fwd_avg_bytes_per_bulk",
    "fwd_avg_packets_per_bulk",
    "fwd_avg_bulk_rate",
    "bwd_avg_bytes_per_bulk",
    "bwd_avg_packets_per_bulk",
    "bwd_avg_bulk_rate",
    "subflow_fwd_packets",
    "subflow_fwd_bytes",
    "subflow_bwd_packets",
    "subflow_bwd_bytes",
    "init_win_bytes_fwd",
    "init_win_bytes_bwd",
    "act_data_pkt_fwd",
    "min_seg_size_forward",
    "active_mean",
    "active_std",
    "active_max",
    "active_min",
    "idle_mean",
    "idle_std",
    "idle_max",
    "idle_min",
]


class AttackExplainer:
    """
    Saldırı açıklama motoru

    SHAP ve Attention-based explanations.
    """

    def __init__(
        self,
        model=None,
        feature_names: List[str] = None,
        use_shap: bool = True,
    ):
        """
        Args:
            model: Keras/sklearn model
            feature_names: Feature isimleri
            use_shap: SHAP kullan (True=daha doğru, False=daha hızlı)
        """
        self.model = model
        self.feature_names = feature_names or CICIDS_FEATURES
        self.use_shap = use_shap

        self.shap_explainer = None
        self.background_data = None

        logger.info("🔍 AttackExplainer initialized")

    def set_model(self, model):
        """Model set et"""
        self.model = model
        self.shap_explainer = None  # Reset

    def compute_feature_importance(
        self,
        X: np.ndarray,
        method: str = "gradient",  # gradient, permutation, shap
    ) -> Dict:
        """
        Feature importance hesapla

        Args:
            X: Input data (batch)
            method: gradient, permutation, or shap

        Returns:
            {feature_name: importance_score}
        """
        if self.model is None:
            # Fallback: variance-based importance
            variances = np.var(X, axis=0)
            importance = variances / np.sum(variances)

            return {
                name: float(imp)
                for name, imp in zip(self.feature_names[: len(importance)], importance)
            }

        if method == "gradient":
            return self._gradient_importance(X)
        elif method == "permutation":
            return self._permutation_importance(X)
        else:
            return self._shap_importance(X)

    def _gradient_importance(self, X: np.ndarray) -> Dict:
        """Gradient-based importance (fast)"""
        try:
            import tensorflow as tf

            X_tensor = tf.convert_to_tensor(X, dtype=tf.float32)

            with tf.GradientTape() as tape:
                tape.watch(X_tensor)
                predictions = self.model(X_tensor)
                if len(predictions.shape) > 1:
                    predictions = tf.reduce_max(predictions, axis=1)

            gradients = tape.gradient(predictions, X_tensor)
            importance = tf.reduce_mean(tf.abs(gradients), axis=0).numpy()

            # Normalize
            importance = importance / (np.sum(importance) + 1e-8)

            return {
                name: float(imp)
                for name, imp in zip(self.feature_names[: len(importance)], importance)
            }
        except Exception as e:
            logger.warning(f"Gradient importance failed: {e}")
            return {}

    def _permutation_importance(self, X: np.ndarray) -> Dict:
        """Permutation importance (slow but accurate)"""
        try:
            baseline_pred = self.model.predict(X, verbose=0)
            baseline_score = np.mean(np.max(baseline_pred, axis=1))

            importance = {}
            n_features = X.shape[1]

            for i in range(min(n_features, len(self.feature_names))):
                X_permuted = X.copy()
                np.random.shuffle(X_permuted[:, i])

                permuted_pred = self.model.predict(X_permuted, verbose=0)
                permuted_score = np.mean(np.max(permuted_pred, axis=1))

                importance[self.feature_names[i]] = float(
                    baseline_score - permuted_score
                )

            # Normalize
            total = sum(abs(v) for v in importance.values()) + 1e-8
            return {k: abs(v) / total for k, v in importance.items()}
        except Exception as e:
            logger.warning(f"Permutation importance failed: {e}")
            return {}

    def _shap_importance(self, X: np.ndarray) -> Dict:
        """SHAP values (most accurate)"""
        try:
            import shap

            if self.shap_explainer is None:
                # Background data
                if self.background_data is None:
                    self.background_data = X[: min(100, len(X))]

                self.shap_explainer = shap.KernelExplainer(
                    lambda x: self.model.predict(x, verbose=0), self.background_data
                )

            shap_values = self.shap_explainer.shap_values(X[: min(50, len(X))])

            if isinstance(shap_values, list):
                shap_values = np.abs(np.array(shap_values)).mean(axis=0)

            importance = np.mean(np.abs(shap_values), axis=0)
            importance = importance / (np.sum(importance) + 1e-8)

            return {
                name: float(imp)
                for name, imp in zip(self.feature_names[: len(importance)], importance)
            }
        except Exception as e:
            logger.warning(f"SHAP importance failed: {e}")
            return self._gradient_importance(X)

    def explain_attack(
        self,
        X: np.ndarray,
        attack_type: str,
        top_n: int = 5,
    ) -> Dict:
        """
        Saldırı için human-readable açıklama

        Args:
            X: Attack sample(s)
            attack_type: Detected attack type
            top_n: Top N önemli feature

        Returns:
            Detailed explanation dict
        """
        # Feature importance hesapla
        importance = self.compute_feature_importance(X)

        # Top features
        sorted_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)
        top_features = sorted_features[:top_n]

        # Domain knowledge
        pattern = ATTACK_FEATURE_PATTERNS.get(attack_type, {})

        # Evidence oluştur
        evidence = []
        for feat_name, imp in top_features:
            if imp > 0.05:  # En az %5 önemli
                direction = (
                    "↑"
                    if np.mean(X[:, self.feature_names.index(feat_name)]) > 0
                    else "↓"
                )
                evidence.append(f"{feat_name} {direction} ({imp*100:.1f}%)")

        # Natural language explanation
        explanation = self._generate_explanation(attack_type, evidence, pattern)

        return {
            "attack_type": attack_type,
            "description": pattern.get("description", f"{attack_type} detected"),
            "top_features": [
                {"name": n, "importance": float(i)} for n, i in top_features
            ],
            "evidence": evidence,
            "indicators": pattern.get("indicators", []),
            "explanation": explanation,
            "confidence_factors": {
                "feature_match": len(evidence) / top_n,
                "pattern_match": self._check_pattern_match(importance, pattern),
            },
            "timestamp": datetime.now().isoformat(),
        }

    def _check_pattern_match(self, importance: Dict, pattern: Dict) -> float:
        """Pattern match skoru hesapla"""
        if not pattern.get("key_features"):
            return 0.5

        key_features = pattern["key_features"]
        match_count = sum(1 for f in key_features if f in importance)

        return match_count / len(key_features)

    def _generate_explanation(
        self,
        attack_type: str,
        evidence: List[str],
        pattern: Dict,
    ) -> str:
        """Natural language explanation oluştur"""

        base = pattern.get("description", f"{attack_type} attack detected")

        if evidence:
            evidence_str = ", ".join(evidence[:3])
            explanation = f"{base}. Key indicators: {evidence_str}."
        else:
            explanation = f"{base}."

        # Add recommendation
        recommendations = {
            "DDoS": "Recommended: Enable rate limiting and implement traffic filtering.",
            "DoS": "Recommended: Block source IP and monitor for additional attempts.",
            "Probe": "Recommended: Review firewall rules and monitor for follow-up attacks.",
            "PortScan": "Recommended: Close unnecessary ports and log scanning activity.",
            "BruteForce": "Recommended: Enable account lockout and implement CAPTCHA.",
            "WebAttack": "Recommended: Update WAF rules and review application logs.",
            "ZERO_DAY": "Recommended: Isolate affected systems and conduct forensic analysis.",
        }

        rec = recommendations.get(
            attack_type, "Recommended: Monitor and investigate further."
        )

        return f"{explanation} {rec}"

    def get_attention_weights(self, model, X: np.ndarray) -> Optional[np.ndarray]:
        """
        Attention-based modelden weight'leri çıkar

        Returns:
            attention_weights: (batch, seq_len, seq_len) or None
        """
        try:
            import tensorflow as tf

            # Find attention layer
            attention_layer = None
            for layer in model.layers:
                if "attention" in layer.name.lower():
                    attention_layer = layer
                    break

            if attention_layer is None:
                return None

            # Create sub-model to get attention output
            attention_model = tf.keras.Model(
                inputs=model.input, outputs=attention_layer.output
            )

            # Get attention weights
            if len(X.shape) == 2:
                X = X.reshape(X.shape[0], 1, X.shape[1])

            attention_output = attention_model.predict(X, verbose=0)

            return attention_output
        except Exception as e:
            logger.warning(f"Could not extract attention weights: {e}")
            return None

    def visualize_explanation(self, explanation: Dict) -> Dict:
        """Görselleştirme için veri hazırla"""
        return {
            "chart_data": {
                "labels": [f["name"] for f in explanation["top_features"]],
                "values": [f["importance"] for f in explanation["top_features"]],
            },
            "summary": {
                "attack": explanation["attack_type"],
                "description": explanation["description"],
                "evidence": explanation["evidence"],
            },
            "recommendation": (
                explanation["explanation"].split(" Recommended: ")[-1]
                if "Recommended:" in explanation["explanation"]
                else ""
            ),
        }


# ============= Factory Function =============


def explain_attack(
    X: np.ndarray,
    attack_type: str,
    model=None,
    feature_names: List[str] = None,
) -> Dict:
    """
    Quick attack explanation

    Args:
        X: Attack sample
        attack_type: Detected type
        model: Optional model for feature importance
        feature_names: Optional feature names
    """
    explainer = AttackExplainer(model=model, feature_names=feature_names)
    return explainer.explain_attack(X, attack_type)


# ============= Test =============

if __name__ == "__main__":
    print("🧪 Attack Explainer Test\n")

    # Simulated attack data
    np.random.seed(42)
    X_ddos = np.random.randn(10, 78).astype(np.float32)
    X_ddos[:, 0] *= 10  # flow_duration high
    X_ddos[:, 1] *= 5  # total_fwd_packets high

    # Explain
    explainer = AttackExplainer()
    explanation = explainer.explain_attack(X_ddos, "DDoS", top_n=5)

    print(f"Attack Type: {explanation['attack_type']}")
    print(f"Description: {explanation['description']}")
    print(f"Top Features:")
    for f in explanation["top_features"]:
        print(f"  - {f['name']}: {f['importance']*100:.1f}%")
    print(f"\nExplanation: {explanation['explanation']}")

    print("\n✅ Test completed!")
