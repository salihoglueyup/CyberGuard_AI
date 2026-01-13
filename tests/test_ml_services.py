"""
CyberGuard AI - ML Service Tests
Tests for ML prediction and model services
"""

import pytest
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestMLPredictor:
    """ML Predictor service tests"""

    def test_import_predictor(self):
        """Test ML predictor can be imported"""
        try:
            from app.services.ml_predictor import predict_threat, get_prediction_stats

            assert True
        except ImportError:
            pytest.skip("ML predictor not available")

    def test_predict_threat(self):
        """Test threat prediction"""
        try:
            from app.services.ml_predictor import predict_threat

            sample_attack = {
                "source": {"country": "CN", "ip": "185.220.101.1"},
                "target": {"country": "TR", "ip": "192.168.1.1"},
                "attack_type": "DDoS",
                "severity": "high",
            }

            result = predict_threat(sample_attack)

            assert "is_threat" in result
            assert "confidence" in result
            assert "severity" in result
            assert 0 <= result["confidence"] <= 1
        except ImportError:
            pytest.skip("ML predictor not available")

    def test_get_prediction_stats(self):
        """Test prediction statistics"""
        try:
            from app.services.ml_predictor import get_prediction_stats

            stats = get_prediction_stats()

            assert isinstance(stats, dict)
        except ImportError:
            pytest.skip("ML predictor not available")


class TestGeoIPService:
    """GeoIP service tests"""

    def test_import_geoip(self):
        """Test GeoIP service can be imported"""
        try:
            from app.services.geoip import get_ip_location

            assert True
        except ImportError:
            pytest.skip("GeoIP service not available")

    def test_get_location_private_ip(self):
        """Test private IP handling"""
        try:
            from app.services.geoip import get_ip_location

            result = get_ip_location("192.168.1.1")

            assert result is not None
            assert "country" in result or result.get("status") == "private"
        except ImportError:
            pytest.skip("GeoIP service not available")

    def test_get_location_public_ip(self):
        """Test public IP lookup"""
        try:
            from app.services.geoip import get_ip_location

            # Google DNS as test
            result = get_ip_location("8.8.8.8")

            assert result is not None
        except ImportError:
            pytest.skip("GeoIP service not available")


class TestAttackSimulator:
    """Attack simulation tests"""

    def test_generate_simulated_attacks(self):
        """Test attack simulation"""
        try:
            from app.api.routes.attack_map import generate_simulated_attacks

            attacks = generate_simulated_attacks(5)

            assert len(attacks) == 5
            for attack in attacks:
                assert "id" in attack
                assert "source" in attack
                assert "target" in attack
                assert "attack_type" in attack
                assert attack["source_type"] == "simulation"
        except ImportError:
            pytest.skip("Attack map not available")

    def test_simulated_attack_structure(self):
        """Test attack data structure"""
        try:
            from app.api.routes.attack_map import generate_simulated_attacks

            attacks = generate_simulated_attacks(1)
            attack = attacks[0]

            # Check source structure
            assert "country" in attack["source"]
            assert "lat" in attack["source"]
            assert "lng" in attack["source"]
            assert "ip" in attack["source"]

            # Check target structure
            assert "country" in attack["target"]
            assert attack["target"]["country"] == "TR"

            # Check other fields
            assert attack["attack_type"] in [
                "DDoS",
                "Brute Force",
                "Malware",
                "Phishing",
                "SQL Injection",
                "XSS",
                "Port Scan",
                "Bot",
            ]
        except ImportError:
            pytest.skip("Attack map not available")


class TestModelRegistry:
    """Model registry tests"""

    def test_model_files_exist(self):
        """Test trained model files exist"""
        models_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models"
        )

        assert os.path.exists(models_dir)

        # Check for at least one model file
        keras_files = [f for f in os.listdir(models_dir) if f.endswith(".keras")]
        assert len(keras_files) > 0, "No trained model files found"

    def test_model_registry_json(self):
        """Test model registry JSON exists"""
        registry_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "models",
            "model_registry.json",
        )

        if os.path.exists(registry_path):
            import json

            with open(registry_path, "r") as f:
                data = json.load(f)
            assert isinstance(data, (dict, list))


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
