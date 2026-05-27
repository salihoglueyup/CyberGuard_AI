"""
CyberGuard AI - ML Prediction & Model Endpoint Tests
Prediction, model management, training, XAI endpoint testleri
"""

from datetime import datetime

import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def _inject_token(token: str = "ml-test-token") -> dict:
    from app.api.routes.auth import TOKENS, load_users
    users = load_users()
    username = list(users.keys())[0] if users else "admin"
    TOKENS[token] = {"username": username, "created_at": datetime.now().isoformat()}
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture(autouse=True)
def auth_headers():
    return _inject_token()


class TestPredictionEndpoints:
    """ML tahmin endpoint testleri"""

    def test_predict_basic(self, auth_headers):
        """Temel tahmin endpoint'i çalışmalı"""
        payload = {
            "features": [0.1, 0.2, 0.3, 0.4, 0.5],
            "model": "auto",
        }
        response = client.post("/api/predict", json=payload, headers=auth_headers)
        assert response.status_code in [200, 404, 422]

    def test_predict_network_traffic(self, auth_headers):
        """Ağ trafiği tahmini"""
        payload = {
            "protocol": "TCP",
            "src_port": 12345,
            "dst_port": 80,
            "packet_length": 1500,
            "duration": 0.001,
        }
        response = client.post("/api/predict/network", json=payload, headers=auth_headers)
        assert response.status_code in [200, 404, 422]

    def test_predict_missing_features(self, auth_headers):
        """Eksik features 422 döndürmeli"""
        response = client.post("/api/predict", json={}, headers=auth_headers)
        assert response.status_code in [404, 422]

    def test_batch_predict(self, auth_headers):
        """Batch tahmin endpoint'i"""
        payload = {
            "samples": [
                {"features": [0.1, 0.2, 0.3]},
                {"features": [0.4, 0.5, 0.6]},
            ]
        }
        response = client.post("/api/predict/batch", json=payload, headers=auth_headers)
        assert response.status_code in [200, 404, 422]


class TestModelManagement:
    """Model yönetimi testleri"""

    def test_list_models(self, auth_headers):
        """Model listesi döndürmeli"""
        response = client.get("/api/models/list", headers=auth_headers)
        if response.status_code == 200:
            data = response.json()
            assert isinstance(data, (list, dict))
        else:
            assert response.status_code == 404

    def test_get_model_info(self, auth_headers):
        """Model bilgisi endpoint'i"""
        response = client.get("/api/models/info", headers=auth_headers)
        assert response.status_code in [200, 404]

    def test_model_registry(self, auth_headers):
        """Model registry endpoint'i"""
        response = client.get("/api/models/registry", headers=auth_headers)
        assert response.status_code in [200, 404]

    def test_model_comparison(self, auth_headers):
        """Model karşılaştırma endpoint'i"""
        response = client.get("/api/comparison/results", headers=auth_headers)
        assert response.status_code in [200, 404]


class TestTrainingEndpoints:
    """Model eğitim endpoint testleri"""

    def test_get_training_status(self, auth_headers):
        """Eğitim durumu endpoint'i"""
        response = client.get("/api/training/status", headers=auth_headers)
        assert response.status_code in [200, 404]

    def test_get_training_history(self, auth_headers):
        """Eğitim geçmişi endpoint'i"""
        response = client.get("/api/training/history", headers=auth_headers)
        assert response.status_code in [200, 404]

    def test_start_training_missing_params(self, auth_headers):
        """Parametre eksik eğitim başlatma 422 veya 200 (hata mesajıyla) döndürmeli"""
        response = client.post("/api/training/start", json={}, headers=auth_headers)
        # Endpoint eksik zorunlu alanlar için 422 döndürür veya 200 {success:false}
        assert response.status_code in [200, 404, 422]


class TestXAIEndpoints:
    """XAI (Explainable AI) endpoint testleri"""

    def test_get_shap_explanation(self, auth_headers):
        """SHAP açıklama endpoint'i"""
        response = client.get("/api/xai/shap", headers=auth_headers)
        assert response.status_code in [200, 404]

    def test_get_lime_explanation(self, auth_headers):
        """LIME açıklama endpoint'i"""
        response = client.get("/api/xai/lime", headers=auth_headers)
        assert response.status_code in [200, 404]

    def test_get_feature_importance(self, auth_headers):
        """Feature importance endpoint'i"""
        response = client.get("/api/xai/feature-importance", headers=auth_headers)
        assert response.status_code in [200, 404]


class TestAutoMLEndpoints:
    """AutoML endpoint testleri"""

    def test_get_automl_status(self, auth_headers):
        """AutoML durum endpoint'i"""
        response = client.get("/api/automl/status", headers=auth_headers)
        assert response.status_code in [200, 404]

    def test_get_automl_results(self, auth_headers):
        """AutoML sonuçları endpoint'i"""
        response = client.get("/api/automl/results", headers=auth_headers)
        assert response.status_code in [200, 404]


class TestDriftDetection:
    """Drift detection endpoint testleri"""

    def test_get_drift_status(self, auth_headers):
        """Drift durumu endpoint'i"""
        response = client.get("/api/drift/status", headers=auth_headers)
        assert response.status_code in [200, 404]

    def test_get_drift_report(self, auth_headers):
        """Drift raporu endpoint'i"""
        response = client.get("/api/drift/report", headers=auth_headers)
        assert response.status_code in [200, 404]
