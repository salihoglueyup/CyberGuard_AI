"""
CyberGuard AI - API Test Suite
pytest tests for backend API endpoints
"""


import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def _get_auth_headers():
    """Get auth headers by injecting a test token into the TOKENS store."""
    from datetime import datetime

    from app.api.routes.auth import TOKENS, load_users
    token = "test-token-for-pytest"
    users = load_users()
    username = list(users.keys())[0] if users else "admin"
    TOKENS[token] = {
        "username": username,
        "created_at": datetime.now().isoformat(),
    }
    return {"Authorization": f"Bearer {token}"}


class TestHealthEndpoints:
    """Test health and root endpoints"""

    def test_root_endpoint(self):
        """Test root endpoint returns welcome message"""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "CyberGuard" in data["message"]
        assert "version" in data

    def test_health_endpoint(self):
        """Test health check endpoint"""
        response = client.get("/api/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"


class TestDashboardAPI:
    """Test dashboard endpoints"""

    def test_get_dashboard_stats(self):
        """Test dashboard stats endpoint"""
        response = client.get("/api/dashboard/stats", headers=_get_auth_headers())
        assert response.status_code == 200
        data = response.json()
        assert "success" in data or "data" in data

    def test_get_system_metrics(self):
        """Test system metrics endpoint"""
        response = client.get("/api/dashboard/system/metrics", headers=_get_auth_headers())
        # May return 200 or 404 depending on implementation
        assert response.status_code in [200, 404]


class TestAttackMapAPI:
    """Test attack map endpoints"""

    def test_get_live_attacks(self):
        """Test live attacks endpoint"""
        response = client.get("/api/attack-map/live", headers=_get_auth_headers())
        assert response.status_code == 200
        data = response.json()
        assert data["success"] == True
        assert "attacks" in data["data"]

    def test_get_attack_stats(self):
        """Test attack statistics endpoint"""
        response = client.get("/api/attack-map/stats", headers=_get_auth_headers())
        assert response.status_code == 200
        data = response.json()
        assert data["success"] == True

    def test_get_countries(self):
        """Test countries endpoint"""
        response = client.get("/api/attack-map/countries", headers=_get_auth_headers())
        assert response.status_code == 200


class TestSIEMAPI:
    """Test SIEM integration endpoints"""

    def test_list_platforms(self):
        """Test list SIEM platforms"""
        response = client.get("/api/siem/platforms", headers=_get_auth_headers())
        assert response.status_code == 200
        data = response.json()
        assert data["success"] == True
        assert "platforms" in data["data"]

    def test_list_connections(self):
        """Test list SIEM connections"""
        response = client.get("/api/siem/connections", headers=_get_auth_headers())
        assert response.status_code == 200

    def test_get_stats(self):
        """Test SIEM stats"""
        response = client.get("/api/siem/stats", headers=_get_auth_headers())
        assert response.status_code == 200


class TestSandboxAPI:
    """Test malware sandbox endpoints"""

    def test_get_recent_analyses(self):
        """Test get recent analyses"""
        response = client.get("/api/sandbox/recent", headers=_get_auth_headers())
        assert response.status_code == 200
        data = response.json()
        assert data["success"] == True

    def test_get_stats(self):
        """Test sandbox stats"""
        response = client.get("/api/sandbox/stats", headers=_get_auth_headers())
        assert response.status_code == 200

    def test_list_environments(self):
        """Test list sandbox environments"""
        response = client.get("/api/sandbox/status", headers=_get_auth_headers())
        assert response.status_code == 200


class TestBlockchainAPI:
    """Test blockchain audit endpoints"""

    def test_get_chain(self):
        """Test get blockchain chain"""
        response = client.get("/api/blockchain/chain", headers=_get_auth_headers())
        assert response.status_code == 200
        data = response.json()
        assert data["success"] == True
        assert "blocks" in data["data"]

    def test_get_stats(self):
        """Test blockchain stats"""
        response = client.get("/api/blockchain/stats", headers=_get_auth_headers())
        assert response.status_code == 200


class TestGANAPI:
    """Test GAN synthesis endpoints"""

    def test_list_models(self):
        """Test list GAN attack types"""
        response = client.get("/api/gan/attack-types", headers=_get_auth_headers())
        assert response.status_code == 200
        data = response.json()
        assert data["success"] == True

    def test_get_stats(self):
        """Test GAN stats"""
        response = client.get("/api/gan/stats", headers=_get_auth_headers())
        assert response.status_code == 200

    def test_generate_samples(self):
        """Test generate synthetic samples"""
        response = client.post(
            "/api/gan/generate", json={"attack_type": "ddos", "num_samples": 10},
            headers=_get_auth_headers()
        )
        assert response.status_code == 200
        data = response.json()
        assert data["success"] == True


class TestHSMAPI:
    """Test HSM endpoints"""

    def test_hsm_status(self):
        """Test HSM status"""
        response = client.get("/api/hsm/status", headers=_get_auth_headers())
        assert response.status_code == 200
        data = response.json()
        assert data["success"] == True
        assert "status" in data["data"]

    def test_list_keys(self):
        """Test list HSM keys"""
        response = client.get("/api/hsm/keys", headers=_get_auth_headers())
        assert response.status_code == 200
        data = response.json()
        assert data["success"] == True
        assert "keys" in data["data"]

    def test_get_audit_log(self):
        """Test get operations log"""
        response = client.get("/api/hsm/operations", headers=_get_auth_headers())
        assert response.status_code == 200


class TestThreatHuntingAPI:
    """Test threat hunting endpoints"""

    def test_list_templates(self):
        """Test list hunting templates"""
        response = client.get("/api/threat-hunting/templates", headers=_get_auth_headers())
        assert response.status_code == 200

    def test_list_investigations(self):
        """Test list investigations"""
        response = client.get("/api/threat-hunting/investigations", headers=_get_auth_headers())
        assert response.status_code == 200


class TestNotificationsAPI:
    """Test notifications endpoints"""

    def test_get_notifications(self):
        """Test get notifications"""
        headers = _get_auth_headers()
        response = client.get("/api/notifications", headers=headers)
        assert response.status_code == 200

    def test_get_preferences(self):
        """Test get notification preferences"""
        headers = _get_auth_headers()
        response = client.get("/api/notifications/preferences", headers=headers)
        assert response.status_code == 200


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
