"""
CyberGuard AI - API Test Suite
pytest tests for backend API endpoints
"""

import pytest
from fastapi.testclient import TestClient
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.main import app

client = TestClient(app)


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
        response = client.get("/api/dashboard/stats")
        assert response.status_code == 200
        data = response.json()
        assert "success" in data or "data" in data

    def test_get_system_metrics(self):
        """Test system metrics endpoint"""
        response = client.get("/api/dashboard/system/metrics")
        # May return 200 or 404 depending on implementation
        assert response.status_code in [200, 404]


class TestAttackMapAPI:
    """Test attack map endpoints"""

    def test_get_live_attacks(self):
        """Test live attacks endpoint"""
        response = client.get("/api/attack-map/live")
        assert response.status_code == 200
        data = response.json()
        assert data["success"] == True
        assert "attacks" in data["data"]

    def test_get_attack_stats(self):
        """Test attack statistics endpoint"""
        response = client.get("/api/attack-map/stats")
        assert response.status_code == 200
        data = response.json()
        assert data["success"] == True

    def test_get_countries(self):
        """Test countries endpoint"""
        response = client.get("/api/attack-map/countries")
        assert response.status_code == 200


class TestSIEMAPI:
    """Test SIEM integration endpoints"""

    def test_list_platforms(self):
        """Test list SIEM platforms"""
        response = client.get("/api/siem/platforms")
        assert response.status_code == 200
        data = response.json()
        assert data["success"] == True
        assert "platforms" in data["data"]

    def test_list_connections(self):
        """Test list SIEM connections"""
        response = client.get("/api/siem/connections")
        assert response.status_code == 200

    def test_get_stats(self):
        """Test SIEM stats"""
        response = client.get("/api/siem/stats")
        assert response.status_code == 200


class TestSandboxAPI:
    """Test malware sandbox endpoints"""

    def test_get_recent_analyses(self):
        """Test get recent analyses"""
        response = client.get("/api/sandbox/recent")
        assert response.status_code == 200
        data = response.json()
        assert data["success"] == True

    def test_get_stats(self):
        """Test sandbox stats"""
        response = client.get("/api/sandbox/stats")
        assert response.status_code == 200

    def test_list_environments(self):
        """Test list sandbox environments"""
        response = client.get("/api/sandbox/environments")
        assert response.status_code == 200


class TestBlockchainAPI:
    """Test blockchain audit endpoints"""

    def test_get_chain(self):
        """Test get blockchain chain"""
        response = client.get("/api/blockchain/chain")
        assert response.status_code == 200
        data = response.json()
        assert data["success"] == True
        assert "chain" in data["data"]

    def test_get_stats(self):
        """Test blockchain stats"""
        response = client.get("/api/blockchain/stats")
        assert response.status_code == 200


class TestGANAPI:
    """Test GAN synthesis endpoints"""

    def test_list_models(self):
        """Test list GAN models"""
        response = client.get("/api/gan/models")
        assert response.status_code == 200
        data = response.json()
        assert data["success"] == True
        assert "models" in data["data"]

    def test_get_stats(self):
        """Test GAN stats"""
        response = client.get("/api/gan/stats")
        assert response.status_code == 200

    def test_generate_samples(self):
        """Test generate synthetic samples"""
        response = client.post(
            "/api/gan/generate", json={"attack_type": "DoS", "num_samples": 10}
        )
        assert response.status_code == 200
        data = response.json()
        assert data["success"] == True
        assert data["data"]["total_generated"] == 10


class TestHSMAPI:
    """Test HSM endpoints"""

    def test_hsm_status(self):
        """Test HSM status"""
        response = client.get("/api/hsm/status")
        assert response.status_code == 200
        data = response.json()
        assert data["success"] == True
        assert "hsm_id" in data["data"]

    def test_list_keys(self):
        """Test list HSM keys"""
        response = client.get("/api/hsm/keys")
        assert response.status_code == 200
        data = response.json()
        assert data["success"] == True
        assert "keys" in data["data"]

    def test_get_audit_log(self):
        """Test get audit log"""
        response = client.get("/api/hsm/audit")
        assert response.status_code == 200


class TestThreatHuntingAPI:
    """Test threat hunting endpoints"""

    def test_list_templates(self):
        """Test list hunting templates"""
        response = client.get("/api/threat-hunting/templates")
        assert response.status_code == 200

    def test_list_investigations(self):
        """Test list investigations"""
        response = client.get("/api/threat-hunting/investigations")
        assert response.status_code == 200


class TestNotificationsAPI:
    """Test notifications endpoints"""

    def test_get_notifications(self):
        """Test get notifications"""
        response = client.get("/api/notifications")
        assert response.status_code == 200

    def test_get_preferences(self):
        """Test get notification preferences"""
        response = client.get("/api/notifications/preferences")
        assert response.status_code == 200


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
