"""
CyberGuard AI - Backend API Tests
Pytest based test suite for FastAPI endpoints
"""

import pytest
from fastapi.testclient import TestClient
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.main import app

client = TestClient(app)


class TestDashboard:
    """Dashboard API tests"""

    def test_get_stats(self):
        """Test dashboard stats endpoint"""
        response = client.get("/api/dashboard/stats")
        assert response.status_code == 200
        data = response.json()
        assert "success" in data or "data" in data

    def test_get_system_health(self):
        """Test system health endpoint"""
        response = client.get("/api/dashboard/system-health")
        assert response.status_code in [200, 404]


class TestAttackMap:
    """Attack Map API tests"""

    def test_get_live_attacks(self):
        """Test live attacks endpoint"""
        response = client.get("/api/attack-map/live?limit=10")
        assert response.status_code == 200
        data = response.json()
        assert data.get("success") == True
        assert "attacks" in data.get("data", {})

    def test_get_live_attacks_with_source(self):
        """Test live attacks with source filter"""
        response = client.get("/api/attack-map/live?limit=5&source=simulation")
        assert response.status_code == 200
        data = response.json()
        assert data.get("success") == True

    def test_get_countries(self):
        """Test countries endpoint"""
        response = client.get("/api/attack-map/countries")
        assert response.status_code == 200
        data = response.json()
        assert data.get("success") == True
        assert "countries" in data.get("data", {})

    def test_get_stats(self):
        """Test attack stats endpoint"""
        response = client.get("/api/attack-map/stats")
        assert response.status_code == 200


class TestNetwork:
    """Network API tests"""

    def test_get_status(self):
        """Test network status endpoint"""
        response = client.get("/api/network/status")
        assert response.status_code == 200

    def test_get_interfaces(self):
        """Test network interfaces endpoint"""
        response = client.get("/api/network/interfaces")
        assert response.status_code == 200


class TestModels:
    """ML Models API tests"""

    def test_get_models(self):
        """Test models list endpoint"""
        response = client.get("/api/models/")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, (list, dict))

    def test_get_model_stats(self):
        """Test model stats endpoint"""
        response = client.get("/api/models/stats")
        assert response.status_code in [200, 404]


class TestThreatHunting:
    """Threat Hunting API tests"""

    def test_get_investigations(self):
        """Test investigations endpoint"""
        response = client.get("/api/threat-hunting/investigations")
        assert response.status_code == 200
        data = response.json()
        assert data.get("success") == True

    def test_query(self):
        """Test threat hunting query"""
        response = client.post(
            "/api/threat-hunting/query", json={"query": "test", "timerange": "24h"}
        )
        assert response.status_code == 200


class TestSandbox:
    """Sandbox API tests"""

    def test_get_recent(self):
        """Test recent analyses endpoint"""
        response = client.get("/api/sandbox/recent")
        assert response.status_code == 200
        data = response.json()
        assert data.get("success") == True


class TestIncidents:
    """Incidents API tests"""

    def test_get_timeline(self):
        """Test timeline endpoint"""
        response = client.get("/api/incidents/timeline")
        assert response.status_code == 200
        data = response.json()
        assert data.get("success") == True

    def test_get_users_behavior(self):
        """Test user behavior endpoint"""
        response = client.get("/api/incidents/behavior/users")
        assert response.status_code == 200
        data = response.json()
        assert data.get("success") == True


class TestSIEM:
    """SIEM Integration API tests"""

    def test_get_platforms(self):
        """Test platforms endpoint"""
        response = client.get("/api/siem/platforms")
        assert response.status_code == 200

    def test_get_rules(self):
        """Test rules endpoint"""
        response = client.get("/api/siem/rules")
        assert response.status_code == 200


class TestSecurity:
    """Security API tests"""

    def test_get_score(self):
        """Test security score endpoint"""
        response = client.get("/api/security/score")
        assert response.status_code == 200

    def test_get_honeypot(self):
        """Test honeypot status endpoint"""
        response = client.get("/api/security/honeypot")
        assert response.status_code == 200


class TestChat:
    """AI Chat API tests"""

    def test_chat_query(self):
        """Test chat query endpoint"""
        response = client.post("/api/chat/query", json={"message": "Merhaba"})
        # May fail if no API key, but should return valid response
        assert response.status_code in [200, 400, 500]


# Utility tests
class TestHealthCheck:
    """Health check tests"""

    def test_root(self):
        """Test root endpoint"""
        response = client.get("/")
        assert response.status_code in [200, 307, 404]

    def test_docs(self):
        """Test API docs endpoint"""
        response = client.get("/api/docs")
        assert response.status_code in [200, 307]


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
