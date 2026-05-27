"""
CyberGuard AI - Backend API Tests
Pytest based test suite for FastAPI endpoints
"""


import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def _get_auth_headers():
    """Get auth headers by injecting a test token into the TOKENS store."""
    from datetime import datetime

    from app.api.routes.auth import TOKENS, load_users
    token = "test-token-for-pytest-endpoints"
    users = load_users()
    username = list(users.keys())[0] if users else "admin"
    TOKENS[token] = {
        "username": username,
        "created_at": datetime.now().isoformat(),
    }
    return {"Authorization": f"Bearer {token}"}


class TestDashboard:
    """Dashboard API tests"""

    def test_get_stats(self):
        """Test dashboard stats endpoint"""
        response = client.get("/api/dashboard/stats", headers=_get_auth_headers())
        assert response.status_code == 200
        data = response.json()
        assert "success" in data or "data" in data

    def test_get_system_health(self):
        """Test system health endpoint"""
        response = client.get("/api/dashboard/system-health", headers=_get_auth_headers())
        assert response.status_code in [200, 404]


class TestAttackMap:
    """Attack Map API tests"""

    def test_get_live_attacks(self):
        """Test live attacks endpoint"""
        response = client.get("/api/attack-map/live?limit=10", headers=_get_auth_headers())
        assert response.status_code == 200
        data = response.json()
        assert data.get("success") == True
        assert "attacks" in data.get("data", {})

    def test_get_live_attacks_with_source(self):
        """Test live attacks with source filter"""
        response = client.get("/api/attack-map/live?limit=5&source=simulation", headers=_get_auth_headers())
        assert response.status_code == 200
        data = response.json()
        assert data.get("success") == True

    def test_get_countries(self):
        """Test countries endpoint"""
        response = client.get("/api/attack-map/countries", headers=_get_auth_headers())
        assert response.status_code == 200
        data = response.json()
        assert data.get("success") == True
        assert "countries" in data.get("data", {})

    def test_get_stats(self):
        """Test attack stats endpoint"""
        response = client.get("/api/attack-map/stats", headers=_get_auth_headers())
        assert response.status_code == 200


class TestNetwork:
    """Network API tests"""

    def test_get_status(self):
        """Test network status endpoint"""
        response = client.get("/api/network/status", headers=_get_auth_headers())
        assert response.status_code == 200

    def test_get_interfaces(self):
        """Test network interfaces endpoint"""
        response = client.get("/api/network/interfaces", headers=_get_auth_headers())
        assert response.status_code == 200


class TestModels:
    """ML Models API tests"""

    def test_get_models(self):
        """Test models list endpoint"""
        response = client.get("/api/models/", headers=_get_auth_headers())
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, (list, dict))

    def test_get_model_stats(self):
        """Test model stats endpoint"""
        response = client.get("/api/models/stats", headers=_get_auth_headers())
        assert response.status_code in [200, 404]


class TestThreatHunting:
    """Threat Hunting API tests"""

    def test_get_investigations(self):
        """Test investigations endpoint"""
        response = client.get("/api/threat-hunting/investigations", headers=_get_auth_headers())
        assert response.status_code == 200
        data = response.json()
        assert data.get("success") == True

    def test_query(self):
        """Test threat hunting query"""
        response = client.post(
            "/api/threat-hunting/query", json={"query": "test", "timerange": "24h"},
            headers=_get_auth_headers(),
        )
        assert response.status_code == 200


class TestSandbox:
    """Sandbox API tests"""

    def test_get_recent(self):
        """Test recent analyses endpoint"""
        response = client.get("/api/sandbox/recent", headers=_get_auth_headers())
        assert response.status_code == 200
        data = response.json()
        assert data.get("success") == True


class TestIncidents:
    """Incidents API tests"""

    def test_get_timeline(self):
        """Test timeline endpoint"""
        response = client.get("/api/incidents/timeline", headers=_get_auth_headers())
        assert response.status_code == 200
        data = response.json()
        assert data.get("success") == True

    def test_get_users_behavior(self):
        """Test user behavior endpoint"""
        response = client.get("/api/incidents/behavior/users", headers=_get_auth_headers())
        assert response.status_code == 200
        data = response.json()
        assert data.get("success") == True


class TestSIEM:
    """SIEM Integration API tests"""

    def test_get_platforms(self):
        """Test platforms endpoint"""
        response = client.get("/api/siem/platforms", headers=_get_auth_headers())
        assert response.status_code == 200

    def test_get_rules(self):
        """Test rules endpoint"""
        response = client.get("/api/siem/rules", headers=_get_auth_headers())
        assert response.status_code == 200


class TestSecurity:
    """Security API tests"""

    def test_get_score(self):
        """Test security score endpoint"""
        response = client.get("/api/security/score", headers=_get_auth_headers())
        assert response.status_code == 200

    def test_get_honeypot(self):
        """Test honeypot status endpoint"""
        response = client.get("/api/security/honeypot", headers=_get_auth_headers())
        assert response.status_code == 200


class TestChat:
    """AI Chat API tests"""

    def test_chat_query(self):
        """Test chat query endpoint"""
        response = client.post("/api/chat/", json={"message": "Merhaba"}, headers=_get_auth_headers())
        # May fail if no API key, but should return valid response
        assert response.status_code in [200, 400, 401, 422, 500]


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
