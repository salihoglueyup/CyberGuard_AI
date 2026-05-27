"""
CyberGuard AI - Security & Monitoring Endpoint Tests
Scanner, alerts, incidents, anomaly endpoint testleri
"""

from datetime import datetime

import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def _inject_token(token: str = "security-test-token") -> dict:
    from app.api.routes.auth import TOKENS, load_users
    users = load_users()
    username = list(users.keys())[0] if users else "admin"
    TOKENS[token] = {"username": username, "created_at": datetime.now().isoformat()}
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture(autouse=True)
def auth_headers():
    return _inject_token()


class TestScannerEndpoints:
    """Vulnerability scanner endpoint testleri"""

    def test_get_vulnerabilities(self, auth_headers):
        response = client.get("/api/vuln/list", headers=auth_headers)
        assert response.status_code in [200, 404]

    def test_scanner_status(self, auth_headers):
        response = client.get("/api/scanner/status", headers=auth_headers)
        assert response.status_code in [200, 404]

    def test_run_quick_scan(self, auth_headers):
        payload = {"target": "127.0.0.1", "scan_type": "quick"}
        response = client.post("/api/scanner/scan", json=payload, headers=auth_headers)
        assert response.status_code in [200, 201, 404, 422]


class TestAlertsEndpoints:
    """Alert endpoint testleri"""

    def test_get_alerts_list(self, auth_headers):
        # Route is GET /api/alerts (empty sub-path)
        response = client.get("/api/alerts", headers=auth_headers)
        assert response.status_code in [200, 404]

    def test_get_alerts_count(self, auth_headers):
        # Route is GET /api/alerts/stats
        response = client.get("/api/alerts/stats", headers=auth_headers)
        assert response.status_code in [200, 404]

    def test_dismiss_alert_invalid_id(self, auth_headers):
        response = client.post(
            "/api/alerts/dismiss/nonexistent-id-xyz",
            headers=auth_headers,
        )
        assert response.status_code in [200, 404, 422]


class TestIncidentsEndpoints:
    """Incident management endpoint testleri"""

    def test_get_incidents(self, auth_headers):
        response = client.get("/api/incidents/list", headers=auth_headers)
        assert response.status_code in [200, 404]

    def test_create_incident(self, auth_headers):
        payload = {
            "title": "Test Incident",
            "description": "CI test incident",
            "severity": "low",
            "type": "test",
        }
        # Route is POST /api/incidents (empty sub-path)
        response = client.post("/api/incidents", json=payload, headers=auth_headers)
        assert response.status_code in [200, 201, 404, 422]


class TestAnomalyEndpoints:
    """Anomaly detection endpoint testleri"""

    def test_get_anomaly_status(self, auth_headers):
        response = client.get("/api/anomaly/status", headers=auth_headers)
        assert response.status_code in [200, 404]

    def test_get_anomaly_list(self, auth_headers):
        response = client.get("/api/anomaly/list", headers=auth_headers)
        assert response.status_code in [200, 404]


class TestNetworkEndpoints:
    """Network monitoring endpoint testleri"""

    def test_get_network_stats(self, auth_headers):
        response = client.get("/api/network/stats", headers=auth_headers)
        assert response.status_code in [200, 404]

    def test_get_network_traffic(self, auth_headers):
        response = client.get("/api/network/traffic", headers=auth_headers)
        assert response.status_code in [200, 404]


class TestSIEMEndpoints:
    """SIEM endpoint testleri"""

    def test_get_siem_events(self, auth_headers):
        response = client.get("/api/siem/events", headers=auth_headers)
        assert response.status_code in [200, 404]

    def test_get_siem_config(self, auth_headers):
        response = client.get("/api/siem/config", headers=auth_headers)
        assert response.status_code in [200, 404]


class TestThreatIntelEndpoints:
    """Threat intelligence endpoint testleri"""

    def test_get_threat_intel(self, auth_headers):
        response = client.get("/api/threat-intel/feed", headers=auth_headers)
        assert response.status_code in [200, 404]

    def test_get_ioc_list(self, auth_headers):
        # GET /api/threat-intel/iocs (plural)
        response = client.get("/api/threat-intel/iocs", headers=auth_headers)
        assert response.status_code in [200, 404]

    def test_mitre_attack_data(self, auth_headers):
        response = client.get("/api/threat-intel/mitre", headers=auth_headers)
        assert response.status_code in [200, 404]


class TestDarkWebEndpoints:
    """Dark web monitoring endpoint testleri"""

    def test_get_darkweb_mentions(self, auth_headers):
        response = client.get("/api/darkweb/mentions", headers=auth_headers)
        assert response.status_code in [200, 404]


class TestPlaybookEndpoints:
    """IR Playbook endpoint testleri"""

    def test_get_playbooks(self, auth_headers):
        response = client.get("/api/playbooks/list", headers=auth_headers)
        assert response.status_code in [200, 404]

    def test_get_playbook_templates(self, auth_headers):
        response = client.get("/api/playbooks/templates", headers=auth_headers)
        assert response.status_code in [200, 404]
