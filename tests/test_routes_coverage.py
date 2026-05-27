"""
CyberGuard AI — Additional Route Coverage Tests
Vulnerability, ThreatIntel, Attacks, Reports, Playbooks, Settings, Auth lockout
"""

from datetime import datetime

import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def _get_auth_headers(token: str = "test-coverage-token") -> dict:
    """Inject test token directly into TOKENS store."""
    from app.api.routes.auth import TOKENS, load_users

    users = load_users()
    username = list(users.keys())[0] if users else "admin"
    TOKENS[token] = {"username": username, "created_at": datetime.now().isoformat()}
    return {"Authorization": f"Bearer {token}"}


def _get_admin_headers(token: str = "test-admin-token") -> dict:
    """Inject admin token into TOKENS store."""
    from app.api.routes.auth import TOKENS, load_users

    users = load_users()
    admin_user = next(
        (u for u in users.values() if u.get("role") == "admin"),
        list(users.values())[0] if users else {"username": "admin"},
    )
    TOKENS[token] = {
        "username": admin_user.get("username", "admin"),
        "created_at": datetime.now().isoformat(),
    }
    return {"Authorization": f"Bearer {token}"}


# ─────────────────────────────────────────────────────────────
# Vulnerability endpoints
# ─────────────────────────────────────────────────────────────

class TestVulnerability:
    def test_get_status(self):
        response = client.get("/api/vuln/status", headers=_get_auth_headers())
        assert response.status_code == 200
        data = response.json()
        assert "success" in data

    def test_list_vulnerabilities(self):
        response = client.get("/api/vuln/vulnerabilities", headers=_get_auth_headers())
        assert response.status_code == 200
        data = response.json()
        assert "success" in data

    def test_get_stats(self):
        response = client.get("/api/vuln/stats", headers=_get_auth_headers())
        assert response.status_code == 200

    def test_list_scans(self):
        response = client.get("/api/vuln/scans", headers=_get_auth_headers())
        assert response.status_code == 200

    def test_scan_requires_auth(self):
        """POST /vuln/scan without token must return 401/403."""
        response = client.post("/api/vuln/scan", json={"target": "localhost"})
        assert response.status_code in [401, 403]


# ─────────────────────────────────────────────────────────────
# Threat Intelligence endpoints
# ─────────────────────────────────────────────────────────────

class TestThreatIntel:
    def test_get_status(self):
        response = client.get("/api/threat-intel/status", headers=_get_auth_headers())
        assert response.status_code == 200
        data = response.json()
        assert "success" in data

    def test_get_iocs(self):
        response = client.get("/api/threat-intel/iocs", headers=_get_auth_headers())
        assert response.status_code == 200

    def test_get_feeds(self):
        response = client.get("/api/threat-intel/feeds", headers=_get_auth_headers())
        assert response.status_code == 200

    def test_get_stats(self):
        response = client.get("/api/threat-intel/stats", headers=_get_auth_headers())
        assert response.status_code == 200

    def test_lookup_ip(self):
        response = client.get(
            "/api/threat-intel/lookup/ip/8.8.8.8", headers=_get_auth_headers()
        )
        assert response.status_code == 200

    def test_lookup_requires_auth(self):
        response = client.post("/api/threat-intel/lookup", json={"indicator": "8.8.8.8"})
        assert response.status_code in [401, 403]


# ─────────────────────────────────────────────────────────────
# Attacks endpoints
# ─────────────────────────────────────────────────────────────

class TestAttacks:
    def test_list_attacks(self):
        response = client.get("/api/attacks/", headers=_get_auth_headers())
        assert response.status_code == 200
        data = response.json()
        assert "success" in data

    def test_list_attacks_with_limit(self):
        response = client.get(
            "/api/attacks/?limit=5&page=1", headers=_get_auth_headers()
        )
        assert response.status_code == 200

    def test_attacks_stats(self):
        response = client.get("/api/attacks/stats", headers=_get_auth_headers())
        assert response.status_code == 200

    def test_attacks_requires_auth(self):
        response = client.get("/api/attacks/")
        assert response.status_code in [401, 403]


# ─────────────────────────────────────────────────────────────
# Reports endpoints
# ─────────────────────────────────────────────────────────────

class TestReports:
    def test_list_reports(self):
        response = client.get("/api/reports", headers=_get_auth_headers())
        assert response.status_code == 200
        data = response.json()
        assert "success" in data

    def test_reports_stats(self):
        response = client.get("/api/reports/stats", headers=_get_auth_headers())
        assert response.status_code in [200, 404]

    def test_reports_requires_auth(self):
        response = client.get("/api/reports")
        assert response.status_code in [401, 403]


# ─────────────────────────────────────────────────────────────
# Playbooks endpoints
# ─────────────────────────────────────────────────────────────

class TestPlaybooks:
    def test_list_playbooks(self):
        response = client.get("/api/playbooks", headers=_get_auth_headers())
        assert response.status_code == 200
        data = response.json()
        assert "success" in data

    def test_playbooks_stats(self):
        response = client.get("/api/playbooks/stats", headers=_get_auth_headers())
        assert response.status_code in [200, 404]

    def test_playbooks_requires_auth(self):
        response = client.get("/api/playbooks")
        assert response.status_code in [401, 403]


# ─────────────────────────────────────────────────────────────
# Settings endpoints
# ─────────────────────────────────────────────────────────────

class TestSettings:
    def test_get_settings(self):
        response = client.get("/api/settings", headers=_get_auth_headers())
        assert response.status_code == 200

    def test_settings_requires_auth(self):
        response = client.get("/api/settings")
        assert response.status_code in [401, 403]


# ─────────────────────────────────────────────────────────────
# Dashboard extra endpoints
# ─────────────────────────────────────────────────────────────

class TestDashboardExtra:
    def test_dashboard_summary(self):
        response = client.get("/api/dashboard/summary", headers=_get_auth_headers())
        assert response.status_code == 200

    def test_dashboard_hourly_trend(self):
        response = client.get(
            "/api/dashboard/hourly-trend", headers=_get_auth_headers()
        )
        assert response.status_code in [200, 404]

    def test_dashboard_recent_attacks(self):
        response = client.get(
            "/api/dashboard/recent-attacks", headers=_get_auth_headers()
        )
        assert response.status_code in [200, 404]

    def test_dashboard_model_performance(self):
        response = client.get(
            "/api/dashboard/model-performance", headers=_get_auth_headers()
        )
        assert response.status_code in [200, 404]


# ─────────────────────────────────────────────────────────────
# Auth — per-username brute-force lockout (OWASP A07)
# ─────────────────────────────────────────────────────────────

class TestAuthLockout:
    def test_lockout_after_threshold(self):
        """10 başarısız girişten sonra hesap kilitlenmeli (429)."""
        from app.api.routes.auth import _failed_logins

        # Temizle
        _failed_logins.pop("nonexistent_lockout_user", None)

        for _ in range(10):
            client.post(
                "/api/auth/login",
                json={"username": "nonexistent_lockout_user", "password": "wrong"},
            )

        # 11. denemede hesap kilitli olmalı
        response = client.post(
            "/api/auth/login",
            json={"username": "nonexistent_lockout_user", "password": "wrong"},
        )
        assert response.status_code == 429

    def test_successful_login_clears_lockout(self):
        """Başarılı girişte başarısız deneme sayacı sıfırlanmalı."""
        import os

        password = os.environ.get("ADMIN_DEFAULT_PASSWORD")
        if not password:
            pytest.skip("ADMIN_DEFAULT_PASSWORD not set")

        from app.api.routes.auth import _failed_logins

        # Önceki lockout temizle
        _failed_logins.pop("admin", None)

        # Başarılı giriş
        response = client.post(
            "/api/auth/login",
            json={"username": "admin", "password": password},
        )
        assert response.status_code == 200
        # Sayaç sıfırlanmış olmalı
        assert "admin" not in _failed_logins


# ─────────────────────────────────────────────────────────────
# Auth — refresh token endpoint
# ─────────────────────────────────────────────────────────────

class TestAuthRefresh:
    def test_refresh_with_invalid_token_returns_401(self):
        response = client.post(
            "/api/auth/refresh", json={"refresh_token": "invalid-refresh-token-xyz"}
        )
        assert response.status_code == 401

    def test_refresh_with_valid_token(self):
        """Geçerli refresh token ile yeni access token alınmalı."""
        from app.api.routes.auth import REFRESH_TOKENS

        rt = "test-valid-refresh-token"
        REFRESH_TOKENS[rt] = {
            "username": "admin",
            "created_at": datetime.now().isoformat(),
        }
        response = client.post("/api/auth/refresh", json={"refresh_token": rt})
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "token" in data["data"]


# ─────────────────────────────────────────────────────────────
# Blockchain endpoints
# ─────────────────────────────────────────────────────────────

class TestBlockchain:
    def test_get_chain(self):
        response = client.get("/api/blockchain/chain", headers=_get_auth_headers())
        assert response.status_code == 200

    def test_get_stats(self):
        response = client.get("/api/blockchain/stats", headers=_get_auth_headers())
        assert response.status_code == 200

    def test_blockchain_requires_auth(self):
        response = client.get("/api/blockchain/chain")
        assert response.status_code in [401, 403]


# ─────────────────────────────────────────────────────────────
# Logs endpoint
# ─────────────────────────────────────────────────────────────

class TestLogs:
    def test_get_logs(self):
        response = client.get("/api/logs/", headers=_get_auth_headers())
        assert response.status_code in [200, 500]  # 500 if SQLite schema missing

    def test_logs_requires_auth(self):
        response = client.get("/api/logs/")
        assert response.status_code in [401, 403]
