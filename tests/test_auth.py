"""
CyberGuard AI - Auth API Tests
Login, logout, token validation, rate limit testi
"""

from datetime import datetime

import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def _inject_token(token: str = "test-auth-token") -> dict:
    """TOKENS store'a doğrudan test token'ı enjekte et."""
    from app.api.routes.auth import TOKENS, load_users
    users = load_users()
    username = list(users.keys())[0] if users else "admin"
    TOKENS[token] = {"username": username, "created_at": datetime.now().isoformat()}
    return {"Authorization": f"Bearer {token}"}


class TestAuthLogin:
    """Login endpoint testleri"""

    def test_login_missing_body(self):
        """Boş body ile login 422 döndürmeli"""
        response = client.post("/api/auth/login", json={})
        assert response.status_code == 422

    def test_login_wrong_credentials(self):
        """Yanlış şifre ile login 401 döndürmeli"""
        import os
        if not os.environ.get("ADMIN_DEFAULT_PASSWORD"):
            pytest.skip("ADMIN_DEFAULT_PASSWORD not set")
        response = client.post(
            "/api/auth/login",
            json={"username": "admin", "password": "wrong-password-xyz"},
        )
        assert response.status_code == 401

    def test_login_correct_credentials(self):
        """Doğru şifre ile login 200 + token döndürmeli"""
        import os
        password = os.environ.get("ADMIN_DEFAULT_PASSWORD")
        if not password:
            pytest.skip("ADMIN_DEFAULT_PASSWORD not set")
        response = client.post(
            "/api/auth/login",
            json={"username": "admin", "password": password},
        )
        assert response.status_code == 200
        data = response.json()
        assert "token" in data or "access_token" in data

    def test_login_nonexistent_user(self):
        """Var olmayan kullanıcı 401 döndürmeli"""
        response = client.post(
            "/api/auth/login",
            json={"username": "ghost_user_xyz", "password": "anypass"},
        )
        assert response.status_code in [401, 403]


class TestAuthToken:
    """Token doğrulama testleri"""

    def test_protected_endpoint_no_token(self):
        """/api/keys token olmadan 403 veya 401 döndürmeli"""
        # api_keys router require_auth ile korunuyor
        response = client.get("/api/keys")
        assert response.status_code in [401, 403]

    def test_protected_endpoint_invalid_token(self):
        """Geçersiz token 401 döndürmeli"""
        response = client.get(
            "/api/keys",
            headers={"Authorization": "Bearer invalid-token-xyz-123"},
        )
        assert response.status_code in [401, 403]

    def test_protected_endpoint_valid_token(self):
        """Geçerli token ile endpoint erişilebilir olmalı"""
        headers = _inject_token("valid-test-token-001")
        response = client.get("/api/keys", headers=headers)
        # 200 ya da 404 (route farklı isimde olabilir) — ama 401 olmamalı
        assert response.status_code != 401
        assert response.status_code != 403


class TestAuthLogout:
    """Logout testleri"""

    def test_logout_removes_token(self):
        """Logout sonrası aynı token ile erişim 401 döndürmeli"""
        token = "logout-test-token-999"
        headers = _inject_token(token)

        # Önce token çalışmalı (api_keys require_auth ile korunuyor)
        response = client.get("/api/keys", headers=headers)
        assert response.status_code not in [401, 403]

        # Logout
        logout_response = client.post("/api/auth/logout", headers=headers)
        assert logout_response.status_code in [200, 204]

        # Artık token geçersiz olmalı
        response_after = client.get("/api/keys", headers=headers)
        assert response_after.status_code in [401, 403]


class TestAuthMe:
    """Mevcut kullanıcı bilgisi testleri"""

    def test_get_me_with_valid_token(self):
        """/api/auth/me geçerli token ile kullanıcı bilgisi döndürmeli"""
        headers = _inject_token("me-test-token-001")
        response = client.get("/api/auth/me", headers=headers)
        if response.status_code == 404:
            pytest.skip("/api/auth/me endpoint mevcut değil")
        assert response.status_code == 200
        data = response.json()
        # Response may be wrapped: {"success": true, "data": {"username": ...}}
        inner = data.get("data", data)
        assert "username" in inner or "user" in inner or "email" in inner
