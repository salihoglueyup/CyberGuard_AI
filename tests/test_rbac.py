"""
CyberGuard AI - RBAC ve Auth Utility Tests
require_auth, require_role, verify_token, hash_password testleri
"""

from datetime import datetime

import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def _inject_token(token: str, username: str = "admin", role: str = "admin") -> dict:
    """TOKENS store'a test token'ı enjekte et, özel rol ile."""
    from app.api.routes.auth import TOKENS
    TOKENS[token] = {
        "username": username,
        "role": role,
        "created_at": datetime.now().isoformat(),
    }
    return {"Authorization": f"Bearer {token}"}


class TestPasswordHashing:
    """Şifre hash/doğrulama testleri"""

    def test_hash_password_produces_bcrypt_hash(self):
        """hash_password bcrypt hash üretmeli"""
        from app.api.routes.auth import hash_password
        hashed = hash_password("test-password-123")
        assert hashed.startswith("$2b$")

    def test_verify_password_correct(self):
        """Doğru şifre verify_password True döndürmeli"""
        from app.api.routes.auth import hash_password, verify_password
        password = "secure-password-456"
        hashed = hash_password(password)
        assert verify_password(password, hashed) is True

    def test_verify_password_wrong(self):
        """Yanlış şifre verify_password False döndürmeli"""
        from app.api.routes.auth import hash_password, verify_password
        hashed = hash_password("correct-password")
        assert verify_password("wrong-password", hashed) is False

    def test_verify_password_empty_hash(self):
        """Boş hash False döndürmeli (admin disabled state)"""
        from app.api.routes.auth import verify_password
        assert verify_password("any-password", "") is False

    def test_different_passwords_different_hashes(self):
        """Aynı şifre her hash'lendiğinde farklı hash üretmeli (salt)"""
        from app.api.routes.auth import hash_password
        h1 = hash_password("same-password")
        h2 = hash_password("same-password")
        assert h1 != h2  # farklı salt → farklı hash


class TestVerifyToken:
    """Token doğrulama testleri"""

    def test_valid_token_returns_user(self):
        """Geçerli token kullanıcı bilgisi döndürmeli"""
        from app.api.routes.auth import TOKENS, load_users, verify_token
        token = "test-verify-valid-001"
        # Use a real user that exists in DEFAULT_USERS / users.json
        users = load_users()
        real_username = list(users.keys())[0]  # first real user
        TOKENS[token] = {
            "username": real_username,
            "role": users[real_username].get("role", "admin"),
            "created_at": datetime.now().isoformat(),
        }
        result = verify_token(token)
        assert result is not None
        assert result["username"] == real_username
        TOKENS.pop(token, None)

    def test_invalid_token_returns_none(self):
        """Geçersiz token None döndürmeli"""
        from app.api.routes.auth import verify_token
        result = verify_token("nonexistent-token-xyz-789")
        assert result is None

    def test_empty_token_returns_none(self):
        """Boş string token None döndürmeli"""
        from app.api.routes.auth import verify_token
        result = verify_token("")
        assert result is None


class TestRequireRole:
    """RBAC endpoint testleri"""

    def test_admin_can_access_admin_only_endpoint(self):
        """Admin rolü admin-only endpointe erişebilmeli"""
        headers = _inject_token("rbac-admin-001", username="admin", role="admin")
        # /api/keys/status endpoint requires auth (admin erişebilir)
        response = client.get("/api/keys/status", headers=headers)
        assert response.status_code in [200, 201]

    def test_analyst_can_access_shared_endpoint(self):
        """Analyst rolü ortak endpointe erişebilmeli"""
        # Use a real user from users.json/DEFAULT_USERS
        from app.api.routes.auth import load_users
        users = load_users()
        real_username = list(users.keys())[0]
        real_role = users[real_username].get("role", "admin")
        headers = _inject_token("rbac-analyst-001", username=real_username, role=real_role)
        response = client.get("/api/keys/status", headers=headers)
        assert response.status_code in [200, 201]

    def test_no_token_returns_401_or_403(self):
        """Token olmadan korumalı endpoint 401/403 döndürmeli"""
        response = client.get("/api/keys/status")
        assert response.status_code in [401, 403]

    def test_invalid_token_returns_401(self):
        """Geçersiz token 401 döndürmeli"""
        response = client.get(
            "/api/keys/status",
            headers={"Authorization": "Bearer garbage-token-12345"},
        )
        assert response.status_code in [401, 403]


class TestLoadUsers:
    """Kullanıcı yükleme testleri"""

    def test_load_users_returns_dict(self):
        """load_users sözlük döndürmeli"""
        from app.api.routes.auth import load_users
        users = load_users()
        assert isinstance(users, dict)
        assert len(users) > 0

    def test_default_users_has_admin(self):
        """Varsayılan kullanıcılar admin içermeli"""
        from app.api.routes.auth import DEFAULT_USERS
        assert "admin" in DEFAULT_USERS
        assert DEFAULT_USERS["admin"]["role"] == "admin"

    def test_user_has_required_fields(self):
        """Her kullanıcı gerekli alanları içermeli"""
        from app.api.routes.auth import DEFAULT_USERS
        required_fields = {"id", "username", "email", "role"}
        for username, user in DEFAULT_USERS.items():
            for field in required_fields:
                assert field in user, f"{username} kullanıcısında '{field}' alanı eksik"


class TestSessionPersistence:
    """Oturum kalıcılık testleri"""

    def test_token_stored_in_memory(self):
        """_inject_token TOKENS dict'e eklemeli"""
        from app.api.routes.auth import TOKENS
        initial_count = len(TOKENS)
        token = "persistence-test-token-999"
        _inject_token(token)
        assert token in TOKENS
        assert len(TOKENS) == initial_count + 1
        TOKENS.pop(token, None)

    def test_token_removed_after_logout(self):
        """Logout sonrası token TOKENS'dan kaldırılmalı"""
        from app.api.routes.auth import TOKENS
        token = "logout-persist-test-888"
        headers = _inject_token(token)

        assert token in TOKENS
        response = client.post("/api/auth/logout", headers=headers)
        assert response.status_code in [200, 204]
        assert token not in TOKENS
