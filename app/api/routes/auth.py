"""
Auth API Routes - CyberGuard AI
Kimlik doğrulama ve yetkilendirme

Dosya Yolu: app/api/routes/auth.py
"""

import logging
import os
import time as _time
from collections import defaultdict
from datetime import datetime, timedelta

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel

from app.paths import PROJECT_ROOT

project_root = str(PROJECT_ROOT)

# Simple in-memory login rate limiter (5 attempts per minute per IP)
_login_attempts: dict = defaultdict(list)
_LOGIN_RATE_LIMIT = 5
_LOGIN_RATE_WINDOW = 60  # seconds
_LOGIN_PRUNE_COUNTER = 0
_LOGIN_PRUNE_INTERVAL = 100  # prune stale IPs every N login attempts

# Per-username brute-force lockout (OWASP A07)
_failed_logins: dict = defaultdict(list)  # username -> [timestamp, ...]
_LOCKOUT_THRESHOLD = 10   # failed attempts before lockout
_LOCKOUT_WINDOW = 300     # seconds (5 minutes) — window + lockout duration
import json
import re
import secrets

import bcrypt

logger = logging.getLogger(__name__)

router = APIRouter()
security = HTTPBearer(auto_error=False)


async def require_auth(
    credentials: HTTPAuthorizationCredentials = Depends(HTTPBearer()),
) -> dict:
    """Reusable auth dependency — returns user dict or raises 401."""
    if not credentials:
        raise HTTPException(status_code=401, detail="Token gerekli")
    user = verify_token(credentials.credentials)
    if not user:
        raise HTTPException(status_code=401, detail="Geçersiz veya süresi dolmuş token")
    return user


def require_role(*roles: str):
    """RBAC dependency factory — kullanım: Depends(require_role('admin', 'analyst'))."""
    async def _check(user: dict = Depends(require_auth)) -> dict:
        if user.get("role") not in roles:
            raise HTTPException(
                status_code=403,
                detail=f"Bu işlem için {' veya '.join(roles)} yetkisi gerekli",
            )
        return user
    return _check

# Basit in-memory user storage (production'da DB kullanılmalı)
USERS_FILE = os.path.join(project_root, "data", "users.json")
SESSIONS_FILE = os.path.join(project_root, "data", "sessions.json")
TOKENS: dict = {}  # token -> session dict (in-memory, persisted on write)
REFRESH_TOKENS: dict = {}  # refresh_token -> username
TOKEN_MAX_AGE = timedelta(hours=24)
REFRESH_TOKEN_MAX_AGE = timedelta(days=7)


def load_sessions() -> dict:
    """Kalıcı oturumları diskten yükle (sunucu başlangıcında çağrılır)."""
    if os.path.exists(SESSIONS_FILE):
        try:
            with open(SESSIONS_FILE, encoding="utf-8") as f:
                return json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            logger.warning(f"sessions.json okunamadı: {e}")
    return {}


def save_session(token: str, session: dict) -> None:
    """Token'ı diske kaydet."""
    try:
        os.makedirs(os.path.dirname(SESSIONS_FILE), exist_ok=True)
        # Var olan dosyadan oku, ekle ve geri yaz
        all_sessions = load_sessions()
        all_sessions[token] = session
        with open(SESSIONS_FILE, "w", encoding="utf-8") as f:
            json.dump(all_sessions, f, indent=2, ensure_ascii=False)
    except OSError as e:
        logger.warning(f"Oturum kaydedilemedi: {e}")


def delete_session(token: str) -> None:
    """Token'ı diskten sil."""
    try:
        all_sessions = load_sessions()
        all_sessions.pop(token, None)
        with open(SESSIONS_FILE, "w", encoding="utf-8") as f:
            json.dump(all_sessions, f, indent=2, ensure_ascii=False)
    except OSError as e:
        logger.warning(f"Oturum silinemedi: {e}")


# Sunucu başlarken kalıcı oturumları belleğe yükle
TOKENS.update(load_sessions())

# Varsayılan admin kullanıcı - şifre ZORUNLU environment variable
_default_admin_password = os.environ.get("ADMIN_DEFAULT_PASSWORD")
if not _default_admin_password:
    logger.warning("ADMIN_DEFAULT_PASSWORD env var not set — admin login disabled until set")
    _admin_hash = ""
else:
    _admin_hash = bcrypt.hashpw(_default_admin_password.encode(), bcrypt.gensalt()).decode()

DEFAULT_USERS = {
    "admin": {
        "id": "1",
        "username": "admin",
        "email": "admin@cyberguard.ai",
        "password_hash": _admin_hash,
        "role": "admin",
        "created_at": datetime.now().isoformat(),
        "must_change_password": True,
    }
}


def load_users():
    """Kullanıcıları yükle"""
    if os.path.exists(USERS_FILE):
        try:
            with open(USERS_FILE, encoding="utf-8") as f:
                return json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            logger.warning(f"Failed to load users file: {e}")
            pass
    return DEFAULT_USERS.copy()


def save_users(users):
    """Kullanıcıları kaydet"""
    os.makedirs(os.path.dirname(USERS_FILE), exist_ok=True)
    with open(USERS_FILE, "w", encoding="utf-8") as f:
        json.dump(users, f, indent=2, ensure_ascii=False)


def hash_password(password: str) -> str:
    """Şifre hash'le (bcrypt)"""
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()


def verify_password(password: str, password_hash: str) -> bool:
    """Şifre doğrula (bcrypt)"""
    if not password_hash:
        return False
    try:
        return bcrypt.checkpw(password.encode(), password_hash.encode())
    except (ValueError, TypeError):
        return False


def generate_token() -> str:
    """Token oluştur"""
    return secrets.token_urlsafe(32)


def cleanup_expired_tokens():
    """Süresi dolmuş tokenları temizle"""
    now = datetime.now()
    expired = [
        t for t, data in TOKENS.items()
        if now - datetime.fromisoformat(data["created_at"]) >= TOKEN_MAX_AGE
    ]
    for t in expired:
        del TOKENS[t]


def verify_token(token: str) -> dict | None:
    """Token doğrula"""
    # Periyodik temizlik
    if len(TOKENS) > 100:
        cleanup_expired_tokens()

    if token in TOKENS:
        token_data = TOKENS[token]
        # Token süresi kontrolü (24 saat)
        created = datetime.fromisoformat(token_data["created_at"])
        if datetime.now() - created < TOKEN_MAX_AGE:
            users = load_users()
            username = token_data["username"]
            if username in users:
                return users[username]
        else:
            del TOKENS[token]
    return None


class LoginRequest(BaseModel):
    username: str
    password: str


class RegisterRequest(BaseModel):
    username: str
    email: str
    password: str
    password_confirm: str


class ChangePasswordRequest(BaseModel):
    current_password: str
    new_password: str


class UserResponse(BaseModel):
    id: str
    username: str
    email: str
    role: str
    created_at: str


def validate_password_strength(password: str):
    """Shared password policy — used by register and change-password."""
    if len(password) < 8:
        raise HTTPException(status_code=400, detail="Şifre en az 8 karakter olmalı")
    if not re.search(r'[A-Z]', password):
        raise HTTPException(status_code=400, detail="Şifre en az bir büyük harf içermeli")
    if not re.search(r'[0-9]', password):
        raise HTTPException(status_code=400, detail="Şifre en az bir rakam içermeli")
    if not re.search(r'[!@#$%^&*(),.?":{}|<>]', password):
        raise HTTPException(status_code=400, detail="Şifre en az bir özel karakter içermeli")


@router.post("/login")
async def login(request: LoginRequest, fastapi_request: Request):
    """Kullanıcı girişi"""
    try:
        # Rate limit check
        client_ip = fastapi_request.client.host if fastapi_request.client else "unknown"
        now = _time.time()
        _login_attempts[client_ip] = [
            t for t in _login_attempts[client_ip] if now - t < _LOGIN_RATE_WINDOW
        ]
        if len(_login_attempts[client_ip]) >= _LOGIN_RATE_LIMIT:
            raise HTTPException(
                status_code=429,
                detail="Çok fazla giriş denemesi. Lütfen 1 dakika bekleyin."
            )
        _login_attempts[client_ip].append(now)

        # Periodically prune stale IPs to prevent unbounded memory growth
        global _LOGIN_PRUNE_COUNTER
        _LOGIN_PRUNE_COUNTER += 1
        if _LOGIN_PRUNE_COUNTER >= _LOGIN_PRUNE_INTERVAL:
            _LOGIN_PRUNE_COUNTER = 0
            stale = [ip for ip, ts in _login_attempts.items() if not ts or now - ts[-1] >= _LOGIN_RATE_WINDOW]
            for ip in stale:
                del _login_attempts[ip]

        users = load_users()

        # Per-username lockout check (OWASP A07 — brute-force protection)
        now_f = _time.time()
        _failed_logins[request.username] = [
            t for t in _failed_logins[request.username] if now_f - t < _LOCKOUT_WINDOW
        ]
        if len(_failed_logins[request.username]) >= _LOCKOUT_THRESHOLD:
            raise HTTPException(
                status_code=429,
                detail="Hesap geçici olarak kilitlendi. Lütfen 5 dakika bekleyin."
            )

        if request.username not in users:
            # Record failed attempt even for nonexistent users (prevents username enumeration timing)
            _failed_logins[request.username].append(now_f)
            raise HTTPException(status_code=401, detail="Kullanıcı adı veya şifre hatalı")

        user = users[request.username]

        if not verify_password(request.password, user.get("password_hash", "")):
            _failed_logins[request.username].append(now_f)
            logger.warning(
                "Failed login attempt",
                extra={"username": request.username, "ip": client_ip,
                       "attempts": len(_failed_logins[request.username])}
            )
            raise HTTPException(status_code=401, detail="Kullanıcı adı veya şifre hatalı")

        # Successful login — clear failed attempt counter
        _failed_logins.pop(request.username, None)

        # Access token oluştur
        token = generate_token()
        session = {
            "username": request.username,
            "created_at": datetime.now().isoformat(),
        }
        TOKENS[token] = session
        save_session(token, session)

        # Refresh token oluştur
        refresh_token = generate_token()
        REFRESH_TOKENS[refresh_token] = {
            "username": request.username,
            "created_at": datetime.now().isoformat(),
        }

        return {
            "success": True,
            "data": {
                "token": token,
                "refresh_token": refresh_token,
                "user": {
                    "id": user["id"],
                    "username": user["username"],
                    "email": user["email"],
                    "role": user["role"],
                },
                "expires_in": 86400,  # 24 saat
            },
        }
    except HTTPException:
        raise
    except Exception as e:
        return {"success": False, "error": str(e)}


@router.post("/register")
async def register(request: RegisterRequest):
    """Yeni kullanıcı kaydı"""
    try:
        users = load_users()

        # Validasyon
        if request.username in users:
            raise HTTPException(
                status_code=400, detail="Bu kullanıcı adı zaten kullanılıyor"
            )

        if request.password != request.password_confirm:
            raise HTTPException(status_code=400, detail="Şifreler eşleşmiyor")

        validate_password_strength(request.password)

        # Email kontrolü
        for user in users.values():
            if user["email"] == request.email:
                raise HTTPException(
                    status_code=400, detail="Bu e-posta zaten kullanılıyor"
                )

        # Yeni kullanıcı oluştur
        new_user = {
            "id": str(len(users) + 1),
            "username": request.username,
            "email": request.email,
            "password_hash": hash_password(request.password),
            "role": "user",
            "created_at": datetime.now().isoformat(),
        }

        users[request.username] = new_user
        save_users(users)

        # Otomatik giriş için token oluştur
        token = generate_token()
        session = {"username": request.username, "created_at": datetime.now().isoformat()}
        TOKENS[token] = session
        save_session(token, session)

        refresh_token = generate_token()
        REFRESH_TOKENS[refresh_token] = {
            "username": request.username,
            "created_at": datetime.now().isoformat(),
        }

        return {
            "success": True,
            "data": {
                "token": token,
                "refresh_token": refresh_token,
                "user": {
                    "id": new_user["id"],
                    "username": new_user["username"],
                    "email": new_user["email"],
                    "role": new_user["role"],
                },
                "message": "Kayıt başarılı!",
            },
        }
    except HTTPException:
        raise
    except Exception as e:
        return {"success": False, "error": str(e)}


@router.post("/logout")
async def logout(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Çıkış yap"""
    try:
        if credentials and credentials.credentials in TOKENS:
            del TOKENS[credentials.credentials]
            delete_session(credentials.credentials)

        return {"success": True, "message": "Başarıyla çıkış yapıldı"}
    except Exception as e:
        return {"success": False, "error": str(e)}


class RefreshRequest(BaseModel):
    refresh_token: str


@router.post("/refresh")
async def refresh_access_token(request: RefreshRequest):
    """Refresh token ile yeni access token al."""
    rt_data = REFRESH_TOKENS.get(request.refresh_token)
    if not rt_data:
        raise HTTPException(status_code=401, detail="Geçersiz refresh token")

    # Refresh token süre kontrolü (7 gün)
    created = datetime.fromisoformat(rt_data["created_at"])
    if datetime.now() - created >= REFRESH_TOKEN_MAX_AGE:
        del REFRESH_TOKENS[request.refresh_token]
        raise HTTPException(status_code=401, detail="Refresh token süresi dolmuş, yeniden giriş yapın")

    username = rt_data["username"]
    users = load_users()
    if username not in users:
        raise HTTPException(status_code=401, detail="Kullanıcı bulunamadı")

    # Yeni access token
    new_token = generate_token()
    session = {"username": username, "created_at": datetime.now().isoformat()}
    TOKENS[new_token] = session
    save_session(new_token, session)

    return {
        "success": True,
        "data": {
            "token": new_token,
            "expires_in": 86400,
        },
    }


@router.get("/me")
async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
):
    """Mevcut kullanıcı bilgilerini getir"""
    try:
        if not credentials:
            raise HTTPException(status_code=401, detail="Token gerekli")

        user = verify_token(credentials.credentials)

        if not user:
            raise HTTPException(
                status_code=401, detail="Geçersiz veya süresi dolmuş token"
            )

        return {
            "success": True,
            "data": {
                "id": user["id"],
                "username": user["username"],
                "email": user["email"],
                "role": user["role"],
                "created_at": user["created_at"],
            },
        }
    except HTTPException:
        raise
    except Exception as e:
        return {"success": False, "error": str(e)}


@router.get("/users")
async def list_users(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Kullanıcı listesi (sadece admin)"""
    try:
        if not credentials:
            raise HTTPException(status_code=401, detail="Token gerekli")

        user = verify_token(credentials.credentials)

        if not user or user.get("role") != "admin":
            raise HTTPException(status_code=403, detail="Yetkiniz yok")

        users = load_users()
        result = []
        for u in users.values():
            result.append(
                {
                    "id": u["id"],
                    "username": u["username"],
                    "email": u["email"],
                    "role": u["role"],
                    "created_at": u["created_at"],
                }
            )

        return {"success": True, "data": result}
    except HTTPException:
        raise
    except Exception as e:
        return {"success": False, "error": str(e)}


@router.post("/change-password")
async def change_password(
    request: ChangePasswordRequest,
    credentials: HTTPAuthorizationCredentials = Depends(security),
):
    """Şifre değiştir"""
    try:
        if not credentials:
            raise HTTPException(status_code=401, detail="Token gerekli")

        user = verify_token(credentials.credentials)

        if not user:
            raise HTTPException(status_code=401, detail="Geçersiz token")

        users = load_users()
        username = user["username"]

        # Mevcut şifre kontrolü
        if not verify_password(request.current_password, users[username]["password_hash"]):
            raise HTTPException(status_code=400, detail="Mevcut şifre hatalı")

        # Apply same password policy as registration
        validate_password_strength(request.new_password)

        # Şifreyi güncelle
        users[username]["password_hash"] = hash_password(request.new_password)
        save_users(users)

        return {"success": True, "message": "Şifre başarıyla değiştirildi"}
    except HTTPException:
        raise
    except Exception as e:
        return {"success": False, "error": str(e)}
