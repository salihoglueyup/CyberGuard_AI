"""
Auth API Routes - CyberGuard AI
Kimlik doğrulama ve yetkilendirme

Dosya Yolu: app/api/routes/auth.py
"""

from fastapi import APIRouter, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, EmailStr
from typing import Optional
import sys
import os
from datetime import datetime, timedelta
import hashlib
import secrets
import json

# Path düzeltmesi
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
sys.path.insert(0, project_root)

router = APIRouter()
security = HTTPBearer(auto_error=False)

# Basit in-memory user storage (production'da DB kullanılmalı)
USERS_FILE = os.path.join(project_root, "data", "users.json")
TOKENS = {}  # token -> user_id mapping

# Varsayılan admin kullanıcı
DEFAULT_USERS = {
    "admin": {
        "id": "1",
        "username": "admin",
        "email": "admin@cyberguard.ai",
        "password_hash": hashlib.sha256("admin123".encode()).hexdigest(),
        "role": "admin",
        "created_at": datetime.now().isoformat(),
    }
}


def load_users():
    """Kullanıcıları yükle"""
    if os.path.exists(USERS_FILE):
        try:
            with open(USERS_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except:
            pass
    return DEFAULT_USERS.copy()


def save_users(users):
    """Kullanıcıları kaydet"""
    os.makedirs(os.path.dirname(USERS_FILE), exist_ok=True)
    with open(USERS_FILE, "w", encoding="utf-8") as f:
        json.dump(users, f, indent=2, ensure_ascii=False)


def hash_password(password: str) -> str:
    """Şifre hash'le"""
    return hashlib.sha256(password.encode()).hexdigest()


def generate_token() -> str:
    """Token oluştur"""
    return secrets.token_urlsafe(32)


def verify_token(token: str) -> Optional[dict]:
    """Token doğrula"""
    if token in TOKENS:
        token_data = TOKENS[token]
        # Token süresi kontrolü (24 saat)
        created = datetime.fromisoformat(token_data["created_at"])
        if datetime.now() - created < timedelta(hours=24):
            users = load_users()
            username = token_data["username"]
            if username in users:
                return users[username]
    return None


class LoginRequest(BaseModel):
    username: str
    password: str


class RegisterRequest(BaseModel):
    username: str
    email: str
    password: str
    password_confirm: str


class UserResponse(BaseModel):
    id: str
    username: str
    email: str
    role: str
    created_at: str


@router.post("/login")
async def login(request: LoginRequest):
    """Kullanıcı girişi"""
    try:
        users = load_users()

        if request.username not in users:
            raise HTTPException(status_code=401, detail="Kullanıcı bulunamadı")

        user = users[request.username]
        password_hash = hash_password(request.password)

        if user["password_hash"] != password_hash:
            raise HTTPException(status_code=401, detail="Şifre hatalı")

        # Token oluştur
        token = generate_token()
        TOKENS[token] = {
            "username": request.username,
            "created_at": datetime.now().isoformat(),
        }

        return {
            "success": True,
            "data": {
                "token": token,
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

        if len(request.password) < 6:
            raise HTTPException(status_code=400, detail="Şifre en az 6 karakter olmalı")

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
        TOKENS[token] = {
            "username": request.username,
            "created_at": datetime.now().isoformat(),
        }

        return {
            "success": True,
            "data": {
                "token": token,
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

        return {"success": True, "message": "Başarıyla çıkış yapıldı"}
    except Exception as e:
        return {"success": False, "error": str(e)}


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
    current_password: str,
    new_password: str,
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
        if users[username]["password_hash"] != hash_password(current_password):
            raise HTTPException(status_code=400, detail="Mevcut şifre hatalı")

        if len(new_password) < 6:
            raise HTTPException(
                status_code=400, detail="Yeni şifre en az 6 karakter olmalı"
            )

        # Şifreyi güncelle
        users[username]["password_hash"] = hash_password(new_password)
        save_users(users)

        return {"success": True, "message": "Şifre başarıyla değiştirildi"}
    except HTTPException:
        raise
    except Exception as e:
        return {"success": False, "error": str(e)}
