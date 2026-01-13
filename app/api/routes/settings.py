"""
Settings API Routes - CyberGuard AI
Kullanıcı ayarları ve tercihlerini yönetim

Dosya Yolu: app/api/routes/settings.py
"""

from fastapi import APIRouter
from pydantic import BaseModel
from typing import Optional, Dict
import json
import os
import sys

# Path düzeltmesi
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
sys.path.insert(0, project_root)

router = APIRouter()

# Settings dosya yolu
SETTINGS_FILE = os.path.join(project_root, "src", "database", "settings.json")


class SettingsModel(BaseModel):
    api_keys: Optional[Dict] = {}
    notifications: Optional[Dict] = {}
    appearance: Optional[Dict] = {}
    language: Optional[str] = "tr"


def load_settings() -> dict:
    """Ayarları yükle"""
    try:
        if os.path.exists(SETTINGS_FILE):
            with open(SETTINGS_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception as e:
        print(f"Settings load error: {e}")

    return {
        "api_keys": {"gemini": "", "openai": "", "virustotal": ""},
        "notifications": {
            "threats": True,
            "training": True,
            "system": False,
            "email": False,
        },
        "appearance": {"theme": "dark", "language": "tr"},
        "language": "tr",
    }


def save_settings(settings: dict) -> bool:
    """Ayarları kaydet"""
    try:
        os.makedirs(os.path.dirname(SETTINGS_FILE), exist_ok=True)
        with open(SETTINGS_FILE, "w", encoding="utf-8") as f:
            json.dump(settings, f, indent=2, ensure_ascii=False)
        return True
    except Exception as e:
        print(f"Settings save error: {e}")
        return False


@router.get("")
async def get_settings():
    """Tüm ayarları getir"""
    try:
        settings = load_settings()

        # API key'leri maskele
        masked = settings.copy()
        if "api_keys" in masked:
            for key in masked["api_keys"]:
                if masked["api_keys"][key]:
                    val = masked["api_keys"][key]
                    masked["api_keys"][key] = (
                        val[:8] + "*" * (len(val) - 12) + val[-4:]
                        if len(val) > 12
                        else "***"
                    )

        return {"success": True, "data": masked}
    except Exception as e:
        return {"success": False, "error": str(e)}


@router.post("")
async def update_settings(settings: SettingsModel):
    """Ayarları güncelle"""
    try:
        current = load_settings()

        # Sadece gönderilen alanları güncelle
        if settings.api_keys:
            # Maskelenmiş key'leri güncelleme
            for key, value in settings.api_keys.items():
                if value and "*" not in value:
                    current.setdefault("api_keys", {})[key] = value

        if settings.notifications:
            current["notifications"] = settings.notifications

        if settings.appearance:
            current["appearance"] = settings.appearance

        if settings.language:
            current["language"] = settings.language

        if save_settings(current):
            return {"success": True, "message": "Ayarlar kaydedildi"}
        else:
            return {"success": False, "error": "Kaydetme hatası"}

    except Exception as e:
        return {"success": False, "error": str(e)}


@router.get("/api-key/{key_name}")
async def get_api_key(key_name: str):
    """Belirli bir API key'i getir (maskelenmiş)"""
    try:
        settings = load_settings()
        value = settings.get("api_keys", {}).get(key_name, "")

        if value:
            masked = (
                value[:8] + "*" * (len(value) - 12) + value[-4:]
                if len(value) > 12
                else "***"
            )
            return {
                "success": True,
                "data": {"key": key_name, "value": masked, "exists": True},
            }

        return {
            "success": True,
            "data": {"key": key_name, "value": "", "exists": False},
        }

    except Exception as e:
        return {"success": False, "error": str(e)}


@router.post("/api-key/{key_name}")
async def set_api_key(key_name: str, value: str):
    """API key kaydet"""
    try:
        settings = load_settings()
        settings.setdefault("api_keys", {})[key_name] = value

        if save_settings(settings):
            return {"success": True, "message": f"{key_name} kaydedildi"}

        return {"success": False, "error": "Kaydetme hatası"}

    except Exception as e:
        return {"success": False, "error": str(e)}


@router.post("/test-api-key/{key_name}")
async def test_api_key(key_name: str):
    """API key'i test et"""
    try:
        settings = load_settings()
        value = settings.get("api_keys", {}).get(key_name, "")

        if not value:
            return {"success": False, "error": "API key bulunamadı"}

        # Gemini test
        if key_name == "gemini":
            try:
                import google.generativeai as genai

                genai.configure(api_key=value)
                model = genai.GenerativeModel("gemini-pro")
                response = model.generate_content("Merhaba")
                return {"success": True, "message": "Gemini API çalışıyor!"}
            except Exception as e:
                return {"success": False, "error": f"Gemini hatası: {str(e)}"}

        # Diğer key'ler için basit kontrol
        return {
            "success": True,
            "message": f"{key_name} key mevcut ({len(value)} karakter)",
        }

    except Exception as e:
        return {"success": False, "error": str(e)}


@router.get("/notifications")
async def get_notifications():
    """Bildirim ayarlarını getir"""
    settings = load_settings()
    return {"success": True, "data": settings.get("notifications", {})}


@router.post("/notifications")
async def update_notifications(notifications: Dict):
    """Bildirim ayarlarını güncelle"""
    try:
        settings = load_settings()
        settings["notifications"] = notifications
        save_settings(settings)
        return {"success": True, "message": "Bildirim ayarları güncellendi"}
    except Exception as e:
        return {"success": False, "error": str(e)}
