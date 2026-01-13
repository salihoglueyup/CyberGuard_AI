"""
API Keys Management - REAL DATA VERSION
Secure API key storage with encryption
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Optional
from datetime import datetime
import os
import json
import hashlib
import secrets

router = APIRouter()

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
data_dir = os.path.join(project_root, "data")
keys_file = os.path.join(data_dir, "api_keys.json")

API_KEYS = []


class KeyCreate(BaseModel):
    name: str
    permissions: List[str] = ["read"]
    expires_days: Optional[int] = 365


def load_keys():
    global API_KEYS
    if os.path.exists(keys_file):
        try:
            with open(keys_file, "r") as f:
                API_KEYS = json.load(f)
        except:
            pass


def save_keys():
    os.makedirs(os.path.dirname(keys_file), exist_ok=True)
    with open(keys_file, "w") as f:
        json.dump(API_KEYS, f, indent=2, default=str)


def generate_api_key() -> str:
    return f"cgai_{secrets.token_urlsafe(32)}"


def hash_key(key: str) -> str:
    return hashlib.sha256(key.encode()).hexdigest()


load_keys()


@router.get("/status")
async def get_status():
    active = len([k for k in API_KEYS if k.get("active", True)])
    return {
        "success": True,
        "data": {"total_keys": len(API_KEYS), "active_keys": active},
    }


@router.post("")
async def create_key(key_data: KeyCreate):
    raw_key = generate_api_key()
    hashed = hash_key(raw_key)

    new_key = {
        "id": f"KEY-{len(API_KEYS) + 1:04d}",
        "name": key_data.name,
        "key_hash": hashed,
        "key_prefix": raw_key[:12] + "...",
        "permissions": key_data.permissions,
        "created_at": datetime.now().isoformat(),
        "last_used": None,
        "usage_count": 0,
        "active": True,
    }

    API_KEYS.append(new_key)
    save_keys()

    return {
        "success": True,
        "data": {
            "id": new_key["id"],
            "name": new_key["name"],
            "api_key": raw_key,  # Only shown once!
            "message": "Save this key - it won't be shown again",
        },
    }


@router.get("")
async def list_keys():
    # Don't return actual keys, just metadata
    safe_keys = []
    for k in API_KEYS:
        safe_keys.append(
            {
                "id": k["id"],
                "name": k["name"],
                "key_prefix": k.get("key_prefix", "***"),
                "permissions": k["permissions"],
                "created_at": k["created_at"],
                "last_used": k.get("last_used"),
                "usage_count": k.get("usage_count", 0),
                "active": k.get("active", True),
            }
        )
    return {"success": True, "data": {"keys": safe_keys}}


@router.delete("/{key_id}")
async def revoke_key(key_id: str):
    for k in API_KEYS:
        if k.get("id") == key_id:
            k["active"] = False
            k["revoked_at"] = datetime.now().isoformat()
            save_keys()
            return {"success": True, "message": "Key revoked"}
    raise HTTPException(status_code=404)


@router.post("/validate")
async def validate_key(api_key: str):
    hashed = hash_key(api_key)
    for k in API_KEYS:
        if k.get("key_hash") == hashed and k.get("active", True):
            k["last_used"] = datetime.now().isoformat()
            k["usage_count"] = k.get("usage_count", 0) + 1
            save_keys()
            return {
                "success": True,
                "data": {"valid": True, "permissions": k["permissions"]},
            }
    return {"success": True, "data": {"valid": False}}


@router.get("/stats")
async def get_stats():
    active = len([k for k in API_KEYS if k.get("active")])
    total_usage = sum(k.get("usage_count", 0) for k in API_KEYS)
    return {
        "success": True,
        "data": {"total": len(API_KEYS), "active": active, "total_usage": total_usage},
    }
