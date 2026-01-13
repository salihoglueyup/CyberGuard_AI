"""
HSM (Hardware Security Module) API - REAL DATA VERSION
Simulates HSM operations with file-based key storage
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Optional
from datetime import datetime
import os
import json
import hashlib
import secrets
import base64

router = APIRouter()

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
data_dir = os.path.join(project_root, "data")
hsm_file = os.path.join(data_dir, "hsm_keys.json")

HSM_DATA = {"keys": [], "operations": []}


class KeyGenRequest(BaseModel):
    name: str
    key_type: str = "aes256"  # aes256, rsa2048, ecdsa
    purpose: str = "encryption"


class EncryptRequest(BaseModel):
    key_id: str
    plaintext: str


def load_hsm():
    global HSM_DATA
    if os.path.exists(hsm_file):
        try:
            with open(hsm_file, "r") as f:
                HSM_DATA.update(json.load(f))
        except:
            pass


def save_hsm():
    os.makedirs(os.path.dirname(hsm_file), exist_ok=True)
    with open(hsm_file, "w") as f:
        json.dump(HSM_DATA, f, indent=2, default=str)


def generate_key(key_type: str) -> str:
    """Generate a cryptographic key"""
    if key_type == "aes256":
        return base64.b64encode(secrets.token_bytes(32)).decode()
    elif key_type == "rsa2048":
        # Simulated RSA key (in real HSM, actual RSA key would be generated)
        return base64.b64encode(secrets.token_bytes(256)).decode()
    elif key_type == "ecdsa":
        return base64.b64encode(secrets.token_bytes(32)).decode()
    else:
        return base64.b64encode(secrets.token_bytes(32)).decode()


def encrypt_data(key: str, plaintext: str) -> str:
    """Simple XOR encryption (for demonstration)"""
    key_bytes = base64.b64decode(key)
    plaintext_bytes = plaintext.encode()

    # Repeat key to match plaintext length
    key_extended = (key_bytes * (len(plaintext_bytes) // len(key_bytes) + 1))[
        : len(plaintext_bytes)
    ]

    encrypted = bytes(a ^ b for a, b in zip(plaintext_bytes, key_extended))
    return base64.b64encode(encrypted).decode()


def decrypt_data(key: str, ciphertext: str) -> str:
    """Simple XOR decryption"""
    key_bytes = base64.b64decode(key)
    ciphertext_bytes = base64.b64decode(ciphertext)

    key_extended = (key_bytes * (len(ciphertext_bytes) // len(key_bytes) + 1))[
        : len(ciphertext_bytes)
    ]

    decrypted = bytes(a ^ b for a, b in zip(ciphertext_bytes, key_extended))
    return decrypted.decode()


load_hsm()


@router.get("/status")
async def get_status():
    return {
        "success": True,
        "data": {
            "status": "active",
            "mode": "software_simulation",
            "total_keys": len(HSM_DATA["keys"]),
            "total_operations": len(HSM_DATA["operations"]),
            "supported_algorithms": ["aes256", "rsa2048", "ecdsa"],
        },
    }


@router.post("/keys")
async def generate_hsm_key(request: KeyGenRequest):
    key_material = generate_key(request.key_type)

    new_key = {
        "id": f"KEY-{secrets.token_hex(4).upper()}",
        "name": request.name,
        "type": request.key_type,
        "purpose": request.purpose,
        "key_hash": hashlib.sha256(key_material.encode()).hexdigest()[:16],
        "created_at": datetime.now().isoformat(),
        "status": "active",
        "_material": key_material,  # In real HSM, this never leaves the device
    }

    HSM_DATA["keys"].append(new_key)
    save_hsm()

    # Don't return actual key material in production
    return {
        "success": True,
        "data": {
            "id": new_key["id"],
            "name": new_key["name"],
            "type": new_key["type"],
            "key_hash": new_key["key_hash"],
        },
    }


@router.get("/keys")
async def list_keys():
    # Never return key material
    safe_keys = []
    for k in HSM_DATA["keys"]:
        safe_keys.append(
            {
                "id": k["id"],
                "name": k["name"],
                "type": k["type"],
                "purpose": k["purpose"],
                "status": k["status"],
                "created_at": k["created_at"],
            }
        )
    return {"success": True, "data": {"keys": safe_keys}}


@router.delete("/keys/{key_id}")
async def delete_key(key_id: str):
    for i, k in enumerate(HSM_DATA["keys"]):
        if k.get("id") == key_id:
            HSM_DATA["keys"].pop(i)
            save_hsm()
            return {"success": True, "message": "Key deleted"}
    raise HTTPException(status_code=404)


@router.post("/encrypt")
async def encrypt(request: EncryptRequest):
    key = None
    for k in HSM_DATA["keys"]:
        if k.get("id") == request.key_id and k.get("status") == "active":
            key = k
            break

    if not key:
        raise HTTPException(status_code=404, detail="Key not found")

    ciphertext = encrypt_data(key["_material"], request.plaintext)

    # Log operation
    HSM_DATA["operations"].append(
        {
            "type": "encrypt",
            "key_id": request.key_id,
            "timestamp": datetime.now().isoformat(),
        }
    )
    save_hsm()

    return {"success": True, "data": {"ciphertext": ciphertext}}


@router.post("/decrypt")
async def decrypt(key_id: str, ciphertext: str):
    key = None
    for k in HSM_DATA["keys"]:
        if k.get("id") == key_id and k.get("status") == "active":
            key = k
            break

    if not key:
        raise HTTPException(status_code=404, detail="Key not found")

    try:
        plaintext = decrypt_data(key["_material"], ciphertext)
    except:
        raise HTTPException(status_code=400, detail="Decryption failed")

    HSM_DATA["operations"].append(
        {"type": "decrypt", "key_id": key_id, "timestamp": datetime.now().isoformat()}
    )
    save_hsm()

    return {"success": True, "data": {"plaintext": plaintext}}


@router.get("/operations")
async def get_operations(limit: int = 100):
    return {"success": True, "data": {"operations": HSM_DATA["operations"][-limit:]}}


@router.get("/stats")
async def get_stats():
    ops = HSM_DATA["operations"]
    by_type = {}
    for o in ops:
        t = o.get("type", "unknown")
        by_type[t] = by_type.get(t, 0) + 1

    return {
        "success": True,
        "data": {
            "total_keys": len(HSM_DATA["keys"]),
            "active_keys": len(
                [k for k in HSM_DATA["keys"] if k.get("status") == "active"]
            ),
            "total_operations": len(ops),
            "operations_by_type": by_type,
        },
    }
