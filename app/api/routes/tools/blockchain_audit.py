"""
Blockchain Audit Trail API - REAL DATA VERSION
File-based immutable audit logging with hash verification
"""

import hashlib
import json
import logging
import os
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from app.api.routes.auth import require_auth

logger = logging.getLogger(__name__)
router = APIRouter(dependencies=[Depends(require_auth)])

from app.paths import PROJECT_ROOT

project_root = str(PROJECT_ROOT)
data_dir = os.path.join(project_root, "data")
chain_file = os.path.join(data_dir, "audit_chain.json")

CHAIN = []
MAX_CHAIN_LENGTH = 10000


class AuditEntry(BaseModel):
    action: str
    actor: str
    resource: str
    details: dict | None = None


def load_chain():
    global CHAIN
    if os.path.exists(chain_file):
        try:
            with open(chain_file) as f:
                CHAIN = json.load(f)
        except Exception as e:
            print(f"[blockchain_audit] Failed to load audit chain: {e}")
    if not CHAIN:
        # Genesis block
        CHAIN = [
            {
                "index": 0,
                "timestamp": datetime.now().isoformat(),
                "action": "genesis",
                "hash": "0" * 64,
                "prev_hash": "0" * 64,
            }
        ]


def save_chain():
    os.makedirs(os.path.dirname(chain_file), exist_ok=True)
    with open(chain_file, "w") as f:
        json.dump(CHAIN, f, indent=2, default=str)


def calculate_hash(block: dict) -> str:
    block_string = json.dumps(
        {k: v for k, v in block.items() if k != "hash"}, sort_keys=True
    )
    return hashlib.sha256(block_string.encode()).hexdigest()


def verify_chain() -> dict:
    if len(CHAIN) <= 1:
        return {"valid": True, "checked": len(CHAIN)}

    for i in range(1, len(CHAIN)):
        current = CHAIN[i]
        previous = CHAIN[i - 1]

        # Check hash linkage
        if current.get("prev_hash") != previous.get("hash"):
            return {"valid": False, "error": f"Broken link at block {i}", "checked": i}

        # Verify current block hash
        expected = calculate_hash(current)
        if current.get("hash") != expected:
            return {"valid": False, "error": f"Invalid hash at block {i}", "checked": i}

    return {"valid": True, "checked": len(CHAIN)}


load_chain()


@router.get("/status")
async def get_status():
    verification = verify_chain()
    return {
        "success": True,
        "data": {
            "chain_length": len(CHAIN),
            "verified": verification["valid"],
            "last_block": CHAIN[-1]["index"] if CHAIN else 0,
        },
    }


@router.post("/log")
async def add_audit_entry(entry: AuditEntry):
    prev_block = CHAIN[-1] if CHAIN else {"hash": "0" * 64}

    new_block = {
        "index": len(CHAIN),
        "timestamp": datetime.now().isoformat(),
        "action": entry.action,
        "actor": entry.actor,
        "resource": entry.resource,
        "details": entry.details or {},
        "prev_hash": prev_block["hash"],
        "hash": "",  # Will be calculated
    }
    new_block["hash"] = calculate_hash(new_block)

    CHAIN.append(new_block)
    if len(CHAIN) > MAX_CHAIN_LENGTH:
        # Archive oldest entries, keep last MAX_CHAIN_LENGTH
        CHAIN[:] = CHAIN[-MAX_CHAIN_LENGTH:]
        # Reset anchor block's prev_hash since referenced blocks are gone
        CHAIN[0]["prev_hash"] = "0" * 64
    save_chain()

    return {
        "success": True,
        "data": {"block_index": new_block["index"], "hash": new_block["hash"]},
    }


@router.get("/chain")
async def get_chain(limit: int = 100, offset: int = 0):
    return {
        "success": True,
        "data": {"blocks": CHAIN[offset : offset + limit], "total": len(CHAIN)},
    }


@router.get("/block/{index}")
async def get_block(index: int):
    if 0 <= index < len(CHAIN):
        return {"success": True, "data": CHAIN[index]}
    raise HTTPException(status_code=404)


@router.get("/verify")
async def verify():
    result = verify_chain()
    return {"success": True, "data": result}


@router.get("/search")
async def search_chain(
    actor: str | None = None,
    action: str | None = None,
    resource: str | None = None,
):
    results = CHAIN[1:]  # Skip genesis
    if actor:
        results = [b for b in results if b.get("actor") == actor]
    if action:
        results = [b for b in results if b.get("action") == action]
    if resource:
        results = [b for b in results if resource in b.get("resource", "")]
    return {"success": True, "data": {"results": results[-100:], "total": len(results)}}


@router.get("/stats")
async def get_stats():
    actions = {}
    actors = {}
    for b in CHAIN[1:]:
        a = b.get("action", "unknown")
        actions[a] = actions.get(a, 0) + 1
        ac = b.get("actor", "unknown")
        actors[ac] = actors.get(ac, 0) + 1
    return {
        "success": True,
        "data": {
            "total_entries": len(CHAIN) - 1,
            "by_action": actions,
            "by_actor": actors,
        },
    }
