"""
Blockchain Audit Trail API - REAL DATA VERSION
File-based immutable audit logging with hash verification
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Optional
from datetime import datetime
import os
import json
import hashlib

router = APIRouter()

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
data_dir = os.path.join(project_root, "data")
chain_file = os.path.join(data_dir, "audit_chain.json")

CHAIN = []


class AuditEntry(BaseModel):
    action: str
    actor: str
    resource: str
    details: Optional[Dict] = None


def load_chain():
    global CHAIN
    if os.path.exists(chain_file):
        try:
            with open(chain_file, "r") as f:
                CHAIN = json.load(f)
        except:
            pass
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


def calculate_hash(block: Dict) -> str:
    block_string = json.dumps(
        {k: v for k, v in block.items() if k != "hash"}, sort_keys=True
    )
    return hashlib.sha256(block_string.encode()).hexdigest()


def verify_chain() -> Dict:
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
    actor: Optional[str] = None,
    action: Optional[str] = None,
    resource: Optional[str] = None,
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
