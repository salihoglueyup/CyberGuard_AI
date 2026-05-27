"""
Federated Learning API - REAL DATA VERSION
Federated learning coordination with node management
"""

import json
import logging
import os
import uuid
from datetime import datetime
from urllib.parse import urlparse

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, field_validator

from app.api.routes.auth import require_auth

logger = logging.getLogger(__name__)

router = APIRouter()

from app.paths import PROJECT_ROOT

project_root = str(PROJECT_ROOT)
data_dir = os.path.join(project_root, "data")
fed_file = os.path.join(data_dir, "federated.json")

FED_DATA = {"nodes": [], "rounds": [], "global_model": None}


class FederatedNode(BaseModel):
    name: str
    endpoint: str
    data_samples: int = 1000

    @field_validator("endpoint")
    @classmethod
    def validate_endpoint(cls, v: str) -> str:
        parsed = urlparse(v)
        hostname = parsed.hostname or ""
        allowed_hosts = {"localhost", "127.0.0.1", "::1"}
        if hostname not in allowed_hosts and not hostname.startswith(("10.", "172.", "192.168.")):
            raise ValueError("Only localhost/private IP endpoints are allowed")
        return v


class TrainingRound(BaseModel):
    min_nodes: int = 2
    epochs: int = 10


def load_data():
    global FED_DATA
    if os.path.exists(fed_file):
        try:
            with open(fed_file) as f:
                FED_DATA.update(json.load(f))
        except (OSError, json.JSONDecodeError) as e:
            logger.warning(f"Failed to load federated data: {e}")


def save_data():
    os.makedirs(os.path.dirname(fed_file), exist_ok=True)
    with open(fed_file, "w") as f:
        json.dump(FED_DATA, f, indent=2, default=str)


load_data()


@router.get("/status")
async def get_status(user: dict = Depends(require_auth)):
    active_nodes = len([n for n in FED_DATA["nodes"] if n.get("status") == "active"])
    return {
        "success": True,
        "data": {
            "status": "active",
            "total_nodes": len(FED_DATA["nodes"]),
            "active_nodes": active_nodes,
            "training_rounds": len(FED_DATA["rounds"]),
            "global_model_version": (FED_DATA.get("global_model") or {}).get("version"),
        },
    }


@router.get("/nodes")
async def list_nodes(user: dict = Depends(require_auth)):
    return {"success": True, "data": {"nodes": FED_DATA["nodes"]}}


@router.post("/nodes")
async def register_node(node: FederatedNode, user: dict = Depends(require_auth)):
    new_node = {
        "id": f"NODE-{str(uuid.uuid4())[:8]}",
        "name": node.name,
        "endpoint": node.endpoint,
        "data_samples": node.data_samples,
        "status": "active",
        "registered_at": datetime.now().isoformat(),
        "last_seen": datetime.now().isoformat(),
        "rounds_participated": 0,
    }
    FED_DATA["nodes"].append(new_node)
    if len(FED_DATA["nodes"]) > 500:
        FED_DATA["nodes"] = FED_DATA["nodes"][-250:]
    save_data()
    return {"success": True, "data": new_node}


@router.delete("/nodes/{node_id}")
async def remove_node(node_id: str, user: dict = Depends(require_auth)):
    for i, n in enumerate(FED_DATA["nodes"]):
        if n.get("id") == node_id:
            FED_DATA["nodes"].pop(i)
            save_data()
            return {"success": True}
    raise HTTPException(status_code=404)


@router.post("/rounds")
async def start_training_round(config: TrainingRound, user: dict = Depends(require_auth)):
    active_nodes = [n for n in FED_DATA["nodes"] if n.get("status") == "active"]

    if len(active_nodes) < config.min_nodes:
        raise HTTPException(
            status_code=400, detail=f"Need at least {config.min_nodes} active nodes"
        )

    new_round = {
        "id": f"ROUND-{len(FED_DATA['rounds']) + 1}",
        "status": "running",
        "started_at": datetime.now().isoformat(),
        "epochs": config.epochs,
        "participating_nodes": [n["id"] for n in active_nodes[: config.min_nodes]],
        "node_updates": [],
        "completed_at": None,
        "aggregated_model": None,
    }

    # Update node participation
    for node_id in new_round["participating_nodes"]:
        for n in FED_DATA["nodes"]:
            if n["id"] == node_id:
                n["rounds_participated"] = n.get("rounds_participated", 0) + 1

    FED_DATA["rounds"].append(new_round)
    if len(FED_DATA["rounds"]) > 1000:
        FED_DATA["rounds"] = FED_DATA["rounds"][-500:]
    save_data()

    return {"success": True, "data": new_round}


@router.get("/rounds")
async def list_rounds(user: dict = Depends(require_auth)):
    return {"success": True, "data": {"rounds": FED_DATA["rounds"]}}


@router.get("/rounds/{round_id}")
async def get_round(round_id: str, user: dict = Depends(require_auth)):
    for r in FED_DATA["rounds"]:
        if r.get("id") == round_id:
            return {"success": True, "data": r}
    raise HTTPException(status_code=404)


@router.post("/rounds/{round_id}/update")
async def submit_update(round_id: str, node_id: str, accuracy: float, user: dict = Depends(require_auth)):
    for r in FED_DATA["rounds"]:
        if r.get("id") == round_id:
            r["node_updates"].append(
                {
                    "node_id": node_id,
                    "accuracy": accuracy,
                    "submitted_at": datetime.now().isoformat(),
                }
            )
            save_data()
            return {"success": True}
    raise HTTPException(status_code=404)


@router.get("/model")
async def get_global_model(user: dict = Depends(require_auth)):
    return {"success": True, "data": {"global_model": FED_DATA.get("global_model")}}


@router.get("/stats")
async def get_stats(user: dict = Depends(require_auth)):
    rounds = FED_DATA["rounds"]
    completed = [r for r in rounds if r.get("status") == "completed"]

    return {
        "success": True,
        "data": {
            "total_nodes": len(FED_DATA["nodes"]),
            "active_nodes": len(
                [n for n in FED_DATA["nodes"] if n.get("status") == "active"]
            ),
            "total_rounds": len(rounds),
            "completed_rounds": len(completed),
            "total_data_samples": sum(
                n.get("data_samples", 0) for n in FED_DATA["nodes"]
            ),
        },
    }
