"""
Federated Learning API - REAL DATA VERSION
Federated learning coordination with node management
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Optional
from datetime import datetime
import os
import json
import uuid

router = APIRouter()

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
data_dir = os.path.join(project_root, "data")
fed_file = os.path.join(data_dir, "federated.json")

FED_DATA = {"nodes": [], "rounds": [], "global_model": None}


class FederatedNode(BaseModel):
    name: str
    endpoint: str
    data_samples: int = 1000


class TrainingRound(BaseModel):
    min_nodes: int = 2
    epochs: int = 10


def load_data():
    global FED_DATA
    if os.path.exists(fed_file):
        try:
            with open(fed_file, "r") as f:
                FED_DATA.update(json.load(f))
        except:
            pass


def save_data():
    os.makedirs(os.path.dirname(fed_file), exist_ok=True)
    with open(fed_file, "w") as f:
        json.dump(FED_DATA, f, indent=2, default=str)


load_data()


@router.get("/status")
async def get_status():
    active_nodes = len([n for n in FED_DATA["nodes"] if n.get("status") == "active"])
    return {
        "success": True,
        "data": {
            "status": "active",
            "total_nodes": len(FED_DATA["nodes"]),
            "active_nodes": active_nodes,
            "training_rounds": len(FED_DATA["rounds"]),
            "global_model_version": FED_DATA.get("global_model", {}).get("version"),
        },
    }


@router.get("/nodes")
async def list_nodes():
    return {"success": True, "data": {"nodes": FED_DATA["nodes"]}}


@router.post("/nodes")
async def register_node(node: FederatedNode):
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
    save_data()
    return {"success": True, "data": new_node}


@router.delete("/nodes/{node_id}")
async def remove_node(node_id: str):
    for i, n in enumerate(FED_DATA["nodes"]):
        if n.get("id") == node_id:
            FED_DATA["nodes"].pop(i)
            save_data()
            return {"success": True}
    raise HTTPException(status_code=404)


@router.post("/rounds")
async def start_training_round(config: TrainingRound):
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
    save_data()

    return {"success": True, "data": new_round}


@router.get("/rounds")
async def list_rounds():
    return {"success": True, "data": {"rounds": FED_DATA["rounds"]}}


@router.get("/rounds/{round_id}")
async def get_round(round_id: str):
    for r in FED_DATA["rounds"]:
        if r.get("id") == round_id:
            return {"success": True, "data": r}
    raise HTTPException(status_code=404)


@router.post("/rounds/{round_id}/update")
async def submit_update(round_id: str, node_id: str, accuracy: float):
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
async def get_global_model():
    return {"success": True, "data": {"global_model": FED_DATA.get("global_model")}}


@router.get("/stats")
async def get_stats():
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
