"""
STIX/TAXII API - REAL DATA VERSION
Threat intelligence sharing with STIX format support
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
stix_file = os.path.join(data_dir, "stix_objects.json")
collections_file = os.path.join(data_dir, "taxii_collections.json")

STIX_OBJECTS = []
COLLECTIONS = []

# Default collections
DEFAULT_COLLECTIONS = [
    {
        "id": "indicators",
        "title": "Threat Indicators",
        "description": "IOCs and threat indicators",
        "can_read": True,
        "can_write": True,
    },
    {
        "id": "campaigns",
        "title": "Attack Campaigns",
        "description": "Known attack campaigns",
        "can_read": True,
        "can_write": False,
    },
    {
        "id": "malware",
        "title": "Malware Repository",
        "description": "Known malware signatures",
        "can_read": True,
        "can_write": True,
    },
]


class STIXIndicator(BaseModel):
    name: str
    pattern: str
    pattern_type: str = "stix"
    valid_from: Optional[str] = None
    labels: List[str] = []


def load_data():
    global STIX_OBJECTS, COLLECTIONS
    if os.path.exists(stix_file):
        try:
            with open(stix_file, "r") as f:
                STIX_OBJECTS = json.load(f)
        except:
            pass
    if os.path.exists(collections_file):
        try:
            with open(collections_file, "r") as f:
                COLLECTIONS = json.load(f)
        except:
            COLLECTIONS = DEFAULT_COLLECTIONS.copy()
    else:
        COLLECTIONS = DEFAULT_COLLECTIONS.copy()


def save_data():
    os.makedirs(os.path.dirname(stix_file), exist_ok=True)
    with open(stix_file, "w") as f:
        json.dump(STIX_OBJECTS, f, indent=2, default=str)
    with open(collections_file, "w") as f:
        json.dump(COLLECTIONS, f, indent=2, default=str)


def create_stix_indicator(indicator: STIXIndicator) -> Dict:
    return {
        "type": "indicator",
        "spec_version": "2.1",
        "id": f"indicator--{str(uuid.uuid4())}",
        "created": datetime.now().isoformat() + "Z",
        "modified": datetime.now().isoformat() + "Z",
        "name": indicator.name,
        "pattern": indicator.pattern,
        "pattern_type": indicator.pattern_type,
        "valid_from": indicator.valid_from or datetime.now().isoformat() + "Z",
        "labels": indicator.labels,
    }


load_data()


@router.get("/status")
async def get_status():
    return {
        "success": True,
        "data": {
            "status": "active",
            "objects": len(STIX_OBJECTS),
            "collections": len(COLLECTIONS),
            "spec_version": "2.1",
        },
    }


# TAXII Discovery
@router.get("/taxii2")
async def taxii_discovery():
    return {
        "title": "CyberGuard AI TAXII Server",
        "description": "Threat intelligence sharing",
        "default": "/taxii2/default",
        "api_roots": ["/taxii2/default"],
    }


@router.get("/taxii2/default")
async def api_root():
    return {
        "title": "Default API Root",
        "versions": ["application/taxii+json;version=2.1"],
        "max_content_length": 10485760,
    }


# Collections
@router.get("/taxii2/default/collections")
async def list_collections():
    return {"collections": COLLECTIONS}


@router.get("/taxii2/default/collections/{collection_id}")
async def get_collection(collection_id: str):
    for c in COLLECTIONS:
        if c["id"] == collection_id:
            return c
    raise HTTPException(status_code=404)


@router.get("/taxii2/default/collections/{collection_id}/objects")
async def get_objects(collection_id: str, limit: int = 100):
    objects = [o for o in STIX_OBJECTS if o.get("collection") == collection_id]
    return {"objects": objects[:limit], "more": len(objects) > limit}


@router.post("/taxii2/default/collections/{collection_id}/objects")
async def add_object(collection_id: str, indicator: STIXIndicator):
    # Check collection
    collection = None
    for c in COLLECTIONS:
        if c["id"] == collection_id:
            collection = c
            break

    if not collection:
        raise HTTPException(status_code=404)
    if not collection.get("can_write"):
        raise HTTPException(status_code=403, detail="Collection is read-only")

    stix_obj = create_stix_indicator(indicator)
    stix_obj["collection"] = collection_id
    STIX_OBJECTS.append(stix_obj)
    save_data()

    return {"id": stix_obj["id"], "success_count": 1, "failure_count": 0}


# STIX objects direct access
@router.get("/stix/objects")
async def list_stix_objects(object_type: Optional[str] = None, limit: int = 100):
    objects = STIX_OBJECTS
    if object_type:
        objects = [o for o in objects if o.get("type") == object_type]
    return {
        "success": True,
        "data": {"objects": objects[:limit], "total": len(objects)},
    }


@router.get("/stix/object/{object_id}")
async def get_stix_object(object_id: str):
    for o in STIX_OBJECTS:
        if o.get("id") == object_id:
            return {"success": True, "data": o}
    raise HTTPException(status_code=404)


@router.delete("/stix/object/{object_id}")
async def delete_stix_object(object_id: str):
    global STIX_OBJECTS
    for i, o in enumerate(STIX_OBJECTS):
        if o.get("id") == object_id:
            STIX_OBJECTS.pop(i)
            save_data()
            return {"success": True}
    raise HTTPException(status_code=404)


@router.get("/stats")
async def get_stats():
    by_type = {}
    for o in STIX_OBJECTS:
        t = o.get("type", "unknown")
        by_type[t] = by_type.get(t, 0) + 1
    return {
        "success": True,
        "data": {
            "total_objects": len(STIX_OBJECTS),
            "by_type": by_type,
            "collections": len(COLLECTIONS),
        },
    }
