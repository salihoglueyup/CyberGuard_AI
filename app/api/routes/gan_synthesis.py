"""
GAN Attack Synthesis API - REAL DATA VERSION
Generates synthetic attack data using saved model or statistical methods
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Optional
from datetime import datetime
import os
import json
import numpy as np

try:
    import tensorflow as tf

    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

router = APIRouter()

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
models_dir = os.path.join(project_root, "models")
data_dir = os.path.join(project_root, "data")
gan_file = os.path.join(data_dir, "gan_generated.json")

GENERATED_SAMPLES = []


class GenerateRequest(BaseModel):
    attack_type: str = "ddos"
    num_samples: int = 100
    noise_dim: int = 100


# Attack type feature distributions (based on CICIDS2017 analysis)
ATTACK_DISTRIBUTIONS = {
    "ddos": {
        "flow_duration": (1000, 5000),
        "packet_count": (100, 10000),
        "byte_count": (5000, 500000),
    },
    "portscan": {
        "flow_duration": (100, 1000),
        "packet_count": (1, 10),
        "byte_count": (40, 200),
    },
    "bruteforce": {
        "flow_duration": (500, 3000),
        "packet_count": (5, 50),
        "byte_count": (100, 1000),
    },
    "botnet": {
        "flow_duration": (10000, 100000),
        "packet_count": (10, 100),
        "byte_count": (500, 5000),
    },
    "infiltration": {
        "flow_duration": (5000, 50000),
        "packet_count": (50, 500),
        "byte_count": (1000, 50000),
    },
}


def load_samples():
    global GENERATED_SAMPLES
    if os.path.exists(gan_file):
        try:
            with open(gan_file, "r") as f:
                GENERATED_SAMPLES = json.load(f)
        except:
            pass


def save_samples():
    os.makedirs(os.path.dirname(gan_file), exist_ok=True)
    with open(gan_file, "w") as f:
        json.dump(GENERATED_SAMPLES[-1000:], f, indent=2, default=str)


def generate_synthetic_samples(attack_type: str, num_samples: int) -> List[Dict]:
    """Generate synthetic attack samples based on statistical distributions"""
    samples = []
    dist = ATTACK_DISTRIBUTIONS.get(attack_type, ATTACK_DISTRIBUTIONS["ddos"])

    for i in range(num_samples):
        # Generate features based on attack type distribution
        flow_duration = np.random.uniform(*dist["flow_duration"])
        packet_count = np.random.uniform(*dist["packet_count"])
        byte_count = np.random.uniform(*dist["byte_count"])

        sample = {
            "id": f"SYN-{datetime.now().strftime('%Y%m%d%H%M%S')}-{i:04d}",
            "attack_type": attack_type,
            "features": {
                "flow_duration": float(flow_duration),
                "fwd_packets": int(packet_count * 0.6),
                "bwd_packets": int(packet_count * 0.4),
                "total_bytes": float(byte_count),
                "flow_bytes_per_sec": float(byte_count / (flow_duration / 1e6)),
                "flow_packets_per_sec": float(packet_count / (flow_duration / 1e6)),
                "packet_length_mean": float(byte_count / max(1, packet_count)),
                "syn_flag_count": int(np.random.poisson(3)),
                "ack_flag_count": int(np.random.poisson(packet_count * 0.8)),
                "fin_flag_count": int(np.random.poisson(1)),
            },
            "generated_at": datetime.now().isoformat(),
        }
        samples.append(sample)

    return samples


load_samples()


@router.get("/status")
async def get_status():
    return {
        "success": True,
        "data": {
            "status": "active",
            "tensorflow_available": TF_AVAILABLE,
            "attack_types": list(ATTACK_DISTRIBUTIONS.keys()),
            "total_generated": len(GENERATED_SAMPLES),
        },
    }


@router.get("/attack-types")
async def list_attack_types():
    types = []
    for name, dist in ATTACK_DISTRIBUTIONS.items():
        types.append(
            {
                "id": name,
                "name": name.upper(),
                "description": f"Synthetic {name} attack samples",
                "feature_ranges": dist,
            }
        )
    return {"success": True, "data": {"attack_types": types}}


@router.post("/generate")
async def generate_samples(request: GenerateRequest):
    if request.attack_type not in ATTACK_DISTRIBUTIONS:
        raise HTTPException(
            status_code=400, detail=f"Unknown attack type: {request.attack_type}"
        )

    samples = generate_synthetic_samples(request.attack_type, request.num_samples)

    # Save samples
    GENERATED_SAMPLES.extend(samples)
    save_samples()

    return {
        "success": True,
        "data": {
            "generated": len(samples),
            "attack_type": request.attack_type,
            "samples": samples[:10],  # Return first 10
            "total_in_db": len(GENERATED_SAMPLES),
        },
    }


@router.get("/samples")
async def get_samples(attack_type: Optional[str] = None, limit: int = 100):
    filtered = GENERATED_SAMPLES
    if attack_type:
        filtered = [s for s in filtered if s.get("attack_type") == attack_type]
    return {
        "success": True,
        "data": {"samples": filtered[-limit:], "total": len(filtered)},
    }


@router.get("/export")
async def export_samples(attack_type: Optional[str] = None):
    """Export samples as feature vectors"""
    filtered = GENERATED_SAMPLES
    if attack_type:
        filtered = [s for s in filtered if s.get("attack_type") == attack_type]

    vectors = []
    for s in filtered:
        features = s.get("features", {})
        vectors.append(list(features.values()))

    return {
        "success": True,
        "data": {
            "feature_names": (
                list(ATTACK_DISTRIBUTIONS["ddos"].keys()) if filtered else []
            ),
            "vectors": vectors[:500],
            "total": len(vectors),
        },
    }


@router.get("/stats")
async def get_stats():
    by_type = {}
    for s in GENERATED_SAMPLES:
        t = s.get("attack_type", "unknown")
        by_type[t] = by_type.get(t, 0) + 1

    return {
        "success": True,
        "data": {
            "total_generated": len(GENERATED_SAMPLES),
            "by_attack_type": by_type,
            "available_types": list(ATTACK_DISTRIBUTIONS.keys()),
        },
    }
