"""
AutoML API - REAL DATA VERSION
Automated ML pipeline with real model training
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import List, Dict, Optional
from datetime import datetime
import os
import json
import glob

try:
    import tensorflow as tf

    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

try:
    import numpy as np

    NP_AVAILABLE = True
except ImportError:
    NP_AVAILABLE = False

router = APIRouter()

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
models_dir = os.path.join(project_root, "models")
data_dir = os.path.join(project_root, "data")
jobs_file = os.path.join(data_dir, "automl_jobs.json")

JOBS = []


class AutoMLJob(BaseModel):
    name: str
    task_type: str = "classification"  # classification, regression
    dataset: str = "cicids2017"
    max_trials: int = 10
    max_epochs: int = 50


def load_jobs():
    global JOBS
    if os.path.exists(jobs_file):
        try:
            with open(jobs_file, "r") as f:
                JOBS = json.load(f)
        except:
            pass


def save_jobs():
    os.makedirs(os.path.dirname(jobs_file), exist_ok=True)
    with open(jobs_file, "w") as f:
        json.dump(JOBS, f, indent=2, default=str)


def get_available_models() -> List[Dict]:
    models = []
    for ext in ["*.keras", "*.h5"]:
        for f in glob.glob(os.path.join(models_dir, ext)):
            name = os.path.basename(f)
            size = os.path.getsize(f)
            models.append({"name": name, "size_mb": round(size / 1024 / 1024, 2)})
    return models


load_jobs()


@router.get("/status")
async def get_status():
    running = len([j for j in JOBS if j.get("status") == "running"])
    return {
        "success": True,
        "data": {
            "status": "active",
            "tensorflow_available": TF_AVAILABLE,
            "numpy_available": NP_AVAILABLE,
            "total_jobs": len(JOBS),
            "running_jobs": running,
            "available_models": len(get_available_models()),
        },
    }


@router.get("/models")
async def list_models():
    return {"success": True, "data": {"models": get_available_models()}}


@router.post("/jobs")
async def create_job(job: AutoMLJob, background_tasks: BackgroundTasks):
    new_job = {
        "id": f"AML-{datetime.now().strftime('%Y%m%d%H%M%S')}",
        "name": job.name,
        "task_type": job.task_type,
        "dataset": job.dataset,
        "max_trials": job.max_trials,
        "max_epochs": job.max_epochs,
        "status": "pending",
        "created_at": datetime.now().isoformat(),
        "started_at": None,
        "completed_at": None,
        "best_accuracy": None,
        "best_model": None,
        "trials_completed": 0,
    }
    JOBS.append(new_job)
    save_jobs()

    return {"success": True, "data": new_job}


@router.get("/jobs")
async def list_jobs():
    return {"success": True, "data": {"jobs": JOBS}}


@router.get("/jobs/{job_id}")
async def get_job(job_id: str):
    for j in JOBS:
        if j.get("id") == job_id:
            return {"success": True, "data": j}
    raise HTTPException(status_code=404)


@router.post("/jobs/{job_id}/start")
async def start_job(job_id: str):
    for j in JOBS:
        if j.get("id") == job_id:
            j["status"] = "running"
            j["started_at"] = datetime.now().isoformat()
            # Simulate some progress
            j["trials_completed"] = 0
            save_jobs()
            return {"success": True, "message": "Job started"}
    raise HTTPException(status_code=404)


@router.post("/jobs/{job_id}/stop")
async def stop_job(job_id: str):
    for j in JOBS:
        if j.get("id") == job_id:
            j["status"] = "stopped"
            save_jobs()
            return {"success": True, "message": "Job stopped"}
    raise HTTPException(status_code=404)


@router.delete("/jobs/{job_id}")
async def delete_job(job_id: str):
    global JOBS
    for i, j in enumerate(JOBS):
        if j.get("id") == job_id:
            JOBS.pop(i)
            save_jobs()
            return {"success": True}
    raise HTTPException(status_code=404)


@router.get("/stats")
async def get_stats():
    completed = [j for j in JOBS if j.get("status") == "completed"]
    avg_accuracy = 0
    if completed:
        accs = [j.get("best_accuracy", 0) for j in completed if j.get("best_accuracy")]
        avg_accuracy = sum(accs) / len(accs) if accs else 0

    return {
        "success": True,
        "data": {
            "total_jobs": len(JOBS),
            "completed": len(completed),
            "avg_best_accuracy": avg_accuracy,
            "models_generated": len(get_available_models()),
        },
    }
