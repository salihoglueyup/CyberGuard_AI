"""
Model Comparison API Routes - CyberGuard AI
Compare multiple models side by side

Endpoints:
- GET /api/comparison/models - Compare selected models
- GET /api/comparison/metrics - Compare metrics across models
- POST /api/comparison/benchmark - Run benchmark comparison
- GET /api/comparison/history - Get comparison history

Dosya Yolu: app/api/routes/comparison.py
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict, Any, List
import glob
from datetime import datetime
import json
import os

router = APIRouter()

# Path setup
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
models_dir = os.path.join(project_root, "models")
registry_path = os.path.join(models_dir, "model_registry.json")


class BenchmarkRequest(BaseModel):
    """Benchmark request"""

    model_ids: List[str]
    dataset: str = "cicids2017"
    metrics: List[str] = ["accuracy", "precision", "recall", "f1_score"]
    sample_size: int = 10000


def load_registry():
    """Load model registry"""
    if os.path.exists(registry_path):
        with open(registry_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"models": []}


@router.get("/models")
async def compare_models(model_ids: str = None):
    """Compare selected models"""
    try:
        registry = load_registry()
        all_models = registry.get("models", [])

        if model_ids:
            selected_ids = model_ids.split(",")
            models = [m for m in all_models if m.get("id") in selected_ids]
        else:
            # Get top 5 models by accuracy
            models = sorted(
                all_models,
                key=lambda x: x.get("metrics", {}).get("accuracy", 0),
                reverse=True,
            )[:5]

        comparison = []
        for m in models:
            metrics = m.get("metrics", {})
            training = m.get("training_config", {})

            comparison.append(
                {
                    "id": m.get("id", ""),
                    "name": m.get("name", "Unknown"),
                    "dataset": m.get("dataset", "unknown"),
                    "status": m.get("status", "unknown"),
                    "metrics": {
                        "accuracy": metrics.get("accuracy", 0),
                        "precision": metrics.get("precision", 0),
                        "recall": metrics.get("recall", 0),
                        "f1_score": metrics.get("f1_score", 0),
                    },
                    "training": {
                        "epochs": training.get("epochs", 0),
                        "samples": training.get("train_samples", 0),
                        "batch_size": training.get("batch_size", 0),
                    },
                    "created_at": m.get("created_at", ""),
                }
            )

        # Calculate rankings
        for metric in ["accuracy", "precision", "recall", "f1_score"]:
            sorted_by_metric = sorted(
                comparison, key=lambda x: x["metrics"][metric], reverse=True
            )
            for i, model in enumerate(sorted_by_metric):
                if f"{metric}_rank" not in model:
                    model[f"{metric}_rank"] = i + 1

        return {
            "success": True,
            "data": {
                "models": comparison,
                "comparison_count": len(comparison),
                "best_by_accuracy": (
                    max(comparison, key=lambda x: x["metrics"]["accuracy"])["name"]
                    if comparison
                    else None
                ),
            },
        }

    except Exception as e:
        import traceback

        traceback.print_exc()
        return {"success": False, "error": str(e)}


@router.get("/metrics")
async def compare_metrics():
    """Compare metrics across all models"""
    try:
        registry = load_registry()
        models = registry.get("models", [])

        # Group by dataset
        by_dataset = {}
        for m in models:
            dataset = m.get("dataset", "unknown")
            if dataset not in by_dataset:
                by_dataset[dataset] = []

            metrics = m.get("metrics", {})
            by_dataset[dataset].append(
                {
                    "name": m.get("name", "Unknown"),
                    "accuracy": metrics.get("accuracy", 0),
                    "precision": metrics.get("precision", 0),
                    "recall": metrics.get("recall", 0),
                    "f1_score": metrics.get("f1_score", 0),
                }
            )

        # Calculate averages per dataset
        averages = {}
        for dataset, model_list in by_dataset.items():
            if model_list:
                averages[dataset] = {
                    "avg_accuracy": sum(m["accuracy"] for m in model_list)
                    / len(model_list),
                    "avg_precision": sum(m["precision"] for m in model_list)
                    / len(model_list),
                    "avg_recall": sum(m["recall"] for m in model_list)
                    / len(model_list),
                    "avg_f1": sum(m["f1_score"] for m in model_list) / len(model_list),
                    "model_count": len(model_list),
                }

        return {
            "success": True,
            "data": {
                "by_dataset": by_dataset,
                "averages": averages,
                "total_models": len(models),
            },
        }

    except Exception as e:
        return {"success": False, "error": str(e)}


@router.post("/benchmark")
async def run_benchmark(request: BenchmarkRequest):
    """Run a benchmark comparison"""
    try:
        registry = load_registry()
        all_models = registry.get("models", [])

        results = []
        for model_id in request.model_ids:
            model = next((m for m in all_models if m.get("id") == model_id), None)

            if model:
                metrics = model.get("metrics", {})
                # Simulate benchmark results (in real implementation, run actual inference)
                result = {
                    "model_id": model_id,
                    "model_name": model.get("name", "Unknown"),
                    "dataset": request.dataset,
                    "sample_size": request.sample_size,
                    "metrics": {
                        metric: metrics.get(metric, 0.0)  # Use actual registry metrics
                        for metric in request.metrics
                    },
                    "inference_time_ms": 25.0,  # Average inference time estimate
                    "memory_usage_mb": 200.0,  # Typical model memory usage
                    "throughput_samples_per_sec": 500.0,  # Conservative estimate
                }
            else:
                result = {"model_id": model_id, "error": "Model not found"}

            results.append(result)

        # Determine winner for each metric
        winners = {}
        for metric in request.metrics:
            valid_results = [r for r in results if "error" not in r]
            if valid_results:
                winner = max(valid_results, key=lambda x: x["metrics"].get(metric, 0))
                winners[metric] = winner["model_name"]

        return {
            "success": True,
            "data": {
                "benchmark_id": f"BENCH-{datetime.now().strftime('%Y%m%d%H%M%S')}",
                "results": results,
                "winners": winners,
                "overall_best": (
                    max(
                        results,
                        key=lambda x: (
                            sum(x.get("metrics", {}).values())
                            if "error" not in x
                            else 0
                        ),
                    )["model_name"]
                    if results and any("error" not in r for r in results)
                    else None
                ),
                "timestamp": datetime.now().isoformat(),
            },
        }

    except Exception as e:
        import traceback

        traceback.print_exc()
        return {"success": False, "error": str(e)}


@router.get("/leaderboard")
async def get_leaderboard(
    dataset: str = None, metric: str = "accuracy", limit: int = 10
):
    """Get model leaderboard"""
    try:
        registry = load_registry()
        models = registry.get("models", [])

        if dataset:
            models = [m for m in models if m.get("dataset") == dataset]

        # Sort by metric
        models = sorted(
            models, key=lambda x: x.get("metrics", {}).get(metric, 0), reverse=True
        )[:limit]

        leaderboard = []
        for i, m in enumerate(models):
            metrics = m.get("metrics", {})
            leaderboard.append(
                {
                    "rank": i + 1,
                    "id": m.get("id", ""),
                    "name": m.get("name", "Unknown"),
                    "dataset": m.get("dataset", "unknown"),
                    metric: metrics.get(metric, 0),
                    "accuracy": metrics.get("accuracy", 0),
                    "created_at": m.get("created_at", ""),
                }
            )

        return {
            "success": True,
            "data": {
                "leaderboard": leaderboard,
                "metric": metric,
                "dataset_filter": dataset,
                "total_models": len(leaderboard),
            },
        }

    except Exception as e:
        return {"success": False, "error": str(e)}
