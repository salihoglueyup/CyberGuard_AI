"""
AI Decision API Routes - CyberGuard AI
========================================

AI Decision Layer için REST API endpoints.

Endpoints:
    - /api/ai/decide - Full AI pipeline
    - /api/ai/zero-day/* - VAE zero-day detection
    - /api/ai/model-select/* - Meta-learning selection
    - /api/ai/threshold/* - RL threshold optimization
    - /api/ai/explain/* - XAI explanations
    - /api/ai/report/* - LLM report generation
    - /api/ai/engine/* - Engine management
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any
from datetime import datetime
import numpy as np
import logging
import sys
import os

# Add project root to path
sys.path.insert(
    0,
    os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    ),
)

logger = logging.getLogger("AIDecisionAPI")

router = APIRouter(prefix="/api/ai", tags=["AI Decision"])


# ============= Pydantic Models =============


class TrafficData(BaseModel):
    """Traffic data for prediction"""

    features: List[float] = Field(..., description="Feature vector")
    source_info: Optional[str] = "Unknown"
    target_info: Optional[str] = "Unknown"


class BatchTrafficData(BaseModel):
    """Batch traffic data"""

    samples: List[List[float]] = Field(..., description="List of feature vectors")
    source_info: Optional[str] = "Unknown"
    target_info: Optional[str] = "Unknown"


class TrainVAERequest(BaseModel):
    """VAE training request"""

    epochs: int = Field(default=30, ge=1, le=300)
    sensitivity: int = Field(default=3, ge=1, le=5)


class TrainRLRequest(BaseModel):
    """RL training request"""

    episodes: int = Field(default=100, ge=10, le=1000)
    steps_per_episode: int = Field(default=100, ge=10, le=500)


class ExplanationRequest(BaseModel):
    """Explanation request"""

    features: List[float]
    attack_type: str
    top_n: int = Field(default=5, ge=1, le=20)


class ReportRequest(BaseModel):
    """Report generation request"""

    attack_type: str
    confidence: float = Field(ge=0, le=1)
    explanation: Optional[Dict] = None
    template: str = Field(default="attack_summary")
    source_info: Optional[str] = "Unknown"
    target_info: Optional[str] = "Unknown"


class ThresholdRequest(BaseModel):
    """Threshold decision request"""

    model_confidence: float = Field(ge=0, le=1)
    anomaly_score: float = Field(ge=0, le=1)
    history: Optional[Dict] = None


# ============= Global Engine Instance =============

_ai_engine = None
_engine_lock = False


def get_engine():
    """Get or create AI Decision Engine"""
    global _ai_engine

    if _ai_engine is None:
        try:
            from src.ai_decision.decision_engine import AIDecisionEngine

            _ai_engine = AIDecisionEngine(input_dim=78, sensitivity=3)
            logger.info("✅ AI Decision Engine created")
        except Exception as e:
            logger.error(f"❌ Failed to create AI Engine: {e}")
            return None

    return _ai_engine


def get_zero_day_detector():
    """Get zero-day detector"""
    engine = get_engine()
    return engine.zero_day_detector if engine else None


def get_explainer():
    """Get explainer"""
    try:
        from src.ai_decision.explainer import AttackExplainer

        return AttackExplainer()
    except Exception as e:
        logger.error(f"Failed to get explainer: {e}")
        return None


def get_reporter():
    """Get LLM reporter"""
    try:
        from src.ai_decision.llm_reporter import LLMReporter

        return LLMReporter()
    except Exception as e:
        logger.error(f"Failed to get reporter: {e}")
        return None


def get_rl_agent():
    """Get RL threshold agent"""
    engine = get_engine()
    return engine.rl_agent if engine else None


def get_meta_selector():
    """Get meta model selector"""
    engine = get_engine()
    return engine.meta_selector if engine else None


# ============= Full Pipeline Endpoints =============


@router.post("/decide")
async def make_decision(data: TrafficData):
    """
    Full AI Decision Pipeline

    1. Zero-Day Detection (VAE)
    2. Model Selection (Meta-learning)
    3. Threshold Decision (RL)
    4. Explanation (XAI)
    5. Report Generation (LLM)
    """
    engine = get_engine()
    if engine is None:
        raise HTTPException(status_code=503, detail="AI Engine not available")

    try:
        X = np.array(data.features).reshape(1, -1).astype(np.float32)

        result = engine.decide(
            X,
            generate_report=True,
            source_info=data.source_info,
            target_info=data.target_info,
        )

        # Convert numpy types to Python types
        result["confidence"] = float(result["confidence"])
        result["anomaly_score"] = float(result["anomaly_score"])

        return {
            "success": True,
            "decision": result,
        }
    except Exception as e:
        logger.error(f"Decision error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/decide/batch")
async def make_batch_decision(data: BatchTrafficData):
    """Batch decision for multiple samples"""
    engine = get_engine()
    if engine is None:
        raise HTTPException(status_code=503, detail="AI Engine not available")

    try:
        X = np.array(data.samples).astype(np.float32)
        results = engine.decide_batch(X, generate_reports=False)

        return {
            "success": True,
            "count": len(results),
            "decisions": results,
        }
    except Exception as e:
        logger.error(f"Batch decision error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============= Zero-Day Detection Endpoints =============


@router.post("/zero-day/detect")
async def detect_zero_day(data: TrafficData):
    """Detect zero-day attacks using VAE"""
    detector = get_zero_day_detector()
    if detector is None:
        raise HTTPException(status_code=503, detail="Zero-Day Detector not available")

    if not detector.is_trained:
        return {
            "success": False,
            "error": "Detector not trained. Call /zero-day/train first.",
            "is_zero_day": False,
        }

    try:
        X = np.array(data.features).reshape(1, -1).astype(np.float32)
        result = detector.detect(X)

        return {
            "success": True,
            "is_zero_day": bool(result["is_zero_day"][0]),
            "anomaly_score": float(result["anomaly_scores"][0]),
            "threshold": float(result["threshold"]),
        }
    except Exception as e:
        logger.error(f"Zero-day detection error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/zero-day/train")
async def train_zero_day(request: TrainVAERequest, background_tasks: BackgroundTasks):
    """Train VAE on normal traffic data"""
    detector = get_zero_day_detector()
    if detector is None:
        raise HTTPException(status_code=503, detail="Zero-Day Detector not available")

    # For now, train on dummy data - in production, this would use real normal traffic
    def train_task():
        try:
            X_normal = np.random.randn(500, 78).astype(np.float32)  # Placeholder
            detector.sensitivity = request.sensitivity
            detector.build()
            result = detector.fit(X_normal, epochs=request.epochs, verbose=0)
            logger.info(f"VAE training completed: {result}")
        except Exception as e:
            logger.error(f"VAE training error: {e}")

    background_tasks.add_task(train_task)

    return {
        "success": True,
        "message": "Training started in background",
        "epochs": request.epochs,
        "sensitivity": request.sensitivity,
    }


@router.get("/zero-day/stats")
async def get_zero_day_stats():
    """Get zero-day detector statistics"""
    detector = get_zero_day_detector()
    if detector is None:
        return {"success": False, "error": "Detector not available"}

    return {
        "success": True,
        "stats": {
            "trained": detector.is_trained,
            "threshold": detector.threshold,
            "sensitivity": detector.sensitivity,
            "input_dim": detector.input_dim,
            "latent_dim": detector.latent_dim,
        },
    }


# ============= Model Selection Endpoints =============


@router.post("/model-select")
async def select_model(data: TrafficData):
    """Select best model for traffic using meta-learning"""
    selector = get_meta_selector()
    if selector is None:
        raise HTTPException(status_code=503, detail="Meta Selector not available")

    try:
        X = np.array(data.features).reshape(1, -1).astype(np.float32)
        scores = selector.select_model(X, return_all=True)
        best_model = max(scores, key=scores.get)

        return {
            "success": True,
            "selected_model": best_model,
            "model_scores": scores,
        }
    except Exception as e:
        logger.error(f"Model selection error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/model-select/stats")
async def get_model_select_stats():
    """Get meta selector statistics"""
    selector = get_meta_selector()
    if selector is None:
        return {"success": False, "error": "Selector not available"}

    return {
        "success": True,
        "stats": selector.get_stats(),
    }


# ============= RL Threshold Endpoints =============


@router.post("/threshold/decide")
async def threshold_decision(request: ThresholdRequest):
    """Get RL-based threshold decision"""
    agent = get_rl_agent()
    if agent is None:
        raise HTTPException(status_code=503, detail="RL Agent not available")

    try:
        recommendation = agent.get_threshold_recommendation(
            request.model_confidence,
            request.anomaly_score,
            request.history,
        )

        return {
            "success": True,
            "decision": recommendation,
        }
    except Exception as e:
        logger.error(f"Threshold decision error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/threshold/train")
async def train_rl_agent(request: TrainRLRequest, background_tasks: BackgroundTasks):
    """Train RL threshold agent"""
    agent = get_rl_agent()
    if agent is None:
        raise HTTPException(status_code=503, detail="RL Agent not available")

    def train_task():
        try:
            result = agent.train(
                episodes=request.episodes,
                steps_per_episode=request.steps_per_episode,
            )
            logger.info(f"RL training completed: {result}")
        except Exception as e:
            logger.error(f"RL training error: {e}")

    background_tasks.add_task(train_task)

    return {
        "success": True,
        "message": "RL training started in background",
        "episodes": request.episodes,
    }


@router.get("/threshold/stats")
async def get_threshold_stats():
    """Get RL agent statistics"""
    agent = get_rl_agent()
    if agent is None:
        return {"success": False, "error": "Agent not available"}

    return {
        "success": True,
        "stats": agent.get_stats(),
    }


# ============= Explainability Endpoints =============


@router.post("/explain")
async def explain_attack(request: ExplanationRequest):
    """Get XAI explanation for attack"""
    explainer = get_explainer()
    if explainer is None:
        raise HTTPException(status_code=503, detail="Explainer not available")

    try:
        X = np.array(request.features).reshape(1, -1).astype(np.float32)
        explanation = explainer.explain_attack(X, request.attack_type, request.top_n)

        return {
            "success": True,
            "explanation": explanation,
        }
    except Exception as e:
        logger.error(f"Explanation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/explain/features")
async def get_feature_info():
    """Get feature information"""
    try:
        from src.ai_decision.explainer import CICIDS_FEATURES, ATTACK_FEATURE_PATTERNS

        return {
            "success": True,
            "features": CICIDS_FEATURES,
            "attack_patterns": ATTACK_FEATURE_PATTERNS,
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


# ============= Report Generation Endpoints =============


@router.post("/report/generate")
async def generate_report(request: ReportRequest):
    """Generate LLM-based attack report"""
    reporter = get_reporter()
    if reporter is None:
        raise HTTPException(status_code=503, detail="Reporter not available")

    try:
        if request.attack_type == "ZERO_DAY":
            report = reporter.generate_zero_day_report(
                anomaly_score=request.confidence,
                raw_error=request.confidence * 0.05,
                threshold=0.03,
            )
        else:
            report = reporter.generate_attack_report(
                attack_type=request.attack_type,
                confidence=request.confidence,
                explanation=request.explanation,
                source_info=request.source_info,
                target_info=request.target_info,
                template=request.template,
            )

        return {
            "success": True,
            "report": report,
        }
    except Exception as e:
        logger.error(f"Report generation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/report/history")
async def get_report_history():
    """Get report generation history"""
    reporter = get_reporter()
    if reporter is None:
        return {"success": False, "error": "Reporter not available"}

    return {
        "success": True,
        "history": reporter.get_report_history(limit=20),
        "stats": reporter.get_stats(),
    }


@router.get("/report/templates")
async def get_report_templates():
    """Get available report templates"""
    try:
        from src.ai_decision.llm_reporter import (
            REPORT_TEMPLATES,
            MITRE_MAPPING,
            SEVERITY_MAP,
        )

        return {
            "success": True,
            "templates": list(REPORT_TEMPLATES.keys()),
            "mitre_mapping": MITRE_MAPPING,
            "severity_levels": SEVERITY_MAP,
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


# ============= Engine Management Endpoints =============


@router.get("/engine/stats")
async def get_engine_stats():
    """Get full engine statistics"""
    engine = get_engine()
    if engine is None:
        return {"success": False, "error": "Engine not available"}

    return {
        "success": True,
        "stats": engine.get_stats(),
    }


@router.post("/engine/initialize")
async def initialize_engine(background_tasks: BackgroundTasks):
    """Initialize AI Decision Engine"""
    engine = get_engine()
    if engine is None:
        raise HTTPException(status_code=503, detail="Engine not available")

    def init_task():
        try:
            X_normal = np.random.randn(500, 78).astype(np.float32)
            result = engine.initialize(X_normal, epochs=20)
            logger.info(f"Engine initialization completed: {result}")
        except Exception as e:
            logger.error(f"Engine initialization error: {e}")

    background_tasks.add_task(init_task)

    return {
        "success": True,
        "message": "Engine initialization started in background",
    }


@router.get("/engine/health")
async def engine_health():
    """Health check for AI Engine"""
    engine = get_engine()

    return {
        "status": "healthy" if engine else "unavailable",
        "engine_available": engine is not None,
        "version": engine.VERSION if engine else None,
        "components": (
            {
                "zero_day": engine.zero_day_detector.is_trained if engine else False,
                "meta_selector": engine.meta_selector is not None if engine else False,
                "rl_agent": engine.rl_agent is not None if engine else False,
            }
            if engine
            else {}
        ),
        "timestamp": datetime.now().isoformat(),
    }
