"""
FastAPI Backend - CyberGuard AI
Modern REST API for cyber security platform

Dosya Yolu: app/main.py
"""

import os
import time

# .env dosyasını yükle (ÖNEMLİ: import'lardan önce)
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
env_path = os.path.join(project_root, ".env")
load_dotenv(env_path, override=True)
print(f"[OK] Environment variables loaded from {env_path}")

# Yapılandırılmış loglama — diğer import'lardan önce başlat
from app.utils.logging import RequestIDMiddleware, setup_logging

setup_logging(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    log_file="cyberguard.log",
    json_console=os.environ.get("JSON_CONSOLE_LOG", "").lower() in ("1", "true"),
)

# API routes - Core
# API routes - Standalone
from app.api.routes import auth
from app.api.routes import websocket as ws_routes
from app.api.routes.core import (
    api_keys,
    dashboard,
    database,
    log_analyzer,
    logs,
    notifications,
    pdf_reports,
    reports,
    settings,
)

# API routes - ML
from app.api.routes.ml import (
    advanced_ml,
    advanced_models,
    ai_decision,
    automl,
    comparison,
    drift_detection,
    models,
    prediction,
    training,
    xai,
)

# API routes - Monitoring
from app.api.routes.monitoring import (
    alerts,
    anomaly,
    incidents,
    network,
    realtime,
    security_advanced,
    siem,
)

# API routes - Security
from app.api.routes.security import (
    adversarial,
    container_security,
    gan_synthesis,
    hsm,
    sandbox,
    scanner,
    vulnerability,
    zeroday,
)

# API routes - Threat
from app.api.routes.threat import (
    attack_map,
    attack_surface,
    attack_training,
    attacks,
    darkweb,
    deception,
    threat_analysis,
    threat_hunting,
    threat_intel,
)

# API routes - Tools
from app.api.routes.tools import blockchain_audit, chat, federated, playbooks, stix_taxii

# Rate Limiting
try:
    from slowapi import Limiter, _rate_limit_exceeded_handler
    from slowapi.errors import RateLimitExceeded
    from slowapi.util import get_remote_address

    limiter = Limiter(key_func=get_remote_address)
    HAS_RATE_LIMIT = True
except ImportError:
    HAS_RATE_LIMIT = False
    print("[Warning] slowapi not installed, rate limiting disabled")

# Prometheus metrics
try:
    from prometheus_fastapi_instrumentator import Instrumentator
    HAS_PROMETHEUS = True
except ImportError:
    HAS_PROMETHEUS = False
    print("[Warning] prometheus-fastapi-instrumentator not installed, metrics disabled")

# FastAPI app
app = FastAPI(
    title="CyberGuard AI API",
    description="Siber güvenlik platformu için REST API",
    version="3.3.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
)

# Rate Limiter ekle
if HAS_RATE_LIMIT:
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# CORS - Frontend erişimi için
# CORS - Frontend erişimi için
_cors_origins = os.environ.get("CORS_ORIGINS", "http://localhost:5173,http://localhost:3000").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type", "Accept", "X-Requested-With"],
)


# Request timing middleware
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(round(process_time * 1000, 2)) + "ms"
    return response

# Request-ID middleware (yapılandırılmış loglama)
app.add_middleware(RequestIDMiddleware)

# Prometheus /metrics endpoint'i aç
if HAS_PROMETHEUS:
    Instrumentator().instrument(app).expose(app, endpoint="/metrics")


# Routers - Core
app.include_router(dashboard.router, prefix="/api/dashboard", tags=["Dashboard"])
app.include_router(attacks.router, prefix="/api/attacks", tags=["Attacks"])
app.include_router(models.router, prefix="/api/models", tags=["Models"])
app.include_router(chat.router, prefix="/api/chat", tags=["Chat"])
app.include_router(training.router, prefix="/api/training", tags=["Training"])
app.include_router(ws_routes.router, tags=["WebSocket"])

# Routers - New
app.include_router(logs.router, prefix="/api/logs", tags=["Logs"])
app.include_router(scanner.router, prefix="/api/scanner", tags=["Scanner"])
app.include_router(network.router, prefix="/api/network", tags=["Network"])
app.include_router(prediction.router, prefix="/api/prediction", tags=["Prediction"])
app.include_router(auth.router, prefix="/api/auth", tags=["Auth"])
app.include_router(database.router, prefix="/api/database", tags=["Database"])
app.include_router(settings.router, prefix="/api/settings", tags=["Settings"])
app.include_router(reports.router, prefix="/api/reports", tags=["Reports"])
app.include_router(
    threat_analysis.router, prefix="/api/threat-analysis", tags=["Threat Analysis"]
)
app.include_router(
    advanced_models.router, prefix="/api/advanced", tags=["Advanced Models"]
)
app.include_router(attack_training.router, prefix="/api", tags=["Attack Training"])
app.include_router(advanced_ml.router, prefix="/api", tags=["Advanced ML"])
app.include_router(ai_decision.router, tags=["AI Decision"])

# New routers - Faz 2-6
app.include_router(xai.router, prefix="/api/xai", tags=["Explainable AI"])
app.include_router(
    adversarial.router, prefix="/api/adversarial", tags=["Adversarial Testing"]
)
app.include_router(
    federated.router, prefix="/api/federated", tags=["Federated Learning"]
)
app.include_router(automl.router, prefix="/api/automl", tags=["AutoML"])

# New routers - Additional Features
app.include_router(
    threat_intel.router, prefix="/api/threat-intel", tags=["Threat Intelligence"]
)
app.include_router(alerts.router, prefix="/api/alerts", tags=["Alerts"])
app.include_router(pdf_reports.router, prefix="/api/pdf-reports", tags=["PDF Reports"])
app.include_router(
    comparison.router, prefix="/api/comparison", tags=["Model Comparison"]
)
app.include_router(anomaly.router, prefix="/api/anomaly", tags=["Anomaly Detection"])
app.include_router(
    security_advanced.router, prefix="/api/security", tags=["Security Advanced"]
)

# New routers - Latest Features
app.include_router(
    vulnerability.router, prefix="/api/vuln", tags=["Vulnerability Scanner"]
)
app.include_router(
    log_analyzer.router, prefix="/api/logs-analysis", tags=["Log Analyzer"]
)
app.include_router(incidents.router, prefix="/api/incidents", tags=["Incidents"])
app.include_router(api_keys.router, prefix="/api/keys", tags=["API Keys"])

# New routers - Mega Update (25 New Features)
app.include_router(
    realtime.router, prefix="/api/realtime", tags=["Real-Time Dashboard"]
)
app.include_router(darkweb.router, prefix="/api/darkweb", tags=["Dark Web Monitoring"])
app.include_router(
    container_security.router, prefix="/api/container", tags=["Container Security"]
)
app.include_router(zeroday.router, prefix="/api/zeroday", tags=["Zero-Day Detection"])
app.include_router(
    threat_hunting.router, prefix="/api/threat-hunting", tags=["Threat Hunting"]
)
app.include_router(playbooks.router, prefix="/api/playbooks", tags=["IR Playbooks"])
app.include_router(
    drift_detection.router, prefix="/api/drift", tags=["Drift Detection"]
)
app.include_router(
    notifications.router, prefix="/api/notifications", tags=["Notification Center"]
)
app.include_router(attack_map.router, prefix="/api/attack-map", tags=["Attack Map"])
app.include_router(siem.router, prefix="/api/siem", tags=["SIEM Integration"])
app.include_router(sandbox.router, prefix="/api/sandbox", tags=["Malware Sandbox"])
app.include_router(
    attack_surface.router, prefix="/api/attack-surface", tags=["Attack Surface"]
)
app.include_router(
    blockchain_audit.router, prefix="/api/blockchain", tags=["Blockchain Audit"]
)
app.include_router(stix_taxii.router, prefix="/api/stix-taxii", tags=["STIX/TAXII"])
app.include_router(deception.router, prefix="/api/deception", tags=["Deception Tech"])
app.include_router(gan_synthesis.router, prefix="/api/gan", tags=["GAN Synthesis"])
app.include_router(hsm.router, prefix="/api/hsm", tags=["HSM"])


@app.get("/")
async def root():
    return {
        "message": "🛡️ CyberGuard AI API",
        "version": "3.3.0",
        "docs": "/api/docs",
        "endpoints": {
            "dashboard": "/api/dashboard",
            "attacks": "/api/attacks",
            "models": "/api/models",
            "training": "/api/training",
            "chat": "/api/chat",
            "logs": "/api/logs",
            "scanner": "/api/scanner",
            "network": "/api/network",
            "websocket": "/ws",
        },
    }


@app.get("/api/health")
async def health_check():
    return {
        "status": "healthy",
        "service": "CyberGuard AI API",
        "version": "3.3.0",
        "rate_limiting": HAS_RATE_LIMIT,
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
