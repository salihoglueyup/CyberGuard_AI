# 📊 Monitoring Guide

CyberGuard AI sistem izleme ve alerting rehberi

---

## 📋 İçindekiler

- [Prometheus & Grafana](#prometheus--grafana)
- [Log Yönetimi](#log-yönetimi)
- [Alerting](#alerting)
- [Health Checks](#health-checks)
- [Dashboard](#dashboard)

---

## 📈 Prometheus & Grafana

### Kurulum

```yaml
# docker-compose.monitoring.yml
version: '3.8'

services:
  prometheus:
    image: prom/prometheus
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
    ports:
      - "9090:9090"

  grafana:
    image: grafana/grafana
    ports:
      - "3000:3000"
    volumes:
      - grafana-data:/var/lib/grafana

volumes:
  grafana-data:
```

### Prometheus Config

```yaml
# prometheus.yml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'cyberguard-api'
    static_configs:
      - targets: ['api:8000']

  - job_name: 'node-exporter'
    static_configs:
      - targets: ['node-exporter:9100']
```

### FastAPI Metrics

```python
# app/metrics.py
from prometheus_client import Counter, Histogram, Gauge

REQUEST_COUNT = Counter(
    'http_requests_total',
    'Total HTTP requests',
    ['method', 'endpoint', 'status']
)

REQUEST_LATENCY = Histogram(
    'http_request_duration_seconds',
    'HTTP request latency',
    ['method', 'endpoint']
)

ACTIVE_CONNECTIONS = Gauge(
    'active_connections',
    'Active WebSocket connections'
)

MODEL_INFERENCE_TIME = Histogram(
    'model_inference_seconds',
    'Model inference time'
)
```

---

## 📝 Log Yönetimi

### Log Formatı

```python
# app/logging_config.py
LOGGING_CONFIG = {
    "version": 1,
    "formatters": {
        "default": {
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        },
        "json": {
            "class": "pythonjsonlogger.jsonlogger.JsonFormatter",
            "format": "%(asctime)s %(name)s %(levelname)s %(message)s"
        }
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "default"
        },
        "file": {
            "class": "logging.handlers.RotatingFileHandler",
            "filename": "logs/app.log",
            "maxBytes": 10485760,
            "backupCount": 5,
            "formatter": "json"
        }
    },
    "root": {
        "level": "INFO",
        "handlers": ["console", "file"]
    }
}
```

### ELK Stack

```yaml
# docker-compose.logging.yml
services:
  elasticsearch:
    image: elasticsearch:8.6.0
    environment:
      - discovery.type=single-node
    ports:
      - "9200:9200"

  logstash:
    image: logstash:8.6.0
    volumes:
      - ./logstash.conf:/usr/share/logstash/pipeline/logstash.conf

  kibana:
    image: kibana:8.6.0
    ports:
      - "5601:5601"
```

---

## 🔔 Alerting

### Alert Rules

```yaml
# alerts.yml
groups:
  - name: cyberguard
    rules:
    - alert: HighErrorRate
      expr: rate(http_requests_total{status=~"5.."}[5m]) > 0.1
      for: 5m
      labels:
        severity: critical
      annotations:
        summary: High error rate detected
    
    - alert: SlowResponse
      expr: histogram_quantile(0.95, http_request_duration_seconds_bucket) > 1
      for: 10m
      labels:
        severity: warning
      annotations:
        summary: API response time is slow
    
    - alert: HighMemoryUsage
      expr: process_resident_memory_bytes > 2e9
      for: 5m
      labels:
        severity: warning
```

### Slack Entegrasyonu

```python
import requests

def send_slack_alert(message, severity="warning"):
    webhook_url = os.getenv("SLACK_WEBHOOK_URL")
    color = "#ff0000" if severity == "critical" else "#ffcc00"
    
    payload = {
        "attachments": [{
            "color": color,
            "title": f"CyberGuard Alert ({severity})",
            "text": message
        }]
    }
    
    requests.post(webhook_url, json=payload)
```

---

## 🏥 Health Checks

### Endpoints

```python
# app/api/routes/health.py
from fastapi import APIRouter

router = APIRouter()

@router.get("/health")
async def health():
    return {"status": "healthy"}

@router.get("/health/ready")
async def readiness():
    # Check DB, Redis, Model
    checks = {
        "database": check_db(),
        "redis": check_redis(),
        "model": check_model()
    }
    
    status = "ready" if all(checks.values()) else "not_ready"
    return {"status": status, "checks": checks}

@router.get("/health/live")
async def liveness():
    return {"status": "alive"}
```

### Docker Health Check

```dockerfile
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1
```

---

## 📊 Dashboard Metrikleri

### Temel Metrikler

| Metrik | Açıklama | Alert Eşik |
|--------|----------|------------|
| Request Rate | req/s | > 1000 |
| Error Rate | % | > 1% |
| Latency P95 | ms | > 500ms |
| CPU Usage | % | > 80% |
| Memory Usage | GB | > 4GB |
| DB Connections | count | > 90% pool |

### Grafana Panel'leri

1. **Request Overview**
   - Total requests
   - Requests by endpoint
   - Error rate

2. **Performance**
   - Response time histogram
   - P50, P95, P99 latencies

3. **System**
   - CPU, Memory, Disk
   - Network I/O

4. **ML Model**
   - Inference count
   - Inference latency
   - Prediction distribution
