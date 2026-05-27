# 📊 Monitoring Guide

CyberGuard AI sistem izleme ve alerting rehberi

> Bu rehber, projeye entegre edilmiş gerçek monitoring stack'i açıklar.
> Dosyalar: `docker-compose.monitoring.yml`, `monitoring/prometheus.yml`, `app/utils/logging.py`

---

## 📋 İçindekiler

- [Prometheus ve Grafana Stack](#prometheus-ve-grafana-stack)
- [FastAPI Metrics Endpoint](#fastapi-metrics-endpoint)
- [Yapılandırılmış Loglama](#yapılandırılmış-loglama)
- [Alerting](#alerting)
- [Health Checks](#health-checks)
- [Dashboard Metrikleri](#dashboard-metrikleri)

---

## 📈 Prometheus ve Grafana Stack

### Başlatma

Proje kökünde tek komutla tüm monitoring stack'ini başlatın:

```bash
docker compose -f docker-compose.monitoring.yml up -d
```

Bu komut 3 servisi ayağa kaldırır:

| Servis | Port | Açıklama |
|--------|------|----------|
| Prometheus | 9090 | Metrik toplayıcı |
| Grafana | 3001 | Görselleştirme (admin / cyberguard2026) |
| Node Exporter | — | Sistem metrikleri (CPU, RAM, Disk) |

### Prometheus Yapılandırması

`monitoring/prometheus.yml` dosyası proje içindedir:

```yaml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  # CyberGuard AI Backend
  - job_name: "cyberguard-api"
    static_configs:
      - targets: ["host.docker.internal:8000"]
    metrics_path: /metrics

  # Sistem metrikleri
  - job_name: "node"
    static_configs:
      - targets: ["node-exporter:9100"]
```

### Grafana Otomatik Provisioning

Grafana açıldığında `monitoring/grafana/dashboards/cyberguard_api.json` otomatik yüklenir.

**İçerik:**
- Request Rate (req/s)
- Request Duration p95 (ms)
- Error Rate (4xx + 5xx)
- Active Requests
- CPU Usage (%)
- Memory Usage
- Top 5 Slowest Endpoints

---

## ⚡ FastAPI Metrics Endpoint

`prometheus-fastapi-instrumentator` paketi, tüm HTTP metriklerini otomatik toplar.

```bash
# Kurulum (requirements.txt'de mevcut)
pip install prometheus-fastapi-instrumentator>=7.0.0
```

`main.py`'de tek satır ile aktif edilir:

```python
from prometheus_fastapi_instrumentator import Instrumentator
Instrumentator().instrument(app).expose(app, endpoint="/metrics")
```

**Erişim:** `http://localhost:8000/metrics`

**Örnek çıktı:**
```
# HELP http_requests_total Total HTTP requests
# TYPE http_requests_total counter
http_requests_total{handler="/api/dashboard/stats",method="GET",status="2xx"} 42.0

# HELP http_request_duration_seconds HTTP request duration
# TYPE http_request_duration_seconds histogram
http_request_duration_seconds_bucket{handler="/api/ml/predict",le="0.1"} 38.0
```

---

## 📝 Yapılandırılmış Loglama

`app/utils/logging.py` modülü JSON tabanlı loglama sağlar.

### Özellikler

| Özellik | Açıklama |
|---------|----------|
| **JSON Formatter** | Her log satırı JSON formatında |
| **Rotating File** | `logs/app/cyberguard.log` (10 MB × 5 yedek) |
| **Request-ID** | Her HTTP isteğine `X-Request-ID` başlığı |
| **Ortam Değişkenleri** | `LOG_LEVEL`, `JSON_CONSOLE_LOG` |

### Kullanım

```python
from app.utils.logging import get_logger

logger = get_logger(__name__)
logger.info("Saldırı tespit edildi", extra={"ip": "1.2.3.4", "type": "DDoS"})
```

### Örnek JSON Log Çıktısı

```json
{
  "timestamp": "2026-04-24T10:30:00.000Z",
  "level": "INFO",
  "logger": "app.api.routes.monitoring.incidents",
  "message": "Incident oluşturuldu: abc-123",
  "module": "incidents",
  "function": "handle_threat",
  "line": 87,
  "ip": "1.2.3.4",
  "type": "DDoS"
}
```

### Ortam Değişkenleri

```env
LOG_LEVEL=INFO              # DEBUG | INFO | WARNING | ERROR
JSON_CONSOLE_LOG=false      # true → konsola da JSON yaz
```

### Log Klasörü

```
logs/
├── app/
│   ├── cyberguard.log      ← aktif log
│   ├── cyberguard.log.1    ← rotate edilmiş yedekler
│   └── cyberguard.log.2
├── training/
└── tensorboard/
```

---

## 🔔 Alerting

### Prometheus Alert Kuralları

`monitoring/alerts.yml` (manuel olarak oluşturup `prometheus.yml`'e ekleyebilirsiniz):

```yaml
groups:
  - name: cyberguard
    rules:
    - alert: HighErrorRate
      expr: rate(http_requests_total{status=~"5.."}[5m]) > 0.1
      for: 5m
      labels:
        severity: critical
      annotations:
        summary: "Yüksek hata oranı tespit edildi"

    - alert: SlowEndpoint
      expr: histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m])) > 1
      for: 10m
      labels:
        severity: warning
      annotations:
        summary: "API yanıt süresi yavaş (p95 > 1s)"

    - alert: HighMemoryUsage
      expr: (node_memory_MemTotal_bytes - node_memory_MemAvailable_bytes) / node_memory_MemTotal_bytes > 0.85
      for: 5m
      labels:
        severity: warning
      annotations:
        summary: "Bellek kullanımı %85 üzerinde"
```

---

## 🏥 Health Checks

### Endpoint'ler

```bash
GET /health        → {"status": "healthy"}
GET /health/ready  → {"status": "ready", "checks": {...}}
GET /metrics       → Prometheus metrikleri (text/plain)
```

### Docker Health Check

`frontend/Dockerfile`'da kullanılabilir:

```dockerfile
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1
```

---

## 📊 Dashboard Metrikleri

### Temel Metrikler

| Metrik | Açıklama | Alert Eşiği |
|--------|----------|-------------|
| Request Rate | req/s | > 1000 |
| Error Rate | % | > 1% |
| Latency P95 | ms | > 500ms |
| CPU Usage | % | > 80% |
| Memory Usage | GB | > 4 GB |
| Cache Hit Rate | % | < 50% (uyarı) |

### Grafana Panel'leri (Otomatik Yüklenen)

1. **Request Overview** — toplam istek, endpoint'e göre, hata oranı
2. **Performance** — P95 yanıt süresi histogram
3. **System** — CPU, Memory (Node Exporter)
4. **Top 5 Slowest Endpoints** — en yavaş 5 endpoint tablosu


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
