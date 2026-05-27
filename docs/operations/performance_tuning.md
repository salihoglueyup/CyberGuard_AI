# ⚡ Performance Tuning Guide

CyberGuard AI performans optimizasyonu rehberi

---

## 📋 İçindekiler

- [Genel Bakış](#genel-bakış)
- [Backend Optimizasyonu](#backend-optimizasyonu)
- [Database Optimizasyonu](#database-optimizasyonu)
- [Model Optimizasyonu](#model-optimizasyonu)
- [Frontend Optimizasyonu](#frontend-optimizasyonu)
- [Caching](#caching)
- [Scaling](#scaling)

---

## 🎯 Genel Bakış

### Performans Hedefleri

| Metrik | Hedef | Kritik |
|--------|-------|--------|
| API Response (P95) | < 200ms | < 500ms |
| Model Inference | < 50ms | < 100ms |
| Page Load | < 2s | < 5s |
| Throughput | 1000 req/s | 500 req/s |

---

## 🖥️ Backend Optimizasyonu

### Async Endpoints

```python
# ❌ Yavaş - Senkron
@app.get("/attacks")
def get_attacks():
    return db.query(Attack).all()

# ✅ Hızlı - Asenkron
@app.get("/attacks")
async def get_attacks():
    return await db.execute(select(Attack)).all()
```

### Connection Pooling

```python
# SQLAlchemy pool ayarları
engine = create_engine(
    DATABASE_URL,
    pool_size=20,
    max_overflow=10,
    pool_pre_ping=True,
    pool_recycle=3600
)
```

### Worker Configuration

```bash
# Uvicorn workers
uvicorn main:app --workers 4 --loop uvloop

# Gunicorn
gunicorn main:app -w 4 -k uvicorn.workers.UvicornWorker
```

---

## 🗄️ Database Optimizasyonu

### Indexler

```sql
-- Sık sorgulanan kolonlara index
CREATE INDEX idx_attacks_type ON attacks(attack_type);
CREATE INDEX idx_attacks_created ON attacks(created_at DESC);
CREATE INDEX idx_attacks_severity ON attacks(severity);

-- Composite index
CREATE INDEX idx_attacks_type_created ON attacks(attack_type, created_at);
```

### Query Optimizasyonu

```python
# ❌ N+1 Query
attacks = db.query(Attack).all()
for attack in attacks:
    print(attack.user.name)  # Her seferinde sorgu

# ✅ Eager Loading
attacks = db.query(Attack).options(joinedload(Attack.user)).all()

# ❌ SELECT *
SELECT * FROM attacks

# ✅ Sadece gerekli kolonlar
SELECT id, attack_type, severity FROM attacks
```

### Pagination

```python
# Offset pagination (büyük tablolarda yavaş)
attacks = db.query(Attack).offset(1000).limit(20).all()

# Cursor pagination (daha hızlı)
attacks = db.query(Attack)\
    .filter(Attack.id > last_id)\
    .limit(20).all()
```

---

## 🧠 Model Optimizasyonu

### Model Warmup

```python
# Başlangıçta model'i ısıt
@app.on_event("startup")
async def warmup():
    predictor.load_models()
    # Dummy prediction
    predictor.predict_single([0.0] * 78)
```

### Batch Prediction

```python
# ❌ Tek tek
for sample in samples:
    results.append(model.predict(sample))

# ✅ Batch
results = model.predict(np.array(samples), batch_size=64)
```

### Model Quantization

```python
import tensorflow as tf

# Float16 quantization
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float16]
tflite_model = converter.convert()
```

### GPU Memory

```python
# Memory growth
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

# Hard limit
tf.config.set_logical_device_configuration(
    gpus[0],
    [tf.config.LogicalDeviceConfiguration(memory_limit=4096)]
)
```

---

## 🎨 Frontend Optimizasyonu

### Code Splitting

```javascript
// Lazy loading
const Dashboard = React.lazy(() => import('./pages/Dashboard'));
const Prediction = React.lazy(() => import('./pages/Prediction'));

function App() {
  return (
    <Suspense fallback={<Spinner />}>
      <Routes>
        <Route path="/dashboard" element={<Dashboard />} />
      </Routes>
    </Suspense>
  );
}
```

### Bundle Size

```bash
# Analyze
npm run build -- --analyze

# Vite rollup options
export default defineConfig({
  build: {
    rollupOptions: {
      output: {
        manualChunks: {
          vendor: ['react', 'react-dom'],
          charts: ['recharts']
        }
      }
    }
  }
})
```

### Image Optimization

```jsx
// WebP format + lazy loading
<img 
  src="image.webp" 
  loading="lazy"
  decoding="async"
/>
```

---

## 💾 Caching

### Gerçek Uygulama: `app/utils/cache.py`

CyberGuard, Redis gerektirmeyen thread-safe in-process TTL cache kullanır:

```python
# app/utils/cache.py — temel kullanım
from app.utils.cache import ttl_cache, cache_invalidate, cache_stats

# Senkron veya async fonksiyonlara uygulanabilir
@ttl_cache(ttl=30)   # 30 saniye önbellek
async def get_dashboard_stats():
    ...              # Pahalı hesaplama — sadece 30s'de bir çalışır

@ttl_cache(ttl=60)
async def get_dashboard_summary():
    ...

@ttl_cache(ttl=10)   # Gerçek zamanlı veri için daha kısa TTL
async def get_live_attacks():
    ...
```

**Mevcut TTL değerleri:**

| Endpoint | TTL | Dosya |
|----------|-----|-------|
| `GET /api/dashboard/stats` | 30s | `app/api/routes/core/dashboard.py` |
| `GET /api/dashboard/summary` | 60s | `app/api/routes/core/dashboard.py` |
| `GET /api/attack-map/live` | 10s | `app/api/routes/threat/attack_map.py` |

**Cache yönetimi:**

```python
# Cache istatistikleri (tüm aktif entry'ler)
stats = cache_stats()
# → {"dashboard_stats:<hash>": {"expires_at": ..., "created_at": ...}, ...}

# Belirli prefix'teki entry'leri sil
cache_invalidate("dashboard")   # dashboard ile başlayan tüm key'ler

# Tüm cache'i temizle
from app.utils.cache import cache_clear_all
cache_clear_all()
```

**Neden Redis değil?**  
Proje SQLite + JSON dosya tabanlıdır ve tek process çalışır. Redis kurulumu gerektirmeden aynı process içinde TTL cache yeterlidir.

### LRU Cache

```python
from functools import lru_cache

@lru_cache(maxsize=1000)
def get_attack_type_name(encoded):
    return ATTACK_TYPES.get(encoded, "Unknown")
```

### HTTP Cache Headers

```python
from fastapi import Response

@app.get("/api/static-data")
async def static_data(response: Response):
    response.headers["Cache-Control"] = "public, max-age=3600"
    return {"data": "..."}
```

---

## 📈 Scaling

### Horizontal Scaling

```yaml
# docker-compose.yml
services:
  api:
    image: cyberguard/api
    deploy:
      replicas: 4
    
  nginx:
    image: nginx
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
```

### Load Balancing

```nginx
# nginx.conf
upstream api {
    least_conn;
    server api1:8000;
    server api2:8000;
    server api3:8000;
}

server {
    location /api {
        proxy_pass http://api;
    }
}
```

### Kubernetes HPA

```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: cyberguard-api-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: cyberguard-api
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
```

---

## 📊 Profiling

### Python Profiling

```bash
# cProfile
python -m cProfile -o output.prof app/main.py

# Visualize
pip install snakeviz
snakeviz output.prof
```

### Memory Profiling

```python
from memory_profiler import profile

@profile
def heavy_function():
    data = [i for i in range(10**7)]
    return sum(data)
```

---

## 📡 Prometheus ile Performans İzleme

CyberGuard, `/metrics` endpoint'i üzerinden Prometheus metriklerini otomatik olarak exposes eder. Yavaş endpoint'leri gerçek zamanlı tespit etmek için Grafana kullanın.

### Stack Başlatma

```bash
docker compose -f docker-compose.monitoring.yml up -d
# Prometheus: http://localhost:9090
# Grafana:    http://localhost:3001  (admin / cyberguard2026)
```

### En Yavaş Endpoint'leri Bulmak (PromQL)

```promql
-- En yüksek p95 gecikme — top 5 endpoint
topk(5,
  histogram_quantile(0.95,
    rate(http_request_duration_seconds_bucket[5m])
  )
)
```

### Grafana Dashboard Panelleri

Otomatik sağlanan `cyberguard_api.json` dashboard'unda:

| Panel | Metrik | Kullanım |
|-------|--------|---------|
| Request Rate | `rate(http_requests_total[1m])` | Trafik yükü |
| p95 Latency | `histogram_quantile(0.95, ...)` | Yavaş endpoint tespiti |
| Error Rate | `rate(http_requests_total{status=~"5.."}[1m])` | Hata oranı |
| Active Requests | `http_requests_in_progress` | Eş zamanlı yük |
| CPU Usage | `node_cpu_seconds_total` | CPU tükenmesi |
| Memory | `node_memory_MemAvailable_bytes` | Bellek durumu |
| Slowest Endpoints | Top-5 tablo | Cache TTL ayarı için rehber |

### TTL Değerlerini Optimize Etmek

Grafana'daki "Slowest Endpoints" tablosunu kullanarak hangi endpoint'lere `@ttl_cache` ekleneceğini belirleyin:

```python
# Yavaş + sık çağrılan endpoint → TTL ekle
@ttl_cache(ttl=30)
async def benim_yavaş_endpointim():
    ...
```

> **İpucu:** p95 gecikme > 200ms olan ve `/static` veya gerçek zamanlı olmayan endpointler TTL cache için idealdir.
