# 🔧 Troubleshooting Guide

CyberGuard AI sorun giderme rehberi

---

## 📋 İçindekiler

- [Kurulum Sorunları](#kurulum-sorunları)
- [Backend Sorunları](#backend-sorunları)
- [Frontend Sorunları](#frontend-sorunları)
- [Database Sorunları](#database-sorunları)
- [Model Sorunları](#model-sorunları)
- [API Sorunları](#api-sorunları)
- [Performans Sorunları](#performans-sorunları)

---

## 🔧 Kurulum Sorunları

### ModuleNotFoundError: No module named 'xxx'

**Sebep**: Bağımlılık eksik veya virtual environment aktif değil.

**Çözüm:**

```bash
# Virtual environment aktif et
# Windows (PowerShell):
.\.venv\Scripts\Activate.ps1
# Linux/macOS:
source .venv/bin/activate

# Bağımlılıkları yükle
pip install -r requirements.txt

# Tek modül
pip install <module_name>
```

### pip install başarısız oluyor

**Sebep**: Network, yetki veya versiyon uyumsuzluğu.

**Çözüm:**

```bash
# pip güncelle
pip install --upgrade pip

# Cache temizle
pip cache purge

# Alternatif mirror
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# Verbose mode
pip install -r requirements.txt -v
```

### npm install başarısız oluyor

**Sebep**: Node versiyonu, network veya cache.

**Çözüm:**

```bash
# Node versiyonu kontrol
node --version  # >= 18.0.0 gerekli

# Cache temizle
npm cache clean --force
rm -rf node_modules package-lock.json
npm install

# Alternative package manager
yarn install   # veya
pnpm install
```

### CUDA/GPU bulunamıyor

**Sebep**: CUDA toolkit kurulu değil veya sürüm uyumsuz.

**Çözüm:**

```bash
# CUDA kontrol
nvidia-smi
nvcc --version

# CPU modunda çalıştır
CUDA_VISIBLE_DEVICES="" python app/main.py

# TensorFlow GPU
pip install tensorflow[and-cuda]
```

---

## 🖥️ Backend Sorunları

### Port zaten kullanımda (Address already in use)

**Sebep**: Başka bir işlem portu kullanıyor.

**Çözüm:**

```bash
# Windows - Port kullanan işlemi bul
netstat -ano | findstr :8000
# PID'yi bul ve sonlandır
taskkill /PID <PID> /F

# Linux/macOS
lsof -i :8000
kill -9 <PID>

# Alternatif port kullan (proje kökünden)
uvicorn app.main:app --port 8001
```

### uvicorn başlatılamıyor

**Sebep**: Import hatası veya syntax error.

**Çözüm:**

```bash
# Syntax kontrol
python -m py_compile app/main.py

# Import kontrol
python -c "from app.main import app"

# Verbose mode (proje kökünden)
uvicorn app.main:app --reload --log-level debug
```

### CORS hatası

**Sebep**: Frontend origin'i backend'de tanımlı değil.

**Çözüm:**

```python
# app/main.py
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

### Internal Server Error (500)

**Sebep**: Backend'de exception.

**Çözüm:**

```bash
# Log kontrol
tail -f logs/app/app.log

# Debug mode (proje kökünden)
DEBUG=true uvicorn app.main:app --reload

# Exception detayı
# Response body'de traceback olacak
```

---

## 🎨 Frontend Sorunları

### Blank page / Nothing renders

**Sebep**: JavaScript error veya build hatası.

**Çözüm:**

```bash
# Console kontrol (F12)
# Build yeniden
npm run build
npm run dev

# Cache temizle
rm -rf .vite dist node_modules/.vite
npm run dev
```

### API calls failing

**Sebep**: Backend çalışmıyor veya URL yanlış.

**Çözüm:**

```javascript
// services/api.js kontrol
const API_URL = 'http://localhost:8000/api';

// Proxy kullan (vite.config.js)
export default defineConfig({
  server: {
    proxy: {
      '/api': 'http://localhost:8000'
    }
  }
})
```

### Slow page load

**Sebep**: Bundle büyük veya network yavaş.

**Çözüm:**

```bash
# Bundle analizi
npm run build -- --analyze

# Lazy loading kullan
const Component = React.lazy(() => import('./Component'));
```

---

## 🗄️ Database Sorunları

### PostgreSQL bağlantı hatası

**Sebep**: Servis çalışmıyor veya credentials yanlış.

**Çözüm:**

```bash
# Servis kontrol
# Windows:
pg_isready
# Linux:
sudo systemctl status postgresql

# Bağlantı test
psql -U postgres -h localhost -d cyberguard

# .env kontrol
DATABASE_URL=postgresql://user:password@localhost:5432/cyberguard
```

### Migration hatası

**Sebep**: Schema mismatch veya migration dosyası eksik.

**Çözüm:**

```bash
# Migration durumu
alembic current
alembic history

# Migration oluştur
alembic revision --autogenerate -m "description"

# Upgrade
alembic upgrade head

# Rollback
alembic downgrade -1
```

### Database full / Disk space

**Sebep**: Log veya eski veri birikimi.

**Çözüm:**

```sql
-- PostgreSQL vacuum
VACUUM FULL;

-- Eski verileri sil
DELETE FROM attacks WHERE created_at < NOW() - INTERVAL '90 days';

-- Table size kontrol
SELECT pg_size_pretty(pg_total_relation_size('attacks'));
```

---

## 🧠 Model Sorunları

### Model yüklenemiyor

**Sebep**: Model dosyası eksik veya corrupt.

**Çözüm:**

```bash
# Model dosyasını kontrol
ls -la models/production/

# Yeniden indir
python scripts/download_models.py

# Manuel yükle
python -c "from tensorflow import keras; keras.models.load_model('models/production/best_model.h5')"
```

### Out of Memory (OOM)

**Sebep**: Model veya batch size çok büyük.

**Çözüm:**

```python
# Batch size küçült
model.predict(X, batch_size=32)

# GPU memory limit
import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)

# Veya sabit limit
tf.config.set_logical_device_configuration(
    gpus[0],
    [tf.config.LogicalDeviceConfiguration(memory_limit=4096)]
)
```

### Yanlış tahminler

**Sebep**: Veri ön işleme uyumsuzluğu veya model drift.

**Çözüm:**

1. Aynı scaler kullanıldığından emin ol
2. Feature sıralamasını kontrol et
3. Model versiyonunu kontrol et
4. Drift detection çalıştır

---

## 🔌 API Sorunları

### 401 Unauthorized

**Sebep**: Token eksik, geçersiz veya süresi dolmuş.

**Çözüm:**

```bash
# 1. Yeni token al (admin şifresini .env'den kontrol edin)
curl -X POST http://localhost:8000/api/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username": "admin", "password": "<ADMIN_DEFAULT_PASSWORD>"}'

# 2. Token ile istek at
curl http://localhost:8000/api/dashboard/stats \
  -H "Authorization: Bearer <token>"

# 3. Token süresi dolmuşsa refresh token kullan
curl -X POST http://localhost:8000/api/auth/refresh \
  -H "Content-Type: application/json" \
  -d '{"refresh_token": "<refresh_token_degeri>"}'
# Yanıt: {"success": true, "data": {"token": "<yeni_token>"}}
```

Token süresi: **24 saat** (access), **7 gün** (refresh).

### 403 Forbidden

**Sebep**: Kullanıcının bu endpoint için gerekli rolü yok.

**Çözüm:**

```bash
# Mevcut rolünüzü kontrol edin
curl http://localhost:8000/api/auth/me \
  -H "Authorization: Bearer <token>"
# {"role": "viewer"} → admin gerektiren endpoint'e erişemezsiniz

# Farklı rol ile giriş yapın veya admin rolü atanmasını isteyin
# POST /api/auth/login ile admin hesabı kullanın
```

**RBAC rolleri:**
| Rol | Yetki |
|-----|-------|
| `admin` | Tüm endpoint'ler, kullanıcı yönetimi, model eğitimi |
| `analyst` | Veri görüntüleme, threat hunting, rapor oluşturma |
| `viewer` | Salt okunur dashboard ve istatistikler |

### 429 Too Many Requests

**Sebep**: Rate limit veya hesap kilidi.

İki ayrı koruma mekanizması vardır:

| Tip | Limit | Süre | Çözüm |
|-----|-------|------|-------|
| IP başına | 5 giriş denemesi | 60 saniye | 60 saniye bekleyin |
| Kullanıcı başına | 10 başarısız giriş | 5 dakika | 5 dakika bekleyin |

```bash
# Rate limit bilgisi
curl -I http://localhost:8000/api/attacks
# X-RateLimit-Remaining header'ını kontrol et

# Bekle veya farklı IP deneyin
```

> **Not**: Hesap kilidi `_failed_logins` sözlüğünde tutulur. Sunucu yeniden başlatılırsa kilid sıfırlanır.

### Timeout

**Sebep**: İşlem çok uzun sürüyor.

**Çözüm:**

```bash
# Timeout artır
curl --max-time 120 http://localhost:8000/api/long-operation

# Background job kullan
POST /api/jobs/start -> {"job_id": "xxx"}
GET /api/jobs/status/xxx -> {"status": "completed"}
```

---

## ⚡ Performans Sorunları

### Yavaş API response

**Çözüm:**

```python
# TTL cache kullan (app/utils/cache.py)
from app.utils.cache import ttl_cache

@router.get("/stats")
@ttl_cache(ttl=30)   # 30 saniye cache
async def get_stats():
    return expensive_db_call()
```

Cache istatistiklerini kontrol etmek için Python'da:

```python
from app.utils.cache import cache_stats
print(cache_stats())
# {"total_entries": 12, "valid_entries": 10, "expired_entries": 2}
```

### Yüksek CPU kullanımı

**Çözüm:**

```bash
# Process kontrol
htop / top

# Prometheus ile en yavaş endpoint'leri bul
# http://localhost:9090/graph
# Sorgu: topk(5, histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m])))

# Worker sayısı
uvicorn app.main:app --workers 4
```

### Yüksek memory kullanımı

**Çözüm:**

```bash
# Memory profiling
pip install memory_profiler
python -m memory_profiler app/main.py

# Cache'i temizle (bellekte biriken entry'ler)
from app.utils.cache import cache_clear_all
cache_clear_all()
```

---

## 📊 Prometheus / Grafana Sorunları

### /metrics endpoint'i boş veya 404 döndürüyor

**Sebep**: `prometheus-fastapi-instrumentator` kurulu değil veya `main.py`'deki import başarısız oldu.

**Çözüm:**

```bash
pip install prometheus-fastapi-instrumentator>=7.0.0

# Doğrulama
python -c "from prometheus_fastapi_instrumentator import Instrumentator; print('OK')"
```

### Grafana'da Prometheus veri kaynağı bağlanamıyor

**Sebep**: Prometheus, `cyberguard-api` job'ında Backend'e erişemiyor.

**Çözüm:**

```yaml
# monitoring/prometheus.yml içindeki target'ı kontrol edin
# Docker içinden host makinesine erişmek için:
- targets: ["host.docker.internal:8000"]   # Windows/macOS
- targets: ["172.17.0.1:8000"]             # Linux (docker0 ağ arayüzü)
```

```bash
# Prometheus log'larını kontrol et
docker compose -f docker-compose.monitoring.yml logs prometheus
```

### Grafana dashboard otomatik yüklenmiyor

**Çözüm:**

```bash
# Provisioning dosyasının doğru konumda olduğunu doğrula
ls monitoring/grafana/provisioning/dashboards/
ls monitoring/grafana/dashboards/cyberguard_api.json

# Grafana container'ını yeniden başlat
docker compose -f docker-compose.monitoring.yml restart grafana
```

---

## 🔐 Auth / Token Sorunları

### "401 Unauthorized" — token geçerli görünüyor ama reddediliyor

**Sebep**: Sunucu yeniden başlatıldı ve eski token bellekte yok.

**Çözüm**: Token'lar artık `data/sessions.json`'a kaydediliyor. Eğer dosya bozuksa:

```bash
# Dosyayı sıfırla
echo "{}" > data/sessions.json
# Ardından tekrar giriş yap
```

### "403 Forbidden" — giriş yapıldı ama endpoint reddediyor

**Sebep**: Kullanıcının rolü bu endpoint için yetersiz (`require_role("admin")` gibi).

**Çözüm:**

```bash
# Kullanıcının rolünü kontrol et
GET /api/auth/me
# {"username": "...", "role": "viewer"}   ← admin yerine viewer

# Admin olarak giriş yap veya kullanıcı rolünü güncelle
```

### Refresh token süresi dolmuş

**Sebep**: Refresh token 7 günlük TTL'yi aştı.

**Çözüm**: Yeniden tam giriş (login) yapın:

```bash
POST /api/auth/login
{"username": "admin", "password": "..."}
# Yeni token + refresh_token alırsınız
```

---

## 🤖 LLM Agent Sorunları

### `/api/incidents/analyze-threat` endpoint'i fallback kullanıyor

**Sebep**: `.env`'de `LLM_PROVIDER` veya `LLM_API_KEY` tanımlı değil.

**Çözüm:**

```env
LLM_PROVIDER=groq
LLM_API_KEY=gsk_xxxx...
LLM_MODEL=llama3-8b-8192
```

```bash
# Test
curl -X POST http://localhost:8000/api/incidents/analyze-threat \
  -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d '{"attack_type": "DDoS", "source_ip": "1.2.3.4", "confidence": 0.9}'
# Yanıtta "source": "llm" ise LLM çalışıyor, "rule-based" ise fallback
```

### LLM JSON parse hatası / geçersiz yanıt

**Sebep**: Model JSON yerine açıklama metni döndürdü.

**Çözüm**: Daha küçük, instruction-following bir model deneyin:

```env
LLM_MODEL=llama3-8b-8192   # Groq'ta iyi sonuç veriyor
# veya
LLM_PROVIDER=ollama
LLM_MODEL=mistral
```

---

## 📝 Loglama Sorunları

### `logs/app/` klasörü oluşmuyor / log dosyası yok

**Sebep**: `setup_logging()` henüz çağrılmadı veya `logs/app/` izin sorunu var.

**Çözüm:**

```bash
# Klasörü manuel oluştur
mkdir -p logs/app

# main.py'de setup_logging çağrısını kontrol et
python -c "from app.utils.logging import setup_logging; setup_logging(); print('OK')"
```

### Konsolda JSON yerine düz metin görünüyor

Bu beklenen davranıştır. JSON konsol logu için:

```env
JSON_CONSOLE_LOG=true
```

---

## 📞 Daha Fazla Yardım

Sorununuz çözülmediyse:

1. **GitHub Issues**: github.com/salihoglueyup/CyberGuard_AI/issues
2. **Discord**: discord.gg/cyberguard
3. **Email**: <support@cyberguard-ai.com>

**Log dosyalarını paylaşmayı unutmayın:** `logs/app/cyberguard.log`

