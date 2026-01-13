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
# Windows:
.\venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

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

# Alternatif port kullan
uvicorn main:app --port 8001
```

### uvicorn başlatılamıyor

**Sebep**: Import hatası veya syntax error.

**Çözüm:**

```bash
# Syntax kontrol
python -m py_compile app/main.py

# Import kontrol
python -c "from app.main import app"

# Verbose mode
uvicorn main:app --reload --log-level debug
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
tail -f logs/app.log

# Debug mode
DEBUG=true python -m uvicorn main:app --reload

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

**Sebep**: Token eksik veya geçersiz.

**Çözüm:**

```bash
# Token al
curl -X POST http://localhost:8000/api/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username": "admin", "password": "admin123"}'

# Token kullan
curl http://localhost:8000/api/dashboard \
  -H "Authorization: Bearer <token>"
```

### 429 Too Many Requests

**Sebep**: Rate limit aşıldı.

**Çözüm:**

```bash
# Rate limit bilgisi
curl -I http://localhost:8000/api/attacks
# X-RateLimit-Remaining header'ını kontrol et

# Bekle veya limit artır
```

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
# Database indexleri
CREATE INDEX idx_attacks_created ON attacks(created_at);
CREATE INDEX idx_attacks_type ON attacks(attack_type);

# Query optimizasyonu
# N+1 query'lerden kaçın

# Caching
from functools import lru_cache
@lru_cache(maxsize=100)
def get_stats():
    ...
```

### Yüksek CPU kullanımı

**Çözüm:**

```bash
# Process kontrol
htop / top

# Model warmup
python -c "from src.models.predictor import AttackPredictor; p = AttackPredictor(); p.load_models()"

# Worker sayısı
uvicorn main:app --workers 4
```

### Yüksek memory kullanımı

**Çözüm:**

```bash
# Memory profiling
pip install memory_profiler
python -m memory_profiler app/main.py

# Garbage collection
import gc
gc.collect()

# Model unload
del model
keras.backend.clear_session()
```

---

## 📞 Daha Fazla Yardım

Sorununuz çözülmediyse:

1. **GitHub Issues**: github.com/salihoglueyup/CyberGuard_AI/issues
2. **Discord**: discord.gg/cyberguard
3. **Email**: <support@cyberguard-ai.com>

**Log dosyalarını paylaşmayı unutmayın!**
