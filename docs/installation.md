# 🚀 Installation Guide

CyberGuard AI kurulum ve yapılandırma rehberi

---

## 📋 İçindekiler

- [Gereksinimler](#gereksinimler)
- [Hızlı Kurulum](#hızlı-kurulum)
- [Manuel Kurulum](#manuel-kurulum)
- [Docker ile Kurulum](#docker-ile-kurulum)
- [Konfigürasyon](#konfigürasyon)
- [Doğrulama](#doğrulama)
- [Sorun Giderme](#sorun-giderme)

---

## 💻 Gereksinimler

### Sistem Gereksinimleri

| Bileşen | Minimum | Önerilen |
|---------|---------|----------|
| **CPU** | 4 cores | 8+ cores |
| **RAM** | 8 GB | 16+ GB |
| **Disk** | 50 GB SSD | 100+ GB SSD |
| **GPU** | - | NVIDIA (CUDA 11+) |
| **OS** | Windows 10, Ubuntu 20.04, macOS 11 | Ubuntu 22.04 |

### Yazılım Gereksinimleri

| Yazılım | Min Versiyon | İndirme |
|---------|--------------|---------|
| **Python** | 3.9+ | [python.org](https://python.org) |
| **Node.js** | 18+ | [nodejs.org](https://nodejs.org) |
| **Git** | 2.30+ | [git-scm.com](https://git-scm.com) |
| **PostgreSQL** | 14+ | [postgresql.org](https://postgresql.org) |

---

## ⚡ Hızlı Kurulum

### Windows (PowerShell)

```powershell
# 1. Repository'yi klonla
git clone https://github.com/salihoglueyup/CyberGuard_AI.git
cd CyberGuard_AI

# 2. Otomatik kurulum scripti
.\scripts\install.ps1

# 3. Servisleri başlat
.\start-servers.bat
```

### Linux/macOS (Bash)

```bash
# 1. Repository'yi klonla
git clone https://github.com/salihoglueyup/CyberGuard_AI.git
cd CyberGuard_AI

# 2. Otomatik kurulum scripti
chmod +x scripts/install.sh
./scripts/install.sh

# 3. Servisleri başlat
./start-servers.sh
```

---

## 🔧 Manuel Kurulum

### Adım 1: Repository'yi Klonla

```bash
git clone https://github.com/salihoglueyup/CyberGuard_AI.git
cd CyberGuard_AI
```

### Adım 2: Python Virtual Environment

```bash
# Virtual environment oluştur
python -m venv venv

# Aktive et
# Windows:
.\venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate
```

### Adım 3: Python Bağımlılıkları

```bash
# Temel bağımlılıklar
pip install --upgrade pip
pip install -r requirements.txt

# GPU desteği için (opsiyonel)
pip install tensorflow-gpu==2.15.0
```

### Adım 4: Frontend Bağımlılıkları

```bash
cd frontend
npm install
cd ..
```

### Adım 5: Environment Variables

```bash
# .env dosyası oluştur
cp .env.example .env

# Düzenle
nano .env  # veya herhangi bir editor
```

**.env dosyası:**

```env
# API Keys
GOOGLE_API_KEY=your_gemini_api_key_here

# Database
DATABASE_URL=postgresql://user:password@localhost:5432/cyberguard
REDIS_URL=redis://localhost:6379

# Security
SECRET_KEY=your_secret_key_here
JWT_SECRET=your_jwt_secret_here

# Server
HOST=localhost
PORT=8000
FRONTEND_PORT=5173
DEBUG=True
```

### Adım 6: Veritabanı Kurulumu

```bash
# PostgreSQL'e bağlan
psql -U postgres

# Database oluştur
CREATE DATABASE cyberguard;
\q

# Migration çalıştır
python -m alembic upgrade head
```

### Adım 7: Model İndirme (Opsiyonel)

```bash
# Pre-trained modelleri indir
python scripts/download_models.py

# veya manuel
gdown https://drive.google.com/uc?id=YOUR_MODEL_ID -O models/production/
```

---

## 🐳 Docker ile Kurulum

### Docker Compose (Önerilen)

```bash
# Docker Compose ile başlat
docker-compose up -d

# Logları görüntüle
docker-compose logs -f
```

### docker-compose.yml

```yaml
version: '3.8'

services:
  backend:
    build: ./app
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=postgresql://postgres:password@db:5432/cyberguard
      - REDIS_URL=redis://redis:6379
    depends_on:
      - db
      - redis
    volumes:
      - ./models:/app/models
      - ./data:/app/data

  frontend:
    build: ./frontend
    ports:
      - "5173:5173"
    depends_on:
      - backend

  db:
    image: postgres:14
    environment:
      - POSTGRES_DB=cyberguard
      - POSTGRES_USER=postgres
      - POSTGRES_PASSWORD=password
    volumes:
      - pgdata:/var/lib/postgresql/data

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"

volumes:
  pgdata:
```

### Tek Container

```bash
# Backend
docker build -t cyberguard-backend ./app
docker run -p 8000:8000 cyberguard-backend

# Frontend
docker build -t cyberguard-frontend ./frontend
docker run -p 5173:5173 cyberguard-frontend
```

---

## ⚙️ Konfigürasyon

### config/config.yaml

```yaml
# Genel ayarlar
general:
  project_name: "CyberGuard AI"
  version: "2.0.0"
  environment: "development"  # development, staging, production
  debug: true
  language: "tr"
  timezone: "Europe/Istanbul"

# Veritabanı
database:
  type: "postgresql"
  host: "localhost"
  port: 5432
  name: "cyberguard"
  user: "postgres"
  password: "${DB_PASSWORD}"
  pool_size: 20

# Redis
redis:
  host: "localhost"
  port: 6379
  db: 0
  password: null

# ML Modeller
models:
  path: "./models"
  default_model: "best_cicids2017"
  auto_load: true
  gpu_memory_limit: 0.5  # GPU bellek limiti (0-1)

# API
api:
  host: "0.0.0.0"
  port: 8000
  workers: 4
  rate_limit: 100  # requests per minute
  cors_origins:
    - "http://localhost:5173"
    - "http://localhost:3000"

# Gemini AI
gemini:
  api_key: "${GOOGLE_API_KEY}"
  model: "gemini-pro"
  max_tokens: 8192
  temperature: 0.7

# Logging
logging:
  level: "INFO"  # DEBUG, INFO, WARNING, ERROR
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
  file: "./logs/app.log"
  max_size: "100MB"
  backup_count: 5

# Güvenlik
security:
  secret_key: "${SECRET_KEY}"
  jwt_algorithm: "HS256"
  jwt_expiry: 3600  # seconds
  password_min_length: 8
  mfa_enabled: false
  rate_limit_enabled: true
```

---

## ✅ Doğrulama

### Backend Test

```bash
# Backend'i başlat
cd app
python -m uvicorn main:app --reload

# Sağlık kontrolü
curl http://localhost:8000/
# Beklenen: {"message": "🛡️ CyberGuard AI API", "version": "2.0.0", ...}

# API Docs
# Tarayıcıda aç: http://localhost:8000/api/docs
```

### Frontend Test

```bash
# Frontend'i başlat
cd frontend
npm run dev

# Tarayıcıda aç: http://localhost:5173
```

### Model Test

```python
# Python test
from src.models.predictor import AttackPredictor

predictor = AttackPredictor()
predictor.load_models()
print("Models loaded successfully!")
```

### Tam Sistem Testi

```bash
# Test suite çalıştır
pytest tests/ -v

# Coverage raporu
pytest tests/ --cov=app --cov-report=html
```

---

## 🔥 Sorun Giderme

### Yaygın Hatalar

#### 1. ModuleNotFoundError

```bash
# Çözüm: Virtual environment aktif değil
source venv/bin/activate  # Linux
.\venv\Scripts\activate   # Windows
pip install -r requirements.txt
```

#### 2. Port Zaten Kullanımda

```bash
# Port'u kullanan işlemi bul
# Windows:
netstat -ano | findstr :8000
taskkill /PID <PID> /F

# Linux:
lsof -i :8000
kill -9 <PID>
```

#### 3. CUDA/GPU Hatası

```bash
# GPU olmadan çalıştır
CUDA_VISIBLE_DEVICES="" python app/main.py

# veya config'de
TF_FORCE_GPU_ALLOW_GROWTH=true
```

#### 4. Database Bağlantı Hatası

```bash
# PostgreSQL çalışıyor mu?
# Windows:
pg_isready

# Linux:
sudo systemctl status postgresql

# Bağlantı testi
psql -U postgres -h localhost -d cyberguard
```

#### 5. npm install Hatası

```bash
# Cache temizle
npm cache clean --force
rm -rf node_modules package-lock.json
npm install
```

### Log Dosyaları

```
logs/
├── app.log          # Uygulama logları
├── error.log        # Hata logları
├── access.log       # Erişim logları
└── model.log        # Model logları
```

---

## 🚀 Sonraki Adımlar

1. **Başlangıç Kılavuzu**: [User Guide](user_guide.md)
2. **API Dokümantasyonu**: [API Reference](api_reference.md)
3. **Model Eğitimi**: [ML Models](ml_models.md)
4. **Deployment**: [Deployment Guide](deployment.md)

---

## 📞 Destek

Sorunlarınız için:

- 📖 [Documentation](https://docs.cyberguard-ai.com)
- 🐛 [GitHub Issues](https://github.com/salihoglueyup/CyberGuard_AI/issues)
- 💬 [Discord](https://discord.gg/cyberguard)
