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

| Yazılım | Min Versiyon | Önerilen | İndirme |
|---------|--------------|---------|----------|
| **Python** | 3.10+ | 3.12 | [python.org](https://python.org) |
| **Node.js** | 18+ | 22 (CI/CD ile aynı) | [nodejs.org](https://nodejs.org) |
| **Git** | 2.30+ | En son | [git-scm.com](https://git-scm.com) |
| **Docker** | 24+ (opsiyonel) | En son | [docker.com](https://docker.com) |

---

## ⚡ Hızlı Kurulum

### Windows (PowerShell)

```powershell
# 1. Repository'yi klonla
git clone https://github.com/salihoglueyup/CyberGuard_AI.git
cd CyberGuard_AI

# 2. Virtual environment oluştur ve aktive et
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 3. Bağımlılıkları yükle
pip install -r requirements.txt

# 4. .env dosyası oluştur
Copy-Item .env.example .env
# Not: .env.example yoksa aşağıdaki Konfigürasyon bölümüne bakın

# 5. Backend'i başlat (proje kökünden)
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### Linux/macOS (Bash)

```bash
# 1. Repository'yi klonla
git clone https://github.com/salihoglueyup/CyberGuard_AI.git
cd CyberGuard_AI

# 2. Virtual environment oluştur ve aktive et
python -m venv .venv
source .venv/bin/activate

# 3. Bağımlılıkları yükle
pip install -r requirements.txt

# 4. Backend'i başlat (proje kökünden)
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
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
python -m venv .venv

# Aktive et
# Windows (PowerShell):
.\.venv\Scripts\Activate.ps1
# Windows (CMD):
.\.venv\Scripts\activate.bat
# Linux/macOS:
source .venv/bin/activate
```

### Adım 3: Python Bağımlılıkları

```bash
# Temel bağımlılıklar
pip install --upgrade pip
pip install -r requirements.txt

# Geliştirici araçları (test, lint, pre-commit)
pip install -r requirements-dev.txt

# Pre-commit hookları kur (opsiyonel)
pre-commit install

# GPU desteği için (opsiyonel)
pip install tensorflow[and-cuda]
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
# ─────────────────────────────────────────────────────────────
# Temel Ayarlar
# ─────────────────────────────────────────────────────────────
# Admin şifresi (ZORUNLU — boş bırakılamaz)
ADMIN_DEFAULT_PASSWORD=your_secure_password_here

# CORS (virgülle ayrılmış izin verilen origin'ler)
CORS_ORIGINS=http://localhost:5173,http://localhost:3000

# ─────────────────────────────────────────────────────────────
# Loglama
# ─────────────────────────────────────────────────────────────
# Seviye: DEBUG | INFO | WARNING | ERROR (varsayılan: INFO)
LOG_LEVEL=INFO
# JSON konsol logu: true → JSON formatında, false → okunabilir
JSON_CONSOLE_LOG=false

# ─────────────────────────────────────────────────────────────
# LLM Entegrasyonu (opsiyonel)
# ─────────────────────────────────────────────────────────────
# LLM Tehdit Agent sağlayıcısı: groq | openai | ollama
LLM_PROVIDER=groq
LLM_API_KEY=your_groq_api_key
LLM_MODEL=llama3-8b-8192

# AI Asistan ek anahtarları (opsiyonel)
GROQ_API_KEY=your_groq_api_key
OPENAI_API_KEY=your_openai_api_key
ANTHROPIC_API_KEY=your_claude_api_key
GOOGLE_API_KEY=your_gemini_api_key

# Yerel Ollama URL (LLM_PROVIDER=ollama ise)
OLLAMA_URL=http://localhost:11434

# ─────────────────────────────────────────────────────────────
# Veritabanı (otomatik — ek kurulum gerekmez)
# DB dosyası: src/database/cyberguard.db (SQLite)
# ─────────────────────────────────────────────────────────────
```

### Adım 6: Veritabanı (Otomatik)

CyberGuard AI, **SQLite** kullanır. Ek kurulum gerekmez; ilk çalıştırmada `src/database/cyberguard.db` otomatik oluşturulur.

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

Projedeki Docker kurulumu `frontend/` klasöründe tanımlanmıştır ve React frontend + Nginx içerir.

```bash
# frontend/ klasörüne git
cd frontend

# Docker Compose ile başlat
docker-compose up -d

# Logları görüntüle
docker-compose logs -f
```

Frontend `http://localhost:80` adresinde, Backend'i `http://localhost:8000` adresinde ayrı başlatmanız gerekir.

Daha fazla bilgi için: [deployment.md](../operations/deployment.md)

### Tek Container (Yalnızca Frontend)

```bash
cd frontend
docker build -t cyberguard-frontend .
docker run -p 80:80 cyberguard-frontend
```

---

## ⚙️ Konfigürasyon

### config/config.yaml

```yaml
# config.yaml — proje kökünde
# ML model parametreleri ve genel ayarlar burada bulunur
# API anahtarları ve şifreler .env dosyasında tanımlanmalıdır

model:
  default: "ssa_lstmids_cicids2017"
  artifacts_path: "model_artifacts/"
  batch_size: 64
  threshold: 0.5

logging:
  level: "INFO"  # DEBUG, INFO, WARNING, ERROR
  dir: "logs/"

api:
  host: "0.0.0.0"
  port: 8000
  # CORS origins .env'den okunur: CORS_ORIGINS=...
```

---

## ✅ Doğrulama

### Backend Test

```bash
# Backend'i başlat (proje kökünden)
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

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

1. **Başlangıç Kılavuzu**: [User Guide](../reference/user_guide.md)
2. **API Dokümantasyonu**: [API Reference](../api/api_reference.md)
3. **Model Eğitimi**: [ML Models](../ml/ml_models.md)
4. **Deployment**: [Deployment Guide](../operations/deployment.md)

---

## 📞 Destek

Sorunlarınız için:

- 📖 [Documentation](https://docs.cyberguard-ai.com)
- 🐛 [GitHub Issues](https://github.com/salihoglueyup/CyberGuard_AI/issues)
- 💬 [Discord](https://discord.gg/cyberguard)
