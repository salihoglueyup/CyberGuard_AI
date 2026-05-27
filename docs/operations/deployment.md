# 🚀 Deployment Guide

CyberGuard AI — Backend (FastAPI) ve Frontend (React + Nginx) Deployment Rehberi

---

## 📋 İçindekiler

- [Genel Bakış](#genel-bakış)
- [Yerel Geliştirme](#yerel-geliştirme)
- [Frontend Docker Deployment](#frontend-docker-deployment)
- [Backend Yapılandırması](#backend-yapılandırması)
- [Nginx Yapılandırması](#nginx-yapılandırması)
- [Ortam Değişkenleri](#ortam-değişkenleri)
- [Production Checklist](#production-checklist)
- [Sorun Giderme](#sorun-giderme)

---

## 🌟 Genel Bakış

CyberGuard AI iki ayrı bileşenden oluşur:

| Bileşen | Teknoloji | Port | Açıklama |
|---------|-----------|------|----------|
| **Backend** | FastAPI + Python | 8000 | REST API, ML modeller, WebSocket |
| **Frontend** | React 19 + Vite + Nginx | 3000 (Docker) / 5173 (dev) | SPA |

Backend ve Frontend **bağımsız** olarak çalışır. Frontend, API isteklerini `http://localhost:8000`'e yönlendirir.

---

## 💻 Yerel Geliştirme

### Backend'i Başlat

```bash
# Proje kökünde (.venv aktif olmalı)
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Backend çalıştıktan sonra:
- API: `http://localhost:8000`
- Swagger UI: `http://localhost:8000/api/docs`
- ReDoc: `http://localhost:8000/api/redoc`

### Frontend'i Başlat (Geliştirme Sunucusu)

```bash
cd frontend
npm install       # ilk çalıştırmada
npm run dev       # Vite dev server — http://localhost:5173
```

### Windows Hızlı Başlangıç

Proje kökündeki `run.bat` dosyasını çalıştırın:

```batch
run.bat
```

---

## 🐳 Frontend Docker Deployment

### Yapı

`frontend/` klasöründe Docker kurulumu bulunur:

```
frontend/
├── Dockerfile          # React build → Nginx
├── docker-compose.yml  # Tek servis: frontend container
├── nginx.conf          # SPA fallback, gzip, güvenlik başlıkları
└── vite.config.js      # Build çıktısı: dist/
```

### docker-compose.yml

```yaml
services:
  frontend:
    build: .
    ports:
      - "3000:80"
    restart: unless-stopped
    environment:
      - NODE_ENV=production
```

Frontend, `http://localhost:3000` adresinde sunulur.

### Çalıştırma

```bash
cd frontend

# Image oluştur ve başlat
docker-compose up -d

# Logları görüntüle
docker-compose logs -f frontend

# Durdur
docker-compose down
```

### Manuel Docker

```bash
cd frontend

# Image oluştur
docker build -t cyberguard-frontend .

# Container başlat
docker run -d -p 3000:80 --name cyberguard-frontend cyberguard-frontend

# Durumu kontrol et
docker ps
```

---

## ⚙️ Backend Yapılandırması

Backend Docker container'ı proje kapsamında mevcut değildir. FastAPI, doğrudan Python ortamında çalıştırılır.

### Üretim Modu

```bash
# Tek worker (geliştirme)
uvicorn app.main:app --host 0.0.0.0 --port 8000

# Çoklu worker (üretim — gunicorn ile)
pip install gunicorn
gunicorn app.main:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

### Systemd Servisi (Linux)

```ini
# /etc/systemd/system/cyberguard-backend.service
[Unit]
Description=CyberGuard AI Backend
After=network.target

[Service]
Type=simple
User=www-data
WorkingDirectory=/opt/CyberGuard_AI_Antigravity
ExecStart=/opt/CyberGuard_AI_Antigravity/.venv/bin/uvicorn app.main:app --host 0.0.0.0 --port 8000
Restart=always
EnvironmentFile=/opt/CyberGuard_AI_Antigravity/.env

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl daemon-reload
sudo systemctl enable cyberguard-backend
sudo systemctl start cyberguard-backend
sudo systemctl status cyberguard-backend
```

---

## 🌐 Nginx Yapılandırması

`frontend/nginx.conf` dosyası Nginx'i yapılandırır:

```nginx
server {
    listen 80;
    server_name _;
    root /usr/share/nginx/html;
    index index.html;

    # Gzip sıkıştırma
    gzip on;
    gzip_types text/plain text/css application/json application/javascript
               text/xml application/xml image/svg+xml;
    gzip_min_length 256;

    # Statik dosya önbelleği (1 yıl)
    location /assets/ {
        expires 1y;
        add_header Cache-Control "public, immutable";
    }

    # React SPA fallback — tüm rotalar index.html'e yönlendirilir
    location / {
        try_files $uri $uri/ /index.html;
    }

    # Güvenlik başlıkları
    add_header X-Frame-Options "SAMEORIGIN" always;
    add_header X-Content-Type-Options "nosniff" always;
    add_header Referrer-Policy "strict-origin-when-cross-origin" always;
}
```

**SPA Fallback:** React Router client-side routing kullandığından, Nginx tüm bilinmeyen yolları `index.html`'e yönlendirmelidir. `try_files $uri $uri/ /index.html;` bu işlevi sağlar.

---

## 🔐 Ortam Değişkenleri

`.env` dosyası proje kökünde oluşturulmalıdır:

```env
# === ZORUNLU ===
ADMIN_DEFAULT_PASSWORD=guclu_bir_sifre_girin

# === CORS ===
# Frontend origin'lerini virgülle ayırın
CORS_ORIGINS=http://localhost:5173,http://localhost:3000

# === LLM API Anahtarları (en az birini girin) ===
GROQ_API_KEY=gsk_...
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
GOOGLE_API_KEY=AIza...

# Ollama (yerel LLM, ücretsiz)
# OLLAMA_BASE_URL=http://localhost:11434
```

**Önemli Notlar:**
- `ADMIN_DEFAULT_PASSWORD` boş bırakılırsa admin girişi devre dışı kalır
- Proje veritabanı olarak **SQLite** kullanır — ek DB kurulumu gerekmez
- `CORS_ORIGINS` frontend URL'ini doğru şekilde içermezse API istekleri reddedilir

---

## ✅ Production Checklist

Canlı ortama geçmeden önce kontrol edin:

### Güvenlik
- [ ] `ADMIN_DEFAULT_PASSWORD` güçlü bir değere ayarlandı
- [ ] `CORS_ORIGINS` yalnızca gerçek frontend domain'ini içeriyor
- [ ] `.env` dosyası `.gitignore`'a eklendi
- [ ] LLM API anahtarları güvenli şekilde saklandı
- [ ] HTTPS için ters proxy (Nginx/Caddy) yapılandırıldı

### Backend
- [ ] Gunicorn ile çoklu worker açıldı
- [ ] Systemd servisi aktif edildi (otomatik yeniden başlatma)
- [ ] `logs/` dizini yazılabilir
- [ ] `model_artifacts/` dizini erişilebilir
- [ ] `data/` dizini yazılabilir (SQLite ve JSON dosyaları)

### Frontend
- [ ] `npm run build` ile üretim build alındı
- [ ] Docker image oluşturuldu ve test edildi
- [ ] Nginx güvenlik başlıkları aktif
- [ ] Gzip sıkıştırma aktif

### Model
- [ ] Gerekli `.keras` model dosyaları `model_artifacts/` içinde mevcut
- [ ] `model_registry.json` doğru yapılandırıldı

---

## 🔥 Sorun Giderme

### Backend Başlamıyor

```bash
# Python sürümü kontrolü
python --version  # 3.10+ gerekli

# Bağımlılıklar yüklü mü?
pip install -r requirements.txt

# Port kullanımda mı?
# Windows:
netstat -ano | findstr :8000
# Linux:
lsof -i :8000

# Log detayı için
uvicorn app.main:app --reload --log-level debug
```

### Frontend Container Başlamıyor

```bash
# Docker durumu
docker ps -a
docker logs cyberguard-frontend

# Image'ı yeniden oluştur
cd frontend
docker-compose down
docker-compose up --build -d
```

### CORS Hatası

Frontend'den API'ye istek gönderildiğinde CORS hatası alıyorsanız:

1. `.env` dosyasındaki `CORS_ORIGINS` değerini kontrol edin
2. Değer, frontend'in çalıştığı origin ile tam eşleşmeli (örn. `http://localhost:5173`)
3. Backend'i yeniden başlatın

### LLM API Bağlantı Hatası

```bash
# Hangi provider'ların yapılandırıldığını görün
curl http://localhost:8000/api/chat/providers

# Groq ücretsiz ve hızlıdır — önce bunu deneyin
# https://console.groq.com/keys adresinden API anahtarı alın
```

---

[⬆️ Back to Top](#-deployment-guide)
