# 🚀 Deployment Guide

CyberGuard AI Deployment Dokümantasyonu

---

## 📋 İçindekiler

- [Genel Bakış](#genel-bakış)
- [Local Deployment](#local-deployment)
- [Streamlit Cloud](#streamlit-cloud)
- [Docker Deployment](#docker-deployment)
- [AWS Deployment](#aws-deployment)
- [Heroku Deployment](#heroku-deployment)
- [Production Checklist](#production-checklist)
- [Monitoring](#monitoring)
- [Troubleshooting](#troubleshooting)

---

## 🌟 Genel Bakış

CyberGuard AI'ı farklı ortamlarda deploy edebilirsiniz:

| Platform | Maliyet | Kolay | Performans | Önerilen |
|----------|---------|-------|------------|----------|
| Local | Ücretsiz | ⭐⭐⭐⭐⭐ | Orta | Dev |
| Streamlit Cloud | Ücretsiz | ⭐⭐⭐⭐⭐ | İyi | Demo |
| Docker | Düşük | ⭐⭐⭐⭐ | İyi | Test |
| AWS | Orta-Yüksek | ⭐⭐⭐ | Mükemmel | Production |
| Heroku | Orta | ⭐⭐⭐⭐ | İyi | MVP |

---

## 💻 Local Deployment

### Gereksinimler

- Python 3.10+
- 8GB+ RAM
- 5GB+ disk space

### Kurulum

```bash
# 1. Repository'yi klonla
git clone https://github.com/yourusername/CyberGuard_AI.git
cd CyberGuard_AI

# 2. Virtual environment oluştur
python -m venv venv

# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate

# 3. Paketleri kur
pip install -r requirements.txt

# 4. .env dosyası oluştur
echo "GOOGLE_API_KEY=your_api_key_here" > .env

# 5. Mock veri oluştur (opsiyonel)
python src/utils/mock_data_generator.py

# 6. Model eğit
python train_model.py

# 7. Çalıştır
cd app
streamlit run main.py
```

### Port Yapılandırması

```bash
# Farklı port kullan
streamlit run main.py --server.port 8080

# Network'e aç
streamlit run main.py --server.address 0.0.0.0
```

---

## ☁️ Streamlit Cloud Deployment

### Avantajlar

- ✅ Ücretsiz (public apps)
- ✅ Otomatik HTTPS
- ✅ GitHub entegrasyonu
- ✅ Kolay güncelleme

### Adım 1: GitHub'a Push

```bash
git add .
git commit -m "Ready for deployment"
git push origin main
```

### Adım 2: Streamlit Cloud'a Bağlan

1. [share.streamlit.io](https://share.streamlit.io) adresine git
2. GitHub ile giriş yap
3. "New app" tıkla
4. Repository seç: `yourusername/CyberGuard_AI`
5. Main file path: `app/main.py`

### Adım 3: Secrets Ekle

Dashboard → App settings → Secrets

```toml
# .streamlit/secrets.toml
GOOGLE_API_KEY = "your_api_key_here"
```

### Adım 4: Deploy

"Deploy!" butonuna tıkla ve bekle (2-5 dakika)

### Config Dosyası

`.streamlit/config.toml` oluştur:

```toml
[theme]
primaryColor = "#667eea"
backgroundColor = "#0e1117"
secondaryBackgroundColor = "#262730"
textColor = "#fafafa"
font = "sans serif"

[server]
headless = true
port = 8501
enableCORS = false
enableXsrfProtection = true

[browser]
gatherUsageStats = false
```

---

## 🐳 Docker Deployment

### Dockerfile

```dockerfile
# Dockerfile
FROM python:3.10-slim

# Çalışma dizini
WORKDIR /app

# Sistem paketleri
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

# Python bağımlılıkları
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Uygulama dosyaları
COPY . .

# Port
EXPOSE 8501

# Health check
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

# Başlat
ENTRYPOINT ["streamlit", "run", "app/main.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

### docker-compose.yml

```yaml
version: '3.8'

services:
  cyberguard-app:
    build: .
    container_name: cyberguard_ai
    ports:
      - "8501:8501"
    environment:
      - GOOGLE_API_KEY=${GOOGLE_API_KEY}
    volumes:
      - ./cyberguard.db:/app/cyberguard.db
      - ./models:/app/models
    restart: unless-stopped
    networks:
      - cyberguard-network

networks:
  cyberguard-network:
    driver: bridge
```

### .dockerignore

```
venv/
__pycache__/
*.pyc
*.pyo
*.pyd
.Python
.env
.git
.gitignore
.vscode
.idea
*.log
temp_*
test_*
```

### Build & Run

```bash
# Build
docker build -t cyberguard-ai .

# Run
docker run -p 8501:8501 \
  -e GOOGLE_API_KEY=your_key \
  -v $(pwd)/cyberguard.db:/app/cyberguard.db \
  cyberguard-ai

# Docker Compose ile
docker-compose up -d

# Logları izle
docker-compose logs -f

# Durdur
docker-compose down
```

---

## ☁️ AWS Deployment

### Architecture

```
Internet → Route 53 → CloudFront → ALB → ECS (Fargate) → RDS
                                           ↓
                                          S3 (models)
```

### 1. EC2 Instance (Basit)

```bash
# 1. EC2 instance oluştur (t2.medium, Ubuntu 22.04)

# 2. SSH ile bağlan
ssh -i your-key.pem ubuntu@your-ec2-ip

# 3. Kurulum
sudo apt update && sudo apt upgrade -y
sudo apt install python3-pip python3-venv -y

# 4. Uygulama deploy
git clone https://github.com/yourusername/CyberGuard_AI.git
cd CyberGuard_AI
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 5. .env oluştur
nano .env
# GOOGLE_API_KEY=your_key

# 6. Systemd service oluştur
sudo nano /etc/systemd/system/cyberguard.service
```

**cyberguard.service:**

```ini
[Unit]
Description=CyberGuard AI
After=network.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/home/ubuntu/CyberGuard_AI
Environment="PATH=/home/ubuntu/CyberGuard_AI/venv/bin"
ExecStart=/home/ubuntu/CyberGuard_AI/venv/bin/streamlit run app/main.py --server.port 8501 --server.address 0.0.0.0
Restart=always

[Install]
WantedBy=multi-user.target
```

```bash
# 7. Servisi başlat
sudo systemctl daemon-reload
sudo systemctl enable cyberguard
sudo systemctl start cyberguard

# 8. Security group'ta 8501 portunu aç
```

### 2. ECS Fargate (Production)

**task-definition.json:**

```json
{
  "family": "cyberguard-task",
  "containerDefinitions": [
    {
      "name": "cyberguard-container",
      "image": "your-account.dkr.ecr.region.amazonaws.com/cyberguard:latest",
      "portMappings": [
        {
          "containerPort": 8501,
          "protocol": "tcp"
        }
      ],
      "environment": [
        {
          "name": "GOOGLE_API_KEY",
          "value": "your_key_here"
        }
      ],
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/cyberguard",
          "awslogs-region": "us-east-1",
          "awslogs-stream-prefix": "ecs"
        }
      }
    }
  ],
  "requiresCompatibilities": ["FARGATE"],
  "networkMode": "awsvpc",
  "cpu": "1024",
  "memory": "2048"
}
```

**Deploy:**

```bash
# 1. ECR'a push
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin your-account.dkr.ecr.us-east-1.amazonaws.com

docker build -t cyberguard .
docker tag cyberguard:latest your-account.dkr.ecr.us-east-1.amazonaws.com/cyberguard:latest
docker push your-account.dkr.ecr.us-east-1.amazonaws.com/cyberguard:latest

# 2. ECS task oluştur
aws ecs register-task-definition --cli-input-json file://task-definition.json

# 3. Service oluştur
aws ecs create-service \
  --cluster cyberguard-cluster \
  --service-name cyberguard-service \
  --task-definition cyberguard-task \
  --desired-count 2 \
  --launch-type FARGATE \
  --network-configuration "awsvpcConfiguration={subnets=[subnet-xxx],securityGroups=[sg-xxx],assignPublicIp=ENABLED}"
```

### 3. S3 + CloudFront (Static Assets)

```bash
# Models ve static dosyaları S3'e yükle
aws s3 cp models/ s3://cyberguard-models/ --recursive

# CloudFront distribution oluştur
aws cloudfront create-distribution --origin-domain-name cyberguard-models.s3.amazonaws.com
```

---

## 🌐 Heroku Deployment

### Procfile

```
web: streamlit run app/main.py --server.port=$PORT --server.address=0.0.0.0
```

### runtime.txt

```
python-3.10.12
```

### Deploy

```bash
# 1. Heroku CLI kur
# https://devcenter.heroku.com/articles/heroku-cli

# 2. Login
heroku login

# 3. App oluştur
heroku create cyberguard-ai

# 4. Config vars ekle
heroku config:set GOOGLE_API_KEY=your_key_here

# 5. Deploy
git push heroku main

# 6. Aç
heroku open

# 7. Logları izle
heroku logs --tail
```

### Buildpack (Opsiyonel)

```bash
heroku buildpacks:set heroku/python
```

---

## ✅ Production Checklist

### Security

- [ ] API keys `.env` dosyasında
- [ ] `.env` gitignore'da
- [ ] HTTPS kullanımı
- [ ] Rate limiting
- [ ] Input validation
- [ ] SQL injection koruması
- [ ] XSS koruması

### Performance

- [ ] Database indexing
- [ ] Caching (@st.cache_resource)
- [ ] Lazy loading
- [ ] Image optimization
- [ ] Gzip compression
- [ ] CDN kullanımı

### Monitoring

- [ ] Error logging
- [ ] Performance monitoring
- [ ] Uptime monitoring
- [ ] Alert sistemi
- [ ] Backup stratejisi

### Documentation

- [ ] README.md güncel
- [ ] API dokümantasyonu
- [ ] Deployment guide
- [ ] User guide
- [ ] Changelog

---

## 📊 Monitoring

### Logs

```bash
# Streamlit logs
tail -f ~/.streamlit/logs/*.log

# Docker logs
docker logs -f cyberguard_ai

# AWS CloudWatch
aws logs tail /ecs/cyberguard --follow
```

### Uptime Monitoring

**UptimeRobot** (Ücretsiz):

```
https://uptimerobot.com
Monitor Type: HTTP(s)
URL: https://your-app-url.com
```

### Application Monitoring

```python
# src/utils/monitoring.py
import time
from functools import wraps

def monitor_performance(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        duration = time.time() - start
        
        logger.info(f"{func.__name__} took {duration:.2f}s")
        return result
    return wrapper
```

---

## 🔧 Troubleshooting

### Common Issues

#### 1. Port Already in Use

```bash
# Port'u kullanımdan kaldır
# Windows
netstat -ano | findstr :8501
taskkill /PID <PID> /F

# Mac/Linux
lsof -ti:8501 | xargs kill -9
```

#### 2. Module Not Found

```bash
# Virtual environment aktif mi?
which python  # venv içinde olmalı

# Paketleri yeniden kur
pip install -r requirements.txt --force-reinstall
```

#### 3. Database Locked

```python
# Timeout artır
import sqlite3
conn = sqlite3.connect('cyberguard.db', timeout=30)
```

#### 4. Memory Error

```bash
# Streamlit memory limit artır
streamlit run app/main.py --server.maxUploadSize=1000
```

#### 5. Streamlit Cloud Secrets

```toml
# .streamlit/secrets.toml oluştur
# Sonra Streamlit Cloud dashboard'dan ekle
```

---

## 🔄 CI/CD Pipeline (GitHub Actions)

**.github/workflows/deploy.yml:**

```yaml
name: Deploy to Production

on:
  push:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      
      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: '3.10'
      
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
      
      - name: Run tests
        run: |
          pytest tests/
  
  deploy:
    needs: test
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      
      - name: Deploy to Streamlit Cloud
        run: |
          # Streamlit Cloud auto-deploys on push
          echo "Deployed to Streamlit Cloud"
      
      # Ya da Docker
      - name: Build and push Docker
        run: |
          docker build -t cyberguard:${{ github.sha }} .
          docker push your-registry/cyberguard:${{ github.sha }}
```

---

## 📈 Scaling

### Vertical Scaling

```bash
# Daha güçlü instance
# AWS: t2.medium → t2.xlarge
# Heroku: Standard-1X → Performance-M
```

### Horizontal Scaling

```yaml
# docker-compose.yml
services:
  cyberguard:
    deploy:
      replicas: 3  # 3 instance
    
  nginx:
    image: nginx
    ports:
      - "80:80"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
```

**nginx.conf:**

```nginx
upstream cyberguard {
    server cyberguard_1:8501;
    server cyberguard_2:8501;
    server cyberguard_3:8501;
}

server {
    listen 80;
    
    location / {
        proxy_pass http://cyberguard;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
    }
}
```

---

## 💰 Cost Estimation

### Free Tier

- **Local**: Ücretsiz
- **Streamlit Cloud**: Ücretsiz (public apps)

### Paid Options

| Platform | Monthly Cost | Specs |
|----------|--------------|-------|
| Heroku Standard | $25-50 | 512MB-1GB RAM |
| AWS EC2 t2.medium | $30-40 | 4GB RAM, 2 vCPU |
| AWS Fargate | $50-100 | 2GB RAM, 1 vCPU |
| DigitalOcean | $12-24 | 2-4GB RAM |

---

## 🚨 Backup & Recovery

### Database Backup

```bash
# Otomatik backup scripti
#!/bin/bash
DATE=$(date +%Y%m%d_%H%M%S)
cp cyberguard.db backups/cyberguard_$DATE.db

# Eski backup'ları sil (30 günden eski)
find backups/ -name "*.db" -mtime +30 -delete
```

### Cron Job

```bash
# Günlük backup (her gün 03:00)
0 3 * * * /path/to/backup.sh
```

### S3'e Yedekleme

```bash
aws s3 sync backups/ s3://cyberguard-backups/
```

---

## 📞 Support

Deployment ile ilgili sorularınız için:

- 📧 Email: devops@cyberguardai.com
- 💬 Discord: [discord.gg/cyberguardai](https://discord.gg/cyberguardai)
- 📖 Docs: [docs.cyberguardai.com/deployment](https://docs.cyberguardai.com/deployment)

---

[⬆️ Back to Top](#-deployment-guide)