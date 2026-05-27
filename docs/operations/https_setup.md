# HTTPS / TLS Yapılandırması

CyberGuard AI frontend'ini HTTPS ile yayınlamak için Nginx ters proxy kurulum rehberi.

## Yöntem 1: Let's Encrypt + Certbot (Üretim)

### Gereksinimler

- Sunucuya yönlendirilmiş bir domain adı (örn. `cyberguard.example.com`)
- Ubuntu/Debian sunucu
- Docker kurulu

### Adım 1: Certbot ile Sertifika Al

```bash
# Certbot kur
sudo apt update && sudo apt install -y certbot

# Standalone mod ile sertifika al (80 portu geçici olarak açık olmalı)
sudo certbot certonly --standalone -d cyberguard.example.com

# Sertifikalar buraya kaydedilir:
# /etc/letsencrypt/live/cyberguard.example.com/fullchain.pem
# /etc/letsencrypt/live/cyberguard.example.com/privkey.pem
```

### Adım 2: nginx-ssl.conf Oluştur

`frontend/nginx-ssl.conf` olarak kaydet:

```nginx
# HTTP → HTTPS yönlendirme
server {
    listen 80;
    server_name cyberguard.example.com;
    return 301 https://$host$request_uri;
}

# HTTPS sunucu
server {
    listen 443 ssl http2;
    server_name cyberguard.example.com;
    root /usr/share/nginx/html;
    index index.html;

    # TLS sertifikaları
    ssl_certificate     /etc/letsencrypt/live/cyberguard.example.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/cyberguard.example.com/privkey.pem;

    # Modern TLS ayarları
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers ECDHE-ECDSA-AES128-GCM-SHA256:ECDHE-RSA-AES128-GCM-SHA256:ECDHE-ECDSA-AES256-GCM-SHA384:ECDHE-RSA-AES256-GCM-SHA384;
    ssl_prefer_server_ciphers off;
    ssl_session_cache shared:SSL:10m;
    ssl_session_timeout 1d;

    # HSTS (tarayıcıya HTTPS'i zorunlu kıl)
    add_header Strict-Transport-Security "max-age=63072000; includeSubDomains; preload" always;

    # Güvenlik başlıkları
    add_header X-Frame-Options "SAMEORIGIN" always;
    add_header X-Content-Type-Options "nosniff" always;
    add_header Referrer-Policy "strict-origin-when-cross-origin" always;
    add_header Content-Security-Policy "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; img-src 'self' data: blob:; connect-src 'self' wss: ws: https:;" always;

    # Gzip
    gzip on;
    gzip_types text/plain text/css application/json application/javascript text/xml application/xml image/svg+xml;
    gzip_min_length 256;

    # Statik dosyalar için agresif cache
    location /assets/ {
        expires 1y;
        add_header Cache-Control "public, immutable";
    }

    # API ters proxy → Backend FastAPI (8000)
    location /api/ {
        proxy_pass http://backend:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }

    # WebSocket ters proxy
    location /ws {
        proxy_pass http://backend:8000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_read_timeout 86400;
    }

    # SPA fallback
    location / {
        try_files $uri $uri/ /index.html;
    }
}
```

### Adım 3: Docker Compose'u Güncelle

```yaml
# frontend/docker-compose-ssl.yml
version: "3.8"
services:
  frontend:
    build: .
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx-ssl.conf:/etc/nginx/conf.d/default.conf
      - /etc/letsencrypt:/etc/letsencrypt:ro
    restart: unless-stopped
```

```bash
# Başlat
docker compose -f docker-compose-ssl.yml up -d
```

### Adım 4: Sertifika Otomatik Yenileme

```bash
# Cron job ekle (her gün 00:00 ve 12:00'da kontrol)
sudo crontab -e

# Şu satırı ekle:
0 0,12 * * * certbot renew --quiet && docker compose -f /path/to/frontend/docker-compose-ssl.yml restart frontend
```

---

## Yöntem 2: Self-Signed Sertifika (Geliştirme / Test)

```bash
# Sertifika oluştur (frontend/ dizininde)
mkdir -p ssl

openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
  -keyout ssl/privkey.pem \
  -out ssl/fullchain.pem \
  -subj "/CN=localhost/O=CyberGuard/C=TR"
```

`nginx-ssl.conf` içinde sertifika yollarını değiştir:

```nginx
ssl_certificate     /etc/nginx/ssl/fullchain.pem;
ssl_certificate_key /etc/nginx/ssl/privkey.pem;
```

```yaml
# docker-compose override
volumes:
  - ./nginx-ssl.conf:/etc/nginx/conf.d/default.conf
  - ./ssl:/etc/nginx/ssl:ro
```

> **Not**: Self-signed sertifika tarayıcı güvenlik uyarısı verir. Sadece geliştirme/test ortamında kullan.

---

## Backend CORS Güncellemesi

HTTPS kullanırken `.env` dosyasını güncelle:

```env
CORS_ORIGINS=https://cyberguard.example.com,http://localhost:5173
```

---

## Güvenlik Kontrol Listesi

| Kontrol | Durum |
|---------|-------|
| TLS 1.2+ kullanılıyor | `ssl_protocols TLSv1.2 TLSv1.3` |
| HSTS aktif | `Strict-Transport-Security` başlığı |
| HTTP → HTTPS yönlendirme | 301 redirect |
| CSP başlığı | `Content-Security-Policy` |
| Sertifika otomatik yenileme | Certbot cron |
| CORS HTTPS ile güncellendi | `.env CORS_ORIGINS` |

---

[⬆️ Geri Dön](../operations/deployment.md)
