# 📋 Release Notes

CyberGuard AI sürüm notları

---

## 🔧 v3.3.0 - Platform Güçlendirme (2026-04-25)

### 🎉 Özet

Bu güncelleme; üretim kalitesi altyapısı, gözlemlenebilirlik, güvenlik ve test kapsamını ekler. 10 temel iyileştirme, 6 yeni dosya, 7 güncellenen dosya.

### ✨ Yeni Özellikler

#### Altyapı & Gözlemlenebilirlik

- **TTL Cache** (`app/utils/cache.py`): Thread-safe in-process cache, `@ttl_cache(ttl=N)` dekoratörü. Dashboard (30s/60s) ve attack-map (10s) entegrasyonu
- **JSON Yapılandırılmış Loglama** (`app/utils/logging.py`): `JSONFormatter`, `RotatingFileHandler` (10MB × 5), `RequestIDMiddleware`. Her istekte `request_id` korelasyonu
- **Prometheus + Grafana Stack** (`docker-compose.monitoring.yml`): prometheus-fastapi-instrumentator ile `/metrics` endpoint'i; Grafana 7 panelli oto-sağlanan dashboard (istek oranı, p95 gecikme, CPU/bellek)

#### Güvenlik

- **Refresh Token** (7 gün TTL): `POST /auth/refresh` endpoint'i; token rotasyonu ile oturum güvenliği
- **RBAC `require_role()`**: Fabrika fonksiyonu ile rol tabanlı erişim kontrolü; admin/analyst/viewer
- **HTTPS/TLS Rehberi** (`docs/operations/https_setup.md`): Nginx ters proxy + Let's Encrypt kurulum dokümantasyonu
- **OWASP A01 — Broken Access Control (Tam Giderim)**: Router-level `APIRouter(dependencies=[Depends(require_auth)])` ile ~23 route dosyasındaki 40+ endpoint korundu. `incidents`, `siem`, `network`, `security_advanced`, `vulnerability`, `scanner`, `threat_intel`, `attacks`, `darkweb`, `deception`, `attack_surface`, `threat_analysis`, `threat_hunting`, `attack_map`, `models`, `prediction`, `training`, `drift_detection`, `log_analyzer`, `logs`, `reports`, `dashboard`, `playbooks`, `blockchain_audit`

#### AI & Analiz

- **LLM Threat Decision Agent** (`src/ai_decision/threat_agent.py`): Groq/OpenAI/Ollama desteği; `ThreatDecisionAgent.handle_threat()`, kural tabanlı yedek, `POST /api/incidents/analyze-threat` endpoint'i

#### Geliştirici Deneyimi

- **GitHub Actions CI** (`.github/workflows/ci.yml`): 3 paralel iş — backend (Python 3.10+3.11, ruff+pytest+coverage), frontend (Node 22, vitest), docker (yalnızca main)
- **Pre-commit Hooks** (`.pre-commit-config.yaml`): ruff-fix, ruff-format, trailing-whitespace, YAML/JSON kontrolleri
- **Test Suite Genişletme**: 3 backend test dosyası (auth, security_monitoring, ml_endpoints) + 3 frontend Vitest test dosyası (ProtectedRoute, NotificationStore, useWebSocket)

### 🗂️ Yeni Dosyalar

| Dosya | Açıklama |
|-------|----------|
| `app/utils/cache.py` | TTL cache utility |
| `app/utils/logging.py` | Yapılandırılmış JSON loglama |
| `app/utils/__init__.py` | utils paketi init |
| `src/ai_decision/threat_agent.py` | LLM Threat Decision Agent |
| `docker-compose.monitoring.yml` | Prometheus + Grafana stack |
| `monitoring/prometheus.yml` | Prometheus scrape config |
| `monitoring/grafana/...` | Grafana datasource + dashboard provisioning |
| `requirements-dev.txt` | Geliştirme bağımlılıkları |
| `.pre-commit-config.yaml` | Pre-commit hook konfigürasyonu |
| `.github/workflows/ci.yml` | CI/CD pipeline |
| `tests/test_auth.py` | Auth endpoint testleri (12 test) |
| `tests/test_security_monitoring.py` | Güvenlik modülü testleri (15 test) |
| `tests/test_ml_endpoints.py` | ML endpoint testleri (22 test) |
| `tests/test_cache.py` | TTL cache testleri (13 test) |
| `tests/test_rbac.py` | RBAC ve auth testleri (17 test) |
| `docs/operations/https_setup.md` | HTTPS/TLS kurulum rehberi |
| `docs/README.md` | Dokümantasyon dizin haritası |

### 📊 Metrikler

| Metrik | Değer |
|--------|-------|
| Yeni Python dosyası | 4 |
| Yeni test dosyası | 8 |
| Backend test sayısı | 122 |
| Frontend test sayısı | 50 |
| Backend coverage (`app/`) | %37.35 |
| Yeni dokümantasyon dosyası | 2 |
| Güncellenen dokümantasyon | 12 |
| CI job sayısı | 3 |
| Auth ile korunan route dosyası | 23 |
| Auth ile korunan toplam endpoint | 40+ |

### 🔧 Değişiklikler

- `app/main.py`: `setup_logging()`, `RequestIDMiddleware`, Prometheus instrumentator eklendi
- `app/api/routes/core/dashboard.py`: `@ttl_cache` dekoratörleri eklendi
- `app/api/routes/threat/attack_map.py`: `@ttl_cache(ttl=10)` eklendi
- `app/api/routes/auth.py`: Refresh token + RBAC eklendi
- `app/api/routes/monitoring/incidents.py`: `/analyze-threat` endpoint eklendi
- `app/api/routes/ml/advanced_ml.py`: Gerçek `.keras` model yükleme implementasyonu (permutation importance)
- `app/api/routes/security/vulnerability.py`: Gerçek CVE ID’leri (CVE-2023-36053 vb.)
- `pyproject.toml`: Coverage kapsamı `app/`, `fail_under=35`
- `frontend/eslint.config.js`: `react-hooks/immutability` + test globals eklendi
- `requirements.txt`: `prometheus-fastapi-instrumentator>=7.0.0` eklendi

---

## 📚 v3.2.0 - Dokümantasyon Yeniden Yapılandırması (2026-04-24)

### 🌟 Özet

33 Markdown dosyası 8 alt kategoriye yeniden düzenlendi. `architecture.md`, `api_reference.md`, `deployment.md`, `installation.md` tam yeniden yazımı yapıldı. Streamlit/Heroku/PostgreSQL referansları temizlendi. Bkz. [changelog [3.2.0]](changelog.md).

---

## 🌐 v3.1.0 - WebSocket & 3D Saldırı Haritası (2026-01-13)

### 🌟 Özet

Gerçek zamanlı 3D saldırı haritası, WebSocket saldırı akışı (`ws://localhost:8000/ws/attacks`), GeoIP servisi ve ML tahmin yayını. Bkz. [changelog [3.1.0]](changelog.md).

---

## 🚀 v3.0.0 - Mega Update (2026-01-10)

### 🎉 Highlights

Bu büyük güncelleme ile CyberGuard AI, orijinal akademik makalenin kapsamının çok ötesine geçerek **25+ yeni özellik** ile tam kapsamlı bir siber güvenlik platformuna dönüşmüştür.

### ✨ Yeni Özellikler

#### API'ler (17 Yeni Modül)

- **XAI (Explainable AI)**: SHAP ve LIME ile model açıklamaları
- **Adversarial Testing**: Model güvenlik testleri
- **Federated Learning**: Dağıtık model eğitimi
- **AutoML**: Otomatik model seçimi ve optimizasyonu
- **Threat Intelligence**: IP/Domain/Hash reputation
- **Email Alerts**: Otomatik bildirim sistemi
- **PDF Reports**: Profesyonel rapor oluşturma
- **Model Comparison**: Model benchmark ve leaderboard
- **Anomaly Detection**: Anomali tespit algoritmaları
- **Security Advanced**: PCAP analizi, Honeypot, Compliance
- **Vulnerability Scanner**: Port tarama, CVE kontrolü
- **Log Analyzer**: ML ile log analizi
- **Incidents**: Olay timeline ve user behavior
- **API Keys**: API anahtar yönetimi

#### Frontend (5 Yeni Sayfa)

- XAI Explainer (`/xai`)
- Security Hub (`/security-hub`)
- AutoML Pipeline (`/automl`)
- Vulnerability Scanner (`/vuln-scanner`)
- Incident Timeline (`/incidents`)

#### Dokümantasyon (14 Yeni Dosya)

- faq.md, troubleshooting.md, glossary.md
- api_endpoints_full.md, testing.md, ci_cd.md
- monitoring.md, backup_recovery.md
- performance_tuning.md, LICENSE.md
- SECURITY_POLICY.md, release_notes.md
- ml_models.md, datasets.md

### 📊 İstatistikler

| Metrik | Değer |
|--------|-------|
| Yeni API Endpoint | 80+ |
| Toplam Endpoint | 150+ |
| Yeni Frontend Sayfa | 5 |
| Yeni Docs Dosyası | 14 |
| Makalede Olmayan Özellik | 25+ |

### 🔧 İyileştirmeler

- Dosya yapısı reorganize edildi
  - scripts/ → training/, optimization/, data/, utils/, archived/
  - models/ → production/, experimental/, archived/
  - docs/ yazım hataları düzeltildi

### 📝 Dokümantasyon

- Tüm yeni özellikler belgelendi
- API endpoint listesi güncellendi
- Changelog v3.0.0 için güncellendi

---

## 🚀 v2.0.0 (2025-01-15)

### ✨ Yeni Özellikler

- AI-Powered Chatbot
- Gemini AI entegrasyonu
- Real-time threat monitoring
- PDF ve Excel export
- MFA desteği
- Enhanced dashboard

### 🔧 İyileştirmeler

- Model accuracy %95+ → %99+
- API response time %40 iyileştirildi
- UI/UX tamamen yenilendi

### 🐛 Düzeltmeler

- Port tarama timeout sorunu
- Database connection pool sızıntısı
- Memory leak

---

## 🚀 v1.5.0 (2024-10-20)

### ✨ Yeni Özellikler

- ML-based threat detection
- Random Forest classifier
- Scheduled scans
- Email notifications
- Slack integration

### 🔧 İyileştirmeler

- Scanner performance %30 artırıldı
- False positive rate azaltıldı

---

## 🚀 v1.0.0 (2024-06-01)

### İlk Stable Sürüm

- Port scanning
- Vulnerability detection
- CVE database integration
- Web dashboard
- REST API
- PostgreSQL support

---

## 📅 Upgrade Guide

### v2.x → v3.0

```bash
# 1. Backup
./scripts/backup_all.sh

# 2. Pull latest
git pull origin main

# 3. Install dependencies
pip install -r requirements.txt
cd frontend && npm install

# 4. Run migrations
alembic upgrade head

# 5. Restart
./start-servers.sh
```

### Breaking Changes

- API v1 endpoints kaldırıldı
- `config.yaml` formatı değişti
- Model dosya yapısı değişti

---

## 📞 Destek

- GitHub Issues
- Discord: discord.gg/cyberguard
- Email: <support@cyberguard-ai.com>
