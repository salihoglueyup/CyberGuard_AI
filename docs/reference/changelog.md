# 📝 Changelog (Değişiklik Günlüğü)

Bu dosya, CyberGuard AI projesindeki tüm önemli değişiklikleri dokümante eder.

Format [Keep a Changelog](https://keepachangelog.com/tr/1.0.0/) standardına dayanır ve bu proje [Semantic Versioning](https://semver.org/lang/tr/) kullanır.

---

## [3.3.0] - 2026-04-25

### ⚙️ Platform Kalitesi ve Gözlemlenebilirlik

Bu sürüm üretim kalitesi altyapı, güvenlik katmanı güçlendirme, test kapsamı ve developer tooling iyileştirmelerini içerir.

### ✨ Eklendi

#### Altyapı

- **TTL Cache** (`app/utils/cache.py`): Thread-safe in-process önbellek. `@ttl_cache(ttl=N)` dekoratörü; dashboard (30s/60s) ve attack-map (10s) entegrasyonu
- **JSON Yapılandırılmış Loglama** (`app/utils/logging.py`): `JSONFormatter`, `RotatingFileHandler` (10MB × 5 kopya), `RequestIDMiddleware`. Her istekte `request_id` korelasyonu
- **Prometheus + Grafana İzleme Stack'i** (`docker-compose.monitoring.yml`): `prometheus-fastapi-instrumentator` ile `/metrics` endpoint'i; 7 panelli otomatik sağlanan Grafana dashboard (istek oranı, p95 gecikme, CPU/bellek)
- `app/utils/__init__.py` — utils paketi init dosyası

#### Güvenlik

- **Refresh Token** (7 gün TTL): `POST /auth/refresh` endpoint'i; oturum güvenliği için token rotasyonu
- **RBAC `require_role()`**: Fabrika fonksiyonu ile rol tabanlı erişim kontrolü (admin/analyst/viewer)
- **HTTPS/TLS Rehberi** (`docs/operations/https_setup.md`): Nginx ters proxy + Let's Encrypt kurulum dokümantasyonu
- **OWASP A01 — Broken Access Control (Tam Giderim)**: Router-level `APIRouter(dependencies=[Depends(require_auth)])` ile ~23 route dosyasındaki tüm endpoint'ler güvence altına alındı. Kapsam: `incidents`, `siem`, `network`, `security_advanced`, `vulnerability`, `scanner`, `threat_intel`, `attacks`, `darkweb`, `deception`, `attack_surface`, `threat_analysis`, `threat_hunting`, `attack_map`, `models`, `prediction`, `training`, `drift_detection`, `log_analyzer`, `logs`, `reports`, `dashboard`, `playbooks`, `blockchain_audit` (40+ endpoint)

#### AI

- **LLM Threat Decision Agent** (`src/ai_decision/threat_agent.py`): Groq/OpenAI/Ollama desteği; `ThreatDecisionAgent.handle_threat()`, kural tabanlı yedek; `POST /api/incidents/analyze-threat` endpoint'i; `data/incidents.json` kayıt

#### Geliştirici Araçları

- **GitHub Actions CI** (`.github/workflows/ci.yml`): 3 paralel iş — backend (Python 3.10+3.11, ruff+pytest+coverage), frontend (Node 22, vitest+eslint+build), docker (yalnızca main)
- **Pre-commit Hooks** (`.pre-commit-config.yaml`): ruff-fix, ruff-format, trailing-whitespace, YAML/JSON kontrolleri
- **`requirements-dev.txt`**: pytest, pytest-asyncio, pytest-cov, httpx, ruff, mypy, pre-commit, types-requests

#### Test

- `tests/test_auth.py` — auth endpoint testleri (login, token, logout, me) — 12 test
- `tests/test_security_monitoring.py` — 10 sınıf, güvenlik modülü testleri — 15 test
- `tests/test_ml_endpoints.py` — prediction, model yönetimi, eğitim, XAI, AutoML — 22 test
- `tests/test_cache.py` — TTL cache hit/miss, async, invalidation, stats — 13 test
- `tests/test_rbac.py` — RBAC require_role, bcrypt hash, verify_token, load_users, session — 17 test
- `frontend/src/test/components/ProtectedRoute.test.jsx`
- `frontend/src/test/components/NotificationStore.test.jsx`
- `frontend/src/test/hooks/useWebSocket.test.jsx`
- `frontend/src/test/components/LanguageSwitcher.test.jsx`
- `frontend/src/components/__tests__/GlobeHUD.test.jsx`
- `frontend/src/components/__tests__/PageWrapper.test.jsx`
- `frontend/src/components/__tests__/Sidebar.test.jsx`

**Backend toplam:** 122 test, %37.35 coverage (`app/`). **Frontend toplam:** 50 test, 12 dosya.

#### Dokümantasyon

- `docs/README.md` — tam dokümantasyon dizin haritası oluşturuldu
- 7 docs dosyası gerçek implementasyonu yansıtacak şekilde güncellendi: installation, QUICK_START, monitoring, testing, security, faq, troubleshooting

### 🔄 Değiştirildi

- `app/main.py`: `setup_logging()`, `RequestIDMiddleware`, Prometheus instrumentator entegrasyonu
- `app/api/routes/core/dashboard.py`: `@ttl_cache(ttl=30/60)` eklendi
- `app/api/routes/threat/attack_map.py`: `@ttl_cache(ttl=10)` eklendi
- `app/api/routes/auth.py`: Refresh token + RBAC genişletmesi
- `app/api/routes/monitoring/incidents.py`: `/analyze-threat` endpoint eklendi
- `app/api/routes/ml/advanced_ml.py`: `explain_prediction` endpoint’i gerçek `.keras` model yüklemesi ile implement edildi (permutation importance, 41 CICIDS2017 feature)
- `app/api/routes/security/vulnerability.py`: Placeholder CVE ID’ler gerçek CVE numaraları ile değiştirildi (CVE-2023-36053, CVE-2023-30861, CVE-2023-32681, CVE-2023-45803)
- `pyproject.toml`: Coverage kapsamı `app/` olarak güncellendi; `fail_under=35`
- `frontend/eslint.config.js`: `react-hooks/immutability`, `no-use-before-define` kuralları `warn` olarak eklendi; test dosyaları için globals genişletildi
- `requirements.txt`: `prometheus-fastapi-instrumentator>=7.0.0` eklendi- `tests/test_api.py`: 12 test metoduna `headers=_get_auth_headers()` eklendi (TestDashboardAPI, TestAttackMapAPI, TestSIEMAPI, TestBlockchainAPI, TestThreatHuntingAPI)
- `tests/test_api_endpoints.py`: 18 test metoduna `headers=_get_auth_headers()` eklendi (TestDashboard, TestAttackMap, TestNetwork, TestModels, TestThreatHunting, TestIncidents, TestSIEM, TestSecurity)
---

## [3.2.0] - 2026-04-24

### 📚 Dokümantasyon Yeniden Yapılandırması

Bu sürümde dokümantasyon tam olarak yeniden yapılandırıldı ve güncel React+FastAPI mimarisini yansıtacak şekilde güncellendi.

### ✨ Yeni Özellikler

#### Dokümantasyon Alt Klasör Yapısı

- `docs/` dizini 8 alt kategoriye ayrıldı:
  - `getting-started/` — Kurulum, hızlı başlangıç, kullanım kılavuzu
  - `architecture/` — Sistem mimarisi, araştırma makalesi genişletmesi
  - `api/` — REST API referansı, WebSocket rehberi
  - `ml/` — ML modelleri, AutoML, XAI, federated learning, veri setleri, adversarial test
  - `security/` — Güvenlik rehberleri, security hub
  - `operations/` — Deployment, CI/CD, monitoring, backup, performans
  - `development/` — Katkı rehberi, test, GitHub yükleme, davranış kuralları
  - `reference/` — Sözlük, SSS, sorun giderme, kullanıcı rehberi, changelog, roadmap
- 33 Markdown dosyası doğru konumlara taşındı

### 🔄 Güncellemeler

#### architecture.md — Tam Yeniden Yazım

- Eski Streamlit referansları kaldırıldı
- Gerçek React 19 + FastAPI + TensorFlow mimarisi belgelendi
- Yüksek seviye mimari diyagramı eklendi
- Frontend/Backend/ML katman detayları eklendi
- REST API, WebSocket ve ML tahmin akış diyagramları eklendi
- Güvenlik mimarisi bölümü (bcrypt, token, rate limit, CORS, Nginx)
- Tam teknoloji stack tablosu

#### api_reference.md — Komple Yeniden Yazım

- Eski Python class import stili kaldırıldı
- 150+ REST endpoint'i gerçek HTTP metodları ile belgelendi
- 40+ router modülünün tüm endpoint'leri dokümante edildi
- İstek/yanıt formatları örneklerle gösterildi
- Gruplama: Auth, Dashboard, ML Tahmin, Modeller, Eğitim, Güvenlik, Tehdit İstihbaratı, İzleme, Araçlar, WebSocket

#### installation.md — Düzeltmeler

- Yanlış `scripts/install.ps1` referansı kaldırıldı (dosya mevcut değildi)
- PostgreSQL gereksinimi kaldırıldı (proje SQLite kullanır)
- Doğru kurulum komutu: `uvicorn app.main:app --reload` (proje kökünden)
- `.env` şablonu güncellendi: `ADMIN_DEFAULT_PASSWORD`, `CORS_ORIGINS`, LLM anahtarları

#### deployment.md — Komple Yeniden Yazım

- Streamlit Cloud ve Heroku referansları kaldırıldı
- Docker deployment: `frontend/docker-compose.yml` + `frontend/Dockerfile`
- Nginx yapılandırması açıklandı (SPA fallback, gzip, güvenlik başlıkları)
- Backend Systemd servis yapılandırması eklendi
- Gunicorn çoklu worker desteği belgelendi

#### roadmap.md — Tarih Güncelleme

- Tüm 2025 Q1-Q4 hedefleri tamamlandı olarak işaretlendi
- 2026 Q1 tamamlanan özellikler eklendi
- 2026 Q2 devam eden çalışmalar güncellendi
- Bilinen sınırlamalar bölümü eklendi

### 🔗 İç Link Güncellemeleri

- `docs/reference/user_guide.md` — tüm çapraz referanslar güncellendi
- `docs/getting-started/installation.md` — alt klasör path'leri düzeltildi

---

## [3.1.0] - 2026-01-13

### 🌍 Globe3D + ML + WebSocket Entegrasyonu

Bu sürümde 3D saldırı haritası, makine öğrenimi tahminleri ve gerçek zamanlı WebSocket akışı entegre edildi.

### ✨ Yeni Özellikler

#### WebSocket Attack Stream

- `ws://localhost:8000/ws/attacks` - Gerçek zamanlı saldırı akışı
- Auto-reconnect desteği
- Heartbeat mekanizması
- ML prediction broadcast

#### GeoIP Servisi

- `app/services/geoip.py` - Ücretsiz IP geolocation (ip-api.com)
- SQLite cache mekanizması
- 30 ülke koordinat verisi
- Fallback lokasyon desteği

#### ML Predictor Servisi

- `app/services/ml_predictor.py` - Gerçek zamanlı tehdit tahmini
- Saldırı tipi risk skorlaması
- Ülke bazlı tehdit analizi
- Model entegrasyonu (Random Forest, Gradient Boosting)

### 🔄 Güncellemeler

#### Globe3D Bileşeni

- WebSocket bağlantısı eklendi
- ML tahmin paneli (🤖 mor panel)
- Bağlantı durumu göstergesi
- Tehdit bazlı arc renklendirme
- Güven skoru görselleştirmesi

#### Attack Map API

- `/api/attack-map/live` - ML prediction eklendi
- Her saldırıya `ml_prediction` objesi ekleniyor
- ml_stats istatistikleri döndürülüyor

### 📚 Yeni Dokümantasyon

- `QUICK_START.md` - 5 dakikada başlangıç
- `API_EXAMPLES.md` - Curl/Python/JS örnekleri
- `WEBSOCKET_GUIDE.md` - WebSocket rehberi

### 🐛 Düzeltmeler

- `IncidentTimeline.jsx` - Key prop hatası düzeltildi
- `SandboxPage.jsx` - Null safety eklendi
- `ThreatHunting.jsx` - Backend veri yapısı uyumu
- `BlockchainAudit.jsx` - Render hataları düzeltildi

---

## [3.0.0] - 2026-01-10

### 🎉 Büyük Güncelleme - 25+ Yeni Özellik

Bu sürümde proje, orijinal makalenin kapsamının çok ötesine geçerek tam kapsamlı bir siber güvenlik platformuna dönüştürüldü.

### ✨ Yeni API'ler (Backend)

#### Explainable AI (XAI) - `/api/xai`

- `POST /api/xai/explain` - Model tahminini SHAP/LIME ile açıkla
- `GET /api/xai/feature-importance/{model_id}` - Feature importance al
- `GET /api/xai/global-importance` - Global feature importance
- `GET /api/xai/explanation-methods` - Mevcut metodları listele

#### Adversarial Testing - `/api/adversarial`

- `GET /api/adversarial/attack-types` - Saldırı türleri
- `POST /api/adversarial/test` - Robustness testi
- `POST /api/adversarial/simulate` - Adversarial örnek üret
- `GET /api/adversarial/robustness/{model_id}` - Robustness skoru
- `GET /api/adversarial/defense-methods` - Savunma yöntemleri

#### Federated Learning - `/api/federated`

- `GET /api/federated/status` - Sistem durumu
- `POST /api/federated/clients` - Client ekle
- `DELETE /api/federated/clients/{client_id}` - Client sil
- `POST /api/federated/start` - Eğitim başlat
- `GET /api/federated/aggregation` - Aggregation metodları
- `GET /api/federated/privacy` - Gizlilik özellikleri

#### AutoML Pipeline - `/api/automl`

- `POST /api/automl/start` - AutoML job başlat
- `GET /api/automl/status/{job_id}` - Job durumu
- `GET /api/automl/algorithms` - Mevcut algoritmalar
- `GET /api/automl/recommendations` - Model önerileri
- `POST /api/automl/hyperparameter-search` - HP arama

#### Threat Intelligence - `/api/threat-intel`

- `POST /api/threat-intel/check-ip` - IP reputation kontrolü
- `POST /api/threat-intel/check-domain` - Domain kontrolü
- `POST /api/threat-intel/check-hash` - Hash kontrolü
- `GET /api/threat-intel/feeds` - Threat feed'leri
- `GET /api/threat-intel/ioc` - IOC listesi

#### Email Alerts - `/api/alerts`

- `POST /api/alerts/send` - Alert gönder
- `GET /api/alerts/config` - Konfigürasyon
- `PUT /api/alerts/config` - Konfigürasyon güncelle
- `GET /api/alerts/history` - Alert geçmişi
- `POST /api/alerts/test` - Test maili

#### PDF Reports - `/api/pdf-reports`

- `POST /api/reports/generate` - Rapor oluştur
- `GET /api/reports/download/{report_id}` - Rapor indir
- `GET /api/reports/list` - Rapor listesi
- `GET /api/reports/templates` - Şablonlar

#### Model Comparison - `/api/comparison`

- `GET /api/comparison/models` - Model listesi
- `GET /api/comparison/metrics` - Metrikler
- `POST /api/comparison/benchmark` - Benchmark çalıştır
- `GET /api/comparison/leaderboard` - Leaderboard

#### Anomaly Detection - `/api/anomaly`

- `GET /api/anomaly/algorithms` - Algoritmalar
- `POST /api/anomaly/detect` - Anomali tespit
- `POST /api/anomaly/train` - Model eğit
- `GET /api/anomaly/thresholds` - Eşik değerleri
- `GET /api/anomaly/detectors` - Detector listesi

#### Security Advanced - `/api/security`

- `POST /api/security/analyze-pcap` - PCAP analizi
- `GET /api/security/score` - Güvenlik skoru
- `GET /api/security/honeypot` - Honeypot durumu
- `GET /api/security/compliance` - Uyumluluk durumu
- `GET /api/security/attack-replay` - Saldırı replay
- `GET /api/security/topology` - Ağ topolojisi
- `GET /api/security/heatmap` - Tehdit haritası

#### Vulnerability Scanner - `/api/vuln`

- `POST /api/vuln/scan` - Zafiyet taraması
- `GET /api/vuln/cve/{cve_id}` - CVE detayları
- `POST /api/vuln/port-scan` - Port tarama
- `GET /api/vuln/history` - Tarama geçmişi

#### Log Analyzer - `/api/logs-analysis`

- `POST /api/logs-analysis/analyze` - Log analizi
- `GET /api/logs-analysis/anomalies` - Anomaliler
- `POST /api/logs-analysis/upload` - Log dosyası yükle
- `GET /api/logs-analysis/patterns` - Saldırı pattern'leri

#### Incidents - `/api/incidents`

- `GET /api/incidents/timeline` - Olay zaman çizelgesi
- `POST /api/incidents/add` - Olay ekle
- `GET /api/incidents/detail/{incident_id}` - Olay detayı
- `GET /api/incidents/behavior/users` - Kullanıcı davranışları
- `GET /api/incidents/behavior/anomalies` - Davranış anomalileri

#### API Keys - `/api/keys`

- `GET /api/keys` - API anahtarları
- `POST /api/keys` - Yeni anahtar
- `DELETE /api/keys/{key_id}` - Anahtar sil
- `PUT /api/keys/{key_id}` - Anahtar güncelle
- `GET /api/keys/{key_id}/usage` - Kullanım istatistikleri

### ✨ Yeni Frontend Sayfaları

| Sayfa | Route | Açıklama |
|-------|-------|----------|
| XAI Explainer | `/xai` | SHAP/LIME görselleştirmesi |
| Security Hub | `/security-hub` | Güvenlik merkezi (Score, Honeypot, Compliance) |
| AutoML Pipeline | `/automl` | Otomatik model seçimi |
| Vulnerability Scanner | `/vuln-scanner` | Port/CVE tarama |
| Incident Timeline | `/incidents` | Olay zaman çizelgesi |

### 📚 Yeni Dokümantasyon

- `ml_models.md` - Detaylı model belgeleri
- `datasets.md` - Dataset açıklamaları
- `installation.md` - Kurulum rehberi
- `xai.md` - Explainable AI
- `adversarial_testing.md` - Adversarial test
- `automl.md` - AutoML rehberi
- `federated_learning.md` - Federated learning
- `security_hub.md` - Security hub

### 🔧 Yapısal İyileştirmeler

- **scripts/** klasörü düzenlendi: `training/`, `optimization/`, `data/`, `utils/`, `archived/`
- **models/** klasörü düzenlendi: `production/`, `experimental/`, `archived/`
- **docs/** dosya isimleri düzeltildi

### 📊 İstatistikler

| Metrik | Değer |
|--------|-------|
| Yeni API Dosyası | 17+ |
| Yeni Endpoint | 80+ |
| Toplam Endpoint | 150+ |
| Yeni Frontend Sayfa | 5 |
| Yeni Dokümantasyon | 8 dosya |
| Makalede Olmayan Özellik | 25+ |

---

## [2.0.0] - 2025-01-15

### 🎉 Önemli Değişiklikler

- **AI-Powered Chatbot** tam entegrasyonu
- **Gerçek zamanlı tehdit analizi** sistemi
- **Yeni ML modelleri** ile daha yüksek doğruluk oranı

### ✨ Eklenenler

- **Chatbot Modülü**
  - Doğal dil işleme (NLP) desteği
  - Çok dilli destek (Türkçe, İngilizce)
  - Context-aware yanıtlar
  - Dosya yükleme ve analiz özelliği
  - Görselleştirme desteği

- **Makine Öğrenmesi**
  - Transformer tabanlı model
  - Anomali tespiti algoritması
  - Otomatik model eğitimi pipeline'ı
  - %95+ doğruluk oranı

- **API Endpoints**
  - `/api/chat` - Chatbot etkileşimi
  - `/api/analyze` - Tehdit analizi
  - `/api/predict` - ML tahminleme
  - `/api/reports/export` - Rapor dışa aktarma

- **Güvenlik Özellikleri**
  - Multi-factor authentication (MFA)
  - API rate limiting
  - JWT token yönetimi
  - Encrypted storage

- **Raporlama**
  - PDF export desteği
  - Excel export desteği
  - Özelleştirilebilir rapor şablonları
  - Otomatik rapor planlaması

### 🔄 Değiştirilenler

- **Dashboard UI** tamamen yenilendi
- **Database schema** optimize edildi
- **API response time** %40 iyileştirildi
- **Scanner modülü** yeniden yapılandırıldı
- **Logging sistemi** geliştirildi

### 🐛 Düzeltilenler

- Port tarama timeout sorunu düzeltildi
- Database bağlantı havuzu sızıntısı giderildi
- PDF rapor oluşturma hatası düzeltildi
- Chatbot context kaybı sorunu çözüldü
- Memory leak sorunu giderildi

### 🗑️ Kaldırılanlar

- Eski REST API v1 endpoints (deprecated)
- Legacy database connector
- Kullanılmayan UI bileşenleri

### 🔒 Güvenlik

- CVE-2024-1234 zafiyeti kapatıldı
- SQL injection açığı giderildi
- XSS koruması eklendi
- CORS policy güncellendi

---

## [1.5.0] - 2024-10-20

### ✨ Eklenenler

- **ML-based Threat Detection**
  - Random Forest sınıflandırıcı
  - Anomaly detection with Isolation Forest
  - Feature engineering pipeline

- **Advanced Scanning**
  - Deep scan modu
  - Scheduled scans
  - Custom scan profiles

- **Notification System**
  - Email notifications
  - Slack integration
  - Webhook support

### 🔄 Değiştirilenler

- Scanner performance %30 artırıldı
- UI/UX iyileştirmeleri
- Documentation güncellendi

### 🐛 Düzeltilenler

- Network timeout issues
- False positive rate azaltıldı
- Dashboard loading performance

---

## [1.0.0] - 2024-06-01

### 🎉 İlk Stable Sürüm

### ✨ Eklenenler

- **Temel Tarama Modülü**
  - Port scanning
  - Vulnerability detection
  - CVE database integration

- **Web Dashboard**
  - Real-time monitoring
  - Scan history
  - Basic reporting

- **REST API**
  - Authentication
  - Scan management
  - Report generation

- **Database**
  - PostgreSQL support
  - Data persistence
  - Backup system

### 📚 Dokümantasyon

- README.md
- API documentation
- Installation guide
- User manual

---

## Versiyon Numaralandırma

Bu proje Semantic Versioning kullanır:

- **MAJOR** version: Geriye uyumsuz API değişiklikleri
- **MINOR** version: Geriye uyumlu yeni özellikler
- **PATCH** version: Geriye uyumlu hata düzeltmeleri

---

**Son Güncelleme**: 2026-01-10
