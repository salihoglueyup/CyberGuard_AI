# 📝 Changelog (Değişiklik Günlüğü)

Bu dosya, CyberGuard AI projesindeki tüm önemli değişiklikleri dokümante eder.

Format [Keep a Changelog](https://keepachangelog.com/tr/1.0.0/) standardına dayanır ve bu proje [Semantic Versioning](https://semver.org/lang/tr/) kullanır.

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
