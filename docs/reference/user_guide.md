# 📖 User Guide

CyberGuard AI Kullanım Kılavuzu

---

## 📋 İçindekiler

- [Giriş](#giriş)
- [Hızlı Başlangıç](#hızlı-başlangıç)
- [Temel Özellikler](#temel-özellikler)
- [Chatbot Kullanımı](#chatbot-kullanımı)
- [Güvenlik Analizi](#güvenlik-analizi)
- [Raporlama](#raporlama)
- [Ayarlar ve Konfigürasyon](#ayarlar-ve-konfigürasyon)
- [Sorun Giderme](#sorun-giderme)
- [SSS](#sss)

---

## 🎯 Giriş

CyberGuard AI, yapay zeka destekli siber güvenlik çözümü sunan kapsamlı bir platformdur. Bu kılavuz, sistemin tüm özelliklerini etkili bir şekilde kullanmanıza yardımcı olacaktır.

### Hedef Kitle

- 🔒 Siber Güvenlik Uzmanları
- 💼 IT Yöneticileri
- 🛡️ SOC Analistleri
- 👨‍💻 Sistem Yöneticileri

---

## 🚀 Hızlı Başlangıç

### İlk Kurulum

1. **Sisteme Giriş**
   ```bash
   # Backend API (önce başlatılmalı)
   uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

   # Frontend geliştirme sunucusu
   cd frontend && npm run dev   # http://localhost:5173

   # API endpointleri
   http://localhost:8000/api/docs      # Swagger UI
   http://localhost:8000/api/redoc     # ReDoc
   ```

   **Varsayılan Admin Bilgileri:**
   ```bash
   # .env dosyasındaki ADMIN_DEFAULT_PASSWORD değeri
   # Yoksa rastgele üretilir — ilk çalışmada terminale yazdırılır
   Username: admin
   Password: (ADMIN_DEFAULT_PASSWORD ortam değişkeninden)
   ```

   > ⚠️ Güvenlik: Üretim ortamında varsayılan şifreyi `.env`'de güçlü bir değerle değiştirin.

2. **İlk Yapılandırma**
    - Dashboard'a gidin
    - Ayarlar menüsünden temel konfigürasyonu yapın
    - API anahtarlarınızı tanımlayın

3. **İlk Tarama**
    - "New Scan" butonuna tıklayın
    - Hedef sistem bilgilerini girin
    - Tarama tipini seçin
    - Başlat!

### Dashboard Gezintisi

```
┌─────────────────────────────────────┐
│  CyberGuard AI Dashboard            │
├─────────────────────────────────────┤
│  📊 Statistics                      │
│  ├─ Active Threats: 0               │
│  ├─ Total Scans: 0                  │
│  └─ System Health: 100%             │
│                                      │
│  🤖 AI Chatbot                      │
│  ├─ Ask security questions          │
│  └─ Get recommendations             │
│                                      │
│  🔍 Quick Actions                   │
│  ├─ New Scan                        │
│  ├─ View Reports                    │
│  └─ Settings                        │
└─────────────────────────────────────┘
```

---

## ⚙️ Temel Özellikler

### 1. 🤖 AI-Powered Chatbot

**Kullanım Senaryoları:**

- ❓ Güvenlik soruları sorma
- 💡 Tehdit analizi isteme
- 🔍 Log analizi yaptırma
- 📚 Best practice önerileri alma

**Örnek Sorgular:**

```
"Bu log dosyasını analiz et"
"Port 443'teki trafik normal mi?"
"DDoS saldırısına karşı ne yapmalıyım?"
"Sistem güvenliğimi nasıl artırabilirim?"
```

**Chatbot Özellikleri:**

- 🧠 Natural Language Processing
- 📖 Context-aware responses
- 🔄 Multi-turn conversations
- 📊 Data visualization support

### 2. 🔍 Güvenlik Taraması

**Tarama Tipleri:**

1. **Quick Scan**
    - Süre: ~5 dakika
    - Temel güvenlik kontrolleri
    - Açık portlar
    - Yaygın zafiyetler

2. **Deep Scan**
    - Süre: ~30 dakika
    - Kapsamlı güvenlik analizi
    - CVE taraması
    - Konfigürasyon kontrolleri

3. **Custom Scan**
    - Özelleştirilebilir parametreler
    - Belirli servislere odaklı
    - Scheduled taramalar

**Tarama Başlatma:**

```python
# Web UI üzerinden
1. "New Scan" → "Scan Type" seç
2. Target IP/Domain gir
3. Options ayarla
4. "Start Scan" tıkla

# CLI üzerinden
python scan.py --type deep --target 192.168.1.1
```

### 3. 📊 Raporlama ve Analiz

**Rapor Tipleri:**

- 📄 Executive Summary
- 🔬 Technical Details
- 📈 Trend Analysis
- 🎯 Risk Assessment

**Rapor Oluşturma:**

```bash
# PDF rapor
Generate Report → Select Scan → PDF Export

# Excel rapor
Generate Report → Select Scan → Excel Export

# API üzerinden
curl -X POST http://localhost:8000/api/reports \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <token>" \
  -d '{"scan_id": "123", "format": "pdf"}'
```

---

## 💬 Chatbot Kullanımı

### Temel Kullanım

1. **Chatbot'u Açma**
    - Dashboard'dan "AI Assistant" butonuna tıklayın
    - Veya `Ctrl + Space` kısayolunu kullanın

2. **Soru Sorma**
   ```
   User: "Son 24 saatteki güvenlik olaylarını göster"
   Bot: "Son 24 saatte 3 güvenlik olayı tespit edildi..."
   ```

3. **Dosya Yükleme**
    - Log dosyalarını drag & drop yapın
    - Chatbot otomatik analiz yapar

### Gelişmiş Özellikler

**1. Context Management**
```
User: "192.168.1.100 IP adresini analiz et"
Bot: "Analiz ediyorum..."

User: "Bu IP için port taraması yap"  # Context'i hatırlar
Bot: "Port taraması başlatılıyor..."
```

**2. Multi-modal Inputs**
```
- 📝 Text queries
- 📁 File uploads (logs, configs)
- 🖼️ Screenshot analysis
- 📊 Data visualization requests
```

**3. Command Shortcuts**
```
/scan <target>          # Quick scan başlat
/report <scan_id>       # Rapor göster
/threats                # Aktif tehditleri listele
/help                   # Yardım menüsü
```

---

## 🔒 Güvenlik Analizi

### Zafiyet Tespiti

**Desteklenen Zafiyet Tipleri:**

- 🔓 Open Ports
- 🐛 Software Vulnerabilities (CVE)
- ⚙️ Misconfigurations
- 🔑 Weak Credentials
- 🌐 Web Application Flaws

**Zafiyet Skorlama:**

```
Critical (9.0-10.0)  🔴 - Acil müdahale gerekli
High     (7.0-8.9)   🟠 - Yüksek öncelikli
Medium   (4.0-6.9)   🟡 - Orta öncelikli
Low      (0.1-3.9)   🟢 - Düşük öncelikli
```

### Tehdit İzleme

**Real-time Monitoring:**

```python
# Dashboard'dan izleme
Monitoring → Real-time Feed

# Görüntülenecek bilgiler:
- Network traffic anomalies
- Failed login attempts
- Suspicious file changes
- Port scan detections
```

**Alert Konfigürasyonu:**

```yaml
# alert_config.yaml
alerts:
  - type: critical_vulnerability
    action: email + slack
    threshold: 8.0
  
  - type: failed_login
    action: email
    threshold: 5 attempts
  
  - type: port_scan
    action: block_ip
    duration: 1h
```

---

## 📈 Raporlama

### Rapor Şablonları

**1. Executive Summary**
- 👔 Yönetici seviyesi
- 📊 High-level istatistikler
- 🎯 Ana bulgular
- 💰 Risk analizi

**2. Technical Report**
- 🔧 Detaylı teknik bilgiler
- 📝 CVE detayları
- 🛠️ Remediation steps
- 📜 Log örnekleri

**3. Compliance Report**
- ✅ Standart uyumluluk (ISO 27001, PCI DSS)
- 📋 Kontrol listesi
- 🚦 Uyumluluk durumu

### Özel Rapor Oluşturma

```python
# Custom report template
{
  "title": "Quarterly Security Assessment",
  "sections": [
    "executive_summary",
    "vulnerability_overview",
    "threat_analysis",
    "recommendations"
  ],
  "filters": {
    "date_range": "last_90_days",
    "severity": ["high", "critical"]
  }
}
```

---

## ⚙️ Ayarlar ve Konfigürasyon

### Sistem Ayarları

**1. Genel Ayarlar**
```yaml
# config/settings.yaml
general:
  language: tr
  timezone: Europe/Istanbul
  theme: dark
  notifications: enabled
```

**2. Tarama Ayarları**
```yaml
scanning:
  max_concurrent_scans: 5
  timeout: 3600
  retry_failed: true
  auto_schedule: false
```

**3. Güvenlik Ayarları**
```yaml
security:
  mfa_enabled: true
  session_timeout: 30m
  password_policy: strong
  api_rate_limit: 100/hour
```

### Kullanıcı Yönetimi

**Rol Tabanlı Erişim:**

| Role | Permissions |
|------|-------------|
| 👑 Admin | Full access |
| 🔧 Analyst | View + Scan |
| 👀 Viewer | View only |
| 🤖 API User | API access |

**Kullanıcı Ekleme:**
```bash
# Web UI'den
Settings → Users → Add New User

# CLI'den
python manage_users.py add --username john --role analyst
```

---

## 🔧 Sorun Giderme

### Yaygın Sorunlar

**1. Chatbot Yanıt Vermiyor**

```bash
# Çözüm 1: Servis restart
systemctl restart cyberguard-chatbot

# Çözüm 2: Log kontrolü
tail -f logs/chatbot.log

# Çözüm 3: Model cache temizleme
python manage.py clear-cache --component chatbot
```

**2. Tarama Başlatılamıyor**

```bash
# Kontrol adımları:
1. Port erişilebilirliği: telnet target_ip port
2. Credentials doğruluğu: test_connection.py
3. Resource kullanımı: top / htop
4. Log analizi: tail -f logs/scanner.log
```

**3. Yavaş Performans**

```python
# Optimizasyon adımları:
1. Database indexing: python manage.py optimize-db
2. Cache temizleme: python manage.py clear-cache
3. Old scan cleanup: python manage.py cleanup --days 30
4. Resource allocation artırma: config/resources.yaml
```

### Log Dosyaları

```
logs/
├── app/
│   └── cyberguard.log    # JSON yapılandırılmış uygulama logları
│                         # (Rotating: 10MB × 5 kopya)
├── tensorboard/          # Model eğitim metrikleri
└── training/             # Eğitim geçmişi
```

**Log Seviyesini Değiştirme:**
```env
# .env
LOG_LEVEL=DEBUG          # DEBUG, INFO, WARNING, ERROR
JSON_CONSOLE_LOG=true    # Konsolda JSON format (varsayılan: false)
```

**Log Örneği:**
```json
{"timestamp": "2026-04-24T12:00:00Z", "level": "INFO", 
 "request_id": "abc123", "method": "POST", 
 "path": "/api/ml/predict", "status_code": 200, "duration_ms": 45}
```

---

## ❓ SSS (Sıkça Sorulan Sorular)

### Genel Sorular

**Q: CyberGuard AI'yı kimler kullanabilir?**
A: Siber güvenlik uzmanları, IT yöneticileri, SOC analistleri ve sistem yöneticileri.

**Q: Lisans gerekli mi?**
A: CyberGuard AI açık kaynaklıdır, lisans gerekmez.

**Q: Hangi işletim sistemlerinde çalışır?**
A: Linux (Ubuntu 20.04+), Windows 10+, macOS 12+ — Python 3.10+ gerektirir.

### Teknik Sorular

**Q: API rate limit nedir?**
A: `slowapi` ile IP tabanlı rate limiting aktif. Varsayılan eşik `config.yaml`'da ayarlanır.

**Q: Maksimum dosya yükleme boyutu?**
A: `uploads/` klasörüne yükleme desteklenir; boyut kısıtı sunucu konfigürasyonuna bağlıdır.

**Q: Kaç eşzamanlı tarama yapılabilir?**
A: FastAPI async mimarisi ile birden fazla eşzamanlı istek desteklenir; kaynak sınırları sunucuya göre değişir.

**Q: Hangi veritabanları destekleniyor?**
A: SQLite (`src/database/cyberguard.db`) + ChromaDB (vektör store) + JSON dosyaları. PostgreSQL/Redis şu an desteklenmez.

### Güvenlik Soruları

**Q: Veriler nasıl korunuyor?**
A: AES-256 encryption, TLS 1.3, end-to-end encryption

**Q: Multi-factor authentication var mı?**
A: Şu an desteklenmiyor; bcrypt + refresh token + RBAC tabanlı auth aktif.

**Q: Token ne kadar süre geçerli?**
A: Access token: oturum süresi boyunca, Refresh token: 7 gün (otomatik yenilenir).

---

## 📞 Destek ve İletişim

### Destek Kanalları

- � Dokümantasyon: [docs/README.md](../README.md)
- 🐛 Sorunlar: Proje GitHub Issues sayfası
- 🔍 API Docs: `http://localhost:8000/api/docs`

---

## 📚 Ek Kaynaklar

### Video Tutorials

- 🎥 [Getting Started (10 min)](https://youtube.com/watch?v=xxx)
- 🎥 [Advanced Scanning (15 min)](https://youtube.com/watch?v=yyy)
- 🎥 [Chatbot Best Practices (8 min)](https://youtube.com/watch?v=zzz)

### Dokümantasyon

- 📖 [API Reference](../api/api_reference.md)
- 🏗️ [Architecture Guide](../architecture/architecture.md)
- 🚀 [Deployment Guide](../operations/deployment.md)

### Blog Yazıları

- 📝 "10 Tips for Effective Security Scanning"
- 📝 "How AI Improves Threat Detection"
- 📝 "Building a SOC with CyberGuard AI"

---

## 🔄 Sürüm Geçmişi

- **v2.0.0** (2025-01) - AI Chatbot entegrasyonu
- **v1.5.0** (2024-10) - ML-based threat detection
- **v1.0.0** (2024-06) - İlk stable sürüm

---

## 📄 Lisans

Bu yazılım MIT lisansı altında dağıtılmaktadır.

---

**🎉 CyberGuard AI'yı seçtiğiniz için teşekkürler!**

*Bu kılavuz sürekli güncellenmektedir. Son sürüm için:*
*https://docs.cyberguard-ai.com/user-guide*