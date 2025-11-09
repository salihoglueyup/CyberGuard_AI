# 📝 Değişiklik Günlüğü

Bu dosya, CyberGuard AI projesindeki tüm önemli değişiklikleri dokümante eder.

Format [Keep a Changelog](https://keepachangelog.com/tr/1.0.0/) standardına dayanır ve bu proje [Semantic Versioning](https://semver.org/lang/tr/) kullanır.

## [Yayınlanmamış]

### Eklenenler
- Yeni özellikler buraya eklenecek

### Değiştirilenler
- Mevcut özelliklerdeki değişiklikler

### Düzeltilenler
- Hata düzeltmeleri

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

## [0.9.0-beta] - 2024-04-15

### ✨ Eklenenler
- Beta release
- Core scanning functionality
- Basic UI
- PostgreSQL integration

### 🐛 Düzeltilenler
- Critical bug fixes
- Performance improvements
- Security patches

---

## [0.5.0-alpha] - 2024-02-10

### ✨ Eklenenler
- Alpha release
- Proof of concept
- Basic port scanner
- Simple CLI interface

---

## Versiyon Numaralandırma

Bu proje Semantic Versioning kullanır:

- **MAJOR** version: Geriye uyumsuz API değişiklikleri
- **MINOR** version: Geriye uyumlu yeni özellikler
- **PATCH** version: Geriye uyumlu hata düzeltmeleri

### Değişiklik Kategorileri

- **Eklenenler**: Yeni özellikler
- **Değiştirilenler**: Mevcut özelliklerdeki değişiklikler
- **Kullanımdan Kaldırılanlar**: Yakında kaldırılacak özellikler
- **Kaldırılanlar**: Artık mevcut olmayan özellikler
- **Düzeltilenler**: Hata düzeltmeleri
- **Güvenlik**: Güvenlik açıkları ve düzeltmeleri

---

## Planlanan Özellikler (Roadmap)

Gelecek sürümler için planlar için [ROADMAP.md](ROADMAP.md) dosyasına bakın.

---

## Changelog Katkıları

Changelog güncellemelerini yaparken:

1. Değişikliği uygun kategoriye ekleyin
2. Açıklayıcı ve net yazın
3. İlgili issue/PR numaralarını ekleyin
4. Tarih formatı: YYYY-MM-DD

**Örnek:**
```markdown
### Eklenenler
- Yeni chatbot NLP modeli (#123) @username
- Real-time notification sistemi (#124) @username
```

---

[Yayınlanmamış]: https://github.com/cyberguard-ai/compare/v2.0.0...HEAD
[2.0.0]: https://github.com/cyberguard-ai/compare/v1.5.0...v2.0.0
[1.5.0]: https://github.com/cyberguard-ai/compare/v1.0.0...v1.5.0
[1.0.0]: https://github.com/cyberguard-ai/compare/v0.9.0-beta...v1.0.0
[0.9.0-beta]: https://github.com/cyberguard-ai/compare/v0.5.0-alpha...v0.9.0-beta
[0.5.0-alpha]: https://github.com/cyberguard-ai/releases/tag/v0.5.0-alpha