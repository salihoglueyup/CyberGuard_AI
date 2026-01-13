# 🛡️ CyberGuard AI - Türkçe Kullanım Kılavuzu

> **Versiyon:** 2.0  
> **Güncelleme:** Ocak 2026  
> **Platform:** Windows / Linux / macOS

---

## 📋 İçindekiler

1. [Başlarken](#-başlarken)
2. [Sistem Gereksinimleri](#-sistem-gereksinimleri)
3. [Kurulum](#-kurulum)
4. [Modül Açıklamaları](#-modül-açıklamaları)
5. [Kullanım Senaryoları](#-kullanım-senaryoları)
6. [Sık Sorulan Sorular](#-sık-sorulan-sorular)

---

## 🚀 Başlarken

CyberGuard AI, yapay zeka destekli bir siber güvenlik platformudur. Ağ trafiğini izler, tehditleri tespit eder ve otomatik yanıt mekanizmaları sunar.

### Hızlı Başlangıç

```bash
# 1. Backend'i başlat
cd app
python -m uvicorn main:app --reload --host 0.0.0.0 --port 8000

# 2. Frontend'i başlat (yeni terminal)
cd frontend
npm run dev
```

**Erişim Adresleri:**

- 🖥️ Frontend: <http://localhost:5173>
- 🔌 Backend API: <http://localhost:8000>
- 📚 API Docs: <http://localhost:8000/api/docs>

---

## 💻 Sistem Gereksinimleri

| Bileşen  | Minimum    | Önerilen    |
| -------- | ---------- | ----------- |
| RAM      | 4 GB       | 8+ GB       |
| CPU      | 2 çekirdek | 4+ çekirdek |
| Disk     | 5 GB       | 20+ GB      |
| Python   | 3.9+       | 3.11+       |
| Node.js  | 18+        | 20+         |

### Gerekli Yazılımlar

- Python 3.9+
- Node.js 18+
- Git
- (İsteğe bağlı) Docker Desktop

---

## 📦 Kurulum

### 1. Projeyi İndirin

```bash
git clone https://github.com/your-repo/CyberGuard_AI.git
cd CyberGuard_AI
```

### 2. Python Bağımlılıklarını Yükleyin

```bash
pip install -r requirements.txt
```

### 3. Frontend Bağımlılıklarını Yükleyin

```bash
cd frontend
npm install
```

### 4. Ortam Değişkenlerini Ayarlayın

`.env` dosyası oluşturun:

```env
# API Anahtarları (opsiyonel)
GROQ_API_KEY=your_groq_key
VIRUSTOTAL_API_KEY=your_vt_key
OPENAI_API_KEY=your_openai_key

# Veritabanı
DATABASE_URL=sqlite:///./cyberguard.db
```

---

## 📊 Modül Açıklamaları

### 🏠 Dashboard (Ana Sayfa)

**Amaç:** Genel güvenlik durumunu tek bakışta görme

**Özellikler:**

- Canlı tehdit sayısı
- Son 24 saat istatistikleri
- Sistem durumu göstergeleri
- Hızlı erişim kısayolları

**Nasıl Kullanılır:**

1. <http://localhost:5173> adresine gidin
2. Dashboard otomatik olarak yüklenir
3. İstatistikler gerçek zamanlı güncellenir

---

### 🌍 Attack Map (Saldırı Haritası)

**Amaç:** Dünya genelindeki saldırıları görselleştirme

**Özellikler:**

- 2D/3D harita görünümü
- Gerçek zamanlı saldırı akışı
- Ülke bazlı istatistikler
- Tehdit seviyesi renk kodlaması

**Nasıl Kullanılır:**

1. Sol menüden "Saldırı Haritası" seçin
2. Sağ üstten 2D/3D moduna geçin
3. Ülkelere tıklayarak detay görün
4. "Canlı Güncelle" ile gerçek zamanlı izleyin

---

### 🔍 Malware Scanner (Zararlı Tarayıcı)

**Amaç:** Dosyaları zararlı yazılımlara karşı tarama

**Özellikler:**

- Dosya yükleme ve tarama
- Hash tabanlı analiz
- VirusTotal entegrasyonu
- Statik analiz sonuçları

**Nasıl Kullanılır:**

1. "Tarayıcı" sayfasına gidin
2. Dosyayı sürükle-bırak veya seç
3. "Tara" butonuna tıklayın
4. Sonuçları inceleyin

---

### 🌐 Network Monitor (Ağ İzleme)

**Amaç:** Ağ trafiğini gerçek zamanlı izleme

**Özellikler:**

- Aktif bağlantılar listesi
- Bandwidth kullanımı
- Interface detayları
- Anomali tespiti

**Nasıl Kullanılır:**

1. "Ağ" menüsüne gidin
2. Aktif interface'leri görün
3. İndirme/yükleme hızlarını izleyin
4. Şüpheli bağlantıları filtreleyin

---

### 🤖 AI Assistant (Yapay Zeka Asistan)

**Amaç:** Güvenlik sorularına AI destekli yanıt

**Özellikler:**

- Doğal dil işleme
- Güvenlik önerileri
- Log analizi
- Tehdit açıklamaları

**Nasıl Kullanılır:**

1. "AI Asistan" sayfasına gidin
2. Sorunuzu yazın (örn: "Bu IP zararlı mı?")
3. Enter tuşuna basın
4. AI yanıtını okuyun

**Örnek Sorular:**

- "192.168.1.100 IP adresi hakkında bilgi ver"
- "DDoS saldırısına karşı ne yapmalıyım?"
- "Log dosyasındaki bu hatayı açıkla"

---

### 📊 ML Models (Makine Öğrenimi)

**Amaç:** Tehdit tespiti için ML modellerini yönetme

**Özellikler:**

- Model eğitimi
- Performans metrikleri
- Model karşılaştırma
- Tahmin yapma

**Nasıl Kullanılır:**

1. "ML Modeller" sayfasına gidin
2. Mevcut modelleri inceleyin
3. "Eğit" ile yeni model oluşturun
4. "Test Et" ile performans ölçün

---

### 🎯 Threat Hunting (Tehdit Avcılığı)

**Amaç:** Proaktif tehdit araştırması

**Özellikler:**

- Sorgu tabanlı arama
- Hazır şablonlar
- IOC arama
- Soruşturma yönetimi

**Nasıl Kullanılır:**

1. "Tehdit Avcılığı" sayfasına gidin
2. Sorgu yazın veya şablon seçin
3. Zaman aralığı belirleyin
4. "Hunt Başlat" tıklayın
5. Sonuçları inceleyin

**Örnek Sorgular:**

```sql
# Brute force tespiti
failed login | authentication failure

# Veri sızıntısı
upload | POST | large transfer

# Zararlı aktivite
malware | virus | trojan
```

---

### 🔐 Security Hub (Güvenlik Merkezi)

**Amaç:** Genel güvenlik durumu ve uyumluluk

**Özellikler:**

- Güvenlik skoru (A-F)
- Uyumluluk kontrolleri
- Ağ topolojisi
- Bal küpü izleme

**Nasıl Kullanılır:**

1. "Güvenlik Merkezi" sayfasına gidin
2. Genel skoru inceleyin
3. Sekmelerde detaylara bakın
4. Önerileri uygulayın

---

### 📦 Container Security (Konteyner Güvenliği)

**Amaç:** Docker konteyner ve imajlarını tarama

**Özellikler:**

- Container listesi
- İmaj güvenlik taraması
- Açıklık tespiti
- CVE raporlama

**Ön Koşul:** Docker Desktop çalışıyor olmalı

**Nasıl Kullanılır:**

1. Docker Desktop'ı başlatın
2. "Container Güvenlik" sayfasına gidin
3. İmaj adı girin ve "Tara" tıklayın
4. Güvenlik açıklarını inceleyin

---

### 🔗 SIEM Integration (SIEM Entegrasyonu)

**Amaç:** Harici SIEM sistemlerine bağlanma

**Desteklenen Platformlar:**

- Splunk Enterprise
- Elastic SIEM
- IBM QRadar
- Microsoft Sentinel
- Wazuh

**Nasıl Kullanılır:**

1. "SIEM" sayfasına gidin
2. Platform seçin
3. Bağlantı bilgilerini girin
4. "Bağlan" tıklayın
5. Event forwarding kuralları oluşturun

---

### 🧪 Sandbox (Kum Havuzu)

**Amaç:** Şüpheli dosyaları izole ortamda analiz

**Özellikler:**

- Dosya yükleme
- Statik analiz
- VirusTotal entegrasyonu
- Risk skorlama

**Nasıl Kullanılır:**

1. "Sandbox" sayfasına gidin
2. Dosya yükleyin
3. Analiz sonuçlarını bekleyin
4. Tehdit raporunu inceleyin

---

### ⛓️ Blockchain Audit (Değişmez Kayıt)

**Amaç:** Güvenlik olaylarının değiştirilemez kaydı

**Özellikler:**

- Olay zinciri
- Hash doğrulama
- Arama
- Bütünlük kontrolü

**Nasıl Kullanılır:**

1. "Blockchain" sayfasına gidin
2. Son blokları inceleyin
3. "Doğrula" ile bütünlük kontrolü yapın
4. Arama ile geçmiş olayları bulun

---

## 📚 Kullanım Senaryoları

### Senaryo 1: Günlük Güvenlik Kontrolü

```bash
1. Dashboard'u açın → Genel durumu kontrol edin
2. Attack Map'e bakın → Aktif tehditleri görün
3. Network Monitor → Şüpheli bağlantıları kontrol edin
4. Security Hub → Güvenlik skorunuzu görün
```

### Senaryo 2: Şüpheli Dosya Analizi

```bash
1. Sandbox'a gidin
2. Dosyayı yükleyin
3. Analiz sonucunu bekleyin
4. Risk skoru yüksekse:
   - AI Assistant'a sorun
   - Threat Hunting yapın
```

### Senaryo 3: Olay Araştırması

```bash
1. Threat Hunting sayfasına gidin
2. Şablon seçin veya sorgu yazın
3. Eşleşmeleri inceleyin
4. Blockchain'de ilgili logları doğrulayın
5. Rapor oluşturun
```

### Senaryo 4: SIEM Entegrasyonu

```bash
1. SIEM sayfasına gidin
2. Platformunuzu seçin (Splunk vb.)
3. API bilgilerini girin
4. Bağlantıyı test edin
5. Forwarding kurallarını aktifleştirin
```

---

## ❓ Sık Sorulan Sorular

### Backend başlamıyor?

```bash
# Port kullanımda olabilir
netstat -ano | findstr :8000
# Farklı port kullanın
uvicorn main:app --port 8001
```

### Frontend hatası alıyorum?

```bash
# Node modules'ü temizleyin
rm -rf node_modules
npm install
npm run dev
```

### AI Assistant yanıt vermiyor?

- `.env` dosyasında `GROQ_API_KEY` veya `OPENAI_API_KEY` olduğundan emin olun
- API limitlerinizi kontrol edin

### Docker bağlantısı yok?

- Docker Desktop'ın çalıştığından emin olun
- WSL2 entegrasyonunu kontrol edin

### 404 hatası alıyorum?

- Backend'in çalıştığından emin olun
- `http://localhost:8000/api/docs` erişilebilir mi kontrol edin

---

## 📞 Destek

**Hata Bildirimi:** GitHub Issues  
**Dokümantasyon:** `/docs` klasörü  
**API Referans:** <http://localhost:8000/api/docs>

---

## 🔐 Güvenlik İpuçları

1. ✅ API anahtarlarını `.env` dosyasında saklayın
2. ✅ `.env` dosyasını git'e eklemeyin
3. ✅ Güçlü parolalar kullanın
4. ✅ Düzenli güncelleme yapın
5. ✅ Log dosyalarını düzenli inceleyin

---

**🛡️ CyberGuard AI ile güvende kalın!**
