# ❓ Sıkça Sorulan Sorular (FAQ)

CyberGuard AI hakkında en çok sorulan sorular ve cevapları

---

## 📋 İçindekiler

- [Genel Sorular](#genel-sorular)
- [Kurulum](#kurulum)
- [Kullanım](#kullanım)
- [ML/AI](#mlai)
- [API](#api)
- [Güvenlik](#güvenlik)
- [Performans](#performans)
- [Lisans ve Destek](#lisans-ve-destek)

---

## 🌟 Genel Sorular

### CyberGuard AI nedir?

CyberGuard AI, yapay zeka destekli siber güvenlik platformudur. SSA-LSTMIDS modeli ile ağ trafiğindeki saldırıları %99+ doğrulukla tespit eder.

### Hangi saldırı türlerini tespit edebilir?

- DDoS (Distributed Denial of Service)
- Port Scanning
- Brute Force
- SQL Injection
- XSS (Cross-Site Scripting)
- Malware
- Botnet aktivitesi
- Ve 15+ diğer saldırı türü

### Hangi veri setleri üzerinde eğitildi?

| Dataset | Kayıt | Accuracy |
|---------|-------|----------|
| NSL-KDD | 148K | 99.36% |
| CICIDS2017 | 2.8M | 99.88% |
| BoT-IoT | 73M | 99.99% |

### Ücretsiz mi?

Community edition ücretsiz ve açık kaynak. Enterprise özellikleri için lisans gereklidir.

---

## 🔧 Kurulum

### Minimum sistem gereksinimleri neler?

| Bileşen | Minimum | Önerilen |
|---------|---------|----------|
| CPU | 4 cores | 8+ cores |
| RAM | 8 GB | 16+ GB |
| Disk | 50 GB SSD | 100+ GB SSD |
| GPU | - | NVIDIA CUDA |

### Hangi işletim sistemlerinde çalışır?

- Windows 10/11, Windows Server 2019+
- Ubuntu 20.04+, CentOS 8+
- macOS 11+
- Docker (herhangi bir platform)

### Python versiyonu?

Python 3.9+ gereklidir. Python 3.11 önerilir.

### Kurulum ne kadar sürer?

- Tam kurulum: 10-15 dakika
- Docker: 5 dakika
- Model indirme: 5-10 dakika (opsiyonel)

### Kurulum hatası alıyorum, ne yapmalıyım?

1. Python versiyonunu kontrol edin: `python --version`
2. Virtual environment aktif mi: `which python`
3. Bağımlılıkları yeniden yükleyin: `pip install -r requirements.txt`
4. Detaylı log: `pip install -r requirements.txt -v`

Bkz: [Troubleshooting](troubleshooting.md)

---

## 💻 Kullanım

### Backend'i nasıl başlatırım?

```bash
cd app
python -m uvicorn main:app --reload
# http://localhost:8000
```

### Frontend'i nasıl başlatırım?

```bash
cd frontend
npm run dev
# http://localhost:5173
```

### API dokümantasyonuna nasıl erişirim?

Backend çalışırken: `http://localhost:8000/api/docs`

### Varsayılan kullanıcı bilgileri nedir?

```
Username: admin
Password: admin123
```

⚠️ İlk girişte şifreyi değiştirin!

### Dashboard'da veriler neden boş görünüyor?

1. Database migration çalıştırın
2. Mock data oluşturun: `python scripts/generate_mock_data.py`
3. API bağlantısını kontrol edin

---

## 🧠 ML/AI

### Hangi ML modelleri kullanılıyor?

| Model | Tür | Accuracy |
|-------|-----|----------|
| SSA-LSTMIDS | Deep Learning | 99.88% |
| BiLSTM | Deep Learning | 99.12% |
| Random Forest | Ensemble | 97.45% |
| XGBoost | Ensemble | 97.21% |

### Model eğitimi ne kadar sürer?

| Dataset | GPU | CPU |
|---------|-----|-----|
| NSL-KDD | 30 min | 2 hours |
| CICIDS2017 | 2 hours | 8 hours |
| BoT-IoT | 4 hours | 16 hours |

### GPU olmadan çalışır mı?

Evet, ama eğitim çok daha yavaş olur. Inference CPU'da sorunsuz çalışır.

### Kendi modelimi eğitebilir miyim?

Evet! Bkz: [Model Training Guide](model_training_guide.md)

```python
python scripts/train_custom_model.py --dataset /path/to/data.csv
```

### XAI (Açıklanabilir AI) nedir?

Model kararlarını açıklamak için SHAP ve LIME kullanıyoruz. Bu sayede modelin neden belirli bir tahminde bulunduğunu anlayabilirsiniz.

Bkz: [XAI Documentation](xai.md)

---

## 🔌 API

### Kaç endpoint var?

150+ endpoint mevcut. Bkz: [API Endpoints Full](api_endpoints_full.md)

### Rate limit nedir?

| Plan | Limit |
|------|-------|
| Community | 100 req/dakika |
| Professional | 1000 req/dakika |
| Enterprise | Sınırsız |

### API key nasıl oluştururum?

```bash
# Web UI
Settings → API Keys → Create New Key

# API
POST /api/keys
{"name": "My API Key", "permissions": ["read", "write"]}
```

### Hangi response formatı kullanılıyor?

JSON formatında standart response:

```json
{
  "success": true,
  "data": {...},
  "message": "İşlem başarılı"
}
```

---

## 🔐 Güvenlik

### Veriler şifreleniyor mu?

Evet, AES-256 encryption kullanılıyor. Transit'te TLS 1.3.

### MFA destekleniyor mu?

Evet, TOTP (Google Authenticator vb.) desteklenir.

### GDPR/KVKK uyumlu mu?

Tasarım gereği uyumlu. Kişisel veri minimum tutulur.

### Güvenlik açığı bulursam ne yapmalıyım?

Lütfen `security@cyberguard-ai.com` adresine bildirin. Bkz: [Security Policy](SECURITY_POLICY.md)

---

## ⚡ Performans

### Ne kadar trafik işleyebilir?

- Single node: 10K req/s
- Cluster: 100K+ req/s

### Bellek kullanımı ne kadar?

- Backend: 500MB-2GB
- Frontend: 100-300MB
- Model inference: 1-4GB

### Yavaş çalışıyor, ne yapmalıyım?

1. Database indekslerini kontrol edin
2. Redis cache aktif mi?
3. Model warmup yapın
4. Resource limitlerini artırın

Bkz: [Performance Tuning](performance_tuning.md)

---

## 📜 Lisans ve Destek

### Lisans türü nedir?

MIT License - Ticari kullanıma açık.

### Destek nasıl alabilirim?

| Kanal | Süre |
|-------|------|
| GitHub Issues | 24-48 saat |
| Email | 24-48 saat |
| Discord | Canlı |
| Enterprise | SLA |

### Katkıda bulunabilir miyim?

Evet! Bkz: [Contributing](contributing.md)

---

## 🔗 Daha Fazla Kaynak

- [Kurulum Rehberi](installation.md)
- [Kullanıcı Rehberi](user_guide.md)
- [API Reference](api_reference.md)
- [Troubleshooting](troubleshooting.md)
