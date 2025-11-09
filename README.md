# 🛡️ CyberGuard AI

<div align="center">

![CyberGuard AI](https://img.shields.io/badge/CyberGuard-AI-blue?style=for-the-badge&logo=security&logoColor=white)
![Python](https://img.shields.io/badge/Python-3.10+-blue?style=for-the-badge&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

**Yapay Zeka Destekli Siber Güvenlik Platformu**

*AI-Powered Cybersecurity Platform with RAG, Machine Learning & Real-time Threat Detection*

[Özellikler](#-özellikler) • [Kurulum](#-kurulum) • [Kullanım](#-kullanım) • [Demo](#-demo) • [Katkıda Bulun](#-katkıda-bulun)

</div>

---

## 📋 İçindekiler

- [Genel Bakış](#-genel-bakış)
- [Özellikler](#-özellikler)
- [Teknoloji Stack](#-teknoloji-stack)
- [Kurulum](#-kurulum)
- [Kullanım](#-kullanım)
- [Proje Yapısı](#-proje-yapısı)
- [Model Eğitimi](#-model-eğitimi)
- [API Dokümantasyonu](#-api-dokümantasyonu)
- [Ekran Görüntüleri](#-ekran-görüntüleri)
- [Katkıda Bulunma](#-katkıda-bulun)
- [Lisans](#-lisans)

---

## 🌟 Genel Bakış

**CyberGuard AI**, siber güvenlik tehditlerini tespit etmek, analiz etmek ve önlemek için yapay zeka ve derin öğrenme teknolojilerini kullanan kapsamlı bir güvenlik platformudur.

### 🎯 Proje Hedefleri

- ✅ **Gerçek Zamanlı Tehdit Tespiti**: Ağ trafiğini anlık izleme ve anomali tespiti
- ✅ **AI Destekli Analiz**: Machine Learning ile saldırı pattern'lerini öğrenme
- ✅ **Akıllı Chatbot**: RAG + Memory ile context-aware güvenlik asistanı
- ✅ **Kapsamlı Raporlama**: PDF raporlar ve detaylı istatistikler
- ✅ **Kullanıcı Dostu Arayüz**: Modern, responsive ve interaktif dashboard

---

## 🚀 Özellikler

### 1️⃣ Security Dashboard
- 📊 **5 Ana KPI**: Toplam saldırı, son 24 saat, engelleme oranı, kritik tehditler
- 📈 **Gerçek Zamanlı Grafikler**: Saatlik trend, saldırı dağılımı, severity analizi
- 🔄 **Otomatik Yenileme**: Canlı izleme (5s, 10s, 30s, 60s)
- 🎯 **Top 10 Saldırganlar**: En aktif IP adresleri
- 🌐 **Port Analizi**: En çok hedeflenen portlar

### 2️⃣ Network Monitor
- 🔍 **4 Tab Yapısı**: Canlı saldırılar, analiz & grafikler, IP sorgulama, port aktivitesi
- 🌐 **Canlı İzleme**: Otomatik yenileme ile real-time monitoring
- 🔎 **IP Analizi**: Detaylı IP geçmişi ve risk skoru
- 📊 **İstatistiksel Analiz**: Saldırı türü, severity, zaman serisi

### 3️⃣ Malware Scanner
- 📁 **Dosya Tarama**: Upload ve gerçek zamanlı analiz
- 🎯 **Risk Skoru**: 0-100 arası detaylı değerlendirme
- 🧬 **İmza Tespiti**: Malware signature detection
- 🗑️ **Karantina Sistemi**: Tehlikeli dosyaları izole etme
- 📊 **Tarama Geçmişi**: Grafik ve istatistiklerle

### 4️⃣ AI Assistant (🔥 En Güçlü Özellik!)
- 🤖 **Gemini Pro Entegrasyonu**: Google'ın en yeni LLM modeli
- 📚 **RAG Sistemi**: Döküman yükleme ve akıllı arama
- 🧠 **Konuşma Hafızası**: Önceki konuşmaları hatırlama
- 🎯 **Attack Vector Analysis**: Database'deki saldırılarla benzerlik analizi
- 💡 **Hızlı Sorular**: Pre-defined sorular ile kolay kullanım
- 🔍 **Context-Aware**: IP, saldırı, sistem bilgilerini otomatik ekler

### 5️⃣ ML Tahmin Sistemi
- 🎯 **%100 Accuracy**: 5000 veri ile eğitilmiş Random Forest modeli
- 🔮 **6 Saldırı Türü**: DDoS, Port Scan, SQL Injection, Brute Force, XSS, Malware
- 📊 **Feature Importance**: En önemli özellikleri gösterir
- 🎨 **Görsel Raporlar**: Confusion matrix, feature importance grafikleri
- ⚡ **Hızlı Test**: Test butonları ile anlık tahmin

### 6️⃣ PDF Rapor Sistemi
- 📄 **Otomatik Raporlar**: 1, 7, 30, 90 günlük dönemler
- 📊 **Grafikler Dahil**: Pie chart, bar chart, tablolar
- 🎯 **Detaylı Analiz**: Saldırı türü, severity, top IP'ler
- 💾 **Tek Tıkla İndirme**: Streamlit download button ile

---

## 🛠️ Teknoloji Stack

### Backend
```python
# Core
Python 3.10+
Streamlit 1.32.0
SQLite (embedded)

# Machine Learning
Scikit-learn 1.5.2      # Random Forest Model
TensorFlow 2.15.0        # Deep Learning (future)

# AI & LLM
Google Gemini Pro        # Chatbot
LangChain               # RAG Framework
ChromaDB                # Vector Database
Sentence Transformers   # Embeddings
```

### Frontend
```javascript
Streamlit UI
Plotly (Interactive Charts)
Matplotlib & Seaborn (Static Charts)
Custom CSS (Gradients, Animations)
```

### Data Processing
```python
Pandas      # Data manipulation
NumPy       # Numerical computing
ReportLab   # PDF generation
PyPDF2      # PDF reading
```

---

## 📦 Kurulum

### Ön Gereksinimler

- Python 3.10 veya üzeri
- pip (Python package manager)
- Git
- Google Gemini API Key ([buradan alın](https://makersuite.google.com/app/apikey))

### Adım 1: Repository'yi Klonlayın

```bash
git clone https://github.com/yourusername/CyberGuard_AI.git
cd CyberGuard_AI
```

### Adım 2: Virtual Environment Oluşturun

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Mac/Linux
python3 -m venv venv
source venv/bin/activate
```

### Adım 3: Gerekli Paketleri Kurun

```bash
pip install -r requirements.txt
```

### Adım 4: Ortam Değişkenlerini Ayarlayın

`.env` dosyası oluşturun:

```bash
GOOGLE_API_KEY=your_gemini_api_key_here
```

### Adım 5: Mock Veri Oluşturun (Opsiyonel)

```bash
python src/utils/mock_data_generator.py
```

### Adım 6: ML Modelini Eğitin

```bash
python train_model.py
```

### Adım 7: Uygulamayı Başlatın

```bash
cd app
streamlit run main.py
```

🎉 **Tebrikler!** Uygulama `http://localhost:8501` adresinde çalışıyor!

---

## 🎮 Kullanım

### Dashboard
1. Ana sayfadan **"📊 Dashboard"** seçin
2. Otomatik yenileme için toggle'ı açın
3. PDF rapor için sidebar'dan **"📥 PDF Rapor İndir"** tıklayın

### Network Monitor
1. **"🔍 Network Monitor"** sayfasına gidin
2. Canlı izleme için toggle'ı açın
3. IP sorgulama için IP adresi girin

### Malware Scanner
1. **"🦠 Malware Scanner"** sayfasına gidin
2. Dosya yükleyin veya test butonlarını kullanın
3. Tarama sonuçlarını inceleyin

### AI Assistant
1. **"🤖 AI Assistant"** sayfasına gidin
2. Hızlı sorular veya chat input kullanın
3. Döküman yüklemek için **"📚 Döküman Yönetimi"** tab'ına gidin
4. Saldırı vektörlerini analiz için **"🎯 Saldırı Vektörleri"** tab'ını kullanın

### ML Tahmin
1. **"🔮 ML Tahmin"** sayfasına gidin
2. Saldırı bilgilerini girin veya test butonları kullanın
3. Tahmin sonuçlarını ve risk skorunu görün

---

## 📂 Proje Yapısı

```
CyberGuard_AI/
│
├── app/                          # Streamlit uygulaması
│   ├── main.py                   # Ana uygulama
│   └── pages/                    # Sayfa modülleri
│       ├── dashboard.py
│       ├── network_monitor.py
│       ├── malware_scanner.py
│       ├── ai_assistant.py
│       └── ml_prediction.py
│
├── src/                          # Kaynak kodlar
│   ├── chatbot/                  # Chatbot modülleri
│   │   ├── gemini_handler.py     # Gemini API
│   │   └── vectorstore/          # RAG sistemi
│   │       ├── rag_manager.py
│   │       ├── memory_manager.py
│   │       └── attack_vectors.py
│   │
│   ├── models/                   # ML modelleri
│   │   ├── random_forest_model.py
│   │   └── predictor.py
│   │
│   └── utils/                    # Yardımcı fonksiyonlar
│       ├── database.py
│       ├── logger.py
│       ├── config.py
│       ├── visualizer.py
│       ├── mock_data_generator.py
│       ├── feature_extractor.py
│       └── pdf_generator.py
│
├── models/                       # Eğitilmiş modeller
│   ├── rf_model.pkl
│   ├── feature_extractor.pkl
│   └── *.png                     # Grafik çıktıları
│
├── data/                         # Veri dosyaları (opsiyonel)
│
├── cyberguard.db                 # SQLite veritabanı
├── config.yaml                   # Konfigürasyon
├── .env                          # Ortam değişkenleri
├── requirements.txt              # Python bağımlılıkları
├── train_model.py               # Model eğitim scripti
└── README.md                     # Bu dosya
```
## 📚 Documentation

Detaylı dokümantasyon için `docs/` klasörüne göz atın:

- 📘 [API Reference](docs/api_reference.md) - API endpoint'leri ve kullanım örnekleri
- 🏗️ [Architecture](docs/architecture.md) - Sistem mimarisi ve component yapısı
- 🚀 [Deployment Guide](docs/deployment.md) - Kurulum ve deployment adımları
- 📖 [User Guide](docs/user_guide.md) - Kapsamlı kullanım kılavuzu

Ek kaynaklar için [Wiki](https://github.com/username/cyberguard-ai/wiki) sayfasını ziyaret edin.
---

## 🧠 Model Eğitimi

### Veri Seti

- **Mock Data**: 5000 saldırı kaydı
- **Sınıflar**: 6 farklı saldırı türü
- **Özellikler**: 8 feature (IP, port, severity, time vb.)

### Model Performansı

```
Model: Random Forest (100 trees)
Accuracy: 100.00%
Precision: 1.00
Recall: 1.00
F1-Score: 1.00
```

### Yeniden Eğitim

```bash
# Yeni veri oluştur
python src/utils/mock_data_generator.py

# Modeli eğit
python train_model.py
```

---

## 📊 Ekran Görüntüleri

### Dashboard
![Dashboard](screenshots/dashboard.png)
*Gerçek zamanlı güvenlik durumu ve istatistikler*

### AI Assistant
![AI Assistant](screenshots/ai_assistant.png)
*RAG + Memory destekli akıllı chatbot*

### ML Tahmin
![ML Prediction](screenshots/ml_prediction.png)
*%100 accuracy ile saldırı tahmini*

### PDF Rapor
![PDF Report](screenshots/pdf_report.png)
*Detaylı güvenlik raporu*

---

## 🎯 Gelecek Özellikler

- [ ] **Real-time Network Capture**: Canlı ağ trafiği yakalama
- [ ] **Email Alerts**: Kritik saldırılarda otomatik email
- [ ] **LSTM Model**: Zamansal pattern'ler için
- [ ] **REST API**: Harici sistemlerle entegrasyon
- [ ] **Docker Support**: Kolay deployment
- [ ] **CICIDS2017 Dataset**: Gerçek veri ile eğitim
- [ ] **Multi-User Support**: Kullanıcı yönetimi
- [ ] **Dark Mode**: Karanlık tema desteği

---

## 🤝 Katkıda Bulun

Katkılarınızı bekliyoruz!

### Nasıl Katkıda Bulunabilirsiniz?

1. **Fork** edin
2. Feature branch oluşturun (`git checkout -b feature/AmazingFeature`)
3. Değişiklikleri commit edin (`git commit -m 'Add some AmazingFeature'`)
4. Branch'e push edin (`git push origin feature/AmazingFeature`)
5. **Pull Request** açın

### Katkı Alanları

- 🐛 Bug fixes
- ✨ Yeni özellikler
- 📝 Dokümantasyon
- 🎨 UI/UX iyileştirmeleri
- 🧪 Test yazımı
- 🌐 Çeviriler

---

## 📄 Lisans

Bu proje MIT lisansı altında lisanslanmıştır. Detaylar için [LICENSE](LICENSE) dosyasına bakın.

---

## 👨‍💻 Geliştirici

**CyberGuard AI Team**

- 📧 Email: contact@cyberguardai.com
- 🌐 Website: [cyberguardai.com](https://cyberguardai.com)
- 💼 LinkedIn: [CyberGuard AI](https://linkedin.com/company/cyberguard-ai)

---

## 🙏 Teşekkürler

- [Streamlit](https://streamlit.io) - Web framework
- [Google Gemini](https://ai.google.dev/) - AI/LLM
- [LangChain](https://langchain.com) - RAG framework
- [Scikit-learn](https://scikit-learn.org) - Machine learning
- [Plotly](https://plotly.com) - Visualizations

---



<div align="center">

**🛡️ CyberGuard AI - Securing the Digital World with AI 🛡️**

Made with ❤️ and lots of ☕

[⬆ Back to Top](#-cyberguard-ai)

</div>