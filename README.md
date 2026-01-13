# 🛡️ CyberGuard AI - Gelişmiş Siber Güvenlik Platformu

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-blue?style=for-the-badge&logo=python" alt="Python">
  <img src="https://img.shields.io/badge/TensorFlow-2.15+-orange?style=for-the-badge&logo=tensorflow" alt="TensorFlow">
  <img src="https://img.shields.io/badge/FastAPI-0.109+-green?style=for-the-badge&logo=fastapi" alt="FastAPI">
  <img src="https://img.shields.io/badge/React-18+-blue?style=for-the-badge&logo=react" alt="React">
  <img src="https://img.shields.io/badge/Accuracy-99.96%25-success?style=for-the-badge" alt="Accuracy">
  <img src="https://img.shields.io/badge/API_Endpoints-150+-purple?style=for-the-badge" alt="Endpoints">
</p>

<p align="center">
  <strong>🎯 SSA-LSTMIDS Makale Implementasyonu | 🤖 5 LLM Provider | 🔬 26+ ML Model | 🌍 3D Globe</strong>
</p>

---

## 📋 İçindekiler

- [Özellikler](#-özellikler)
- [Teknoloji Stack](#-teknoloji-stack)
- [Hızlı Başlangıç](#-hızlı-başlangıç)
- [Proje Yapısı](#-proje-yapısı)
- [ML Modelleri](#-ml-modelleri)
- [AI Assistant](#-ai-assistant)
- [API Dokümantasyonu](#-api-dokümantasyonu)
- [Frontend Sayfaları](#-frontend-sayfaları)
- [Konfigürasyon](#️-konfigürasyon)

---

## ✨ Özellikler

### 🔬 Makale Implementasyonu (SSA-LSTMIDS)

| Özellik | Açıklama |
|---------|----------|
| **SSA-LSTMIDS** | Makale mimarisi birebir uygulandı |
| **Accuracy** | %99.96 (Makaleden %0.54 daha iyi) |
| **Datasets** | CICIDS2017, NSL-KDD desteği |

### 🤖 AI Assistant

| Özellik | Açıklama |
|---------|----------|
| **5 LLM Provider** | Groq, OpenAI, Claude, Gemini, Ollama |
| **Smart Actions** | "DDoS analizi yap" → Otomatik çalıştır |
| **Conversation Memory** | SQLite kalıcı hafıza |

### 🧠 State-of-the-Art ML

| Model | Mimari |
|-------|--------|
| **Attention** | CNN → BiLSTM → Multi-Head Attention |
| **Transformer** | Positional Encoding → Encoder × N |
| **AutoML** | Grid/Random/Bayesian hyperparameter search |
| **XAI** | SHAP, LIME feature importance |

### 📊 Ek Özellikler

- ✅ Real-time IDS
- ✅ A/B Model Testing
- ✅ Drift Detection (PSI, KS)
- ✅ Federated Learning
- ✅ SMOTE/ADASYN data augmentation
- ✅ PSO/SSA feature selection

---

## 🛠 Teknoloji Stack

### Backend

```
Python 3.10+    FastAPI      TensorFlow 2.15+
SQLite          Pandas       Scikit-learn
SHAP            LIME         LangChain
```

### Frontend

```
React 18+       Vite         Axios
Zustand         Recharts     TailwindCSS
```

### AI/LLM

```
Groq (Llama 3.3)   OpenAI (GPT-4o)   Claude 3.5
Gemini Pro         Ollama (Local)
```

---

## 🚀 Hızlı Başlangıç

### 1. Depoyu Klonla

```bash
git clone https://github.com/username/CyberGuard_AI.git
cd CyberGuard_AI
```

### 2. Backend Kurulumu

```bash
python -m venv .venv
.venv\Scripts\activate  # Windows
pip install -r requirements.txt
```

### 3. Environment Variables

```bash
cp .env.example .env
# .env dosyasını düzenle:
# GROQ_API_KEY=gsk_...
# OPENAI_API_KEY=sk-...  (opsiyonel)
```

### 4. Frontend Kurulumu

```bash
cd frontend
npm install
```

### 5. Başlat

```bash
# Terminal 1 - Backend
python -m uvicorn app.main:app --reload --port 8000

# Terminal 2 - Frontend
cd frontend && npm run dev
```

### 6. Aç

- **Frontend:** <http://localhost:5173>
- **API Docs:** <http://localhost:8000/docs>

---

## 📁 Proje Yapısı

```
CyberGuard_AI/
├── app/                        # FastAPI Backend
│   ├── api/routes/             # API endpoints
│   │   ├── dashboard.py
│   │   ├── chat.py
│   │   ├── models.py
│   │   ├── training.py
│   │   └── advanced_ml.py
│   └── main.py
│
├── src/                        # Core Modüller
│   ├── chatbot/                # AI Assistant
│   │   ├── providers/          # LLM handlers
│   │   ├── memory/             # Conversation memory
│   │   └── integration/        # Model integration
│   │
│   ├── network_detection/      # IDS Modülleri
│   │   ├── models/             # LSTM, Attention, Transformer
│   │   ├── data/               # Augmentation, Feature Selection
│   │   ├── training/           # Trainer, Evaluator
│   │   └── inference/          # Real-time IDS
│   │
│   ├── ml/                     # Gelişmiş ML
│   │   ├── automl.py
│   │   ├── explainability.py
│   │   ├── ab_testing.py
│   │   ├── drift_detection.py
│   │   └── federated.py
│   │
│   └── utils/                  # Yardımcı modüller
│
├── frontend/                   # React Frontend
│   └── src/
│       ├── pages/              # 34 sayfa
│       ├── components/         # 15+ component (Globe3D, Network3D, UI)
│       └── services/           # API servisleri
│
├── models/                     # Eğitilmiş modeller
├── data/                       # Datasets
├── scripts/                    # Utility scripts
└── config.yaml                 # Konfigürasyon
```

---

## 🧠 ML Modelleri

### IDS Modelleri

| Model | Mimari | Accuracy | Dosya |
|-------|--------|----------|-------|
| **SSA-LSTMIDS** | Conv1D(30) → LSTM(120) → Dense(512) | **99.96%** | `ssa_lstmids.py` |
| **CNN+BiLSTM+Attention** | CNN → BiLSTM → Multi-Head Attention | ~99% | `attention.py` |
| **Transformer IDS** | Positional → Encoder × 3 | ~98% | `transformer_ids.py` |
| **Informer** | ProbSparse Attention | ~98% | `transformer_ids.py` |

### Kullanım

```python
# SSA-LSTMIDS (Makale)
from src.network_detection.models import build_ssa_lstmids
model = build_ssa_lstmids(input_shape=(10, 41), num_classes=5)

# Attention Model
from src.network_detection.models import build_cnn_bilstm_attention
model = build_cnn_bilstm_attention(input_shape=(10, 41), num_classes=5)

# Transformer
from src.network_detection.models import build_transformer_ids
model = build_transformer_ids(input_shape=(10, 41), num_classes=5)
```

---

## 🤖 AI Assistant

### LLM Providers

| Provider | Model | API Key |
|----------|-------|---------|
| **Groq** | Llama 3.3 70B | `GROQ_API_KEY` |
| **OpenAI** | GPT-4o | `OPENAI_API_KEY` |
| **Claude** | Claude 3.5 Sonnet | `ANTHROPIC_API_KEY` |
| **Gemini** | Gemini Pro | `GEMINI_API_KEY` |
| **Ollama** | Local | `ollama serve` |

### Kullanım

```python
from src.chatbot.providers import get_provider_manager

pm = get_provider_manager()
response = pm.chat("DDoS saldırısı nasıl tespit edilir?")
```

### Smart Actions

```
"DDoS analizi yap"     → DDoS modeli çalıştır
"Model karşılaştır"    → Tablo döndür
"MITRE mapping"        → ATT&CK tactics
"IDS durumu"           → Real-time status
```

---

## 📚 API Dokümantasyonu

### Endpoints

| Kategori | Endpoint | Açıklama |
|----------|----------|----------|
| **Dashboard** | `GET /api/dashboard/stats` | İstatistikler |
| **Chat** | `POST /api/chat/` | AI sohbet |
| **Models** | `GET /api/models/` | Model listesi |
| **Training** | `POST /api/training/start` | Eğitim başlat |
| **Predictions** | `POST /api/prediction/predict` | Tahmin yap |
| **AutoML** | `POST /api/ml/automl/search` | AutoML başlat |
| **XAI** | `GET /api/ml/xai/feature-importance` | Feature importance |

### Swagger UI

```
http://localhost:8000/docs
```

---

## 🖥 Frontend Sayfaları

| Sayfa | URL | Açıklama |
|-------|-----|----------|
| Dashboard | `/` | Genel bakış |
| AI Assistant | `/ai-assistant` | Chatbot |
| ML Models | `/models` | Model yönetimi |
| Network Monitor | `/network` | Ağ izleme |
| Threat Analysis | `/threat-analysis` | Tehdit analizi |
| Attack Logs | `/attacks` | Saldırı logları |
| Training | `/training` | Model eğitimi |
| Settings | `/settings` | Ayarlar |

---

## ⚙️ Konfigürasyon

### config.yaml

```yaml
model:
  default_model: "SSA-LSTMIDS"
  confidence_threshold: 0.8
  
training:
  epochs: 100
  batch_size: 64
  learning_rate: 0.001
  
database:
  path: "data/cyberguard.db"
```

### Environment Variables

```env
# LLM Providers
GROQ_API_KEY=gsk_...
OPENAI_API_KEY=sk-...           # Opsiyonel
ANTHROPIC_API_KEY=sk-ant-...    # Opsiyonel
GEMINI_API_KEY=...              # Opsiyonel

# Database
DATABASE_PATH=data/cyberguard.db
SECRET_KEY=your-secret-key
```

---

## 📊 Datasets

| Dataset | Boyut | Sınıflar | Durum |
|---------|-------|----------|-------|
| **CICIDS2017** | ~1.15 GB | 15 | ✅ Destekleniyor |
| **NSL-KDD** | ~3 MB | 5 | ✅ Destekleniyor |
| **BoT-IoT** | ~17 GB | 10 | ⚠️ Opsiyonel |

---

## 🧪 Test

```bash
# Backend tests
pytest scripts/tests/ -v

# Frontend tests
cd frontend && npm test
```

---

## 🐳 Docker (Opsiyonel)

```bash
# Build
docker-compose build

# Run
docker-compose up -d
```

---

## 📈 Performans

### SSA-LSTMIDS (CICIDS2017)

| Metrik | Değer |
|--------|-------|
| **Accuracy** | 99.96% |
| **Precision** | 99.96% |
| **Recall** | 99.96% |
| **F1-Score** | 99.96% |

### Makale Karşılaştırması

| Model | Makale | Bizim |
|-------|--------|-------|
| SSA-LSTMIDS | 99.42% | **99.96%** ✅ |

---

## 🤝 Katkıda Bulunma

1. Fork'la
2. Feature branch oluştur (`git checkout -b feature/amazing`)
3. Commit et (`git commit -m 'Add amazing feature'`)
4. Push et (`git push origin feature/amazing`)
5. Pull Request aç

---

## 📄 Lisans

MIT License - Detaylar için [LICENSE](LICENSE) dosyasına bakın.

---

## 👨‍💻 Geliştirici

**CyberGuard AI Team**

---

<p align="center">
  Made with ❤️ for Cybersecurity
</p>
