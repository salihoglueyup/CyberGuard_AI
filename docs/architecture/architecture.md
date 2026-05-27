# 🏗️ Architecture

CyberGuard AI Sistem Mimarisi

---

## 📋 İçindekiler

- [Genel Bakış](#genel-bakış)
- [Yüksek Seviye Mimari](#yüksek-seviye-mimari)
- [Katman Detayları](#katman-detayları)
- [Backend Modül Yapısı](#backend-modül-yapısı)
- [Frontend Sayfa Yapısı](#frontend-sayfa-yapısı)
- [ML / AI Katmanı](#ml--ai-katmanı)
- [Veri Akışı](#veri-akışı)
- [Teknoloji Stack](#teknoloji-stack)
- [Güvenlik Mimarisi](#güvenlik-mimarisi)

---

## 🌟 Genel Bakış

CyberGuard AI; **React 19 + FastAPI** üzerine inşa edilmiş, SSA-LSTMIDS makalesini (Scientific Reports 2025) gerçek bir platforma taşıyan tam yığın siber güvenlik uygulamasıdır.

### Core Principles

- 🎯 **Modularity**: 40+ router modülü, 6 kategoride ayrılmış
- 🔄 **API-First**: Tüm işlevler REST + WebSocket üzerinden erişilebilir
- 📈 **Scalability**: Stateless FastAPI backend, Docker ile dağıtılabilir
- 🛡️ **Security First**: JWT benzeri token auth, bcrypt, rate limiting, güvenli CORS
- 🚀 **Performance**: React lazy loading, TF model önbelleği, GeoIP SQLite cache

---

## 🏛️ Yüksek Seviye Mimari

```
┌──────────────────────────────────────────────────────────────┐
│                   React 19 + Vite Frontend                    │
│   34 sayfa · Zustand · TailwindCSS · Three.js · React Router │
└─────────────────────┬────────────────────────────────────────┘
                      │ HTTP REST / WebSocket
┌─────────────────────▼────────────────────────────────────────┐
│                  FastAPI Backend (v2.0)                        │
│   150+ endpoint · 40+ router · Rate Limiting · JWT Auth       │
├──────────────┬───────────────┬───────────────┬───────────────┤
│  ML/AI Layer │  Security     │  Threat Intel │  Monitoring   │
│  TF 2.15+    │  Scanner/HSM  │  MITRE ATT&CK │  SIEM/Alerts  │
│  26+ model   │  Sandbox      │  Dark Web     │  Anomaly Det. │
│  AutoML/XAI  │  Adversarial  │  Threat Hunt  │  Incidents    │
└──────────────┴───────────────┴───────────────┴───────────────┘
                      │
┌─────────────────────▼────────────────────────────────────────┐
│                      Veri Katmanı                             │
│   SQLite (cyberguard.db) · ChromaDB (RAG) · JSON veri dosyaları │
└──────────────────────────────────────────────────────────────┘
```

---

## 🔩 Katman Detayları

### Frontend Katmanı (`frontend/src/`)

| Alt Dizin | İçerik |
|-----------|--------|
| `pages/core/` | Dashboard, Logs, Reports, Settings, Analytics, Database |
| `pages/ai/` | AIAssistant, AIHub, MLModels, XAIExplainer, AutoMLPipeline, Predictions, AttackTraining |
| `pages/security/` | MalwareScanner, VulnScanner, SecurityHub, Sandbox, ContainerSecurity, BlockchainAudit, ThreatHunting, DarkWeb, Compliance, Forensics, Honeypot, ApiSecurity, UserBehavior, Pentest |
| `pages/monitoring/` | NetworkMonitor, Network3D, AttackMap, IncidentTimeline, ThreatIntel, SIEM, GlobeView, TopologyMap |
| `components/` | Layout, Header, Sidebar, ThreatMap, NotificationBell, ErrorBoundary, ProtectedRoute |
| `store/` | Zustand global state (`store/index.js`) |
| `hooks/` | Custom React hooks |
| `services/` | API istemci fonksiyonları |

Tüm sayfalar `React.lazy()` + `<Suspense>` ile lazy load edilir.

### Backend Katmanı (`app/`)

```
app/
├── main.py                 ← FastAPI app, router kayıtları, middleware
├── paths.py                ← Proje köküne path çözümlemeleri
├── api/
│   ├── websocket.py        ← WebSocket endpoint'leri
│   └── routes/
│       ├── auth.py         ← /api/auth  (login, register, token doğrulama)
│       ├── core/           ← dashboard, settings, database, reports, pdf_reports,
│       │                      api_keys, notifications, logs, log_analyzer
│       ├── ml/             ← models, training, prediction, comparison, automl,
│       │                      advanced_ml, advanced_models, ai_decision, xai, drift_detection
│       ├── security/       ← scanner, sandbox, vulnerability, zeroday,
│       │                      adversarial, container_security, hsm, gan_synthesis
│       ├── threat/         ← threat_intel, threat_analysis, threat_hunting,
│       │                      darkweb, attack_map, attack_surface, attack_training,
│       │                      attacks, deception
│       ├── monitoring/     ← alerts, incidents, siem, realtime,
│       │                      network, anomaly, security_advanced
│       └── tools/          ← blockchain_audit, stix_taxii, playbooks, chat, federated
```

### ML / AI Katmanı (`src/`)

```
src/
├── ml/
│   ├── automl.py           ← Grid/Random/Bayesian hyperparameter search
│   ├── drift_detection.py  ← Data drift izleme
│   ├── explainability.py   ← SHAP + LIME açıklamaları
│   ├── federated.py        ← Federated Learning protokolü
│   └── ab_testing.py       ← A/B model karşılaştırma
├── models/
│   ├── tensorflow_model.py ← SSA-LSTMIDS ve diğer mimariler
│   ├── model_manager.py    ← Model yükleme / önbellekleme
│   ├── model_evaluator.py  ← Metrik hesaplama
│   └── train_tensorflow_model.py
├── chatbot/
│   ├── providers/          ← Groq, OpenAI, Claude, Gemini, Ollama adaptörleri
│   ├── integration/        ← LangChain entegrasyonu
│   ├── memory/             ← SQLite konuşma hafızası
│   └── vectorstore/        ← ChromaDB RAG sistemi
├── services/
│   ├── geoip.py            ← IP geolocation (ip-api.com + SQLite cache)
│   ├── ml_predictor.py     ← Gerçek zamanlı tehdit tahmini
│   └── training_api.py     ← Eğitim API sarmalayıcısı
└── database/
    ├── cyberguard.db       ← SQLite veritabanı
    └── init_db.py          ← Şema başlatma
```

---

## 🔄 Veri Akışı

### REST API İstek Akışı

```
Tarayıcı (React)
    │  HTTP + Authorization: Bearer <token>
    ▼
FastAPI  ──► CORS Middleware ──► Request Timing Middleware
    │
    ▼
Router (örn. /api/prediction)
    │
    ├──► require_auth() dependency → token doğrula
    │
    ▼
Route Handler
    │
    ├──► src/models/model_manager.py  (ML tahmini)
    ├──► src/database/cyberguard.db   (veri okuma/yazma)
    └──► src/services/geoip.py        (IP zenginleştirme)
    │
    ▼
JSON Response
```

### WebSocket Gerçek Zamanlı Akış

```
React Bileşeni
    │  ws://host:8000/ws/attacks
    ▼
app/api/websocket.py
    │
    ├──► GeoIP servisi (saldırı kaynağını coğrafi konuma çevir)
    ├──► ML Predictor (risk skoru hesapla)
    └──► Broadcast (tüm bağlı istemcilere gönder)
    │
    ▼
Globe3D / AttackMap bileşeni (Three.js render)
```

### ML Tahmin Akışı

```
Ham Ağ Trafiği (78 özellik)
    │
    ▼
Normalizasyon + Sequence oluşturma (uzunluk: 10)
    │
    ▼
SSA-LSTMIDS Modeli
  Conv1D(30) → Conv1D(60) → LSTM(120) → MultiHeadAttention → Dense(512) → Softmax
    │
    ▼
Sınıf tahmini (15 sınıf: Normal + 14 saldırı tipi)
    │
    ▼
XAI açıklaması (SHAP / LIME) — isteğe bağlı
    │
    ▼
JSON Response → Frontend görselleştirme
```

---

## 🔐 Güvenlik Mimarisi

### Kimlik Doğrulama Akışı

```
POST /api/auth/login  {email, password}
    │
    ▼
IP başına rate limit kontrolü (5 deneme / 60 saniye)
    │
    ▼
bcrypt.checkpw(password, stored_hash)
    │
    ▼
secrets.token_urlsafe(32)  →  TOKENS dict'e kaydet (24 saat TTL)
    │
    ▼
{token, user}  →  React localStorage'a kaydeder
    │
    ▼
Sonraki istekler: Authorization: Bearer <token>  →  require_auth() dependency
```

### Güvenlik Katmanları

| Katman | Uygulama |
|--------|----------|
| Şifre | bcrypt hash |
| Token | `secrets.token_urlsafe(32)`, 24 saat TTL |
| Rate Limiting | `slowapi` — IP başına 5 login/dk |
| CORS | `CORS_ORIGINS` env var'dan okunur |
| Request Headers | `X-Frame-Options`, `X-Content-Type-Options` (Nginx) |
| Input Validation | Pydantic v2 model doğrulama |

---

## 🛠️ Teknoloji Stack

### Frontend

| Teknoloji | Versiyon | Kullanım |
|-----------|----------|----------|
| React | 19 | UI framework |
| Vite | 6+ | Build tool |
| React Router | 7 | SPA routing |
| Zustand | — | State yönetimi |
| TailwindCSS | 3 | Stil |
| Three.js | — | 3D görselleştirme (Globe, Network3D) |
| i18next | — | Çoklu dil altyapısı |

### Backend

| Teknoloji | Versiyon | Kullanım |
|-----------|----------|----------|
| Python | 3.10+ | Ana dil |
| FastAPI | 0.104+ | REST framework |
| Uvicorn | 0.24+ | ASGI sunucu |
| Pydantic | v2 | Veri doğrulama |
| slowapi | — | Rate limiting |
| bcrypt | — | Şifre hash |
| python-dotenv | — | Ortam değişkenleri |

### ML / AI

| Teknoloji | Kullanım |
|-----------|----------|
| TensorFlow 2.15+ | Derin öğrenme modelleri |
| Scikit-learn | Klasik ML, metrikler |
| LangChain | RAG, LLM zinciri |
| ChromaDB | Vektör veritabanı |
| Sentence Transformers | Embedding |
| SHAP / LIME | Açıklanabilirlik |
| Groq, OpenAI, Claude, Gemini, Ollama | LLM sağlayıcıları |

### Altyapı

| Teknoloji | Kullanım |
|-----------|----------|
| SQLite | Ana veritabanı |
| Nginx | Frontend reverse proxy (Docker) |
| Docker + docker-compose | Konteyner dağıtımı |
| PyYAML | `config.yaml` konfigürasyonu |

---

## 📁 Proje Kök Yapısı

```
CyberGuard_AI/
├── app/                    ← FastAPI backend
├── frontend/               ← React + Vite frontend
├── src/                    ← ML, AI, servis modülleri
├── scripts/                ← Model eğitim scriptleri
├── model_artifacts/        ← Eğitilmiş .keras modeller
├── data/                   ← JSON veri dosyaları
├── tests/                  ← pytest test dosyaları
├── docs/                   ← Bu dokümantasyon
├── logs/                   ← Uygulama logları
├── config.yaml             ← Model ve uygulama konfigürasyonu
├── requirements.txt        ← Python bağımlılıkları
└── run.bat                 ← Windows hızlı başlatma scripti
```

```
┌─────────────────────────────────────────────────────────────────┐
│                        Presentation Layer                        │
│  ┌──────────┬──────────┬──────────┬──────────┬──────────────┐  │
│  │Dashboard │ Network  │ Malware  │   AI     │  ML Predict  │  │
│  │          │ Monitor  │ Scanner  │Assistant │              │  │
│  └──────────┴──────────┴──────────┴──────────┴──────────────┘  │
└───────────────────────────┬─────────────────────────────────────┘
                            │
┌───────────────────────────┴─────────────────────────────────────┐
│                        Application Layer                         │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │                     Core Services                          │ │
│  │  ┌──────────┬──────────┬──────────┬──────────┬─────────┐ │ │
│  │  │ Chatbot  │   RAG    │  Memory  │  Attack  │   ML    │ │ │
│  │  │ Service  │  System  │ Manager  │ Vectors  │  Model  │ │ │
│  │  └──────────┴──────────┴──────────┴──────────┴─────────┘ │ │
│  └────────────────────────────────────────────────────────────┘ │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │                     Utility Services                       │ │
│  │  ┌────────┬─────────┬──────────┬──────────┬────────────┐ │ │
│  │  │Database│ Logger  │  Config  │Visualizer│PDF Generator│ │ │
│  │  └────────┴─────────┴──────────┴──────────┴────────────┘ │ │
│  └────────────────────────────────────────────────────────────┘ │
└───────────────────────────┬─────────────────────────────────────┘
                            │
┌───────────────────────────┴─────────────────────────────────────┐
│                         Data Layer                              │
│  ┌────────────────┬─────────────────┬────────────────────────┐ │
│  │  SQLite DB     │  Vector Store   │   ML Models (.pkl)     │ │
│  │ (cyberguard.db)│  (ChromaDB)     │  (Random Forest)       │ │
│  └────────────────┴─────────────────┴────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🧩 Component Diagram

### Frontend Components

```
┌─────────────────────────────────────────┐
│           Streamlit Pages               │
│                                         │
│  ┌────────────────────────────────────┐ │
│  │  main.py (Router)                 │ │
│  │  ├─ Auto-refresh logic            │ │
│  │  ├─ Session state management      │ │
│  │  └─ Page navigation               │ │
│  └────────────────────────────────────┘ │
│                                         │
│  ┌────────────────────────────────────┐ │
│  │  Pages (app/pages/)               │ │
│  │  ├─ dashboard.py                  │ │
│  │  ├─ network_monitor.py            │ │
│  │  ├─ malware_scanner.py            │ │
│  │  ├─ ai_assistant.py               │ │
│  │  └─ ml_prediction.py              │ │
│  └────────────────────────────────────┘ │
└─────────────────────────────────────────┘
```

### Backend Components

```
┌─────────────────────────────────────────┐
│           Core Services                 │
│                                         │
│  ┌────────────────────────────────────┐ │
│  │  Chatbot (src/chatbot/)           │ │
│  │  ├─ gemini_handler.py             │ │
│  │  └─ vectorstore/                  │ │
│  │     ├─ rag_manager.py             │ │
│  │     ├─ memory_manager.py          │ │
│  │     └─ attack_vectors.py          │ │
│  └────────────────────────────────────┘ │
│                                         │
│  ┌────────────────────────────────────┐ │
│  │  ML Models (src/models/)          │ │
│  │  ├─ random_forest_model.py        │ │
│  │  └─ predictor.py                  │ │
│  └────────────────────────────────────┘ │
│                                         │
│  ┌────────────────────────────────────┐ │
│  │  Utilities (src/utils/)           │ │
│  │  ├─ database.py                   │ │
│  │  ├─ logger.py                     │ │
│  │  ├─ config.py                     │ │
│  │  ├─ visualizer.py                 │ │
│  │  ├─ feature_extractor.py          │ │
│  │  ├─ pdf_generator.py              │ │
│  │  └─ mock_data_generator.py        │ │
│  └────────────────────────────────────┘ │
└─────────────────────────────────────────┘
```

---

## 🔄 Data Flow

### Request-Response Flow

```
User Action
    │
    ▼
┌─────────────────┐
│ Streamlit Page  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Page Handler   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐     ┌──────────────┐
│  Core Service   │────▶│   Database   │
└────────┬────────┘     └──────────────┘
         │
         ▼
┌─────────────────┐
│  Process Data   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Return Result  │
└────────┬────────┘
         │
         ▼
     Display
```

### ML Prediction Flow

```
User Input
    │
    ▼
┌─────────────────────┐
│ Feature Extraction  │
│  (IP, Port, Time)   │
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│   Normalization     │
│   (Scaler)          │
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  Random Forest      │
│  Model Prediction   │
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  Risk Calculation   │
│  (Risk Score 0-100) │
└─────────┬───────────┘
          │
          ▼
     Result Display
```

### RAG System Flow

```
User Question
    │
    ▼
┌──────────────────────┐
│  Query Embedding     │
│  (Sentence Transform)│
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│  Vector Search       │
│  (ChromaDB)          │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│  Retrieve Documents  │
│  (Top K Results)     │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│  Context Building    │
│  (Combine Results)   │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│  Gemini Pro          │
│  (Generate Answer)   │
└──────────┬───────────┘
           │
           ▼
      Response
```

---

## 📦 Module Structure

### Chatbot Module

```python
src/chatbot/
├── gemini_handler.py           # Main chatbot interface
│   ├── GeminiHandler           # Core class
│   │   ├── chat()              # Send message
│   │   ├── get_attack_context()
│   │   ├── get_ip_context()
│   │   └── get_system_context()
│
└── vectorstore/                # RAG system
    ├── rag_manager.py          # Document management
    │   ├── RAGManager
    │   │   ├── add_text_document()
    │   │   ├── add_pdf_document()
    │   │   └── search()
    │
    ├── memory_manager.py       # Conversation memory
    │   ├── MemoryManager
    │   │   ├── add_conversation()
    │   │   ├── search_memory()
    │   │   └── get_relevant_memory()
    │
    └── attack_vectors.py       # Attack vectorization
        ├── AttackVectorManager
        │   ├── vectorize_attacks()
        │   ├── find_similar_attacks()
        │   └── analyze_attack_pattern()
```

### ML Module

```python
src/models/
├── random_forest_model.py      # Model definition
│   ├── CyberAttackModel
│   │   ├── train()
│   │   ├── predict()
│   │   ├── predict_proba()
│   │   └── evaluate()
│
└── predictor.py                # Prediction interface
    ├── AttackPredictor
    │   ├── load_models()
    │   ├── predict_single()
    │   ├── predict_batch()
    │   └── get_model_info()
```

### Utils Module

```python
src/utils/
├── database.py                 # Database operations
│   ├── DatabaseManager
│   │   ├── add_attack()
│   │   ├── get_attacks()
│   │   └── get_database_stats()
│
├── feature_extractor.py        # ML feature extraction
│   ├── FeatureExtractor
│   │   ├── prepare_features()
│   │   └── prepare_labels()
│
└── pdf_generator.py            # PDF report generation
    ├── PDFReportGenerator
    │   ├── generate_report()
    │   └── get_attack_stats()
```

---

## 🛠️ Technology Stack

### Frontend Layer

| Technology | Version | Purpose |
|------------|---------|---------|
| Streamlit | 1.32.0 | Web framework |
| Plotly | 5.20.0 | Interactive charts |
| Matplotlib | 3.9.2 | Static charts |
| Custom CSS | - | Styling |

### Application Layer

| Technology | Version | Purpose |
|------------|---------|---------|
| Python | 3.10+ | Core language |
| Google Gemini | 2.5 Flash | LLM/AI |
| LangChain | 0.2.0 | RAG framework |
| Scikit-learn | 1.5.2 | ML models |
| TensorFlow | 2.15.0 | Deep learning |

### Data Layer

| Technology | Version | Purpose |
|------------|---------|---------|
| SQLite | 3.x | Relational DB |
| ChromaDB | 0.4.24 | Vector DB |
| Pandas | 2.2.1 | Data processing |

### Infrastructure

| Technology | Purpose |
|------------|---------|
| Virtual Environment | Dependency isolation |
| Git | Version control |
| PyPI | Package management |

---

## 🎨 Design Patterns

### 1. Singleton Pattern

**Kullanım:** Database Manager, Config Manager

```python
class DatabaseManager:
    _instance = None
    
    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super().__new__(cls)
        return cls._instance
```

**Avantaj:** Tek bir database connection

### 2. Factory Pattern

**Kullanım:** Model creation

```python
class ModelFactory:
    @staticmethod
    def create_model(model_type: str):
        if model_type == 'random_forest':
            return RandomForestModel()
        elif model_type == 'lstm':
            return LSTMModel()
```

**Avantaj:** Esnek model seçimi

### 3. Strategy Pattern

**Kullanım:** Feature extraction

```python
class FeatureExtractor:
    def __init__(self, strategy: FeatureStrategy):
        self.strategy = strategy
    
    def extract(self, data):
        return self.strategy.extract(data)
```

**Avantaj:** Farklı extraction yöntemleri

### 4. Observer Pattern

**Kullanım:** Real-time updates

```python
class AttackObserver:
    def __init__(self):
        self.observers = []
    
    def notify(self, attack):
        for observer in self.observers:
            observer.update(attack)
```

**Avantaj:** Event-driven architecture

### 5. Repository Pattern

**Kullanım:** Data access

```python
class AttackRepository:
    def __init__(self, db):
        self.db = db
    
    def get_all(self):
        return self.db.query("SELECT * FROM attacks")
    
    def get_by_id(self, id):
        return self.db.query(f"SELECT * FROM attacks WHERE id={id}")
```

**Avantaj:** Data layer abstraction

---

## 📈 Scalability

### Horizontal Scaling

```
┌──────────────┐      ┌──────────────┐
│  Streamlit   │      │  Streamlit   │
│  Instance 1  │      │  Instance 2  │
└──────┬───────┘      └──────┬───────┘
       │                     │
       └──────────┬──────────┘
                  │
          ┌───────┴────────┐
          │  Load Balancer │
          └───────┬────────┘
                  │
          ┌───────┴────────┐
          │  Shared DB     │
          └────────────────┘
```

### Vertical Scaling

- **CPU**: Artırılabilir (ML model için)
- **RAM**: Artırılabilir (vector store için)
- **Storage**: Artırılabilir (database için)

### Caching Strategy

```python
@st.cache_resource
def load_model():
    return AttackPredictor()

@st.cache_data(ttl=3600)
def get_attack_stats():
    return db.get_database_stats()
```

### Performance Optimization

1. **Database Indexing**
```sql
CREATE INDEX idx_timestamp ON attacks(timestamp);
CREATE INDEX idx_source_ip ON attacks(source_ip);
CREATE INDEX idx_attack_type ON attacks(attack_type);
```

2. **Batch Processing**
```python
# Tek tek yerine batch olarak
db.add_attacks_batch(attacks_list)
```

3. **Lazy Loading**
```python
# İhtiyaç duyulduğunda yükle
if user_requests_chart:
    chart = generate_chart()
```

---

## 🔐 Security Architecture

### Authentication Flow

```
User Login
    │
    ▼
┌──────────────┐
│  Credentials │
└──────┬───────┘
       │
       ▼
┌──────────────┐
│  Validate    │
└──────┬───────┘
       │
       ▼
┌──────────────┐
│ Create Token │
└──────┬───────┘
       │
       ▼
   Session State
```

### Data Protection

- ✅ API keys in `.env` (not in code)
- ✅ SQL injection prevention (parameterized queries)
- ✅ Input validation
- ✅ XSS protection (Streamlit built-in)
- ✅ HTTPS recommended (deployment)

---

## 🧪 Testing Architecture

### Test Pyramid

```
        ┌─────────┐
       E2E Tests
      ┌────────────┐
   Integration Tests
  ┌──────────────────┐
      Unit Tests
```

### Test Coverage

```python
tests/
├── unit/
│   ├── test_database.py
│   ├── test_model.py
│   └── test_utils.py
├── integration/
│   ├── test_chatbot.py
│   └── test_rag.py
└── e2e/
    └── test_dashboard.py
```

---

## 🚀 Deployment Architecture

### Local Deployment

```
Developer Machine
    ├── venv/
    ├── streamlit run app/main.py
    └── http://localhost:8501
```

### Cloud Deployment (Planned)

```
┌─────────────────┐
│   CloudFlare    │  ← CDN
└────────┬────────┘
         │
┌────────┴────────┐
│  Load Balancer  │
└────────┬────────┘
         │
    ┌────┴────┐
    │         │
┌───┴───┐ ┌──┴────┐
│ App 1 │ │ App 2 │  ← Streamlit instances
└───┬───┘ └──┬────┘
    │        │
    └───┬────┘
        │
┌───────┴────────┐
│   PostgreSQL   │  ← Database (cloud)
└────────────────┘
```

---

## 📊 Monitoring & Logging

### Logging Architecture

```python
Logger
  ├── Console Handler (DEBUG)
  ├── File Handler (INFO)
  └── Error Handler (ERROR/CRITICAL)
```

### Metrics Collection

```python
metrics = {
    'request_count': Counter,
    'response_time': Histogram,
    'error_rate': Gauge,
    'active_users': Gauge
}
```

---

## 🔄 CI/CD Pipeline (Future)

```
Git Push
    │
    ▼
┌──────────────┐
│  GitHub      │
└──────┬───────┘
       │
       ▼
┌──────────────┐
│  Run Tests   │
└──────┬───────┘
       │
       ▼
┌──────────────┐
│  Build Docker│
└──────┬───────┘
       │
       ▼
┌──────────────┐
│  Deploy      │
└──────────────┘
```

---

## 📚 Best Practices

### Code Organization

✅ **Modular structure**
✅ **Single responsibility**
✅ **DRY (Don't Repeat Yourself)**
✅ **Clear naming conventions**
✅ **Comprehensive documentation**

### Performance

✅ **Caching (@st.cache_resource)**
✅ **Lazy loading**
✅ **Batch processing**
✅ **Database indexing**
✅ **Vector store optimization**

### Security

✅ **Environment variables**
✅ **Input validation**
✅ **Error handling**
✅ **Secure communication**
✅ **Access control**

---

## 🔮 Future Architecture Enhancements

### Microservices (v2.0)

```
┌──────────┐  ┌──────────┐  ┌──────────┐
│  Auth    │  │ Chatbot  │  │ ML Model │
│ Service  │  │ Service  │  │ Service  │
└────┬─────┘  └────┬─────┘  └────┬─────┘
     │             │              │
     └─────────────┴──────────────┘
                   │
            ┌──────┴──────┐
            │  API Gateway │
            └─────────────┘
```

### Real-time Processing

```
Attack Data → Kafka → Stream Processing → Alert System
```

### Multi-tenant Architecture

```
Tenant 1 ──┐
Tenant 2 ──┼─→ Shared App ──→ Isolated DB per Tenant
Tenant 3 ──┘
```

---

## 📖 References

- [FastAPI Documentation](https://fastapi.tiangolo.com)
- [React 19 Docs](https://react.dev)
- [TensorFlow 2.x Guide](https://www.tensorflow.org/guide)
- [ChromaDB Design](https://docs.trychroma.com)
- [SSA-LSTMIDS Paper (Scientific Reports 2025)](https://www.nature.com/srep/)

---

[⬆️ Back to Top](#-architecture)