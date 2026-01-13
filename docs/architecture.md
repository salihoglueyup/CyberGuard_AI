# 🏗️ Architecture

CyberGuard AI Sistem Mimarisi

---

## 📋 İçindekiler

- [Genel Bakış](#genel-bakış)
- [System Architecture](#system-architecture)
- [Component Diagram](#component-diagram)
- [Data Flow](#data-flow)
- [Module Structure](#module-structure)
- [Technology Stack](#technology-stack)
- [Design Patterns](#design-patterns)
- [Scalability](#scalability)

---

## 🌟 Genel Bakış

CyberGuard AI, **modüler** ve **scalable** bir mimariye sahiptir. Her component bağımsız olarak geliştirilebilir ve test edilebilir.

### Core Principles

- 🎯 **Modularity**: Her modül bağımsız
- 🔄 **Reusability**: Tekrar kullanılabilir componentler
- 📈 **Scalability**: Yatay ve dikey ölçeklenebilir
- 🛡️ **Security First**: Güvenlik odaklı tasarım
- 🚀 **Performance**: Optimize edilmiş algoritmalar

---

## 🏛️ System Architecture

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Streamlit Web Interface                   │
│              (User Interaction & Visualization)              │
└────────────────────┬────────────────────────────────────────┘
                     │
        ┌────────────┼────────────┐
        │            │            │
        ▼            ▼            ▼
┌──────────────┬──────────────┬──────────────┐
│   Frontend   │   Business   │   Backend    │
│    Layer     │    Logic     │    Layer     │
└──────────────┴──────────────┴──────────────┘
```

### Detailed Architecture

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

- [Streamlit Documentation](https://docs.streamlit.io)
- [LangChain Architecture](https://docs.langchain.com)
- [ChromaDB Design](https://docs.trychroma.com)
- [Scikit-learn Best Practices](https://scikit-learn.org)

---

[⬆️ Back to Top](#-architecture)