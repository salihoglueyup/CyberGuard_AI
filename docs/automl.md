# 🤖 AutoML Pipeline Dokümantasyonu

Otomatik model seçimi ve hiperparametre optimizasyonu - Detaylı Rehber

---

## 📋 İçindekiler

- [Genel Bakış](#genel-bakış)
- [AutoML Nedir?](#automl-nedir)
- [Desteklenen Algoritmalar](#desteklenen-algoritmalar)
- [API Endpoints](#api-endpoints)
- [Hiperparametre Arama](#hiperparametre-arama)
- [Model Değerlendirme](#model-değerlendirme)
- [Kullanım Senaryoları](#kullanım-senaryoları)
- [Best Practices](#best-practices)

---

## 🌟 Genel Bakış

AutoML modülü, veri setiniz için en iyi makine öğrenmesi modelini otomatik olarak bulur ve optimize eder.

### Özellikler

```
┌─────────────────────────────────────────────────────────────────┐
│                    AutoML Pipeline                               │
├─────────────────────────────────────────────────────────────────┤
│  🎯 Otomatik model seçimi                                        │
│  ⚙️ Hiperparametre optimizasyonu                                 │
│  📊 Model karşılaştırma ve leaderboard                           │
│  💡 Akıllı öneriler                                              │
│  🔄 Cross-validation                                             │
│  📈 Ensemble oluşturma                                           │
│  ⏱️ Zaman limiti kontrolü                                        │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🎯 AutoML Nedir?

### Klasik ML vs AutoML

```
Klasik ML Workflow:
┌─────────┐    ┌──────────────┐    ┌─────────────┐    ┌─────────┐
│  Data   │ -> │  Feature Eng │ -> │   Model     │ -> │ Tuning  │
│  Prep   │    │  (Manual)    │    │   Selection │    │ (Grid)  │
└─────────┘    └──────────────┘    │  (Manual)   │    └─────────┘
                                   └─────────────┘
                ⏱️ Günler - Haftalar

AutoML Workflow:
┌─────────┐    ┌─────────────────────────────────────┐    ┌─────────┐
│  Data   │ -> │           AutoML Engine             │ -> │  Best   │
│  Prep   │    │  Feature Eng + Model + Tuning       │    │  Model  │
└─────────┘    └─────────────────────────────────────┘    └─────────┘
                ⏱️ Dakikalar - Saatler
```

### AutoML Bileşenleri

| Bileşen | Açıklama | Zorluk |
|---------|----------|--------|
| Algorithm Selection | En iyi algoritma seçimi | Yüksek |
| Hyperparameter Tuning | Parametre optimizasyonu | Çok Yüksek |
| Feature Engineering | Otomatik özellik oluşturma | Orta |
| Ensemble Creation | Model kombinasyonu | Orta |
| Neural Architecture Search | DL mimarisi arama | Çok Yüksek |

---

## 📚 Desteklenen Algoritmalar

### Deep Learning Modelleri

#### LSTM (Long Short-Term Memory)

```python
config = {
    "algorithm": "lstm",
    "hyperparameters": {
        "units": [64, 128, 256],
        "dropout": [0.1, 0.2, 0.3],
        "layers": [1, 2, 3],
        "learning_rate": [0.001, 0.0001]
    }
}
```

| Özellik | Değer |
|---------|-------|
| **Kullanım** | Time-series, sequential data |
| **Complexity** | Medium |
| **Training Time** | 10-30 min |
| **GPU Gerekli** | Önerilen |

#### BiLSTM (Bidirectional LSTM)

```python
config = {
    "algorithm": "bilstm",
    "hyperparameters": {
        "units": [64, 128],
        "dropout": [0.2, 0.3],
        "attention": [True, False]
    }
}
```

| Özellik | Değer |
|---------|-------|
| **Kullanım** | Forward + backward context |
| **Complexity** | Medium-High |
| **Training Time** | 15-45 min |

#### GRU (Gated Recurrent Unit)

```python
config = {
    "algorithm": "gru",
    "hyperparameters": {
        "units": [64, 128],
        "reset_after": [True, False]
    }
}
```

| Özellik | Değer |
|---------|-------|
| **Kullanım** | Faster LSTM alternative |
| **Complexity** | Medium |
| **Training Time** | 8-25 min |

#### CNN-LSTM Hybrid

```python
config = {
    "algorithm": "cnn_lstm",
    "hyperparameters": {
        "filters": [32, 64],
        "kernel_size": [3, 5],
        "lstm_units": [64, 128]
    }
}
```

| Özellik | Değer |
|---------|-------|
| **Kullanım** | Feature extraction + sequence |
| **Complexity** | High |
| **Training Time** | 20-60 min |

#### Transformer

```python
config = {
    "algorithm": "transformer",
    "hyperparameters": {
        "num_heads": [4, 8],
        "d_model": [64, 128],
        "num_layers": [2, 4]
    }
}
```

| Özellik | Değer |
|---------|-------|
| **Kullanım** | Self-attention mechanisms |
| **Complexity** | Very High |
| **Training Time** | 30-90 min |

### Ensemble Methods

#### Random Forest

```python
config = {
    "algorithm": "random_forest",
    "hyperparameters": {
        "n_estimators": [100, 200, 500],
        "max_depth": [10, 20, None],
        "min_samples_split": [2, 5, 10]
    }
}
```

#### XGBoost

```python
config = {
    "algorithm": "xgboost",
    "hyperparameters": {
        "n_estimators": [100, 300, 500],
        "max_depth": [5, 10, 15],
        "learning_rate": [0.01, 0.1, 0.3]
    }
}
```

### Algoritma Karşılaştırması

```
                    Accuracy              Training Speed
LSTM                ████████████████░░░░  ██████████░░░░░░░░░░
BiLSTM              █████████████████░░░  ████████░░░░░░░░░░░░
GRU                 ███████████████░░░░░  █████████████░░░░░░░
CNN-LSTM            ██████████████████░░  ███████░░░░░░░░░░░░░
Transformer         █████████████████░░░  ██████░░░░░░░░░░░░░░
Random Forest       █████████████░░░░░░░  ██████████████████░░
XGBoost             ██████████████░░░░░░  ████████████████░░░░
```

---

## 🔌 API Endpoints

### POST /api/automl/start

AutoML job başlat

**Request:**

```json
{
  "dataset_name": "cicids2017",
  "task_type": "classification",
  "target_metric": "accuracy",
  "max_models": 5,
  "time_limit_minutes": 30,
  "include_deep_learning": true,
  "cross_validation_folds": 5,
  "algorithms": ["lstm", "bilstm", "random_forest"]
}
```

**Response:**

```json
{
  "success": true,
  "data": {
    "job_id": "AUTOML-20260110-abc123",
    "status": "completed",
    "started_at": "2026-01-10T12:00:00",
    "completed_at": "2026-01-10T12:28:00",
    "best_model": {
      "algorithm": "bilstm",
      "algorithm_name": "Bidirectional LSTM",
      "hyperparameters": {
        "units": 128,
        "dropout": 0.3,
        "learning_rate": 0.001
      },
      "metrics": {
        "accuracy": 0.9888,
        "precision": 0.9876,
        "recall": 0.9892,
        "f1_score": 0.9884
      },
      "training_time_seconds": 1245
    },
    "leaderboard": [
      {"rank": 1, "algorithm": "bilstm", "accuracy": 0.9888},
      {"rank": 2, "algorithm": "lstm", "accuracy": 0.9845},
      {"rank": 3, "algorithm": "random_forest", "accuracy": 0.9756}
    ]
  }
}
```

### GET /api/automl/status/{job_id}

Job durumunu kontrol et

**Response:**

```json
{
  "success": true,
  "data": {
    "job_id": "AUTOML-20260110-abc123",
    "status": "running",
    "progress": 65,
    "current_model": "lstm_config_3",
    "models_completed": 3,
    "models_total": 5,
    "elapsed_time_seconds": 1200,
    "estimated_remaining_seconds": 600
  }
}
```

### GET /api/automl/algorithms

```json
{
  "success": true,
  "data": {
    "algorithms": [
      {
        "id": "lstm",
        "name": "LSTM",
        "type": "deep_learning",
        "description": "Long Short-Term Memory network",
        "complexity": "medium",
        "training_time": "10-30 min",
        "best_for": ["time_series", "sequential", "network_traffic"],
        "hyperparameters": [
          {"name": "units", "type": "int", "range": [32, 512]},
          {"name": "dropout", "type": "float", "range": [0.0, 0.5]}
        ]
      }
    ]
  }
}
```

### GET /api/automl/recommendations

**Request:**

```
GET /api/automl/recommendations?dataset_type=network_traffic&objective=accuracy
```

**Response:**

```json
{
  "success": true,
  "data": {
    "recommendation": {
      "dataset_type": "network_traffic",
      "top_pick": {
        "algorithm": "bilstm",
        "why": "Best for temporal patterns in network data",
        "expected_accuracy": "98-99%"
      },
      "alternatives": [
        {"algorithm": "cnn_lstm", "expected_accuracy": "97-99%"},
        {"algorithm": "transformer", "expected_accuracy": "97-98%"}
      ],
      "not_recommended": ["svm", "logistic_regression"],
      "tips": [
        "Use SMOTE for class imbalance",
        "Consider time-based features",
        "Normalize packet sizes"
      ]
    }
  }
}
```

### POST /api/automl/hyperparameter-search

```json
{
  "algorithm": "lstm",
  "search_method": "bayesian",
  "max_trials": 50,
  "hyperparameters": {
    "units": {"type": "int", "min": 32, "max": 256},
    "dropout": {"type": "float", "min": 0.1, "max": 0.5}
  }
}
```

---

## 🔍 Hiperparametre Arama

### Arama Yöntemleri

#### 1. Grid Search

```python
# Tüm kombinasyonları dene
search_space = {
    "units": [64, 128, 256],
    "dropout": [0.1, 0.2, 0.3]
}
# 3 x 3 = 9 kombinasyon
```

- **Avantaj**: Kapsamlı
- **Dezavantaj**: Yavaş (O(n^k))

#### 2. Random Search

```python
# Rastgele kombinasyonlar dene
search_space = {
    "units": scipy.stats.randint(32, 256),
    "dropout": scipy.stats.uniform(0.0, 0.5)
}
# max_trials kadar dene
```

- **Avantaj**: Hızlı, büyük arama alanı
- **Dezavantaj**: Global optimum garantisi yok

#### 3. Bayesian Optimization (Önerilen)

```python
# Akıllı arama
from hyperopt import fmin, tpe, hp

search_space = {
    "units": hp.quniform("units", 32, 256, 32),
    "dropout": hp.uniform("dropout", 0.0, 0.5)
}

best = fmin(
    fn=objective,
    space=search_space,
    algo=tpe.suggest,
    max_evals=50
)
```

- **Avantaj**: Verimli, az deneme ile iyi sonuç
- **Dezavantaj**: Paralel zor

### Karşılaştırma

| Yöntem | Efficiency | Parallelizable | Best For |
|--------|------------|----------------|----------|
| Grid | Low | ✅ | Small search space |
| Random | Medium | ✅ | Large search space |
| Bayesian | High | ⚠️ | Limited budget |

---

## 📊 Model Değerlendirme

### Metrikler

| Metrik | Açıklama | Kullanım |
|--------|----------|----------|
| **Accuracy** | Doğru tahmin oranı | Balanced data |
| **Precision** | TP / (TP + FP) | Minimize FP |
| **Recall** | TP / (TP + FN) | Minimize FN |
| **F1-Score** | Harmonic mean of P & R | Imbalanced data |
| **AUC-ROC** | Area under ROC curve | Binary classification |

### Cross-Validation

```python
# 5-fold cross-validation
from sklearn.model_selection import cross_val_score

scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
print(f"Mean: {scores.mean():.4f} (+/- {scores.std():.4f})")
```

### Leaderboard Formatı

```
╔═══════╤══════════════════╤══════════╤══════════╤═══════════╗
║ Rank  │ Algorithm        │ Accuracy │ F1-Score │ Time (s)  ║
╠═══════╪══════════════════╪══════════╪══════════╪═══════════╣
║  🥇   │ BiLSTM           │ 98.88%   │ 98.84%   │ 1245      ║
║  🥈   │ CNN-LSTM         │ 98.45%   │ 98.41%   │ 1567      ║
║  🥉   │ LSTM             │ 98.12%   │ 98.08%   │ 987       ║
║  4    │ Transformer      │ 97.89%   │ 97.85%   │ 2134      ║
║  5    │ Random Forest    │ 97.56%   │ 97.52%   │ 234       ║
╚═══════╧══════════════════╧══════════╧══════════╧═══════════╝
```

---

## 💻 Kullanım Senaryoları

### 1. Hızlı Baseline

```python
# 5 dakikada baseline model
response = requests.post("/api/automl/start", json={
    "dataset_name": "cicids2017",
    "max_models": 3,
    "time_limit_minutes": 5,
    "algorithms": ["random_forest", "xgboost"]
})
```

### 2. En İyi Model Arama

```python
# Kapsamlı arama
response = requests.post("/api/automl/start", json={
    "dataset_name": "cicids2017",
    "max_models": 10,
    "time_limit_minutes": 60,
    "include_deep_learning": True,
    "cross_validation_folds": 5
})
```

### 3. Belirli Algoritma Optimizasyonu

```python
# LSTM hiperparametre optimizasyonu
response = requests.post("/api/automl/hyperparameter-search", json={
    "algorithm": "lstm",
    "search_method": "bayesian",
    "max_trials": 100,
    "hyperparameters": {
        "units": {"type": "int", "min": 64, "max": 512},
        "dropout": {"type": "float", "min": 0.1, "max": 0.5},
        "layers": {"type": "int", "min": 1, "max": 3}
    }
})
```

---

## 📝 Best Practices

### 1. Data Preparation

```python
# ✅ Do
- Clean missing values
- Handle class imbalance (SMOTE)
- Normalize features
- Split before AutoML (avoid leakage)

# ❌ Don't
- Include test data in AutoML
- Ignore class imbalance
- Use raw categorical features
```

### 2. Time Budget

| Data Size | Recommended Time |
|-----------|-----------------|
| < 10K | 5-15 min |
| 10K-100K | 15-60 min |
| 100K-1M | 1-4 hours |
| > 1M | 4-24 hours |

### 3. Algorithm Selection

```
Network Traffic → BiLSTM, CNN-LSTM
Malware Detection → CNN-LSTM, Random Forest
IoT Data → LSTM, GRU
Tabular Data → XGBoost, Random Forest
```

---

## 📚 Referanslar

- [AutoML: A Survey](https://arxiv.org/abs/1908.00709)
- [Hyperparameter Optimization](https://www.automl.org/book/)
- [Neural Architecture Search](https://arxiv.org/abs/1808.05377)
- [Auto-Keras](https://autokeras.com/)
- [Auto-sklearn](https://automl.github.io/auto-sklearn/)
