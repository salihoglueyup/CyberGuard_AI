# 🧠 Machine Learning Models Dokümantasyonu

CyberGuard AI'da kullanılan tüm makine öğrenmesi modelleri

---

## 📋 İçindekiler

- [Model Mimarisi](#model-mimarisi)
- [SSA-LSTMIDS (Ana Model)](#ssa-lstmids-ana-model)
- [Desteklenen Modeller](#desteklenen-modeller)
- [Model Performansları](#model-performansları)
- [Model Eğitimi](#model-eğitimi)
- [Model Kullanımı](#model-kullanımı)

---

## 🏗️ Model Mimarisi

### SSA-LSTMIDS Mimarisi

```
┌─────────────────────────────────────────────────────────────┐
│                      INPUT LAYER                             │
│                    (78 features)                             │
└─────────────────────────┬───────────────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────────┐
│                   CONV1D BLOCK 1                             │
│  Conv1D(30, kernel=3) → BatchNorm → ReLU → MaxPool(2)       │
└─────────────────────────┬───────────────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────────┐
│                   CONV1D BLOCK 2                             │
│  Conv1D(60, kernel=3) → BatchNorm → ReLU → MaxPool(2)       │
└─────────────────────────┬───────────────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────────┐
│                     LSTM LAYER                               │
│           LSTM(120 units, return_sequences=True)             │
└─────────────────────────┬───────────────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────────┐
│                   ATTENTION LAYER                            │
│            MultiHeadAttention(num_heads=4)                   │
└─────────────────────────┬───────────────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────────┐
│                    DENSE LAYERS                              │
│      Dense(512) → Dropout(0.2) → Dense(256) → Dropout(0.2)  │
└─────────────────────────┬───────────────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────────┐
│                   OUTPUT LAYER                               │
│              Dense(num_classes, softmax)                     │
└─────────────────────────────────────────────────────────────┘
```

---

## 🎯 SSA-LSTMIDS (Ana Model)

### Genel Bilgiler

| Özellik | Değer |
|---------|-------|
| **Model Adı** | SSA-LSTMIDS (Sparrow Search Algorithm - LSTM IDS) |
| **Kaynak Makale** | "An optimized LSTM-based deep learning model for anomaly network intrusion detection" |
| **Yayın** | Scientific Reports, 2025 |
| **Optimizasyon** | SSA (Sparrow Search Algorithm) |

### SSA Optimizasyonu

SSA (Sparrow Search Algorithm), serçelerin yiyecek arama davranışından ilham alan metaheuristik bir optimizasyon algoritmasıdır.

**Optimize Edilen Hiperparametreler:**

- Conv1D filter sayısı (30)
- LSTM unit sayısı (120)
- Dense layer units (512)
- Dropout oranı (0.2)
- Epoch sayısı (300)
- Batch size (120)

### Performans Sonuçları

| Dataset | Accuracy | Precision | Recall | F1-Score |
|---------|----------|-----------|--------|----------|
| **NSL-KDD** | 99.36% | 99.37% | 99.36% | 99.36% |
| **CICIDS2017** | 99.88% | 99.89% | 99.88% | 99.88% |
| **BoT-IoT** | 99.99% | 99.99% | 99.99% | 99.99% |

---

## 📚 Desteklenen Modeller

### 1. Deep Learning Modelleri

#### LSTM (Long Short-Term Memory)

```python
# Basit LSTM
model = keras.Sequential([
    keras.layers.LSTM(128, return_sequences=True),
    keras.layers.LSTM(64),
    keras.layers.Dense(num_classes, activation='softmax')
])
```

- **Kullanım**: Temporal pattern recognition
- **Accuracy**: ~96-98%

#### BiLSTM (Bidirectional LSTM)

```python
# Bidirectional LSTM
model = keras.Sequential([
    keras.layers.Bidirectional(keras.layers.LSTM(128, return_sequences=True)),
    keras.layers.Bidirectional(keras.layers.LSTM(64)),
    keras.layers.Dense(num_classes, activation='softmax')
])
```

- **Kullanım**: Forward + backward context
- **Accuracy**: ~97-99%

#### CNN-LSTM Hybrid

```python
# CNN + LSTM
model = keras.Sequential([
    keras.layers.Conv1D(64, 3, activation='relu'),
    keras.layers.MaxPooling1D(2),
    keras.layers.LSTM(128),
    keras.layers.Dense(num_classes, activation='softmax')
])
```

- **Kullanım**: Feature extraction + sequence learning
- **Accuracy**: ~98-99.5%

#### Transformer

```python
# Attention-based model
inputs = keras.layers.Input(shape=(timesteps, features))
x = keras.layers.MultiHeadAttention(num_heads=4, key_dim=64)(inputs, inputs)
x = keras.layers.GlobalAveragePooling1D()(x)
outputs = keras.layers.Dense(num_classes, activation='softmax')(x)
```

- **Kullanım**: Self-attention mechanisms
- **Accuracy**: ~97-99%

### 2. Traditional ML Modelleri

#### Random Forest

```python
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(
    n_estimators=200,
    max_depth=20,
    random_state=42
)
```

- **Kullanım**: Baseline, hızlı inference
- **Accuracy**: ~92-96%

#### XGBoost

```python
import xgboost as xgb

model = xgb.XGBClassifier(
    n_estimators=300,
    max_depth=10,
    learning_rate=0.1
)
```

- **Kullanım**: Gradient boosting
- **Accuracy**: ~94-97%

#### Support Vector Machine

```python
from sklearn.svm import SVC

model = SVC(
    kernel='rbf',
    C=1.0,
    gamma='auto'
)
```

- **Kullanım**: Linear/non-linear classification
- **Accuracy**: ~90-94%

---

## 📊 Model Performansları

### Dataset Bazında Karşılaştırma

#### NSL-KDD Dataset

| Model | Accuracy | Precision | Recall | F1-Score | Training Time |
|-------|----------|-----------|--------|----------|---------------|
| SSA-LSTMIDS | **99.36%** | 99.37% | 99.36% | 99.36% | 45 min |
| BiLSTM | 98.52% | 98.54% | 98.52% | 98.53% | 35 min |
| CNN-LSTM | 98.21% | 98.23% | 98.21% | 98.22% | 40 min |
| Random Forest | 96.15% | 96.18% | 96.15% | 96.16% | 5 min |
| XGBoost | 95.82% | 95.85% | 95.82% | 95.83% | 8 min |

#### CICIDS2017 Dataset

| Model | Accuracy | Precision | Recall | F1-Score | Training Time |
|-------|----------|-----------|--------|----------|---------------|
| SSA-LSTMIDS | **99.88%** | 99.89% | 99.88% | 99.88% | 2 hours |
| BiLSTM | 99.12% | 99.14% | 99.12% | 99.13% | 1.5 hours |
| CNN-LSTM | 98.95% | 98.97% | 98.95% | 98.96% | 1.8 hours |
| Random Forest | 97.45% | 97.48% | 97.45% | 97.46% | 20 min |
| XGBoost | 97.21% | 97.24% | 97.21% | 97.22% | 30 min |

#### BoT-IoT Dataset

| Model | Accuracy | Precision | Recall | F1-Score | Training Time |
|-------|----------|-----------|--------|----------|---------------|
| SSA-LSTMIDS | **99.99%** | 99.99% | 99.99% | 99.99% | 4 hours |
| BiLSTM | 99.85% | 99.86% | 99.85% | 99.85% | 3 hours |
| CNN-LSTM | 99.78% | 99.79% | 99.78% | 99.78% | 3.5 hours |
| Random Forest | 99.12% | 99.15% | 99.12% | 99.13% | 45 min |
| XGBoost | 99.05% | 99.08% | 99.05% | 99.06% | 1 hour |

---

## 🔧 Model Eğitimi

### Eğitim Scripti

```bash
# Full training pipeline
python scripts/train_cicids_full_ssa.py

# Specific dataset
python scripts/train_nsl_kdd.py
python scripts/train_botiot.py

# Fine-tuning
python scripts/finetune_deep_ssa.py
```

### Eğitim Parametreleri

```python
# Optimum parametreler
training_config = {
    'epochs': 300,
    'batch_size': 120,
    'learning_rate': 0.001,
    'optimizer': 'adam',
    'loss': 'sparse_categorical_crossentropy',
    'early_stopping_patience': 10,
    'reduce_lr_patience': 5,
    'validation_split': 0.2
}
```

### Data Augmentation

```python
# SMOTE for imbalanced classes
from imblearn.over_sampling import SMOTE

smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
```

---

## 💻 Model Kullanımı

### Python API

```python
from src.models.predictor import AttackPredictor

# Model yükle
predictor = AttackPredictor()
predictor.load_models()

# Tek tahmin
result = predictor.predict_single(features)
print(f"Attack Type: {result['predicted_type']}")
print(f"Confidence: {result['confidence']:.2%}")

# Toplu tahmin
results = predictor.predict_batch(features_list)
```

### REST API

```bash
# Tahmin endpoint'i
curl -X POST http://localhost:8000/api/prediction/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [0.1, 0.2, ...], "model_id": "best_cicids2017"}'
```

### Response Format

```json
{
  "success": true,
  "data": {
    "predicted_type": "DDoS",
    "confidence": 0.9876,
    "probabilities": {
      "Normal": 0.0012,
      "DDoS": 0.9876,
      "PortScan": 0.0089,
      "BruteForce": 0.0023
    },
    "risk_level": "critical"
  }
}
```

---

## 📁 Model Dosyaları

```
models/
├── production/
│   ├── best_cicids2017_model.h5
│   ├── best_nslkdd_model.h5
│   └── best_botiot_model.h5
├── experimental/
│   ├── transformer_v1.h5
│   └── bilstm_attention.h5
├── archived/
│   └── old_models/
├── scalers/
│   ├── cicids2017_scaler.pkl
│   ├── nslkdd_scaler.pkl
│   └── botiot_scaler.pkl
└── model_registry.json
```

---

## 📝 Referanslar

- [An optimized LSTM-based deep learning model](https://doi.org/10.1038/s41598-025-85248-z)
- [NSL-KDD Dataset](https://www.unb.ca/cic/datasets/nsl.html)
- [CICIDS2017 Dataset](https://www.unb.ca/cic/datasets/ids-2017.html)
- [BoT-IoT Dataset](https://research.unsw.edu.au/projects/bot-iot-dataset)
