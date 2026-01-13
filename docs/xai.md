# 🔍 Explainable AI (XAI) Dokümantasyonu

CyberGuard AI projesindeki Explainable AI özellikleri - Detaylı Rehber

---

## 📋 İçindekiler

- [Genel Bakış](#genel-bakış)
- [Neden XAI?](#neden-xai)
- [API Endpoints](#api-endpoints)
- [SHAP Açıklamaları](#shap-açıklamaları)
- [LIME Açıklamaları](#lime-açıklamaları)
- [Feature Importance](#feature-importance)
- [Görselleştirmeler](#görselleştirmeler)
- [Kullanım Örnekleri](#kullanım-örnekleri)
- [Best Practices](#best-practices)

---

## 🌟 Genel Bakış

XAI modülü, makine öğrenmesi modellerinin kararlarını açıklamak için SHAP (SHapley Additive exPlanations) ve LIME (Local Interpretable Model-agnostic Explanations) yöntemlerini kullanır.

### Desteklenen Açıklama Yöntemleri

| Yöntem | Tür | Açıklama |
|--------|-----|----------|
| **SHAP** | Global + Local | Shapley değerleri ile açıklama |
| **LIME** | Local | Lokal yorumlanabilir model |
| **Feature Importance** | Global | Model bazlı önem sıralaması |
| **Permutation Importance** | Global | Permütasyon tabanlı önem |

---

## 🎯 Neden XAI?

### Siber Güvenlikte Önem

```
┌─────────────────────────────────────────────────────────────────┐
│                    XAI'ın Faydaları                              │
├─────────────────────────────────────────────────────────────────┤
│  🔍 ŞEFFAFLİK      │ Model kararlarının neden verildiğini anlama│
│  🤝 GÜVEN          │ Kullanıcıların AI önerilerine güvenmesi    │
│  🐛 DEBUG          │ Model hatalarını tespit etmek              │
│  ⚖️ COMPLIANCE     │ GDPR, KVKK gibi düzenlemelere uyum         │
│  🎓 EĞİTİM         │ Güvenlik analistlerini eğitmek             │
│  ✅ VALİDASYON     │ Model davranışını doğrulamak               │
└─────────────────────────────────────────────────────────────────┘
```

### Yasal Gereksinimler

- **GDPR Article 22**: Automated decision-making, including profiling
- **KVKK Madde 11**: Kişinin, kendisiyle ilgili otomatik işleme dayalı kararlar hakkında bilgi edinme hakkı
- **ISO 27001**: Information security management

---

## 🔌 API Endpoints

### POST /api/xai/explain

Model tahminini açıkla

**Request:**

```json
{
  "model_id": "best_cicids2017",
  "features": [0.1, 0.2, 0.3, ...],
  "num_features": 10,
  "method": "shap"
}
```

**Response:**

```json
{
  "success": true,
  "data": {
    "prediction": "DDoS",
    "confidence": 0.98,
    "explanation": {
      "method": "shap",
      "base_value": 0.12,
      "top_features": [
        {
          "feature": "Flow Duration",
          "value": 15234.5,
          "shap_value": 0.35,
          "contribution": "positive",
          "rank": 1
        },
        {
          "feature": "Total Fwd Packets",
          "value": 892,
          "shap_value": -0.12,
          "contribution": "negative",
          "rank": 2
        }
      ]
    },
    "feature_values": [...],
    "timestamp": "2026-01-10T12:00:00"
  }
}
```

### GET /api/xai/feature-importance/{model_id}

**Response:**

```json
{
  "success": true,
  "data": {
    "model_id": "best_cicids2017",
    "method": "mean_shap",
    "feature_importance": [
      {"feature": "Flow Duration", "importance": 0.15, "rank": 1},
      {"feature": "Total Fwd Packets", "importance": 0.12, "rank": 2},
      {"feature": "Fwd Packet Length Mean", "importance": 0.10, "rank": 3}
    ]
  }
}
```

### GET /api/xai/global-importance

Tüm modeller için ortalama feature importance

### GET /api/xai/explanation-methods

**Response:**

```json
{
  "success": true,
  "data": {
    "methods": [
      {
        "id": "shap",
        "name": "SHAP",
        "description": "SHapley Additive exPlanations",
        "type": "global_local",
        "pros": ["Teorik tutarlılık", "Global açıklamalar"],
        "cons": ["Yavaş hesaplama", "Yüksek bellek"]
      },
      {
        "id": "lime",
        "name": "LIME",
        "description": "Local Interpretable Model-agnostic Explanations",
        "type": "local",
        "pros": ["Hızlı", "Model-agnostik"],
        "cons": ["Tutarsız olabilir", "Sadece lokal"]
      }
    ]
  }
}
```

---

## 📊 SHAP Açıklamaları

### Teorik Arka Plan

SHAP, oyun teorisinden gelen Shapley değerlerini kullanarak her özelliğin tahmine katkısını hesaplar.

**Shapley Değeri Formülü:**

```
φᵢ = Σ [|S|! (n-|S|-1)! / n!] × [f(S ∪ {i}) - f(S)]
```

### SHAP Türleri

| Tür | Kullanım | Hız |
|-----|----------|-----|
| TreeSHAP | Tree-based modeller | ⚡ Çok Hızlı |
| DeepSHAP | Deep learning | ⚡ Hızlı |
| KernelSHAP | Herhangi model | 🐢 Yavaş |
| LinearSHAP | Lineer modeller | ⚡ Çok Hızlı |

### Python Kullanımı

```python
import shap

# Model yükle
model = load_model("best_cicids2017")

# SHAP explainer oluştur
explainer = shap.TreeExplainer(model)  # veya DeepExplainer

# Açıklama üret
shap_values = explainer.shap_values(X_test)

# Tek örnek için açıklama
shap.force_plot(explainer.expected_value, shap_values[0], X_test[0])

# Özet plot
shap.summary_plot(shap_values, X_test)
```

### SHAP Görselleri

```
Force Plot (Tek Örnek):
┌──────────────────────────────────────────────────────────────┐
│  Base: 0.12                                                   │
│  ────────────────┬─────────────────────────┬────────────────  │
│  Flow Duration   │  Total Fwd Packets      │  Final: 0.98    │
│  +0.35           │  -0.12                   │                  │
│  ████████████████│▒▒▒▒▒▒                    │                  │
└──────────────────┴─────────────────────────┴────────────────┘

Summary Plot (Tüm Örnekler):
┌──────────────────────────────────────────────────────────────┐
│  Feature            │ SHAP Value Impact                      │
│  ───────────────────┼────────────────────────────────────────│
│  Flow Duration      │ ████████████████████ High              │
│  Total Fwd Packets  │ ██████████████░░░░░░ Medium            │
│  Fwd Packet Length  │ █████████░░░░░░░░░░░ Low               │
└──────────────────────────────────────────────────────────────┘
```

---

## 🍋 LIME Açıklamaları

### Nasıl Çalışır?

1. Tahmin noktası çevresinde perturbation samples oluştur
2. Her sample için orijinal model tahmini al
3. Weighted linear model eğit
4. Linear model katsayılarını açıklama olarak kullan

### Python Kullanımı

```python
from lime import lime_tabular

# LIME explainer oluştur
explainer = lime_tabular.LimeTabularExplainer(
    X_train,
    feature_names=feature_names,
    class_names=class_names,
    mode='classification'
)

# Açıklama üret
explanation = explainer.explain_instance(
    X_test[0],
    model.predict_proba,
    num_features=10
)

# Görselle
explanation.show_in_notebook()

# Liste olarak
print(explanation.as_list())
# [('Flow Duration > 1000', 0.25), ('Total Fwd Packets > 500', 0.18), ...]
```

### LIME vs SHAP

| Özellik | SHAP | LIME |
|---------|------|------|
| Teorik Tutarlılık | ✅ | ❌ |
| Hız | 🐢 | ⚡ |
| Global Açıklama | ✅ | ❌ |
| Model-Agnostik | ✅ | ✅ |
| Stabilite | ✅ | ⚠️ |
| Bellek Kullanımı | Yüksek | Düşük |

---

## 🎯 Feature Importance

### Global Importance

Tüm tahminlerde hangi özelliklerin genel olarak önemli olduğunu gösterir.

```python
# Random Forest feature importance
importance = model.feature_importances_
feature_importance = pd.DataFrame({
    'feature': feature_names,
    'importance': importance
}).sort_values('importance', ascending=False)
```

### Lokal Importance

Tek bir tahmin için hangi özelliklerin belirleyici olduğunu gösterir.

### CyberGuard AI'daki En Önemli Özellikler

| Sıra | Özellik | Önemi | Açıklama |
|------|---------|-------|----------|
| 1 | Flow Duration | 15% | Akış süresi |
| 2 | Total Fwd Packets | 12% | Forward paket sayısı |
| 3 | Fwd Packet Length Mean | 10% | Ortalama forward paket uzunluğu |
| 4 | Bwd Packet Length Mean | 9% | Ortalama backward paket uzunluğu |
| 5 | Flow Bytes/s | 8% | Saniye başına byte |

---

## 📈 Görselleştirmeler

### Frontend Görselleştirmeleri

```jsx
// XAIExplainer.jsx'te kullanım

// Bar Chart
{explanation.top_features.map(feature => (
  <div className="bar-container">
    <span>{feature.feature}</span>
    <div 
      className="bar"
      style={{ 
        width: `${Math.abs(feature.shap_value) * 100}%`,
        backgroundColor: feature.contribution === 'positive' ? 'green' : 'red'
      }}
    />
    <span>{feature.shap_value.toFixed(4)}</span>
  </div>
))}
```

### API ile Görsel

```python
import requests
import matplotlib.pyplot as plt

# Açıklama al
response = requests.post("/api/xai/explain", json={
    "model_id": "best_cicids2017",
    "features": sample_features,
    "method": "shap"
})

data = response.json()["data"]["explanation"]["top_features"]

# Plot
features = [f["feature"] for f in data]
values = [f["shap_value"] for f in data]
colors = ['green' if v > 0 else 'red' for v in values]

plt.barh(features, values, color=colors)
plt.xlabel("SHAP Value")
plt.title("Feature Contributions")
plt.show()
```

---

## 💻 Kullanım Örnekleri

### 1. Saldırı Açıklaması

```python
# Bir saldırı tahmini için açıklama
attack_sample = get_attack_sample("DDoS")

explanation = requests.post("/api/xai/explain", json={
    "model_id": "best_cicids2017",
    "features": attack_sample.tolist(),
    "method": "shap"
}).json()

print(f"Tahmin: {explanation['data']['prediction']}")
print(f"Güven: {explanation['data']['confidence']:.2%}")
print("\nÖnemli Faktörler:")
for f in explanation['data']['explanation']['top_features'][:5]:
    print(f"  {f['feature']}: {f['shap_value']:+.4f}")
```

### 2. Model Karşılaştırması

```python
# İki model için aynı örneğin açıklaması
models = ["lstm_model", "random_forest_model"]

for model_id in models:
    exp = requests.post("/api/xai/explain", json={
        "model_id": model_id,
        "features": sample.tolist(),
        "method": "shap"
    }).json()
    
    print(f"\n{model_id}:")
    print(f"Tahmin: {exp['data']['prediction']}")
```

### 3. Batch Açıklama

```python
# Birden fazla örnek için açıklama
results = []
for sample in samples:
    exp = requests.post("/api/xai/explain", json={
        "model_id": "best_cicids2017",
        "features": sample.tolist(),
        "method": "lime"  # LIME daha hızlı
    }).json()
    results.append(exp["data"])
```

---

## 📝 Best Practices

### 1. Yöntem Seçimi

| Senaryo | Önerilen Yöntem |
|---------|-----------------|
| Hızlı açıklama | LIME |
| Detaylı analiz | SHAP |
| Tree-based model | TreeSHAP |
| Deep learning | DeepSHAP |
| Global görünüm | SHAP Summary |

### 2. Performans İyileştirmeleri

```python
# SHAP için sample kullan
shap_values = explainer.shap_values(X_test[:100])  # İlk 100 örnek

# Background data limitle
explainer = shap.KernelExplainer(
    model.predict, 
    shap.sample(X_train, 100)  # 100 background sample
)
```

### 3. Açıklama Kalitesi

- En az 5-10 özellik göster
- Pozitif/negatif katkıları renklendir
- Özellik değerlerini de göster
- Güven aralığı ekle

---

## 📚 Referanslar

- [SHAP Paper](https://arxiv.org/abs/1705.07874) - Lundberg & Lee (2017)
- [LIME Paper](https://arxiv.org/abs/1602.04938) - Ribeiro et al. (2016)
- [Interpretable ML Book](https://christophm.github.io/interpretable-ml-book/)
- [SHAP Documentation](https://shap.readthedocs.io/)
- [LIME Documentation](https://lime-ml.readthedocs.io/)
