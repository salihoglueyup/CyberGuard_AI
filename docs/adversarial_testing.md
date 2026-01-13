# ⚔️ Adversarial Testing Dokümantasyonu

Model güvenliği ve adversarial attack test sistemi - Detaylı Rehber

---

## 📋 İçindekiler

- [Genel Bakış](#genel-bakış)
- [Neden Adversarial Testing?](#neden-adversarial-testing)
- [Desteklenen Saldırı Türleri](#desteklenen-saldırı-türleri)
- [API Endpoints](#api-endpoints)
- [Robustness Değerlendirmesi](#robustness-değerlendirmesi)
- [Savunma Yöntemleri](#savunma-yöntemleri)
- [Kullanım Örnekleri](#kullanım-örnekleri)
- [Best Practices](#best-practices)

---

## 🌟 Genel Bakış

Adversarial Testing modülü, ML modellerinin kasıtlı olarak tasarlanmış saldırılara karşı dayanıklılığını test eder.

### Adversarial Attack Nedir?

```
┌─────────────────────────────────────────────────────────────────┐
│                   ADVERSARIAL ATTACK                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Normal Input          Perturbation         Adversarial Input   │
│  ┌──────────┐          ┌──────────┐         ┌──────────┐        │
│  │  DDoS    │    +     │  noise   │    =    │  Normal  │        │
│  │  %99     │          │  ε=0.01  │         │  %95     │        │
│  └──────────┘          └──────────┘         └──────────┘        │
│                                                                  │
│  Model görünmez bir perturbation ile kandırılır!                │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🎯 Neden Adversarial Testing?

### Güvenlik Riskleri

| Risk | Açıklama | Etki |
|------|----------|------|
| **Evasion Attack** | Saldırıyı normal trafik gibi gösterme | IDS bypass |
| **Poisoning Attack** | Eğitim verisini manipüle etme | Model bozulması |
| **Model Stealing** | Model parametrelerini çalma | IP kaybı |
| **Inference Attack** | Hassas veri çıkarımı | Gizlilik ihlali |

### Gerçek Dünya Senaryoları

1. **Malware evasion**: Zararlı yazılımı antivirüsten gizleme
2. **Spam bypass**: Spam filtresini atlama
3. **IDS evasion**: Saldırı trafiğini normal gösterme
4. **Fraud masking**: Sahte işlemleri gizleme

---

## 🎯 Desteklenen Saldırı Türleri

### 1. FGSM (Fast Gradient Sign Method)

**Açıklama**: Gradient yönünde tek adım perturbation

```python
# FGSM formülü
x_adv = x + ε * sign(∇_x L(θ, x, y))
```

**Özellikler:**

| Özellik | Değer |
|---------|-------|
| Hız | ⚡ Çok Hızlı |
| Etkililik | ★★★☆☆ |
| Algılanabilirlik | Orta |
| Parametre | ε (epsilon) |

**Kullanım:**

```python
from art.attacks.evasion import FastGradientMethod

attack = FastGradientMethod(estimator=classifier, eps=0.1)
x_adv = attack.generate(x=X_test)
```

### 2. PGD (Projected Gradient Descent)

**Açıklama**: İteratif gradient-based saldırı

```python
# PGD algoritması
for i in range(num_iterations):
    x_adv = x_adv + α * sign(∇_x L(θ, x_adv, y))
    x_adv = clip(x_adv, x - ε, x + ε)
```

**Özellikler:**

| Özellik | Değer |
|---------|-------|
| Hız | ⚡ Orta |
| Etkililik | ★★★★☆ |
| Algılanabilirlik | Düşük |
| Parametreler | ε, α, iterations |

**Kullanım:**

```python
from art.attacks.evasion import ProjectedGradientDescent

attack = ProjectedGradientDescent(
    estimator=classifier,
    eps=0.1,
    eps_step=0.01,
    max_iter=40
)
x_adv = attack.generate(x=X_test)
```

### 3. C&W (Carlini & Wagner)

**Açıklama**: Optimizasyon tabanlı güçlü saldırı

```python
# C&W objective
minimize ||δ||_p + c * f(x + δ)
subject to x + δ ∈ [0, 1]^n
```

**Özellikler:**

| Özellik | Değer |
|---------|-------|
| Hız | 🐢 Yavaş |
| Etkililik | ★★★★★ |
| Algılanabilirlik | Çok Düşük |
| Parametreler | c, κ, learning_rate |

**Kullanım:**

```python
from art.attacks.evasion import CarliniL2Method

attack = CarliniL2Method(
    classifier=classifier,
    targeted=False,
    max_iter=100
)
x_adv = attack.generate(x=X_test)
```

### 4. DeepFool

**Açıklama**: Minimum perturbation bulma

**Özellikler:**

| Özellik | Değer |
|---------|-------|
| Hız | ⚡ Orta |
| Etkililik | ★★★★☆ |
| Algılanabilirlik | Çok Düşük |

### 5. JSMA (Jacobian-based Saliency Map Attack)

**Açıklama**: Saliency map tabanlı hedefli saldırı

**Özellikler:**

| Özellik | Değer |
|---------|-------|
| Hız | 🐢 Yavaş |
| Etkililik | ★★★☆☆ |
| Sparse perturbation | ✅ |

### Saldırı Karşılaştırması

```
            Hız                    Etkililik
FGSM        █████████████████████  ██████░░░░░░░░░░░
PGD         ████████████░░░░░░░░░  ████████████░░░░░
C&W         ████░░░░░░░░░░░░░░░░░  █████████████████
DeepFool    ████████████░░░░░░░░░  ████████████░░░░░
JSMA        ████░░░░░░░░░░░░░░░░░  ██████████░░░░░░░
```

---

## 🔌 API Endpoints

### GET /api/adversarial/attack-types

```json
{
  "success": true,
  "data": {
    "attack_types": [
      {
        "id": "fgsm",
        "name": "Fast Gradient Sign Method",
        "description": "Single-step gradient attack",
        "speed": "very_fast",
        "effectiveness": "medium",
        "parameters": ["epsilon"]
      }
    ]
  }
}
```

### POST /api/adversarial/test

Model robustness testi çalıştır

**Request:**

```json
{
  "model_id": "best_cicids2017",
  "attack_type": "fgsm",
  "epsilon": 0.1,
  "iterations": 40,
  "sample_size": 1000,
  "targeted": false
}
```

**Response:**

```json
{
  "success": true,
  "data": {
    "test_id": "ADV-20260110-abc123",
    "model_id": "best_cicids2017",
    "attack_type": "fgsm",
    "results": {
      "original_accuracy": 99.88,
      "accuracy_under_attack": 85.42,
      "accuracy_drop": 14.46,
      "attack_success_rate": 14.46,
      "avg_perturbation": 0.087,
      "max_perturbation": 0.1
    },
    "verdict": "moderately_robust",
    "recommendations": [
      "Consider adversarial training",
      "Lower epsilon tolerance recommended"
    ]
  }
}
```

### POST /api/adversarial/simulate

Adversarial örnek oluştur

**Request:**

```json
{
  "model_id": "best_cicids2017",
  "original_features": [0.1, 0.2, ...],
  "attack_type": "pgd",
  "target_class": null
}
```

**Response:**

```json
{
  "success": true,
  "data": {
    "original_prediction": "DDoS",
    "adversarial_prediction": "Normal",
    "original_features": [...],
    "adversarial_features": [...],
    "perturbation": [...],
    "l2_distance": 0.0234,
    "linf_distance": 0.0087
  }
}
```

### GET /api/adversarial/robustness/{model_id}

```json
{
  "success": true,
  "data": {
    "model_id": "best_cicids2017",
    "robustness_score": 78,
    "tests": {
      "fgsm_0.01": {"accuracy": 98.2, "status": "pass"},
      "fgsm_0.1": {"accuracy": 85.4, "status": "warning"},
      "pgd_0.1": {"accuracy": 82.1, "status": "warning"},
      "cw": {"accuracy": 75.8, "status": "fail"}
    },
    "overall_verdict": "moderately_robust"
  }
}
```

### GET /api/adversarial/defense-methods

```json
{
  "success": true,
  "data": {
    "defense_methods": [
      {
        "id": "adversarial_training",
        "name": "Adversarial Training",
        "description": "Train with adversarial examples",
        "effectiveness": "high",
        "overhead": "medium"
      }
    ]
  }
}
```

---

## 📊 Robustness Değerlendirmesi

### Robustness Skoru (0-100)

| Skor | Derece | Açıklama |
|------|--------|----------|
| 90-100 | A | Çok Robust |
| 80-89 | B | Robust |
| 70-79 | C | Orta |
| 60-69 | D | Zayıf |
| 0-59 | F | Kritik Risk |

### Test Metrikleri

| Metrik | Açıklama |
|--------|----------|
| Accuracy under attack | Saldırı altında doğruluk |
| Attack success rate | Saldırı başarı oranı |
| Average perturbation | Ortalama perturbation miktarı |
| Certified radius | Garantili güvenli yarıçap |

---

## 🛡️ Savunma Yöntemleri

### 1. Adversarial Training

**Açıklama**: Adversarial örneklerle model eğitimi

```python
# Adversarial training
for epoch in range(epochs):
    for x, y in train_loader:
        # Normal eğitim
        loss_normal = criterion(model(x), y)
        
        # Adversarial örnek üret
        x_adv = generate_adversarial(x, y)
        loss_adv = criterion(model(x_adv), y)
        
        # Combine losses
        loss = loss_normal + α * loss_adv
        loss.backward()
        optimizer.step()
```

**Etkililik**: ★★★★★

### 2. Input Preprocessing

**Açıklama**: Giriş verilerini temizleme

```python
# JPEG compression
from torchvision import transforms
preprocess = transforms.Compose([
    transforms.Lambda(lambda x: jpeg_compress(x, quality=75)),
    transforms.GaussianBlur(kernel_size=3)
])
```

**Etkililik**: ★★★☆☆

### 3. Defensive Distillation

**Açıklama**: Model çıktılarını yumuşatma

```python
# Temperature scaling
def softmax_with_temperature(logits, T):
    return F.softmax(logits / T, dim=1)

# Train with high temperature
teacher_output = softmax_with_temperature(teacher_logits, T=20)
```

**Etkililik**: ★★★★☆

### 4. Gradient Masking

**Açıklama**: Gradient bilgisini gizleme

**Uyarı**: ⚠️ Güvenli değil, bypass edilebilir!

### 5. Ensemble Methods

**Açıklama**: Birden fazla model kullanma

```python
# Ensemble voting
predictions = []
for model in ensemble:
    predictions.append(model.predict(x))
final_prediction = majority_vote(predictions)
```

**Etkililik**: ★★★★☆

---

## 💻 Kullanım Örnekleri

### 1. Temel Robustness Testi

```python
import requests

# Test çalıştır
response = requests.post("/api/adversarial/test", json={
    "model_id": "best_cicids2017",
    "attack_type": "fgsm",
    "epsilon": 0.1,
    "sample_size": 1000
})

result = response.json()["data"]
print(f"Original Accuracy: {result['results']['original_accuracy']}%")
print(f"Under Attack: {result['results']['accuracy_under_attack']}%")
print(f"Verdict: {result['verdict']}")
```

### 2. Epsilon Sweep

```python
# Farklı epsilon değerleri ile test
epsilons = [0.01, 0.05, 0.1, 0.2, 0.3]
results = []

for eps in epsilons:
    resp = requests.post("/api/adversarial/test", json={
        "model_id": "best_cicids2017",
        "attack_type": "fgsm",
        "epsilon": eps
    })
    results.append({
        "epsilon": eps,
        "accuracy": resp.json()["data"]["results"]["accuracy_under_attack"]
    })

# Plot
import matplotlib.pyplot as plt
plt.plot([r["epsilon"] for r in results], [r["accuracy"] for r in results])
plt.xlabel("Epsilon")
plt.ylabel("Accuracy (%)")
plt.title("Robustness vs Perturbation Size")
plt.show()
```

### 3. Adversarial Örnek Görselleştirme

```python
# Adversarial örnek oluştur
response = requests.post("/api/adversarial/simulate", json={
    "model_id": "best_cicids2017",
    "original_features": sample.tolist(),
    "attack_type": "pgd"
})

data = response.json()["data"]
print(f"Original: {data['original_prediction']}")
print(f"Adversarial: {data['adversarial_prediction']}")
print(f"L2 Distance: {data['l2_distance']:.6f}")
```

---

## 📝 Best Practices

### 1. Test Stratejisi

```
┌─────────────────────────────────────────────────────────────────┐
│ 1. Başlangıç: FGSM ile hızlı test (ε=0.01, 0.05, 0.1)          │
│ 2. Derinlemesine: PGD ile iteratif test                         │
│ 3. En kötü durum: C&W ile güçlü saldırı                         │
│ 4. Sonuç: Robustness raporu oluştur                             │
└─────────────────────────────────────────────────────────────────┘
```

### 2. Kabul Kriterleri

| Saldırı | Min Accuracy | Epsilon |
|---------|--------------|---------|
| FGSM | 90% | 0.1 |
| PGD | 85% | 0.1 |
| C&W | 75% | - |

### 3. Continuous Testing

```yaml
# CI/CD pipeline'da adversarial test
adversarial_test:
  - fgsm_eps_0.05: min_accuracy: 95%
  - pgd_eps_0.1: min_accuracy: 85%
  - notify_on_failure: true
```

---

## 📚 Referanslar

- [Explaining and Harnessing Adversarial Examples](https://arxiv.org/abs/1412.6572) - Goodfellow et al.
- [Towards Evaluating the Robustness of Neural Networks](https://arxiv.org/abs/1608.04644) - Carlini & Wagner
- [Adversarial examples in the physical world](https://arxiv.org/abs/1607.02533)
- [Adversarial Robustness Toolbox (ART)](https://github.com/Trusted-AI/adversarial-robustness-toolbox)
- [CleverHans Library](https://github.com/cleverhans-lab/cleverhans)
