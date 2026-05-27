# CyberGuard AI - Tüm Dokümantasyon

**Oluşturulma Tarihi:** 2026-01-13

---

# İÇİNDEKİLER

- [adversarial_testing]
- [API_EXAMPLES]
- [api_endpoints_full]
- [api_reference]
- [architecture]
- [automl]
- [backup_recovery]
- [beyond_paper]
- [changelog]
- [ci_cd]
- [code_of_conduct]
- [contributing]
- [datasets]
- [deployment]
- [faq]
- [federated_learning]
- [github_upload]
- [github_upload_guide]
- [glossary]
- [installation]
- [KULLANIM_KILAVUZU]
- [LICENSE]
- [ml_models]
- [monitoring]
- [performance_tuning]
- [QUICK_START]
- [release_notes]
- [roadmap]
- [SECURITY_POLICY]
- [security]
- [security_hub]
- [testing]
- [troubleshooting]
- [user_guide]
- [WEBSOCKET_GUIDE]
- [xai]

---



# ADVERSARİAL_TESTİNG

# âš”ï¸ Adversarial Testing DokÃ¼mantasyonu

Model gÃ¼venliÄŸi ve adversarial attack test sistemi - DetaylÄ± Rehber

---

## ğŸ“‹ Ä°Ã§indekiler

- [Genel BakÄ±ÅŸ](#genel-bakÄ±ÅŸ)
- [Neden Adversarial Testing?](#neden-adversarial-testing)
- [Desteklenen SaldÄ±rÄ± TÃ¼rleri](#desteklenen-saldÄ±rÄ±-tÃ¼rleri)
- [API Endpoints](#api-endpoints)
- [Robustness DeÄŸerlendirmesi](#robustness-deÄŸerlendirmesi)
- [Savunma YÃ¶ntemleri](#savunma-yÃ¶ntemleri)
- [KullanÄ±m Ã–rnekleri](#kullanÄ±m-Ã¶rnekleri)
- [Best Practices](#best-practices)

---

## ğŸŒŸ Genel BakÄ±ÅŸ

Adversarial Testing modÃ¼lÃ¼, ML modellerinin kasÄ±tlÄ± olarak tasarlanmÄ±ÅŸ saldÄ±rÄ±lara karÅŸÄ± dayanÄ±klÄ±lÄ±ÄŸÄ±nÄ± test eder.

### Adversarial Attack Nedir?

```
â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
â”‚                   ADVERSARIAL ATTACK                             â”‚
â”œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¤
â”‚                                                                  â”‚
â”‚  Normal Input          Perturbation         Adversarial Input   â”‚
â”‚  â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”          â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”         â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”        â”‚
â”‚  â”‚  DDoS    â”‚    +     â”‚  noise   â”‚    =    â”‚  Normal  â”‚        â”‚
â”‚  â”‚  %99     â”‚          â”‚  Îµ=0.01  â”‚         â”‚  %95     â”‚        â”‚
â”‚  â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜          â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜         â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜        â”‚
â”‚                                                                  â”‚
â”‚  Model gÃ¶rÃ¼nmez bir perturbation ile kandÄ±rÄ±lÄ±r!                â”‚
â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜
```

---

## ğŸ¯ Neden Adversarial Testing?

### GÃ¼venlik Riskleri

| Risk | AÃ§Ä±klama | Etki |
|------|----------|------|
| **Evasion Attack** | SaldÄ±rÄ±yÄ± normal trafik gibi gÃ¶sterme | IDS bypass |
| **Poisoning Attack** | EÄŸitim verisini manipÃ¼le etme | Model bozulmasÄ± |
| **Model Stealing** | Model parametrelerini Ã§alma | IP kaybÄ± |
| **Inference Attack** | Hassas veri Ã§Ä±karÄ±mÄ± | Gizlilik ihlali |

### GerÃ§ek DÃ¼nya SenaryolarÄ±

1. **Malware evasion**: ZararlÄ± yazÄ±lÄ±mÄ± antivirÃ¼sten gizleme
2. **Spam bypass**: Spam filtresini atlama
3. **IDS evasion**: SaldÄ±rÄ± trafiÄŸini normal gÃ¶sterme
4. **Fraud masking**: Sahte iÅŸlemleri gizleme

---

## ğŸ¯ Desteklenen SaldÄ±rÄ± TÃ¼rleri

### 1. FGSM (Fast Gradient Sign Method)

**AÃ§Ä±klama**: Gradient yÃ¶nÃ¼nde tek adÄ±m perturbation

```python
# FGSM formÃ¼lÃ¼
x_adv = x + Îµ * sign(âˆ‡_x L(Î¸, x, y))
```

**Ã–zellikler:**

| Ã–zellik | DeÄŸer |
|---------|-------|
| HÄ±z | âš¡ Ã‡ok HÄ±zlÄ± |
| Etkililik | â˜…â˜…â˜…â˜†â˜† |
| AlgÄ±lanabilirlik | Orta |
| Parametre | Îµ (epsilon) |

**KullanÄ±m:**

```python
from art.attacks.evasion import FastGradientMethod

attack = FastGradientMethod(estimator=classifier, eps=0.1)
x_adv = attack.generate(x=X_test)
```

### 2. PGD (Projected Gradient Descent)

**AÃ§Ä±klama**: Ä°teratif gradient-based saldÄ±rÄ±

```python
# PGD algoritmasÄ±
for i in range(num_iterations):
    x_adv = x_adv + Î± * sign(âˆ‡_x L(Î¸, x_adv, y))
    x_adv = clip(x_adv, x - Îµ, x + Îµ)
```

**Ã–zellikler:**

| Ã–zellik | DeÄŸer |
|---------|-------|
| HÄ±z | âš¡ Orta |
| Etkililik | â˜…â˜…â˜…â˜…â˜† |
| AlgÄ±lanabilirlik | DÃ¼ÅŸÃ¼k |
| Parametreler | Îµ, Î±, iterations |

**KullanÄ±m:**

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

**AÃ§Ä±klama**: Optimizasyon tabanlÄ± gÃ¼Ã§lÃ¼ saldÄ±rÄ±

```python
# C&W objective
minimize ||Î´||_p + c * f(x + Î´)
subject to x + Î´ âˆˆ [0, 1]^n
```

**Ã–zellikler:**

| Ã–zellik | DeÄŸer |
|---------|-------|
| HÄ±z | ğŸ¢ YavaÅŸ |
| Etkililik | â˜…â˜…â˜…â˜…â˜… |
| AlgÄ±lanabilirlik | Ã‡ok DÃ¼ÅŸÃ¼k |
| Parametreler | c, Îº, learning_rate |

**KullanÄ±m:**

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

**AÃ§Ä±klama**: Minimum perturbation bulma

**Ã–zellikler:**

| Ã–zellik | DeÄŸer |
|---------|-------|
| HÄ±z | âš¡ Orta |
| Etkililik | â˜…â˜…â˜…â˜…â˜† |
| AlgÄ±lanabilirlik | Ã‡ok DÃ¼ÅŸÃ¼k |

### 5. JSMA (Jacobian-based Saliency Map Attack)

**AÃ§Ä±klama**: Saliency map tabanlÄ± hedefli saldÄ±rÄ±

**Ã–zellikler:**

| Ã–zellik | DeÄŸer |
|---------|-------|
| HÄ±z | ğŸ¢ YavaÅŸ |
| Etkililik | â˜…â˜…â˜…â˜†â˜† |
| Sparse perturbation | âœ… |

### SaldÄ±rÄ± KarÅŸÄ±laÅŸtÄ±rmasÄ±

```
            HÄ±z                    Etkililik
FGSM        â–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆ  â–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–‘â–‘â–‘â–‘â–‘â–‘â–‘â–‘â–‘â–‘â–‘
PGD         â–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–‘â–‘â–‘â–‘â–‘â–‘â–‘â–‘â–‘  â–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–‘â–‘â–‘â–‘â–‘
C&W         â–ˆâ–ˆâ–ˆâ–ˆâ–‘â–‘â–‘â–‘â–‘â–‘â–‘â–‘â–‘â–‘â–‘â–‘â–‘â–‘â–‘â–‘â–‘  â–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆ
DeepFool    â–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–‘â–‘â–‘â–‘â–‘â–‘â–‘â–‘â–‘  â–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–‘â–‘â–‘â–‘â–‘
JSMA        â–ˆâ–ˆâ–ˆâ–ˆâ–‘â–‘â–‘â–‘â–‘â–‘â–‘â–‘â–‘â–‘â–‘â–‘â–‘â–‘â–‘â–‘â–‘  â–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–‘â–‘â–‘â–‘â–‘â–‘â–‘
```

---

## ğŸ”Œ API Endpoints

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

Model robustness testi Ã§alÄ±ÅŸtÄ±r

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

Adversarial Ã¶rnek oluÅŸtur

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

## ğŸ“Š Robustness DeÄŸerlendirmesi

### Robustness Skoru (0-100)

| Skor | Derece | AÃ§Ä±klama |
|------|--------|----------|
| 90-100 | A | Ã‡ok Robust |
| 80-89 | B | Robust |
| 70-79 | C | Orta |
| 60-69 | D | ZayÄ±f |
| 0-59 | F | Kritik Risk |

### Test Metrikleri

| Metrik | AÃ§Ä±klama |
|--------|----------|
| Accuracy under attack | SaldÄ±rÄ± altÄ±nda doÄŸruluk |
| Attack success rate | SaldÄ±rÄ± baÅŸarÄ± oranÄ± |
| Average perturbation | Ortalama perturbation miktarÄ± |
| Certified radius | Garantili gÃ¼venli yarÄ±Ã§ap |

---

## ğŸ›¡ï¸ Savunma YÃ¶ntemleri

### 1. Adversarial Training

**AÃ§Ä±klama**: Adversarial Ã¶rneklerle model eÄŸitimi

```python
# Adversarial training
for epoch in range(epochs):
    for x, y in train_loader:
        # Normal eÄŸitim
        loss_normal = criterion(model(x), y)
        
        # Adversarial Ã¶rnek Ã¼ret
        x_adv = generate_adversarial(x, y)
        loss_adv = criterion(model(x_adv), y)
        
        # Combine losses
        loss = loss_normal + Î± * loss_adv
        loss.backward()
        optimizer.step()
```

**Etkililik**: â˜…â˜…â˜…â˜…â˜…

### 2. Input Preprocessing

**AÃ§Ä±klama**: GiriÅŸ verilerini temizleme

```python
# JPEG compression
from torchvision import transforms
preprocess = transforms.Compose([
    transforms.Lambda(lambda x: jpeg_compress(x, quality=75)),
    transforms.GaussianBlur(kernel_size=3)
])
```

**Etkililik**: â˜…â˜…â˜…â˜†â˜†

### 3. Defensive Distillation

**AÃ§Ä±klama**: Model Ã§Ä±ktÄ±larÄ±nÄ± yumuÅŸatma

```python
# Temperature scaling
def softmax_with_temperature(logits, T):
    return F.softmax(logits / T, dim=1)

# Train with high temperature
teacher_output = softmax_with_temperature(teacher_logits, T=20)
```

**Etkililik**: â˜…â˜…â˜…â˜…â˜†

### 4. Gradient Masking

**AÃ§Ä±klama**: Gradient bilgisini gizleme

**UyarÄ±**: âš ï¸ GÃ¼venli deÄŸil, bypass edilebilir!

### 5. Ensemble Methods

**AÃ§Ä±klama**: Birden fazla model kullanma

```python
# Ensemble voting
predictions = []
for model in ensemble:
    predictions.append(model.predict(x))
final_prediction = majority_vote(predictions)
```

**Etkililik**: â˜…â˜…â˜…â˜…â˜†

---

## ğŸ’» KullanÄ±m Ã–rnekleri

### 1. Temel Robustness Testi

```python
import requests

# Test Ã§alÄ±ÅŸtÄ±r
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
# FarklÄ± epsilon deÄŸerleri ile test
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

### 3. Adversarial Ã–rnek GÃ¶rselleÅŸtirme

```python
# Adversarial Ã¶rnek oluÅŸtur
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

## ğŸ“ Best Practices

### 1. Test Stratejisi

```
â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
â”‚ 1. BaÅŸlangÄ±Ã§: FGSM ile hÄ±zlÄ± test (Îµ=0.01, 0.05, 0.1)          â”‚
â”‚ 2. Derinlemesine: PGD ile iteratif test                         â”‚
â”‚ 3. En kÃ¶tÃ¼ durum: C&W ile gÃ¼Ã§lÃ¼ saldÄ±rÄ±                         â”‚
â”‚ 4. SonuÃ§: Robustness raporu oluÅŸtur                             â”‚
â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜
```

### 2. Kabul Kriterleri

| SaldÄ±rÄ± | Min Accuracy | Epsilon |
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

## ğŸ“š Referanslar

- [Explaining and Harnessing Adversarial Examples](https://arxiv.org/abs/1412.6572) - Goodfellow et al.
- [Towards Evaluating the Robustness of Neural Networks](https://arxiv.org/abs/1608.04644) - Carlini & Wagner
- [Adversarial examples in the physical world](https://arxiv.org/abs/1607.02533)
- [Adversarial Robustness Toolbox (ART)](https://github.com/Trusted-AI/adversarial-robustness-toolbox)
- [CleverHans Library](https://github.com/cleverhans-lab/cleverhans)


---


# API_EXAMPLES

# ğŸ”Œ CyberGuard AI - API Ã–rnekleri

Bu dokÃ¼manda CyberGuard AI API'sini kullanmak iÃ§in Ã¶rnek kodlar bulabilirsiniz.

---

## ğŸ“‹ Ä°Ã§indekiler

1. [Curl Ã–rnekleri](#curl-Ã¶rnekleri)
2. [Python Ã–rnekleri](#python-Ã¶rnekleri)
3. [JavaScript Ã–rnekleri](#javascript-Ã¶rnekleri)
4. [YaygÄ±n KullanÄ±m SenaryolarÄ±](#yaygÄ±n-kullanÄ±m-senaryolarÄ±)

---

## ğŸ”§ Curl Ã–rnekleri

### Dashboard Verisi

```bash
curl -X GET "http://localhost:8000/api/dashboard/stats" \
  -H "Content-Type: application/json"
```

### CanlÄ± SaldÄ±rÄ±lar

```bash
curl -X GET "http://localhost:8000/api/attack-map/live?limit=20" \
  -H "Content-Type: application/json"
```

### Ãœlke Ä°statistikleri

```bash
curl -X GET "http://localhost:8000/api/attack-map/countries" \
  -H "Content-Type: application/json"
```

### AÄŸ Durumu

```bash
curl -X GET "http://localhost:8000/api/network/status" \
  -H "Content-Type: application/json"
```

### Threat Hunting Sorgusu

```bash
curl -X POST "http://localhost:8000/api/threat-hunting/query" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "failed login",
    "timerange": "24h"
  }'
```

### AI Chat

```bash
curl -X POST "http://localhost:8000/api/chat/query" \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Bu IP zararlÄ± mÄ±: 192.168.1.100"
  }'
```

---

## ğŸ Python Ã–rnekleri

### Kurulum

```bash
pip install requests
```

### Temel KullanÄ±m

```python
import requests

BASE_URL = "http://localhost:8000/api"

# Dashboard verisi al
def get_dashboard():
    response = requests.get(f"{BASE_URL}/dashboard/stats")
    return response.json()

# CanlÄ± saldÄ±rÄ±larÄ± al
def get_live_attacks(limit=50):
    response = requests.get(f"{BASE_URL}/attack-map/live", params={"limit": limit})
    return response.json()

# AÄŸ durumu
def get_network_status():
    response = requests.get(f"{BASE_URL}/network/status")
    return response.json()

# KullanÄ±m
if __name__ == "__main__":
    print("Dashboard:", get_dashboard())
    print("Attacks:", get_live_attacks(10))
```

### ML Tahmin Ã–rneÄŸi

```python
import requests

def predict_threat(data):
    """ML modeli ile tehdit tahmini yap"""
    response = requests.post(
        "http://localhost:8000/api/prediction/predict",
        json={"features": data}
    )
    return response.json()

# Ã–rnek veri
sample_data = {
    "source_ip": "185.220.101.1",
    "target_port": 22,
    "protocol": "TCP",
    "bytes_sent": 1500,
    "duration": 3.5
}

result = predict_threat(sample_data)
print(f"Tehdit Skoru: {result.get('threat_score', 0)}")
print(f"SÄ±nÄ±flandÄ±rma: {result.get('classification', 'unknown')}")
```

### Threat Hunting

```python
import requests

def hunt_threats(query, timerange="24h"):
    """Tehdit avlama sorgusu Ã§alÄ±ÅŸtÄ±r"""
    response = requests.post(
        "http://localhost:8000/api/threat-hunting/query",
        json={
            "query": query,
            "timerange": timerange
        }
    )
    return response.json()

# Brute force tespiti
results = hunt_threats("failed login | authentication failure")
print(f"EÅŸleÅŸme sayÄ±sÄ±: {len(results.get('data', {}).get('results', []))}")
```

### Sandbox Analizi

```python
import requests

def analyze_file(file_path):
    """DosyayÄ± sandbox'ta analiz et"""
    with open(file_path, 'rb') as f:
        files = {'file': f}
        response = requests.post(
            "http://localhost:8000/api/sandbox/analyze",
            files=files
        )
    return response.json()

# Ã–rnek kullanÄ±m
result = analyze_file("suspicious_file.exe")
print(f"Risk Skoru: {result.get('data', {}).get('risk_score', 0)}")
print(f"SonuÃ§: {result.get('data', {}).get('verdict', 'unknown')}")
```

---

## ğŸ“œ JavaScript Ã–rnekleri

### Fetch API

```javascript
const BASE_URL = 'http://localhost:8000/api';

// Dashboard verisi
async function getDashboard() {
    const response = await fetch(`${BASE_URL}/dashboard/stats`);
    return response.json();
}

// CanlÄ± saldÄ±rÄ±lar
async function getLiveAttacks(limit = 50) {
    const response = await fetch(`${BASE_URL}/attack-map/live?limit=${limit}`);
    return response.json();
}

// AI Chat
async function askAI(message) {
    const response = await fetch(`${BASE_URL}/chat/query`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message })
    });
    return response.json();
}

// KullanÄ±m
getDashboard().then(data => console.log('Dashboard:', data));
askAI('DDoS saldÄ±rÄ±sÄ±na karÅŸÄ± ne yapmalÄ±yÄ±m?').then(data => console.log('AI:', data));
```

### Axios

```javascript
import axios from 'axios';

const api = axios.create({
    baseURL: 'http://localhost:8000/api',
    timeout: 10000,
    headers: { 'Content-Type': 'application/json' }
});

// Dashboard
const getDashboard = async () => {
    const { data } = await api.get('/dashboard/stats');
    return data;
};

// SaldÄ±rÄ±lar
const getAttacks = async (limit = 50) => {
    const { data } = await api.get('/attack-map/live', { params: { limit } });
    return data;
};

// Threat Hunting
const huntThreats = async (query, timerange = '24h') => {
    const { data } = await api.post('/threat-hunting/query', { query, timerange });
    return data;
};

export { getDashboard, getAttacks, huntThreats };
```

---

## ğŸ“Š YaygÄ±n KullanÄ±m SenaryolarÄ±

### Senaryo 1: GÃ¼venlik Dashboard OluÅŸturma

```python
import requests
import time

def create_security_dashboard():
    """GÃ¼venlik Ã¶zeti oluÅŸtur"""
    base = "http://localhost:8000/api"
    
    # Verileri topla
    dashboard = requests.get(f"{base}/dashboard/stats").json()
    attacks = requests.get(f"{base}/attack-map/live?limit=10").json()
    network = requests.get(f"{base}/network/status").json()
    security = requests.get(f"{base}/security/score").json()
    
    # Ã–zet oluÅŸtur
    summary = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "security_score": security.get("data", {}).get("score", 0),
        "active_attacks": len(attacks.get("data", {}).get("attacks", [])),
        "network_status": network.get("data", {}).get("status", "unknown"),
        "alerts": dashboard.get("data", {}).get("alerts", 0)
    }
    
    return summary

print(create_security_dashboard())
```

### Senaryo 2: Otomatik Tehdit Tespiti

```python
import requests
import time

def monitor_threats(interval=60):
    """Tehdit izleme dÃ¶ngÃ¼sÃ¼"""
    while True:
        attacks = requests.get(
            "http://localhost:8000/api/attack-map/live?limit=50"
        ).json()
        
        for attack in attacks.get("data", {}).get("attacks", []):
            if attack.get("ml_prediction", {}).get("is_threat"):
                print(f"âš ï¸ TEHDIT: {attack.get('source', {}).get('ip')} -> {attack.get('target', {}).get('ip')}")
                print(f"   Tip: {attack.get('attack_type')}")
                print(f"   GÃ¼ven: {attack.get('ml_prediction', {}).get('confidence', 0):.1%}")
        
        time.sleep(interval)

# monitor_threats(30)  # Her 30 saniyede kontrol
```

### Senaryo 3: Rapor OluÅŸturma

```python
import requests
import json
from datetime import datetime

def generate_report():
    """GÃ¼nlÃ¼k gÃ¼venlik raporu"""
    base = "http://localhost:8000/api"
    
    report = {
        "title": "GÃ¼nlÃ¼k GÃ¼venlik Raporu",
        "date": datetime.now().isoformat(),
        "sections": {}
    }
    
    # SaldÄ±rÄ± Ã¶zeti
    attacks = requests.get(f"{base}/attack-map/stats").json()
    report["sections"]["attacks"] = attacks.get("data", {})
    
    # Ãœlke daÄŸÄ±lÄ±mÄ±
    countries = requests.get(f"{base}/attack-map/countries").json()
    report["sections"]["countries"] = countries.get("data", {}).get("countries", [])[:5]
    
    # ML istatistikleri
    ml_stats = requests.get(f"{base}/models/stats").json()
    report["sections"]["ml"] = ml_stats.get("data", {})
    
    # Kaydet
    with open(f"report_{datetime.now().strftime('%Y%m%d')}.json", "w") as f:
        json.dump(report, f, indent=2)
    
    return report

print(json.dumps(generate_report(), indent=2))
```

---

## ğŸ”— API Endpoint Listesi

| Kategori | Endpoint | Metod | AÃ§Ä±klama |
| -------- | -------- | ----- | -------- |
| Dashboard | `/dashboard/stats` | GET | Genel istatistikler |
| Attack Map | `/attack-map/live` | GET | CanlÄ± saldÄ±rÄ±lar |
| Attack Map | `/attack-map/countries` | GET | Ãœlke bazlÄ± veriler |
| Network | `/network/status` | GET | AÄŸ durumu |
| Network | `/network/interfaces` | GET | Interface listesi |
| Threat Hunting | `/threat-hunting/query` | POST | Sorgu Ã§alÄ±ÅŸtÄ±r |
| Security | `/security/score` | GET | GÃ¼venlik skoru |
| Chat | `/chat/query` | POST | AI sohbet |
| Sandbox | `/sandbox/analyze` | POST | Dosya analizi |

**Tam liste iÃ§in:** <http://localhost:8000/api/docs>

---

**ğŸ”Œ Kolay entegrasyon, gÃ¼Ã§lÃ¼ gÃ¼venlik!**


---


# APİ_ENDPOİNTS_FULL

# ğŸ”Œ API Endpoints - Tam Liste

CyberGuard AI'daki tÃ¼m API endpoint'leri

---

## ğŸ“Š Genel BakÄ±ÅŸ

| Kategori | Endpoint SayÄ±sÄ± |
|----------|-----------------|
| Authentication | 5 |
| Dashboard | 8 |
| Prediction | 10 |
| Network | 12 |
| Reports | 8 |
| Chatbot | 6 |
| XAI | 4 |
| Adversarial | 5 |
| Federated | 6 |
| AutoML | 5 |
| Threat Intelligence | 5 |
| Alerts | 5 |
| Security Advanced | 10 |
| Vulnerability | 4 |
| Log Analyzer | 5 |
| Incidents | 6 |
| API Keys | 5 |
| Settings | 4 |
| **TOPLAM** | **113+** |

---

## ğŸ” Authentication

| Method | Endpoint | AÃ§Ä±klama |
|--------|----------|----------|
| POST | `/api/auth/login` | KullanÄ±cÄ± giriÅŸi |
| POST | `/api/auth/logout` | Ã‡Ä±kÄ±ÅŸ |
| POST | `/api/auth/refresh` | Token yenile |
| POST | `/api/auth/register` | KayÄ±t (admin) |
| GET | `/api/auth/me` | Mevcut kullanÄ±cÄ± |

---

## ğŸ“Š Dashboard

| Method | Endpoint | AÃ§Ä±klama |
|--------|----------|----------|
| GET | `/api/dashboard` | Ana dashboard |
| GET | `/api/dashboard/stats` | Ä°statistikler |
| GET | `/api/dashboard/threats` | Tehdit Ã¶zeti |
| GET | `/api/dashboard/timeline` | 24 saat timeline |
| GET | `/api/dashboard/models` | Model durumlarÄ± |
| GET | `/api/dashboard/system` | Sistem metrikleri |
| GET | `/api/dashboard/recent` | Son aktiviteler |
| GET | `/api/dashboard/quick-actions` | HÄ±zlÄ± eylemler |

---

## ğŸ¯ Prediction

| Method | Endpoint | AÃ§Ä±klama |
|--------|----------|----------|
| POST | `/api/prediction/predict` | Tek tahmin |
| POST | `/api/prediction/bulk` | Toplu tahmin |
| GET | `/api/prediction/models` | Model listesi |
| PUT | `/api/prediction/model` | Aktif model deÄŸiÅŸtir |
| GET | `/api/prediction/stats` | Tahmin istatistikleri |
| POST | `/api/prediction/realtime` | GerÃ§ek zamanlÄ± tahmin |
| GET | `/api/prediction/history` | Tahmin geÃ§miÅŸi |
| GET | `/api/prediction/confidence` | GÃ¼ven eÅŸikleri |
| POST | `/api/prediction/validate` | Input validasyon |
| GET | `/api/prediction/classes` | SÄ±nÄ±f listesi |

---

## ğŸŒ Network

| Method | Endpoint | AÃ§Ä±klama |
|--------|----------|----------|
| GET | `/api/network/attacks` | SaldÄ±rÄ± listesi |
| GET | `/api/network/attacks/{id}` | SaldÄ±rÄ± detayÄ± |
| GET | `/api/network/stats` | AÄŸ istatistikleri |
| GET | `/api/network/traffic` | Trafik verileri |
| GET | `/api/network/top-ips` | En aktif IP'ler |
| GET | `/api/network/geo` | CoÄŸrafi daÄŸÄ±lÄ±m |
| GET | `/api/network/timeline` | Zaman Ã§izelgesi |
| GET | `/api/network/protocols` | Protokol daÄŸÄ±lÄ±mÄ± |
| GET | `/api/network/ports` | Port istatistikleri |
| POST | `/api/network/analyze` | Trafik analizi |
| GET | `/api/network/flows` | Flow verileri |
| GET | `/api/network/bandwidth` | Bant geniÅŸliÄŸi |

---

## ğŸ“‹ Reports

| Method | Endpoint | AÃ§Ä±klama |
|--------|----------|----------|
| POST | `/api/reports/create` | Rapor oluÅŸtur |
| GET | `/api/reports/list` | Rapor listesi |
| GET | `/api/reports/{id}` | Rapor detayÄ± |
| GET | `/api/reports/{id}/download` | Rapor indir |
| DELETE | `/api/reports/{id}` | Rapor sil |
| POST | `/api/reports/schedule` | Planla |
| GET | `/api/reports/templates` | Åablonlar |
| POST | `/api/reports/export` | DÄ±ÅŸa aktar |

---

## ğŸ¤– Chatbot

| Method | Endpoint | AÃ§Ä±klama |
|--------|----------|----------|
| POST | `/api/chatbot/chat` | Mesaj gÃ¶nder |
| GET | `/api/chatbot/history` | GeÃ§miÅŸ |
| DELETE | `/api/chatbot/clear` | GeÃ§miÅŸi temizle |
| POST | `/api/chatbot/analyze` | Dosya analizi |
| GET | `/api/chatbot/suggestions` | Ã–neriler |
| POST | `/api/chatbot/command` | Komut Ã§alÄ±ÅŸtÄ±r |

---

## ğŸ” XAI (Explainable AI)

| Method | Endpoint | AÃ§Ä±klama |
|--------|----------|----------|
| POST | `/api/xai/explain` | Model aÃ§Ä±klamasÄ± |
| GET | `/api/xai/feature-importance/{model_id}` | Feature importance |
| GET | `/api/xai/global-importance` | Global importance |
| GET | `/api/xai/explanation-methods` | Mevcut metodlar |

---

## âš”ï¸ Adversarial

| Method | Endpoint | AÃ§Ä±klama |
|--------|----------|----------|
| GET | `/api/adversarial/attack-types` | SaldÄ±rÄ± tÃ¼rleri |
| POST | `/api/adversarial/test` | Robustness testi |
| POST | `/api/adversarial/simulate` | SaldÄ±rÄ± simÃ¼lasyonu |
| GET | `/api/adversarial/robustness/{model_id}` | Robustness skoru |
| GET | `/api/adversarial/defense-methods` | Savunma yÃ¶ntemleri |

---

## ğŸ”— Federated Learning

| Method | Endpoint | AÃ§Ä±klama |
|--------|----------|----------|
| GET | `/api/federated/status` | Sistem durumu |
| POST | `/api/federated/clients` | Client ekle |
| DELETE | `/api/federated/clients/{id}` | Client sil |
| POST | `/api/federated/start` | EÄŸitim baÅŸlat |
| GET | `/api/federated/aggregation` | Aggregation metodlarÄ± |
| GET | `/api/federated/privacy` | Gizlilik Ã¶zellikleri |

---

## ğŸ¤– AutoML

| Method | Endpoint | AÃ§Ä±klama |
|--------|----------|----------|
| POST | `/api/automl/start` | Job baÅŸlat |
| GET | `/api/automl/status/{job_id}` | Job durumu |
| GET | `/api/automl/algorithms` | Algoritmalar |
| GET | `/api/automl/recommendations` | Ã–neriler |
| POST | `/api/automl/hyperparameter-search` | HP aramasÄ± |

---

## ğŸ•µï¸ Threat Intelligence

| Method | Endpoint | AÃ§Ä±klama |
|--------|----------|----------|
| POST | `/api/threat-intel/check-ip` | IP kontrolÃ¼ |
| POST | `/api/threat-intel/check-domain` | Domain kontrolÃ¼ |
| POST | `/api/threat-intel/check-hash` | Hash kontrolÃ¼ |
| GET | `/api/threat-intel/feeds` | Threat feed'leri |
| GET | `/api/threat-intel/ioc` | IOC listesi |

---

## ğŸ“§ Alerts

| Method | Endpoint | AÃ§Ä±klama |
|--------|----------|----------|
| POST | `/api/alerts/send` | Alert gÃ¶nder |
| GET | `/api/alerts/config` | KonfigÃ¼rasyon |
| PUT | `/api/alerts/config` | Config gÃ¼ncelle |
| GET | `/api/alerts/history` | Alert geÃ§miÅŸi |
| POST | `/api/alerts/test` | Test gÃ¶nder |

---

## ğŸ›¡ï¸ Security Advanced

| Method | Endpoint | AÃ§Ä±klama |
|--------|----------|----------|
| POST | `/api/security/analyze-pcap` | PCAP analizi |
| GET | `/api/security/score` | GÃ¼venlik skoru |
| GET | `/api/security/honeypot` | Honeypot durumu |
| GET | `/api/security/compliance` | Uyumluluk |
| GET | `/api/security/attack-replay` | SaldÄ±rÄ± replay |
| GET | `/api/security/topology` | AÄŸ topolojisi |
| GET | `/api/security/heatmap` | Tehdit haritasÄ± |
| POST | `/api/security/scan-network` | AÄŸ tarama |
| GET | `/api/security/audit-log` | Audit log |
| GET | `/api/security/risk-scores` | Risk skorlarÄ± |

---

## ğŸ” Vulnerability Scanner

| Method | Endpoint | AÃ§Ä±klama |
|--------|----------|----------|
| POST | `/api/vuln/scan` | Zafiyet taramasÄ± |
| GET | `/api/vuln/cve/{cve_id}` | CVE detayÄ± |
| POST | `/api/vuln/port-scan` | Port tarama |
| GET | `/api/vuln/history` | Tarama geÃ§miÅŸi |

---

## ğŸ“‹ Log Analyzer

| Method | Endpoint | AÃ§Ä±klama |
|--------|----------|----------|
| POST | `/api/logs-analysis/analyze` | Log analizi |
| GET | `/api/logs-analysis/anomalies` | Anomaliler |
| POST | `/api/logs-analysis/upload` | Log yÃ¼kle |
| GET | `/api/logs-analysis/patterns` | SaldÄ±rÄ± pattern'leri |
| GET | `/api/logs-analysis/stats` | Ä°statistikler |

---

## â±ï¸ Incidents

| Method | Endpoint | AÃ§Ä±klama |
|--------|----------|----------|
| GET | `/api/incidents/timeline` | Olay zaman Ã§izelgesi |
| POST | `/api/incidents/add` | Olay ekle |
| GET | `/api/incidents/detail/{id}` | Olay detayÄ± |
| GET | `/api/incidents/behavior/users` | KullanÄ±cÄ± davranÄ±ÅŸlarÄ± |
| GET | `/api/incidents/behavior/anomalies` | DavranÄ±ÅŸ anomalileri |
| GET | `/api/incidents/behavior/user/{id}` | KullanÄ±cÄ± detayÄ± |

---

## ğŸ”‘ API Keys

| Method | Endpoint | AÃ§Ä±klama |
|--------|----------|----------|
| GET | `/api/keys` | Anahtar listesi |
| POST | `/api/keys` | Yeni anahtar |
| PUT | `/api/keys/{key_id}` | GÃ¼ncelle |
| DELETE | `/api/keys/{key_id}` | Sil |
| GET | `/api/keys/{key_id}/usage` | KullanÄ±m istatistikleri |

---

## âš™ï¸ Settings

| Method | Endpoint | AÃ§Ä±klama |
|--------|----------|----------|
| GET | `/api/settings/general` | Genel ayarlar |
| PUT | `/api/settings/general` | AyarlarÄ± gÃ¼ncelle |
| GET | `/api/settings/notifications` | Bildirim ayarlarÄ± |
| PUT | `/api/settings/notifications` | Bildirim gÃ¼ncelle |

---

## ğŸ“ Response Format

### BaÅŸarÄ±lÄ±

```json
{
  "success": true,
  "data": {...},
  "message": "Ä°ÅŸlem baÅŸarÄ±lÄ±"
}
```

### Hata

```json
{
  "success": false,
  "error": "Error type",
  "message": "Hata aÃ§Ä±klamasÄ±"
}
```

---

## ğŸ” Authentication

TÃ¼m endpoint'ler (auth hariÃ§) JWT token gerektirir:

```
Authorization: Bearer <token>
```

---

## âš¡ Rate Limits

| Plan | Limit |
|------|-------|
| Community | 100/dakika |
| Pro | 1000/dakika |
| Enterprise | Unlimited |


---


# APİ_REFERENCE

# ğŸ”Œ API Reference

CyberGuard AI API dokÃ¼mantasyonu

---

## ğŸ“‹ Ä°Ã§indekiler

- [Genel BakÄ±ÅŸ](#genel-bakÄ±ÅŸ)
- [Authentication](#authentication)
- [Core Modules](#core-modules)
- [Chatbot API](#chatbot-api)
- [ML Model API](#ml-model-api)
- [Database API](#database-api)
- [Utilities API](#utilities-api)

---

## ğŸŒŸ Genel BakÄ±ÅŸ

CyberGuard AI, modÃ¼ler bir yapÄ±ya sahiptir. Her modÃ¼l baÄŸÄ±msÄ±z olarak kullanÄ±labilir.

### Base Configuration

```python
from src.utils import get_config

config = get_config()
# config.yaml dosyasÄ±nÄ± yÃ¼kler
```

---

## ğŸ” Authentication

### Gemini API Key

```python
# .env dosyasÄ±nda
GOOGLE_API_KEY=your_api_key_here

# KullanÄ±m
from src.chatbot import GeminiHandler

chatbot = GeminiHandler()
```

---

## ğŸ§© Core Modules

### Database Manager

```python
from src.utils.database import DatabaseManager

db = DatabaseManager(db_path='cyberguard.db')

# Ä°statistik al
stats = db.get_database_stats()

# SaldÄ±rÄ± ekle
db.add_attack({
    'attack_type': 'DDoS',
    'source_ip': '192.168.1.100',
    'destination_ip': '192.168.0.10',
    'port': 80,
    'severity': 'high',
    'blocked': True
})

# SaldÄ±rÄ±larÄ± Ã§ek
attacks = db.get_attacks(limit=100)
```

### Logger

```python
from src.utils.logger import Logger

logger = Logger("MyModule")

logger.info("Bilgi mesajÄ±")
logger.warning("UyarÄ± mesajÄ±")
logger.error("Hata mesajÄ±")
logger.critical("Kritik hata")
```

### Config Manager

```python
from src.utils.config import Config

config = Config()

# DeÄŸer al
db_path = config.get('database', 'path')

# DeÄŸer set et
config.set('model', 'accuracy', 0.95)
```

---

## ğŸ¤– Chatbot API

### GeminiHandler

```python
from src.chatbot.gemini_handler import GeminiHandler

chatbot = GeminiHandler()

# Basit sohbet
response = chatbot.chat("Merhaba!")

# Context ile sohbet
context = {
    'total_attacks': 5000,
    'by_severity': {'critical': 10, 'high': 20}
}
response = chatbot.chat("Son saldÄ±rÄ±larÄ± gÃ¶ster", context=context)

# KonuÅŸma geÃ§miÅŸini temizle
chatbot.clear_history()

# KonuÅŸmayÄ± kaydet
filename = chatbot.export_conversation()
```

#### Available Methods

| Method | Parameters | Returns | Description |
|--------|-----------|---------|-------------|
| `chat()` | `message: str, context: dict` | `str` | Mesaj gÃ¶nder, cevap al |
| `clear_history()` | - | - | KonuÅŸma geÃ§miÅŸini temizle |
| `export_conversation()` | `filename: str` | `str` | KonuÅŸmayÄ± JSON'a aktar |
| `get_attack_context()` | `hours: int` | `dict` | SaldÄ±rÄ± context'i oluÅŸtur |
| `get_ip_context()` | `ip: str` | `dict` | IP context'i oluÅŸtur |
| `get_system_context()` | - | `dict` | Sistem context'i oluÅŸtur |

---

## ğŸ§  RAG System API

### RAG Manager

```python
from src.chatbot.vectorstore.rag_manager import RAGManager

rag = RAGManager()

# DÃ¶kÃ¼man ekle
rag.add_text_document(
    text="DDoS saldÄ±rÄ±sÄ± nedir...",
    metadata={'title': 'DDoS Rehberi', 'category': 'Security'}
)

# PDF ekle
rag.add_pdf_document(
    pdf_path='security_guide.pdf',
    metadata={'title': 'Security Guide'}
)

# Arama yap
results = rag.search("DDoS saldÄ±rÄ±sÄ±ndan nasÄ±l korunurum?", k=3)

# Context oluÅŸtur
context = rag.get_context_for_query("DDoS nedir?", k=3)

# Ä°statistikler
stats = rag.get_stats()

# TÃ¼mÃ¼nÃ¼ sil
rag.delete_all_documents()
```

### Memory Manager

```python
from src.chatbot.vectorstore.memory_manager import MemoryManager

memory = MemoryManager(user_id="user123")

# KonuÅŸma ekle
memory.add_conversation(
    user_message="DDoS nedir?",
    bot_response="DDoS, distributed denial of service...",
    context={'source': 'chatbot'}
)

# HafÄ±zada ara
results = memory.search_memory("saldÄ±rÄ± sayÄ±sÄ±", k=3)

# Son konuÅŸmalarÄ± al
context = memory.get_recent_context(n=5)

# Ä°lgili konuÅŸmalarÄ± al
relevant = memory.get_relevant_memory_for_query("DDoS", k=2)

# Temizle
memory.clear_short_term()
memory.clear_all_memory()
```

### Attack Vector Manager

```python
from src.chatbot.vectorstore.attack_vectors import AttackVectorManager

attack_vectors = AttackVectorManager()

# Database'i vektÃ¶rleÅŸtir
attack_vectors.vectorize_attacks(limit=1000)

# Benzer saldÄ±rÄ± bul
results = attack_vectors.find_similar_attacks("DDoS saldÄ±rÄ±sÄ±", k=5)

# Pattern analizi
analysis = attack_vectors.analyze_attack_pattern("DDoS")

# Chatbot iÃ§in Ã¶zet
summary = attack_vectors.get_attack_summary_for_chatbot("Port Scan")

# Temizle
attack_vectors.clear_vectors()
```

---

## ğŸ¯ ML Model API

### Model Predictor

```python
from src.models.predictor import AttackPredictor

predictor = AttackPredictor()

# Model yÃ¼kle
predictor.load_models()

# Tahmin yap
attack_data = {
    'source_ip': '192.168.1.105',
    'destination_ip': '192.168.0.10',
    'port': 80,
    'severity': 'critical',
    'blocked': 1,
    'timestamp': '2024-10-29 14:30:00'
}

result = predictor.predict_single(attack_data)
# Returns: {
#     'predicted_type': 'DDoS',
#     'confidence': 0.98,
#     'probabilities': {...},
#     'risk_level': 'critical'
# }

# Toplu tahmin
results = predictor.predict_batch([attack1, attack2, attack3])

# Model bilgisi
info = predictor.get_model_info()
```

### Model Training

```python
from train_model import ModelTrainer

trainer = ModelTrainer(db_path='cyberguard.db')

# Tam eÄŸitim pipeline
trainer.run_full_training(limit=5000)

# Manuel eÄŸitim
df = trainer.load_data_from_db(limit=1000)
X_train, X_test, y_train, y_test = trainer.prepare_data(df)
trainer.train_model(X_train, y_train)
metrics = trainer.evaluate_model(X_test, y_test)
trainer.save_models()
```

---

## ğŸ’¾ Database API

### Attack Operations

```python
from src.utils.database import DatabaseManager

db = DatabaseManager()

# SaldÄ±rÄ± ekle
attack_id = db.add_attack({
    'attack_type': 'SQL Injection',
    'source_ip': '10.0.0.1',
    'destination_ip': '192.168.1.1',
    'port': 3306,
    'severity': 'high',
    'blocked': True,
    'description': 'SQL injection attempt detected'
})

# SaldÄ±rÄ±larÄ± Ã§ek
attacks = db.get_attacks(limit=100)

# Filtreleme
ddos_attacks = db.get_attacks_by_type('DDoS')
critical_attacks = db.get_attacks_by_severity('critical')

# IP bazlÄ± arama
ip_attacks = db.get_attacks_by_ip('192.168.1.100')

# Zaman aralÄ±ÄŸÄ±
recent_attacks = db.get_attacks_last_hours(24)

# Ä°statistikler
stats = db.get_database_stats()
# Returns: {
#     'attacks': 5000,
#     'network_logs': 10000,
#     'scan_results': 500,
#     'db_size_mb': 25.5
# }
```

### Scan Operations

```python
# Tarama ekle
scan_id = db.add_scan_result({
    'filename': 'suspicious.exe',
    'file_hash': 'abc123...',
    'scan_result': 'Threat Detected',
    'threat_type': 'Trojan',
    'risk_score': 95,
    'is_malicious': True
})

# Tarama geÃ§miÅŸi
scans = db.get_scan_history(limit=50)

# ZararlÄ± dosyalar
malicious = db.get_malicious_files()
```

---

## ğŸ› ï¸ Utilities API

### Mock Data Generator

```python
from src.utils.mock_data_generator import MockDataGenerator

generator = MockDataGenerator(db_path='cyberguard.db')

# Veri Ã¼ret
generator.generate_all(
    attack_count=5000,
    log_count=10000,
    scan_count=2500,
    clear_first=True
)

# Manuel Ã¼retim
attacks = generator.generate_attacks(count=100)
logs = generator.generate_logs(count=200)
scans = generator.generate_network_scans(count=50)

# Database'e ekle
generator.insert_to_database(attacks, logs, scans)

# Temizle
generator.clear_database()
```

### PDF Report Generator

```python
from src.utils.pdf_generator import PDFReportGenerator

pdf_gen = PDFReportGenerator(db_path='cyberguard.db')

# Rapor oluÅŸtur
filename = pdf_gen.generate_report(
    output_filename='security_report.pdf',
    days=7,
    include_charts=True
)

# Ä°statistik al
stats = pdf_gen.get_attack_stats(days=7)

# Grafik oluÅŸtur
pie_chart = pdf_gen.create_pie_chart(data, title='SaldÄ±rÄ± DaÄŸÄ±lÄ±mÄ±')
bar_chart = pdf_gen.create_bar_chart(data, title='Severity')
```

### Feature Extractor

```python
from src.utils.feature_extractor import FeatureExtractor

extractor = FeatureExtractor()

# DataFrame'den Ã¶zellik Ã§Ä±kar
X = extractor.prepare_features(df, fit=True)
y = extractor.prepare_labels(df, fit=True)

# Kaydet/YÃ¼kle
extractor.save('models/feature_extractor.pkl')
extractor.load('models/feature_extractor.pkl')

# SÄ±nÄ±f ismi al
attack_name = extractor.get_attack_type_name(encoded_label=5)
```

---

## ğŸ”§ Error Handling

TÃ¼m API fonksiyonlarÄ± exception fÄ±rlatabilir:

```python
try:
    result = predictor.predict_single(attack_data)
except FileNotFoundError:
    print("Model dosyasÄ± bulunamadÄ±!")
except ValueError:
    print("GeÃ§ersiz veri!")
except Exception as e:
    print(f"Beklenmeyen hata: {e}")
```

---

## ğŸ“Š Response Formats

### Standard Response

```json
{
  "success": true,
  "data": {...},
  "message": "Ä°ÅŸlem baÅŸarÄ±lÄ±",
  "timestamp": "2024-10-29T14:30:00"
}
```

### Error Response

```json
{
  "success": false,
  "error": "ValueError",
  "message": "GeÃ§ersiz veri formatÄ±",
  "timestamp": "2024-10-29T14:30:00"
}
```

---

## ğŸ”— Ã–rnek KullanÄ±m SenaryolarÄ±

### Senaryo 1: GerÃ§ek ZamanlÄ± SaldÄ±rÄ± Tespiti

```python
from src.models.predictor import AttackPredictor
from src.utils.database import DatabaseManager

# Model yÃ¼kle
predictor = AttackPredictor()
db = DatabaseManager()

# Yeni trafik verisi geldiÄŸinde
new_traffic = {
    'source_ip': '192.168.1.105',
    'destination_ip': '192.168.0.10',
    'port': 80,
    'severity': 'high',
    'blocked': 0,
    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
}

# Tahmin yap
result = predictor.predict_single(new_traffic)

# EÄŸer saldÄ±rÄ±ysa database'e kaydet
if result['predicted_type'] != 'Normal':
    new_traffic['attack_type'] = result['predicted_type']
    db.add_attack(new_traffic)
    
    # Alarm gÃ¶nder
    if result['risk_level'] == 'critical':
        send_alert(f"Kritik saldÄ±rÄ±: {result['predicted_type']}")
```

### Senaryo 2: AI Asistan ile Analiz

```python
from src.chatbot.gemini_handler import GeminiHandler
from src.chatbot.vectorstore.attack_vectors import AttackVectorManager

chatbot = GeminiHandler()
attack_vectors = AttackVectorManager()

# KullanÄ±cÄ± sorusu
user_question = "Son 24 saatte en Ã§ok hangi tÃ¼r saldÄ±rÄ± oldu?"

# Context oluÅŸtur
context = chatbot.get_attack_context(hours=24)
similar_attacks = attack_vectors.get_attack_summary_for_chatbot(user_question)

# Context'i birleÅŸtir
full_context = {**context, 'similar_attacks': similar_attacks}

# Cevap al
response = chatbot.chat(user_question, context=full_context)
print(response)
```

### Senaryo 3: HaftalÄ±k Rapor Otomasyonu

```python
from src.utils.pdf_generator import PDFReportGenerator
import schedule
import time

def weekly_report():
    pdf_gen = PDFReportGenerator()
    filename = pdf_gen.generate_report(
        output_filename=f'weekly_report_{datetime.now().strftime("%Y%m%d")}.pdf',
        days=7,
        include_charts=True
    )
    send_email_with_attachment(filename)

# Her pazartesi 09:00'da Ã§alÄ±ÅŸtÄ±r
schedule.every().monday.at("09:00").do(weekly_report)

while True:
    schedule.run_pending()
    time.sleep(60)
```

---

## ğŸ“š Ä°leri Seviye KullanÄ±m

### Custom Model Training

```python
from src.models.random_forest_model import CyberAttackModel
from src.utils.feature_extractor import FeatureExtractor
import pandas as pd

# Kendi verinizi yÃ¼kleyin
df = pd.read_csv('my_attack_data.csv')

# Feature extraction
extractor = FeatureExtractor()
X = extractor.prepare_features(df, fit=True)
y = extractor.prepare_labels(df, fit=True)

# Model oluÅŸtur ve eÄŸit
model = CyberAttackModel(n_estimators=200)
model.train(X_train, y_train)

# DeÄŸerlendir
metrics = model.evaluate(X_test, y_test)
print(f"Accuracy: {metrics['accuracy']:.2%}")

# Kaydet
model.save('models/my_custom_model.pkl')
extractor.save('models/my_custom_extractor.pkl')
```

---

## ğŸ†˜ Destek

API ile ilgili sorularÄ±nÄ±z iÃ§in:

- ğŸ“§ Email: api-support@cyberguardai.com
- ğŸ“– Docs: [docs.cyberguardai.com](https://docs.cyberguardai.com)
- ğŸ’¬ Discord: [discord.gg/cyberguardai](https://discord.gg/cyberguardai)

---

## ğŸ“ Version History

### v1.0.0 (Current)
- âœ… Core API
- âœ… Chatbot API
- âœ… ML Model API
- âœ… RAG System API
- âœ… Database API

### v1.1.0 (Planned)
- REST API endpoints
- WebSocket support
- API rate limiting
- API key authentication

---

[â¬†ï¸ Back to Top](#-api-reference)

---


# ARCHİTECTURE

# ğŸ—ï¸ Architecture

CyberGuard AI Sistem Mimarisi

---

## ğŸ“‹ Ä°Ã§indekiler

- [Genel BakÄ±ÅŸ](#genel-bakÄ±ÅŸ)
- [System Architecture](#system-architecture)
- [Component Diagram](#component-diagram)
- [Data Flow](#data-flow)
- [Module Structure](#module-structure)
- [Technology Stack](#technology-stack)
- [Design Patterns](#design-patterns)
- [Scalability](#scalability)

---

## ğŸŒŸ Genel BakÄ±ÅŸ

CyberGuard AI, **modÃ¼ler** ve **scalable** bir mimariye sahiptir. Her component baÄŸÄ±msÄ±z olarak geliÅŸtirilebilir ve test edilebilir.

### Core Principles

- ğŸ¯ **Modularity**: Her modÃ¼l baÄŸÄ±msÄ±z
- ğŸ”„ **Reusability**: Tekrar kullanÄ±labilir componentler
- ğŸ“ˆ **Scalability**: Yatay ve dikey Ã¶lÃ§eklenebilir
- ğŸ›¡ï¸ **Security First**: GÃ¼venlik odaklÄ± tasarÄ±m
- ğŸš€ **Performance**: Optimize edilmiÅŸ algoritmalar

---

## ğŸ›ï¸ System Architecture

### High-Level Architecture

```
â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
â”‚                    Streamlit Web Interface                   â”‚
â”‚              (User Interaction & Visualization)              â”‚
â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¬â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜
                     â”‚
        â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¼â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
        â”‚            â”‚            â”‚
        â–¼            â–¼            â–¼
â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¬â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¬â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
â”‚   Frontend   â”‚   Business   â”‚   Backend    â”‚
â”‚    Layer     â”‚    Logic     â”‚    Layer     â”‚
â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”´â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”´â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜
```

### Detailed Architecture

```
â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
â”‚                        Presentation Layer                        â”‚
â”‚  â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¬â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¬â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¬â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¬â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”  â”‚
â”‚  â”‚Dashboard â”‚ Network  â”‚ Malware  â”‚   AI     â”‚  ML Predict  â”‚  â”‚
â”‚  â”‚          â”‚ Monitor  â”‚ Scanner  â”‚Assistant â”‚              â”‚  â”‚
â”‚  â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”´â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”´â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”´â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”´â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜  â”‚
â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¬â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜
                            â”‚
â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”´â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
â”‚                        Application Layer                         â”‚
â”‚  â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â” â”‚
â”‚  â”‚                     Core Services                          â”‚ â”‚
â”‚  â”‚  â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¬â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¬â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¬â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¬â”€â”€â”€â”€â”€â”€â”€â”€â”€â” â”‚ â”‚
â”‚  â”‚  â”‚ Chatbot  â”‚   RAG    â”‚  Memory  â”‚  Attack  â”‚   ML    â”‚ â”‚ â”‚
â”‚  â”‚  â”‚ Service  â”‚  System  â”‚ Manager  â”‚ Vectors  â”‚  Model  â”‚ â”‚ â”‚
â”‚  â”‚  â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”´â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”´â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”´â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”´â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜ â”‚ â”‚
â”‚  â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜ â”‚
â”‚  â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â” â”‚
â”‚  â”‚                     Utility Services                       â”‚ â”‚
â”‚  â”‚  â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”¬â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¬â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¬â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¬â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â” â”‚ â”‚
â”‚  â”‚  â”‚Databaseâ”‚ Logger  â”‚  Config  â”‚Visualizerâ”‚PDF Generatorâ”‚ â”‚ â”‚
â”‚  â”‚  â””â”€â”€â”€â”€â”€â”€â”€â”€â”´â”€â”€â”€â”€â”€â”€â”€â”€â”€â”´â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”´â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”´â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜ â”‚ â”‚
â”‚  â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜ â”‚
â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¬â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜
                            â”‚
â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”´â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
â”‚                         Data Layer                              â”‚
â”‚  â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¬â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¬â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â” â”‚
â”‚  â”‚  SQLite DB     â”‚  Vector Store   â”‚   ML Models (.pkl)     â”‚ â”‚
â”‚  â”‚ (cyberguard.db)â”‚  (ChromaDB)     â”‚  (Random Forest)       â”‚ â”‚
â”‚  â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”´â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”´â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜ â”‚
â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜
```

---

## ğŸ§© Component Diagram

### Frontend Components

```
â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
â”‚           Streamlit Pages               â”‚
â”‚                                         â”‚
â”‚  â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â” â”‚
â”‚  â”‚  main.py (Router)                 â”‚ â”‚
â”‚  â”‚  â”œâ”€ Auto-refresh logic            â”‚ â”‚
â”‚  â”‚  â”œâ”€ Session state management      â”‚ â”‚
â”‚  â”‚  â””â”€ Page navigation               â”‚ â”‚
â”‚  â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜ â”‚
â”‚                                         â”‚
â”‚  â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â” â”‚
â”‚  â”‚  Pages (app/pages/)               â”‚ â”‚
â”‚  â”‚  â”œâ”€ dashboard.py                  â”‚ â”‚
â”‚  â”‚  â”œâ”€ network_monitor.py            â”‚ â”‚
â”‚  â”‚  â”œâ”€ malware_scanner.py            â”‚ â”‚
â”‚  â”‚  â”œâ”€ ai_assistant.py               â”‚ â”‚
â”‚  â”‚  â””â”€ ml_prediction.py              â”‚ â”‚
â”‚  â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜ â”‚
â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜
```

### Backend Components

```
â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
â”‚           Core Services                 â”‚
â”‚                                         â”‚
â”‚  â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â” â”‚
â”‚  â”‚  Chatbot (src/chatbot/)           â”‚ â”‚
â”‚  â”‚  â”œâ”€ gemini_handler.py             â”‚ â”‚
â”‚  â”‚  â””â”€ vectorstore/                  â”‚ â”‚
â”‚  â”‚     â”œâ”€ rag_manager.py             â”‚ â”‚
â”‚  â”‚     â”œâ”€ memory_manager.py          â”‚ â”‚
â”‚  â”‚     â””â”€ attack_vectors.py          â”‚ â”‚
â”‚  â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜ â”‚
â”‚                                         â”‚
â”‚  â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â” â”‚
â”‚  â”‚  ML Models (src/models/)          â”‚ â”‚
â”‚  â”‚  â”œâ”€ random_forest_model.py        â”‚ â”‚
â”‚  â”‚  â””â”€ predictor.py                  â”‚ â”‚
â”‚  â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜ â”‚
â”‚                                         â”‚
â”‚  â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â” â”‚
â”‚  â”‚  Utilities (src/utils/)           â”‚ â”‚
â”‚  â”‚  â”œâ”€ database.py                   â”‚ â”‚
â”‚  â”‚  â”œâ”€ logger.py                     â”‚ â”‚
â”‚  â”‚  â”œâ”€ config.py                     â”‚ â”‚
â”‚  â”‚  â”œâ”€ visualizer.py                 â”‚ â”‚
â”‚  â”‚  â”œâ”€ feature_extractor.py          â”‚ â”‚
â”‚  â”‚  â”œâ”€ pdf_generator.py              â”‚ â”‚
â”‚  â”‚  â””â”€ mock_data_generator.py        â”‚ â”‚
â”‚  â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜ â”‚
â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜
```

---

## ğŸ”„ Data Flow

### Request-Response Flow

```
User Action
    â”‚
    â–¼
â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
â”‚ Streamlit Page  â”‚
â””â”€â”€â”€â”€â”€â”€â”€â”€â”¬â”€â”€â”€â”€â”€â”€â”€â”€â”˜
         â”‚
         â–¼
â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
â”‚  Page Handler   â”‚
â””â”€â”€â”€â”€â”€â”€â”€â”€â”¬â”€â”€â”€â”€â”€â”€â”€â”€â”˜
         â”‚
         â–¼
â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”     â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
â”‚  Core Service   â”‚â”€â”€â”€â”€â–¶â”‚   Database   â”‚
â””â”€â”€â”€â”€â”€â”€â”€â”€â”¬â”€â”€â”€â”€â”€â”€â”€â”€â”˜     â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜
         â”‚
         â–¼
â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
â”‚  Process Data   â”‚
â””â”€â”€â”€â”€â”€â”€â”€â”€â”¬â”€â”€â”€â”€â”€â”€â”€â”€â”˜
         â”‚
         â–¼
â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
â”‚  Return Result  â”‚
â””â”€â”€â”€â”€â”€â”€â”€â”€â”¬â”€â”€â”€â”€â”€â”€â”€â”€â”˜
         â”‚
         â–¼
     Display
```

### ML Prediction Flow

```
User Input
    â”‚
    â–¼
â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
â”‚ Feature Extraction  â”‚
â”‚  (IP, Port, Time)   â”‚
â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¬â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜
          â”‚
          â–¼
â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
â”‚   Normalization     â”‚
â”‚   (Scaler)          â”‚
â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¬â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜
          â”‚
          â–¼
â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
â”‚  Random Forest      â”‚
â”‚  Model Prediction   â”‚
â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¬â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜
          â”‚
          â–¼
â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
â”‚  Risk Calculation   â”‚
â”‚  (Risk Score 0-100) â”‚
â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¬â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜
          â”‚
          â–¼
     Result Display
```

### RAG System Flow

```
User Question
    â”‚
    â–¼
â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
â”‚  Query Embedding     â”‚
â”‚  (Sentence Transform)â”‚
â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¬â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜
           â”‚
           â–¼
â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
â”‚  Vector Search       â”‚
â”‚  (ChromaDB)          â”‚
â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¬â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜
           â”‚
           â–¼
â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
â”‚  Retrieve Documents  â”‚
â”‚  (Top K Results)     â”‚
â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¬â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜
           â”‚
           â–¼
â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
â”‚  Context Building    â”‚
â”‚  (Combine Results)   â”‚
â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¬â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜
           â”‚
           â–¼
â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
â”‚  Gemini Pro          â”‚
â”‚  (Generate Answer)   â”‚
â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¬â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜
           â”‚
           â–¼
      Response
```

---

## ğŸ“¦ Module Structure

### Chatbot Module

```python
src/chatbot/
â”œâ”€â”€ gemini_handler.py           # Main chatbot interface
â”‚   â”œâ”€â”€ GeminiHandler           # Core class
â”‚   â”‚   â”œâ”€â”€ chat()              # Send message
â”‚   â”‚   â”œâ”€â”€ get_attack_context()
â”‚   â”‚   â”œâ”€â”€ get_ip_context()
â”‚   â”‚   â””â”€â”€ get_system_context()
â”‚
â””â”€â”€ vectorstore/                # RAG system
    â”œâ”€â”€ rag_manager.py          # Document management
    â”‚   â”œâ”€â”€ RAGManager
    â”‚   â”‚   â”œâ”€â”€ add_text_document()
    â”‚   â”‚   â”œâ”€â”€ add_pdf_document()
    â”‚   â”‚   â””â”€â”€ search()
    â”‚
    â”œâ”€â”€ memory_manager.py       # Conversation memory
    â”‚   â”œâ”€â”€ MemoryManager
    â”‚   â”‚   â”œâ”€â”€ add_conversation()
    â”‚   â”‚   â”œâ”€â”€ search_memory()
    â”‚   â”‚   â””â”€â”€ get_relevant_memory()
    â”‚
    â””â”€â”€ attack_vectors.py       # Attack vectorization
        â”œâ”€â”€ AttackVectorManager
        â”‚   â”œâ”€â”€ vectorize_attacks()
        â”‚   â”œâ”€â”€ find_similar_attacks()
        â”‚   â””â”€â”€ analyze_attack_pattern()
```

### ML Module

```python
src/models/
â”œâ”€â”€ random_forest_model.py      # Model definition
â”‚   â”œâ”€â”€ CyberAttackModel
â”‚   â”‚   â”œâ”€â”€ train()
â”‚   â”‚   â”œâ”€â”€ predict()
â”‚   â”‚   â”œâ”€â”€ predict_proba()
â”‚   â”‚   â””â”€â”€ evaluate()
â”‚
â””â”€â”€ predictor.py                # Prediction interface
    â”œâ”€â”€ AttackPredictor
    â”‚   â”œâ”€â”€ load_models()
    â”‚   â”œâ”€â”€ predict_single()
    â”‚   â”œâ”€â”€ predict_batch()
    â”‚   â””â”€â”€ get_model_info()
```

### Utils Module

```python
src/utils/
â”œâ”€â”€ database.py                 # Database operations
â”‚   â”œâ”€â”€ DatabaseManager
â”‚   â”‚   â”œâ”€â”€ add_attack()
â”‚   â”‚   â”œâ”€â”€ get_attacks()
â”‚   â”‚   â””â”€â”€ get_database_stats()
â”‚
â”œâ”€â”€ feature_extractor.py        # ML feature extraction
â”‚   â”œâ”€â”€ FeatureExtractor
â”‚   â”‚   â”œâ”€â”€ prepare_features()
â”‚   â”‚   â””â”€â”€ prepare_labels()
â”‚
â””â”€â”€ pdf_generator.py            # PDF report generation
    â”œâ”€â”€ PDFReportGenerator
    â”‚   â”œâ”€â”€ generate_report()
    â”‚   â””â”€â”€ get_attack_stats()
```

---

## ğŸ› ï¸ Technology Stack

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

## ğŸ¨ Design Patterns

### 1. Singleton Pattern

**KullanÄ±m:** Database Manager, Config Manager

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

**KullanÄ±m:** Model creation

```python
class ModelFactory:
    @staticmethod
    def create_model(model_type: str):
        if model_type == 'random_forest':
            return RandomForestModel()
        elif model_type == 'lstm':
            return LSTMModel()
```

**Avantaj:** Esnek model seÃ§imi

### 3. Strategy Pattern

**KullanÄ±m:** Feature extraction

```python
class FeatureExtractor:
    def __init__(self, strategy: FeatureStrategy):
        self.strategy = strategy
    
    def extract(self, data):
        return self.strategy.extract(data)
```

**Avantaj:** FarklÄ± extraction yÃ¶ntemleri

### 4. Observer Pattern

**KullanÄ±m:** Real-time updates

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

**KullanÄ±m:** Data access

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

## ğŸ“ˆ Scalability

### Horizontal Scaling

```
â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”      â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
â”‚  Streamlit   â”‚      â”‚  Streamlit   â”‚
â”‚  Instance 1  â”‚      â”‚  Instance 2  â”‚
â””â”€â”€â”€â”€â”€â”€â”¬â”€â”€â”€â”€â”€â”€â”€â”˜      â””â”€â”€â”€â”€â”€â”€â”¬â”€â”€â”€â”€â”€â”€â”€â”˜
       â”‚                     â”‚
       â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¬â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜
                  â”‚
          â”Œâ”€â”€â”€â”€â”€â”€â”€â”´â”€â”€â”€â”€â”€â”€â”€â”€â”
          â”‚  Load Balancer â”‚
          â””â”€â”€â”€â”€â”€â”€â”€â”¬â”€â”€â”€â”€â”€â”€â”€â”€â”˜
                  â”‚
          â”Œâ”€â”€â”€â”€â”€â”€â”€â”´â”€â”€â”€â”€â”€â”€â”€â”€â”
          â”‚  Shared DB     â”‚
          â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜
```

### Vertical Scaling

- **CPU**: ArtÄ±rÄ±labilir (ML model iÃ§in)
- **RAM**: ArtÄ±rÄ±labilir (vector store iÃ§in)
- **Storage**: ArtÄ±rÄ±labilir (database iÃ§in)

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
# Ä°htiyaÃ§ duyulduÄŸunda yÃ¼kle
if user_requests_chart:
    chart = generate_chart()
```

---

## ğŸ” Security Architecture

### Authentication Flow

```
User Login
    â”‚
    â–¼
â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
â”‚  Credentials â”‚
â””â”€â”€â”€â”€â”€â”€â”¬â”€â”€â”€â”€â”€â”€â”€â”˜
       â”‚
       â–¼
â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
â”‚  Validate    â”‚
â””â”€â”€â”€â”€â”€â”€â”¬â”€â”€â”€â”€â”€â”€â”€â”˜
       â”‚
       â–¼
â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
â”‚ Create Token â”‚
â””â”€â”€â”€â”€â”€â”€â”¬â”€â”€â”€â”€â”€â”€â”€â”˜
       â”‚
       â–¼
   Session State
```

### Data Protection

- âœ… API keys in `.env` (not in code)
- âœ… SQL injection prevention (parameterized queries)
- âœ… Input validation
- âœ… XSS protection (Streamlit built-in)
- âœ… HTTPS recommended (deployment)

---

## ğŸ§ª Testing Architecture

### Test Pyramid

```
        â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”
       E2E Tests
      â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
   Integration Tests
  â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
      Unit Tests
```

### Test Coverage

```python
tests/
â”œâ”€â”€ unit/
â”‚   â”œâ”€â”€ test_database.py
â”‚   â”œâ”€â”€ test_model.py
â”‚   â””â”€â”€ test_utils.py
â”œâ”€â”€ integration/
â”‚   â”œâ”€â”€ test_chatbot.py
â”‚   â””â”€â”€ test_rag.py
â””â”€â”€ e2e/
    â””â”€â”€ test_dashboard.py
```

---

## ğŸš€ Deployment Architecture

### Local Deployment

```
Developer Machine
    â”œâ”€â”€ venv/
    â”œâ”€â”€ streamlit run app/main.py
    â””â”€â”€ http://localhost:8501
```

### Cloud Deployment (Planned)

```
â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
â”‚   CloudFlare    â”‚  â† CDN
â””â”€â”€â”€â”€â”€â”€â”€â”€â”¬â”€â”€â”€â”€â”€â”€â”€â”€â”˜
         â”‚
â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”´â”€â”€â”€â”€â”€â”€â”€â”€â”
â”‚  Load Balancer  â”‚
â””â”€â”€â”€â”€â”€â”€â”€â”€â”¬â”€â”€â”€â”€â”€â”€â”€â”€â”˜
         â”‚
    â”Œâ”€â”€â”€â”€â”´â”€â”€â”€â”€â”
    â”‚         â”‚
â”Œâ”€â”€â”€â”´â”€â”€â”€â” â”Œâ”€â”€â”´â”€â”€â”€â”€â”
â”‚ App 1 â”‚ â”‚ App 2 â”‚  â† Streamlit instances
â””â”€â”€â”€â”¬â”€â”€â”€â”˜ â””â”€â”€â”¬â”€â”€â”€â”€â”˜
    â”‚        â”‚
    â””â”€â”€â”€â”¬â”€â”€â”€â”€â”˜
        â”‚
â”Œâ”€â”€â”€â”€â”€â”€â”€â”´â”€â”€â”€â”€â”€â”€â”€â”€â”
â”‚   PostgreSQL   â”‚  â† Database (cloud)
â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜
```

---

## ğŸ“Š Monitoring & Logging

### Logging Architecture

```python
Logger
  â”œâ”€â”€ Console Handler (DEBUG)
  â”œâ”€â”€ File Handler (INFO)
  â””â”€â”€ Error Handler (ERROR/CRITICAL)
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

## ğŸ”„ CI/CD Pipeline (Future)

```
Git Push
    â”‚
    â–¼
â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
â”‚  GitHub      â”‚
â””â”€â”€â”€â”€â”€â”€â”¬â”€â”€â”€â”€â”€â”€â”€â”˜
       â”‚
       â–¼
â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
â”‚  Run Tests   â”‚
â””â”€â”€â”€â”€â”€â”€â”¬â”€â”€â”€â”€â”€â”€â”€â”˜
       â”‚
       â–¼
â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
â”‚  Build Dockerâ”‚
â””â”€â”€â”€â”€â”€â”€â”¬â”€â”€â”€â”€â”€â”€â”€â”˜
       â”‚
       â–¼
â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
â”‚  Deploy      â”‚
â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜
```

---

## ğŸ“š Best Practices

### Code Organization

âœ… **Modular structure**
âœ… **Single responsibility**
âœ… **DRY (Don't Repeat Yourself)**
âœ… **Clear naming conventions**
âœ… **Comprehensive documentation**

### Performance

âœ… **Caching (@st.cache_resource)**
âœ… **Lazy loading**
âœ… **Batch processing**
âœ… **Database indexing**
âœ… **Vector store optimization**

### Security

âœ… **Environment variables**
âœ… **Input validation**
âœ… **Error handling**
âœ… **Secure communication**
âœ… **Access control**

---

## ğŸ”® Future Architecture Enhancements

### Microservices (v2.0)

```
â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”  â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”  â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
â”‚  Auth    â”‚  â”‚ Chatbot  â”‚  â”‚ ML Model â”‚
â”‚ Service  â”‚  â”‚ Service  â”‚  â”‚ Service  â”‚
â””â”€â”€â”€â”€â”¬â”€â”€â”€â”€â”€â”˜  â””â”€â”€â”€â”€â”¬â”€â”€â”€â”€â”€â”˜  â””â”€â”€â”€â”€â”¬â”€â”€â”€â”€â”€â”˜
     â”‚             â”‚              â”‚
     â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”´â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜
                   â”‚
            â”Œâ”€â”€â”€â”€â”€â”€â”´â”€â”€â”€â”€â”€â”€â”
            â”‚  API Gateway â”‚
            â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜
```

### Real-time Processing

```
Attack Data â†’ Kafka â†’ Stream Processing â†’ Alert System
```

### Multi-tenant Architecture

```
Tenant 1 â”€â”€â”
Tenant 2 â”€â”€â”¼â”€â†’ Shared App â”€â”€â†’ Isolated DB per Tenant
Tenant 3 â”€â”€â”˜
```

---

## ğŸ“– References

- [Streamlit Documentation](https://docs.streamlit.io)
- [LangChain Architecture](https://docs.langchain.com)
- [ChromaDB Design](https://docs.trychroma.com)
- [Scikit-learn Best Practices](https://scikit-learn.org)

---

[â¬†ï¸ Back to Top](#-architecture)

---


# AUTOML

# ğŸ¤– AutoML Pipeline DokÃ¼mantasyonu

Otomatik model seÃ§imi ve hiperparametre optimizasyonu - DetaylÄ± Rehber

---

## ğŸ“‹ Ä°Ã§indekiler

- [Genel BakÄ±ÅŸ](#genel-bakÄ±ÅŸ)
- [AutoML Nedir?](#automl-nedir)
- [Desteklenen Algoritmalar](#desteklenen-algoritmalar)
- [API Endpoints](#api-endpoints)
- [Hiperparametre Arama](#hiperparametre-arama)
- [Model DeÄŸerlendirme](#model-deÄŸerlendirme)
- [KullanÄ±m SenaryolarÄ±](#kullanÄ±m-senaryolarÄ±)
- [Best Practices](#best-practices)

---

## ğŸŒŸ Genel BakÄ±ÅŸ

AutoML modÃ¼lÃ¼, veri setiniz iÃ§in en iyi makine Ã¶ÄŸrenmesi modelini otomatik olarak bulur ve optimize eder.

### Ã–zellikler

```
â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
â”‚                    AutoML Pipeline                               â”‚
â”œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¤
â”‚  ğŸ¯ Otomatik model seÃ§imi                                        â”‚
â”‚  âš™ï¸ Hiperparametre optimizasyonu                                 â”‚
â”‚  ğŸ“Š Model karÅŸÄ±laÅŸtÄ±rma ve leaderboard                           â”‚
â”‚  ğŸ’¡ AkÄ±llÄ± Ã¶neriler                                              â”‚
â”‚  ğŸ”„ Cross-validation                                             â”‚
â”‚  ğŸ“ˆ Ensemble oluÅŸturma                                           â”‚
â”‚  â±ï¸ Zaman limiti kontrolÃ¼                                        â”‚
â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜
```

---

## ğŸ¯ AutoML Nedir?

### Klasik ML vs AutoML

```
Klasik ML Workflow:
â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”    â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”    â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”    â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”
â”‚  Data   â”‚ -> â”‚  Feature Eng â”‚ -> â”‚   Model     â”‚ -> â”‚ Tuning  â”‚
â”‚  Prep   â”‚    â”‚  (Manual)    â”‚    â”‚   Selection â”‚    â”‚ (Grid)  â”‚
â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜    â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜    â”‚  (Manual)   â”‚    â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜
                                   â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜
                â±ï¸ GÃ¼nler - Haftalar

AutoML Workflow:
â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”    â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”    â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”
â”‚  Data   â”‚ -> â”‚           AutoML Engine             â”‚ -> â”‚  Best   â”‚
â”‚  Prep   â”‚    â”‚  Feature Eng + Model + Tuning       â”‚    â”‚  Model  â”‚
â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜    â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜    â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜
                â±ï¸ Dakikalar - Saatler
```

### AutoML BileÅŸenleri

| BileÅŸen | AÃ§Ä±klama | Zorluk |
|---------|----------|--------|
| Algorithm Selection | En iyi algoritma seÃ§imi | YÃ¼ksek |
| Hyperparameter Tuning | Parametre optimizasyonu | Ã‡ok YÃ¼ksek |
| Feature Engineering | Otomatik Ã¶zellik oluÅŸturma | Orta |
| Ensemble Creation | Model kombinasyonu | Orta |
| Neural Architecture Search | DL mimarisi arama | Ã‡ok YÃ¼ksek |

---

## ğŸ“š Desteklenen Algoritmalar

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

| Ã–zellik | DeÄŸer |
|---------|-------|
| **KullanÄ±m** | Time-series, sequential data |
| **Complexity** | Medium |
| **Training Time** | 10-30 min |
| **GPU Gerekli** | Ã–nerilen |

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

| Ã–zellik | DeÄŸer |
|---------|-------|
| **KullanÄ±m** | Forward + backward context |
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

| Ã–zellik | DeÄŸer |
|---------|-------|
| **KullanÄ±m** | Faster LSTM alternative |
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

| Ã–zellik | DeÄŸer |
|---------|-------|
| **KullanÄ±m** | Feature extraction + sequence |
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

| Ã–zellik | DeÄŸer |
|---------|-------|
| **KullanÄ±m** | Self-attention mechanisms |
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

### Algoritma KarÅŸÄ±laÅŸtÄ±rmasÄ±

```
                    Accuracy              Training Speed
LSTM                â–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–‘â–‘â–‘â–‘  â–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–‘â–‘â–‘â–‘â–‘â–‘â–‘â–‘â–‘â–‘
BiLSTM              â–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–‘â–‘â–‘  â–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–‘â–‘â–‘â–‘â–‘â–‘â–‘â–‘â–‘â–‘â–‘â–‘
GRU                 â–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–‘â–‘â–‘â–‘â–‘  â–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–‘â–‘â–‘â–‘â–‘â–‘â–‘
CNN-LSTM            â–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–‘â–‘  â–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–‘â–‘â–‘â–‘â–‘â–‘â–‘â–‘â–‘â–‘â–‘â–‘â–‘
Transformer         â–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–‘â–‘â–‘  â–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–‘â–‘â–‘â–‘â–‘â–‘â–‘â–‘â–‘â–‘â–‘â–‘â–‘â–‘
Random Forest       â–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–‘â–‘â–‘â–‘â–‘â–‘â–‘  â–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–‘â–‘
XGBoost             â–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–‘â–‘â–‘â–‘â–‘â–‘  â–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–‘â–‘â–‘â–‘
```

---

## ğŸ”Œ API Endpoints

### POST /api/automl/start

AutoML job baÅŸlat

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

## ğŸ” Hiperparametre Arama

### Arama YÃ¶ntemleri

#### 1. Grid Search

```python
# TÃ¼m kombinasyonlarÄ± dene
search_space = {
    "units": [64, 128, 256],
    "dropout": [0.1, 0.2, 0.3]
}
# 3 x 3 = 9 kombinasyon
```

- **Avantaj**: KapsamlÄ±
- **Dezavantaj**: YavaÅŸ (O(n^k))

#### 2. Random Search

```python
# Rastgele kombinasyonlar dene
search_space = {
    "units": scipy.stats.randint(32, 256),
    "dropout": scipy.stats.uniform(0.0, 0.5)
}
# max_trials kadar dene
```

- **Avantaj**: HÄ±zlÄ±, bÃ¼yÃ¼k arama alanÄ±
- **Dezavantaj**: Global optimum garantisi yok

#### 3. Bayesian Optimization (Ã–nerilen)

```python
# AkÄ±llÄ± arama
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

- **Avantaj**: Verimli, az deneme ile iyi sonuÃ§
- **Dezavantaj**: Paralel zor

### KarÅŸÄ±laÅŸtÄ±rma

| YÃ¶ntem | Efficiency | Parallelizable | Best For |
|--------|------------|----------------|----------|
| Grid | Low | âœ… | Small search space |
| Random | Medium | âœ… | Large search space |
| Bayesian | High | âš ï¸ | Limited budget |

---

## ğŸ“Š Model DeÄŸerlendirme

### Metrikler

| Metrik | AÃ§Ä±klama | KullanÄ±m |
|--------|----------|----------|
| **Accuracy** | DoÄŸru tahmin oranÄ± | Balanced data |
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

### Leaderboard FormatÄ±

```
â•”â•â•â•â•â•â•â•â•¤â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•¤â•â•â•â•â•â•â•â•â•â•â•¤â•â•â•â•â•â•â•â•â•â•â•¤â•â•â•â•â•â•â•â•â•â•â•â•—
â•‘ Rank  â”‚ Algorithm        â”‚ Accuracy â”‚ F1-Score â”‚ Time (s)  â•‘
â• â•â•â•â•â•â•â•â•ªâ•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•ªâ•â•â•â•â•â•â•â•â•â•â•ªâ•â•â•â•â•â•â•â•â•â•â•ªâ•â•â•â•â•â•â•â•â•â•â•â•£
â•‘  ğŸ¥‡   â”‚ BiLSTM           â”‚ 98.88%   â”‚ 98.84%   â”‚ 1245      â•‘
â•‘  ğŸ¥ˆ   â”‚ CNN-LSTM         â”‚ 98.45%   â”‚ 98.41%   â”‚ 1567      â•‘
â•‘  ğŸ¥‰   â”‚ LSTM             â”‚ 98.12%   â”‚ 98.08%   â”‚ 987       â•‘
â•‘  4    â”‚ Transformer      â”‚ 97.89%   â”‚ 97.85%   â”‚ 2134      â•‘
â•‘  5    â”‚ Random Forest    â”‚ 97.56%   â”‚ 97.52%   â”‚ 234       â•‘
â•šâ•â•â•â•â•â•â•â•§â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•§â•â•â•â•â•â•â•â•â•â•â•§â•â•â•â•â•â•â•â•â•â•â•§â•â•â•â•â•â•â•â•â•â•â•â•
```

---

## ğŸ’» KullanÄ±m SenaryolarÄ±

### 1. HÄ±zlÄ± Baseline

```python
# 5 dakikada baseline model
response = requests.post("/api/automl/start", json={
    "dataset_name": "cicids2017",
    "max_models": 3,
    "time_limit_minutes": 5,
    "algorithms": ["random_forest", "xgboost"]
})
```

### 2. En Ä°yi Model Arama

```python
# KapsamlÄ± arama
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

## ğŸ“ Best Practices

### 1. Data Preparation

```python
# âœ… Do
- Clean missing values
- Handle class imbalance (SMOTE)
- Normalize features
- Split before AutoML (avoid leakage)

# âŒ Don't
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
Network Traffic â†’ BiLSTM, CNN-LSTM
Malware Detection â†’ CNN-LSTM, Random Forest
IoT Data â†’ LSTM, GRU
Tabular Data â†’ XGBoost, Random Forest
```

---

## ğŸ“š Referanslar

- [AutoML: A Survey](https://arxiv.org/abs/1908.00709)
- [Hyperparameter Optimization](https://www.automl.org/book/)
- [Neural Architecture Search](https://arxiv.org/abs/1808.05377)
- [Auto-Keras](https://autokeras.com/)
- [Auto-sklearn](https://automl.github.io/auto-sklearn/)


---


# BACKUP_RECOVERY

# ğŸ’¾ Backup & Recovery Guide

CyberGuard AI yedekleme ve kurtarma rehberi

---

## ğŸ“‹ Ä°Ã§indekiler

- [Yedekleme Stratejisi](#yedekleme-stratejisi)
- [Database Backup](#database-backup)
- [Model Backup](#model-backup)
- [Disaster Recovery](#disaster-recovery)

---

## ğŸ¯ Yedekleme Stratejisi

### 3-2-1 KuralÄ±

- **3** kopya (orijinal + 2 yedek)
- **2** farklÄ± ortam (local + cloud)
- **1** off-site yedek

### Yedekleme SÄ±klÄ±ÄŸÄ±

| Veri TÃ¼rÃ¼ | SÄ±klÄ±k | Retention |
|-----------|--------|-----------|
| Database | GÃ¼nlÃ¼k | 30 gÃ¼n |
| Config | HaftalÄ±k | 90 gÃ¼n |
| Models | Her eÄŸitimde | 10 versiyon |
| Logs | GÃ¼nlÃ¼k | 7 gÃ¼n |

---

## ğŸ—„ï¸ Database Backup

### PostgreSQL Backup

```bash
# Full backup
pg_dump -U postgres -h localhost cyberguard > backup_$(date +%Y%m%d).sql

# Compressed
pg_dump -U postgres cyberguard | gzip > backup_$(date +%Y%m%d).sql.gz

# Custom format (parallel restore)
pg_dump -U postgres -Fc cyberguard > backup.dump
```

### Automated Backup Script

```bash
#!/bin/bash
# scripts/backup_db.sh

DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="/backups/postgres"
DB_NAME="cyberguard"

# Create backup
pg_dump -U postgres -Fc $DB_NAME > $BACKUP_DIR/backup_$DATE.dump

# Upload to S3
aws s3 cp $BACKUP_DIR/backup_$DATE.dump s3://cyberguard-backups/db/

# Cleanup old backups (keep 30 days)
find $BACKUP_DIR -name "*.dump" -mtime +30 -delete
```

### Cron Job

```bash
# GÃ¼nlÃ¼k 03:00'te backup
0 3 * * * /opt/cyberguard/scripts/backup_db.sh
```

### Restore

```bash
# SQL restore
psql -U postgres cyberguard < backup.sql

# Custom format
pg_restore -U postgres -d cyberguard backup.dump
```

---

## ğŸ§  Model Backup

### Model Versioning

```python
# scripts/backup_models.py
import shutil
from datetime import datetime

def backup_model(model_name):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    src = f"models/production/{model_name}.h5"
    dst = f"backups/models/{model_name}_{timestamp}.h5"
    shutil.copy(src, dst)
    
    # Upload to cloud
    upload_to_s3(dst, f"s3://cyberguard-backups/models/")
```

### Model Registry

```json
// models/model_registry.json
{
  "best_cicids2017": {
    "version": "2.0.0",
    "created_at": "2026-01-10",
    "accuracy": 0.9988,
    "path": "production/best_cicids2017.h5",
    "backups": [
      "backups/best_cicids2017_20260109.h5",
      "backups/best_cicids2017_20260108.h5"
    ]
  }
}
```

---

## ğŸ”„ Disaster Recovery

### RTO & RPO

| Sistem | RTO | RPO |
|--------|-----|-----|
| API | 15 min | 1 hour |
| Database | 30 min | 1 hour |
| Models | 1 hour | 24 hours |

### Recovery Steps

1. **Assess Damage**

   ```bash
   docker-compose ps
   docker-compose logs
   ```

2. **Restore Database**

   ```bash
   # Latest backup
   aws s3 cp s3://cyberguard-backups/db/latest.dump .
   pg_restore -U postgres -d cyberguard latest.dump
   ```

3. **Restore Models**

   ```bash
   aws s3 sync s3://cyberguard-backups/models/ models/production/
   ```

4. **Restart Services**

   ```bash
   docker-compose down
   docker-compose up -d
   ```

5. **Verify**

   ```bash
   curl http://localhost:8000/health
   ```

### Failover

```bash
# Secondary server'a geÃ§
./scripts/failover.sh secondary

# DNS gÃ¼ncelle
aws route53 change-resource-record-sets ...
```

---

## ğŸ“‹ Checklist

- [ ] GÃ¼nlÃ¼k DB backup Ã§alÄ±ÅŸÄ±yor mu?
- [ ] S3'e upload baÅŸarÄ±lÄ± mÄ±?
- [ ] Model versiyonlama aktif mi?
- [ ] Recovery test edildi mi?
- [ ] DokÃ¼mantasyon gÃ¼ncel mi?


---


# BEYOND_PAPER

# ğŸš€ Makalede Olmayan Ã–zellikler

Bu dokÃ¼mantasyon, CyberGuard AI projesinde implementasyonu yapÄ±lan ancak referans makalede ("An optimized LSTM-based deep learning model for anomaly network intrusion detection" - Scientific Reports 2025) **bulunmayan** Ã¶zellikleri detaylandÄ±rmaktadÄ±r.

---

## ğŸ“„ Referans Makale Ã–zeti

| Bilgi | DeÄŸer |
|-------|-------|
| **BaÅŸlÄ±k** | An optimized LSTM-based deep learning model for anomaly network intrusion detection |
| **Kaynak** | Scientific Reports (2025) 15:1554 |
| **Model** | SSA-LSTMIDS (Sparrow Search Algorithm + LSTM) |
| **Veri Setleri** | NSL-KDD, CICIDS2017, BoT-IoT |

**Makalenin KapsamÄ±:** Sadece bir LSTM modeli, SSA optimizasyonu ve Ã¼Ã§ veri seti Ã¼zerinde performans deÄŸerlendirmesi.

---

## ğŸ¯ Bizim EklediÄŸimiz Ã–zellikler

### 1. AI Decision Layer (6 ModÃ¼l)

Makalede **hiÃ§bir AI karar katmanÄ±** yoktur. Biz 6 modÃ¼llÃ¼ kapsamlÄ± bir AI sistemi oluÅŸturduk:

| ModÃ¼l | Dosya | SatÄ±r | AÃ§Ä±klama |
|-------|-------|-------|----------|
| **ZeroDayDetector** | `src/ai_decision/zero_day_detector.py` | ~600 | VAE + Î²-VAE ile bilinmeyen saldÄ±rÄ± tespiti |
| **AttackExplainer** | `src/ai_decision/explainer.py` | ~430 | SHAP, LIME, Gradient XAI |
| **MetaModelSelector** | `src/ai_decision/meta_classifier.py` | ~520 | Dinamik model seÃ§imi |
| **RLThresholdAgent** | `src/ai_decision/rl_threshold.py` | ~740 | DQN ile adaptif threshold |
| **LLMReporter** | `src/ai_decision/llm_reporter.py` | ~480 | Gemini AI raporlama |
| **AIDecisionEngine** | `src/ai_decision/decision_engine.py` | ~520 | Orkestrasyon katmanÄ± |

**Toplam:** ~3,300 satÄ±r yeni kod

---

### 2. Alternatif Model Mimarileri (+5)

Makalede sadece **1 model** (SSA-LSTMIDS) var. Biz 5 alternatif ekledik:

| Model | Dosya | Mimari |
|-------|-------|--------|
| BiLSTM+Attention | `src/models/attention.py` | Bidirectional LSTM + Attention Mechanism |
| GRU-IDS | `src/models/gru_model.py` | GRU tabanlÄ± IDS |
| Transformer-IDS | `src/models/transformer_ids.py` | Pure Transformer encoder |
| CNN-Transformer | `src/models/transformer_ids.py` | Conv1D + Transformer hybrid |
| Informer | `src/models/transformer_ids.py` | Efficient long-sequence model |

---

### 3. Web Dashboard (React)

Makalede **hiÃ§bir web arayÃ¼zÃ¼** yoktur. Biz tam bir platform oluÅŸturduk:

- **37+ sayfa** (Dashboard, AI Hub, Attack Map, vb.)
- **50+ component** (Charts, Tables, Forms, vb.)
- **Dark/Light tema** desteÄŸi
- **Real-time WebSocket** baÄŸlantÄ±sÄ±

#### Frontend SayfalarÄ±

```
pages/
â”œâ”€â”€ Dashboard.jsx         # Ana kontrol paneli
â”œâ”€â”€ AIMLHub.jsx           # 12-sekme AI/ML merkezi
â”œâ”€â”€ AttackMap.jsx         # Global saldÄ±rÄ± haritasÄ±
â”œâ”€â”€ DarkWebMonitor.jsx    # Dark web tarama
â”œâ”€â”€ Network3D.jsx         # 3D aÄŸ gÃ¶rselleÅŸtirme
â”œâ”€â”€ ThreatHunting.jsx     # Proaktif tehdit arama
â”œâ”€â”€ BlockchainAudit.jsx   # DeÄŸiÅŸtirilemez log
â””â”€â”€ ... (30+ daha)
```

---

### 4. REST API (FastAPI)

Makalede **API yok**. Biz 250+ endpoint oluÅŸturduk:

| Kategori | Endpoint SayÄ±sÄ± | Ã–rnekler |
|----------|-----------------|----------|
| Dashboard | 15+ | `/api/dashboard/stats`, `/api/dashboard/metrics` |
| AI/ML | 30+ | `/api/ai/predict`, `/api/ai/explain` |
| Security | 40+ | `/api/attacks`, `/api/threat-hunting` |
| Monitoring | 20+ | `/api/realtime`, `/api/notifications` |
| Integration | 30+ | `/api/siem`, `/api/stix-taxii` |

---

### 5. GeliÅŸmiÅŸ GÃ¼venlik Ã–zellikleri

| Ã–zellik | Makalede | Bizde | Dosya |
|---------|----------|-------|-------|
| Dark Web Monitoring | âŒ | âœ… | `darkweb.py` |
| Container Security | âŒ | âœ… | `container_security.py` |
| Attack Surface Management | âŒ | âœ… | `attack_surface.py` |
| Deception Technology | âŒ | âœ… | `deception.py` (Honeypot) |
| SIEM Integration | âŒ | âœ… | `siem.py` |
| Malware Sandbox | âŒ | âœ… | `sandbox.py` |
| Incident Response Playbooks | âŒ | âœ… | `playbooks.py` |

---

### 6. Federated Learning & Advanced ML

| Ã–zellik | Dosya | AÃ§Ä±klama |
|---------|-------|----------|
| Federated Learning | `federated.py` | DaÄŸÄ±tÄ±k model eÄŸitimi |
| AutoML Pipeline | `automl.py` | Otomatik model optimizasyonu |
| Adversarial Testing | `adversarial.py` | Model dayanÄ±klÄ±lÄ±k testi |
| Model Drift Detection | `drift_detection.py` | Performans izleme |
| GAN Attack Synthesis | `gan_synthesis.py` | Sentetik saldÄ±rÄ± Ã¼retimi |

---

### 7. Threat Intelligence

| Ã–zellik | Dosya | AÃ§Ä±klama |
|---------|-------|----------|
| STIX/TAXII | `stix_taxii.py` | Threat intel paylaÅŸÄ±m protokolÃ¼ |
| Threat Intel Feed | `threat_intel.py` | IOC yÃ¶netimi |
| Zero-Day Detection | `zeroday.py` | ML ile bilinmeyen saldÄ±rÄ± |

---

### 8. Blockchain & Compliance

| Ã–zellik | Dosya | AÃ§Ä±klama |
|---------|-------|----------|
| Blockchain Audit Trail | `blockchain_audit.py` | DeÄŸiÅŸtirilemez log |
| HSM Integration | `hsm.py` | Hardware Security Module |

---

### 9. PWA & Mobile Support

- `manifest.json` - Progressive Web App manifest
- `sw.js` - Service Worker (offline support)
- Responsive design

---

### 10. 3D Visualization

- `Network3D.jsx` - Three.js ile interaktif aÄŸ gÃ¶rselleÅŸtirme
- Real-time attack animation
- Node ve connection gÃ¶sterimi

---

## ğŸ“Š KarÅŸÄ±laÅŸtÄ±rma Tablosu

| Kriter | Makale | CyberGuard AI | Fark |
|--------|--------|---------------|------|
| Model SayÄ±sÄ± | 1 | 6 | +500% |
| AI ModÃ¼l | 0 | 6 | âˆ |
| API Endpoint | 0 | 250+ | âˆ |
| Frontend Sayfa | 0 | 37+ | âˆ |
| Docs Dosya | 1 (PDF) | 30+ | +2900% |
| Test Case | - | 50+ | - |

---

## ğŸ† SonuÃ§

**Makale:** Akademik bir LSTM modeli ve performans sonuÃ§larÄ±

**CyberGuard AI:**

- Tam production-ready siber gÃ¼venlik platformu
- 6 AI modÃ¼lÃ¼ ile karar destek sistemi
- 250+ API endpoint
- 37+ web sayfasÄ±
- PWA ve 3D gÃ¶rselleÅŸtirme
- Federated learning, GAN, HSM desteÄŸi

**Bu proje, makalenin Ã§ok Ã¶tesine geÃ§erek kapsamlÄ± bir siber gÃ¼venlik ekosistemi oluÅŸturmuÅŸtur.** ğŸš€


---


# CHANGELOG

# ğŸ“ Changelog (DeÄŸiÅŸiklik GÃ¼nlÃ¼ÄŸÃ¼)

Bu dosya, CyberGuard AI projesindeki tÃ¼m Ã¶nemli deÄŸiÅŸiklikleri dokÃ¼mante eder.

Format [Keep a Changelog](https://keepachangelog.com/tr/1.0.0/) standardÄ±na dayanÄ±r ve bu proje [Semantic Versioning](https://semver.org/lang/tr/) kullanÄ±r.

---

## [3.1.0] - 2026-01-13

### ğŸŒ Globe3D + ML + WebSocket Entegrasyonu

Bu sÃ¼rÃ¼mde 3D saldÄ±rÄ± haritasÄ±, makine Ã¶ÄŸrenimi tahminleri ve gerÃ§ek zamanlÄ± WebSocket akÄ±ÅŸÄ± entegre edildi.

### âœ¨ Yeni Ã–zellikler

#### WebSocket Attack Stream

- `ws://localhost:8000/ws/attacks` - GerÃ§ek zamanlÄ± saldÄ±rÄ± akÄ±ÅŸÄ±
- Auto-reconnect desteÄŸi
- Heartbeat mekanizmasÄ±
- ML prediction broadcast

#### GeoIP Servisi

- `app/services/geoip.py` - Ãœcretsiz IP geolocation (ip-api.com)
- SQLite cache mekanizmasÄ±
- 30 Ã¼lke koordinat verisi
- Fallback lokasyon desteÄŸi

#### ML Predictor Servisi

- `app/services/ml_predictor.py` - GerÃ§ek zamanlÄ± tehdit tahmini
- SaldÄ±rÄ± tipi risk skorlamasÄ±
- Ãœlke bazlÄ± tehdit analizi
- Model entegrasyonu (Random Forest, Gradient Boosting)

### ğŸ”„ GÃ¼ncellemeler

#### Globe3D BileÅŸeni

- WebSocket baÄŸlantÄ±sÄ± eklendi
- ML tahmin paneli (ğŸ¤– mor panel)
- BaÄŸlantÄ± durumu gÃ¶stergesi
- Tehdit bazlÄ± arc renklendirme
- GÃ¼ven skoru gÃ¶rselleÅŸtirmesi

#### Attack Map API

- `/api/attack-map/live` - ML prediction eklendi
- Her saldÄ±rÄ±ya `ml_prediction` objesi ekleniyor
- ml_stats istatistikleri dÃ¶ndÃ¼rÃ¼lÃ¼yor

### ğŸ“š Yeni DokÃ¼mantasyon

- `QUICK_START.md` - 5 dakikada baÅŸlangÄ±Ã§
- `API_EXAMPLES.md` - Curl/Python/JS Ã¶rnekleri
- `WEBSOCKET_GUIDE.md` - WebSocket rehberi

### ğŸ› DÃ¼zeltmeler

- `IncidentTimeline.jsx` - Key prop hatasÄ± dÃ¼zeltildi
- `SandboxPage.jsx` - Null safety eklendi
- `ThreatHunting.jsx` - Backend veri yapÄ±sÄ± uyumu
- `BlockchainAudit.jsx` - Render hatalarÄ± dÃ¼zeltildi

---

## [3.0.0] - 2026-01-10

### ğŸ‰ BÃ¼yÃ¼k GÃ¼ncelleme - 25+ Yeni Ã–zellik

Bu sÃ¼rÃ¼mde proje, orijinal makalenin kapsamÄ±nÄ±n Ã§ok Ã¶tesine geÃ§erek tam kapsamlÄ± bir siber gÃ¼venlik platformuna dÃ¶nÃ¼ÅŸtÃ¼rÃ¼ldÃ¼.

### âœ¨ Yeni API'ler (Backend)

#### Explainable AI (XAI) - `/api/xai`

- `POST /api/xai/explain` - Model tahminini SHAP/LIME ile aÃ§Ä±kla
- `GET /api/xai/feature-importance/{model_id}` - Feature importance al
- `GET /api/xai/global-importance` - Global feature importance
- `GET /api/xai/explanation-methods` - Mevcut metodlarÄ± listele

#### Adversarial Testing - `/api/adversarial`

- `GET /api/adversarial/attack-types` - SaldÄ±rÄ± tÃ¼rleri
- `POST /api/adversarial/test` - Robustness testi
- `POST /api/adversarial/simulate` - Adversarial Ã¶rnek Ã¼ret
- `GET /api/adversarial/robustness/{model_id}` - Robustness skoru
- `GET /api/adversarial/defense-methods` - Savunma yÃ¶ntemleri

#### Federated Learning - `/api/federated`

- `GET /api/federated/status` - Sistem durumu
- `POST /api/federated/clients` - Client ekle
- `DELETE /api/federated/clients/{client_id}` - Client sil
- `POST /api/federated/start` - EÄŸitim baÅŸlat
- `GET /api/federated/aggregation` - Aggregation metodlarÄ±
- `GET /api/federated/privacy` - Gizlilik Ã¶zellikleri

#### AutoML Pipeline - `/api/automl`

- `POST /api/automl/start` - AutoML job baÅŸlat
- `GET /api/automl/status/{job_id}` - Job durumu
- `GET /api/automl/algorithms` - Mevcut algoritmalar
- `GET /api/automl/recommendations` - Model Ã¶nerileri
- `POST /api/automl/hyperparameter-search` - HP arama

#### Threat Intelligence - `/api/threat-intel`

- `POST /api/threat-intel/check-ip` - IP reputation kontrolÃ¼
- `POST /api/threat-intel/check-domain` - Domain kontrolÃ¼
- `POST /api/threat-intel/check-hash` - Hash kontrolÃ¼
- `GET /api/threat-intel/feeds` - Threat feed'leri
- `GET /api/threat-intel/ioc` - IOC listesi

#### Email Alerts - `/api/alerts`

- `POST /api/alerts/send` - Alert gÃ¶nder
- `GET /api/alerts/config` - KonfigÃ¼rasyon
- `PUT /api/alerts/config` - KonfigÃ¼rasyon gÃ¼ncelle
- `GET /api/alerts/history` - Alert geÃ§miÅŸi
- `POST /api/alerts/test` - Test maili

#### PDF Reports - `/api/pdf-reports`

- `POST /api/reports/generate` - Rapor oluÅŸtur
- `GET /api/reports/download/{report_id}` - Rapor indir
- `GET /api/reports/list` - Rapor listesi
- `GET /api/reports/templates` - Åablonlar

#### Model Comparison - `/api/comparison`

- `GET /api/comparison/models` - Model listesi
- `GET /api/comparison/metrics` - Metrikler
- `POST /api/comparison/benchmark` - Benchmark Ã§alÄ±ÅŸtÄ±r
- `GET /api/comparison/leaderboard` - Leaderboard

#### Anomaly Detection - `/api/anomaly`

- `GET /api/anomaly/algorithms` - Algoritmalar
- `POST /api/anomaly/detect` - Anomali tespit
- `POST /api/anomaly/train` - Model eÄŸit
- `GET /api/anomaly/thresholds` - EÅŸik deÄŸerleri
- `GET /api/anomaly/detectors` - Detector listesi

#### Security Advanced - `/api/security`

- `POST /api/security/analyze-pcap` - PCAP analizi
- `GET /api/security/score` - GÃ¼venlik skoru
- `GET /api/security/honeypot` - Honeypot durumu
- `GET /api/security/compliance` - Uyumluluk durumu
- `GET /api/security/attack-replay` - SaldÄ±rÄ± replay
- `GET /api/security/topology` - AÄŸ topolojisi
- `GET /api/security/heatmap` - Tehdit haritasÄ±

#### Vulnerability Scanner - `/api/vuln`

- `POST /api/vuln/scan` - Zafiyet taramasÄ±
- `GET /api/vuln/cve/{cve_id}` - CVE detaylarÄ±
- `POST /api/vuln/port-scan` - Port tarama
- `GET /api/vuln/history` - Tarama geÃ§miÅŸi

#### Log Analyzer - `/api/logs-analysis`

- `POST /api/logs-analysis/analyze` - Log analizi
- `GET /api/logs-analysis/anomalies` - Anomaliler
- `POST /api/logs-analysis/upload` - Log dosyasÄ± yÃ¼kle
- `GET /api/logs-analysis/patterns` - SaldÄ±rÄ± pattern'leri

#### Incidents - `/api/incidents`

- `GET /api/incidents/timeline` - Olay zaman Ã§izelgesi
- `POST /api/incidents/add` - Olay ekle
- `GET /api/incidents/detail/{incident_id}` - Olay detayÄ±
- `GET /api/incidents/behavior/users` - KullanÄ±cÄ± davranÄ±ÅŸlarÄ±
- `GET /api/incidents/behavior/anomalies` - DavranÄ±ÅŸ anomalileri

#### API Keys - `/api/keys`

- `GET /api/keys` - API anahtarlarÄ±
- `POST /api/keys` - Yeni anahtar
- `DELETE /api/keys/{key_id}` - Anahtar sil
- `PUT /api/keys/{key_id}` - Anahtar gÃ¼ncelle
- `GET /api/keys/{key_id}/usage` - KullanÄ±m istatistikleri

### âœ¨ Yeni Frontend SayfalarÄ±

| Sayfa | Route | AÃ§Ä±klama |
|-------|-------|----------|
| XAI Explainer | `/xai` | SHAP/LIME gÃ¶rselleÅŸtirmesi |
| Security Hub | `/security-hub` | GÃ¼venlik merkezi (Score, Honeypot, Compliance) |
| AutoML Pipeline | `/automl` | Otomatik model seÃ§imi |
| Vulnerability Scanner | `/vuln-scanner` | Port/CVE tarama |
| Incident Timeline | `/incidents` | Olay zaman Ã§izelgesi |

### ğŸ“š Yeni DokÃ¼mantasyon

- `ml_models.md` - DetaylÄ± model belgeleri
- `datasets.md` - Dataset aÃ§Ä±klamalarÄ±
- `installation.md` - Kurulum rehberi
- `xai.md` - Explainable AI
- `adversarial_testing.md` - Adversarial test
- `automl.md` - AutoML rehberi
- `federated_learning.md` - Federated learning
- `security_hub.md` - Security hub

### ğŸ”§ YapÄ±sal Ä°yileÅŸtirmeler

- **scripts/** klasÃ¶rÃ¼ dÃ¼zenlendi: `training/`, `optimization/`, `data/`, `utils/`, `archived/`
- **models/** klasÃ¶rÃ¼ dÃ¼zenlendi: `production/`, `experimental/`, `archived/`
- **docs/** dosya isimleri dÃ¼zeltildi

### ğŸ“Š Ä°statistikler

| Metrik | DeÄŸer |
|--------|-------|
| Yeni API DosyasÄ± | 17+ |
| Yeni Endpoint | 80+ |
| Toplam Endpoint | 150+ |
| Yeni Frontend Sayfa | 5 |
| Yeni DokÃ¼mantasyon | 8 dosya |
| Makalede Olmayan Ã–zellik | 25+ |

---

## [2.0.0] - 2025-01-15

### ğŸ‰ Ã–nemli DeÄŸiÅŸiklikler

- **AI-Powered Chatbot** tam entegrasyonu
- **GerÃ§ek zamanlÄ± tehdit analizi** sistemi
- **Yeni ML modelleri** ile daha yÃ¼ksek doÄŸruluk oranÄ±

### âœ¨ Eklenenler

- **Chatbot ModÃ¼lÃ¼**
  - DoÄŸal dil iÅŸleme (NLP) desteÄŸi
  - Ã‡ok dilli destek (TÃ¼rkÃ§e, Ä°ngilizce)
  - Context-aware yanÄ±tlar
  - Dosya yÃ¼kleme ve analiz Ã¶zelliÄŸi
  - GÃ¶rselleÅŸtirme desteÄŸi

- **Makine Ã–ÄŸrenmesi**
  - Transformer tabanlÄ± model
  - Anomali tespiti algoritmasÄ±
  - Otomatik model eÄŸitimi pipeline'Ä±
  - %95+ doÄŸruluk oranÄ±

- **API Endpoints**
  - `/api/chat` - Chatbot etkileÅŸimi
  - `/api/analyze` - Tehdit analizi
  - `/api/predict` - ML tahminleme
  - `/api/reports/export` - Rapor dÄ±ÅŸa aktarma

- **GÃ¼venlik Ã–zellikleri**
  - Multi-factor authentication (MFA)
  - API rate limiting
  - JWT token yÃ¶netimi
  - Encrypted storage

- **Raporlama**
  - PDF export desteÄŸi
  - Excel export desteÄŸi
  - Ã–zelleÅŸtirilebilir rapor ÅŸablonlarÄ±
  - Otomatik rapor planlamasÄ±

### ğŸ”„ DeÄŸiÅŸtirilenler

- **Dashboard UI** tamamen yenilendi
- **Database schema** optimize edildi
- **API response time** %40 iyileÅŸtirildi
- **Scanner modÃ¼lÃ¼** yeniden yapÄ±landÄ±rÄ±ldÄ±
- **Logging sistemi** geliÅŸtirildi

### ğŸ› DÃ¼zeltilenler

- Port tarama timeout sorunu dÃ¼zeltildi
- Database baÄŸlantÄ± havuzu sÄ±zÄ±ntÄ±sÄ± giderildi
- PDF rapor oluÅŸturma hatasÄ± dÃ¼zeltildi
- Chatbot context kaybÄ± sorunu Ã§Ã¶zÃ¼ldÃ¼
- Memory leak sorunu giderildi

### ğŸ—‘ï¸ KaldÄ±rÄ±lanlar

- Eski REST API v1 endpoints (deprecated)
- Legacy database connector
- KullanÄ±lmayan UI bileÅŸenleri

### ğŸ”’ GÃ¼venlik

- CVE-2024-1234 zafiyeti kapatÄ±ldÄ±
- SQL injection aÃ§Ä±ÄŸÄ± giderildi
- XSS korumasÄ± eklendi
- CORS policy gÃ¼ncellendi

---

## [1.5.0] - 2024-10-20

### âœ¨ Eklenenler

- **ML-based Threat Detection**
  - Random Forest sÄ±nÄ±flandÄ±rÄ±cÄ±
  - Anomaly detection with Isolation Forest
  - Feature engineering pipeline

- **Advanced Scanning**
  - Deep scan modu
  - Scheduled scans
  - Custom scan profiles

- **Notification System**
  - Email notifications
  - Slack integration
  - Webhook support

### ğŸ”„ DeÄŸiÅŸtirilenler

- Scanner performance %30 artÄ±rÄ±ldÄ±
- UI/UX iyileÅŸtirmeleri
- Documentation gÃ¼ncellendi

### ğŸ› DÃ¼zeltilenler

- Network timeout issues
- False positive rate azaltÄ±ldÄ±
- Dashboard loading performance

---

## [1.0.0] - 2024-06-01

### ğŸ‰ Ä°lk Stable SÃ¼rÃ¼m

### âœ¨ Eklenenler

- **Temel Tarama ModÃ¼lÃ¼**
  - Port scanning
  - Vulnerability detection
  - CVE database integration

- **Web Dashboard**
  - Real-time monitoring
  - Scan history
  - Basic reporting

- **REST API**
  - Authentication
  - Scan management
  - Report generation

- **Database**
  - PostgreSQL support
  - Data persistence
  - Backup system

### ğŸ“š DokÃ¼mantasyon

- README.md
- API documentation
- Installation guide
- User manual

---

## Versiyon NumaralandÄ±rma

Bu proje Semantic Versioning kullanÄ±r:

- **MAJOR** version: Geriye uyumsuz API deÄŸiÅŸiklikleri
- **MINOR** version: Geriye uyumlu yeni Ã¶zellikler
- **PATCH** version: Geriye uyumlu hata dÃ¼zeltmeleri

---

**Son GÃ¼ncelleme**: 2026-01-10


---


# Cİ_CD

# ğŸ”„ CI/CD Pipeline Guide

CyberGuard AI iÃ§in CI/CD kurulumu

---

## ğŸ“‹ Ä°Ã§indekiler

- [Genel BakÄ±ÅŸ](#genel-bakÄ±ÅŸ)
- [GitHub Actions](#github-actions)
- [Docker Build](#docker-build)
- [Deployment](#deployment)
- [Secrets Management](#secrets-management)

---

## ğŸŒŸ Genel BakÄ±ÅŸ

```
â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”    â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”    â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”    â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
â”‚    Code     â”‚ -> â”‚    Test     â”‚ -> â”‚    Build    â”‚ -> â”‚   Deploy    â”‚
â”‚    Push     â”‚    â”‚   (pytest)  â”‚    â”‚   (Docker)  â”‚    â”‚  (K8s/VPS)  â”‚
â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜    â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜    â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜    â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜
```

---

## ğŸ™ GitHub Actions

### Ana Workflow

```yaml
# .github/workflows/main.yml
name: CI/CD Pipeline

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

env:
  PYTHON_VERSION: '3.11'
  NODE_VERSION: '18'

jobs:
  # Lint Job
  lint:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v4
    
    - name: Set up Python
      uses: actions/setup-python@v5
      with:
        python-version: ${{ env.PYTHON_VERSION }}
    
    - name: Install linters
      run: pip install flake8 black isort
    
    - name: Lint Python
      run: |
        flake8 app/ --max-line-length=100
        black --check app/
        isort --check-only app/

  # Test Job
  test:
    runs-on: ubuntu-latest
    needs: lint
    
    services:
      postgres:
        image: postgres:14
        env:
          POSTGRES_DB: test_db
          POSTGRES_PASSWORD: test_pass
        ports:
          - 5432:5432
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Set up Python
      uses: actions/setup-python@v5
      with:
        python-version: ${{ env.PYTHON_VERSION }}
        cache: 'pip'
    
    - name: Install dependencies
      run: pip install -r requirements.txt
    
    - name: Run tests
      run: pytest tests/ -v --cov=app --cov-report=xml
      env:
        DATABASE_URL: postgresql://postgres:test_pass@localhost/test_db
    
    - name: Upload coverage
      uses: codecov/codecov-action@v4
      with:
        files: coverage.xml

  # Frontend Test
  frontend-test:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Setup Node
      uses: actions/setup-node@v4
      with:
        node-version: ${{ env.NODE_VERSION }}
        cache: 'npm'
        cache-dependency-path: frontend/package-lock.json
    
    - name: Install & Test
      working-directory: frontend
      run: |
        npm ci
        npm run lint
        npm run test

  # Build Docker
  build:
    runs-on: ubuntu-latest
    needs: [test, frontend-test]
    if: github.ref == 'refs/heads/main'
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Login to Docker Hub
      uses: docker/login-action@v3
      with:
        username: ${{ secrets.DOCKER_USERNAME }}
        password: ${{ secrets.DOCKER_PASSWORD }}
    
    - name: Build and Push
      uses: docker/build-push-action@v5
      with:
        context: .
        push: true
        tags: |
          cyberguard/api:latest
          cyberguard/api:${{ github.sha }}

  # Deploy
  deploy:
    runs-on: ubuntu-latest
    needs: build
    if: github.ref == 'refs/heads/main'
    
    steps:
    - name: Deploy to server
      uses: appleboy/ssh-action@v1.0.0
      with:
        host: ${{ secrets.DEPLOY_HOST }}
        username: ${{ secrets.DEPLOY_USER }}
        key: ${{ secrets.DEPLOY_KEY }}
        script: |
          cd /opt/cyberguard
          docker-compose pull
          docker-compose up -d
          docker system prune -f
```

### PR Checks

```yaml
# .github/workflows/pr.yml
name: PR Checks

on:
  pull_request:
    branches: [main, develop]

jobs:
  check:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v4
    
    - name: Check commit message
      uses: wagoid/commitlint-github-action@v5
    
    - name: Check PR size
      uses: codelytv/pr-size-labeler@v1
      with:
        GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
```

---

## ğŸ³ Docker Build

### Dockerfile

```dockerfile
# Dockerfile
FROM python:3.11-slim

WORKDIR /app

# Dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# App
COPY app/ ./app/
COPY models/ ./models/

# Run
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### docker-compose.yml

```yaml
version: '3.8'

services:
  api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=postgresql://postgres:password@db/cyberguard
    depends_on:
      - db
      - redis
    restart: unless-stopped

  frontend:
    build: ./frontend
    ports:
      - "80:80"
    depends_on:
      - api

  db:
    image: postgres:14-alpine
    volumes:
      - pgdata:/var/lib/postgresql/data
    environment:
      - POSTGRES_DB=cyberguard
      - POSTGRES_PASSWORD=password

  redis:
    image: redis:7-alpine
    volumes:
      - redisdata:/data

volumes:
  pgdata:
  redisdata:
```

---

## ğŸš€ Deployment

### Kubernetes (Ã–rnek)

```yaml
# k8s/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: cyberguard-api
spec:
  replicas: 3
  selector:
    matchLabels:
      app: cyberguard-api
  template:
    metadata:
      labels:
        app: cyberguard-api
    spec:
      containers:
      - name: api
        image: cyberguard/api:latest
        ports:
        - containerPort: 8000
        resources:
          limits:
            memory: "1Gi"
            cpu: "500m"
        env:
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: cyberguard-secrets
              key: database-url
```

---

## ğŸ” Secrets Management

### GitHub Secrets

| Secret | AÃ§Ä±klama |
|--------|----------|
| `DOCKER_USERNAME` | Docker Hub username |
| `DOCKER_PASSWORD` | Docker Hub password |
| `DEPLOY_HOST` | Server IP |
| `DEPLOY_USER` | SSH user |
| `DEPLOY_KEY` | SSH private key |
| `DATABASE_URL` | Production DB URL |

### .env.example

```env
# API
GOOGLE_API_KEY=xxx

# Database
DATABASE_URL=postgresql://user:pass@host/db

# Security
SECRET_KEY=xxx
JWT_SECRET=xxx
```

---

## ğŸ“Š Pipeline Metrikleri

| Metrik | Hedef |
|--------|-------|
| Build Time | < 5 min |
| Test Coverage | > 80% |
| Deploy Time | < 2 min |
| Rollback Time | < 1 min |


---


# CODE_OF_CONDUCT

# ğŸ‘¥ DavranÄ±ÅŸ KurallarÄ±

## TaahhÃ¼dÃ¼mÃ¼z

AÃ§Ä±k ve misafirperver bir ortam saÄŸlamak adÄ±na, katÄ±lÄ±mcÄ±lar ve yÃ¶neticiler olarak projemizi herkes iÃ§in taciz iÃ§ermeyen bir deneyim haline getirmeyi taahhÃ¼t ediyoruz. Bu, yaÅŸ, beden Ã¶lÃ§Ã¼sÃ¼, engellilik, etnik kÃ¶ken, cinsiyet kimliÄŸi ve ifadesi, deneyim seviyesi, milliyet, kiÅŸisel gÃ¶rÃ¼nÃ¼m, Ä±rk, din veya cinsel kimlik ve yÃ¶nelim fark etmeksizin geÃ§erlidir.

## StandartlarÄ±mÄ±z

**Olumlu bir ortam yaratmaya katkÄ±da bulunan davranÄ±ÅŸ Ã¶rnekleri:**

- âœ… KarÅŸÄ±lÄ±klÄ± saygÄ±lÄ± ve kapsayÄ±cÄ± dil kullanmak
- âœ… FarklÄ± bakÄ±ÅŸ aÃ§Ä±larÄ±na ve deneyimlere saygÄ± gÃ¶stermek
- âœ… YapÄ±cÄ± eleÅŸtiriyi nazikÃ§e kabul etmek
- âœ… Topluluk iÃ§in en iyi olana odaklanmak
- âœ… DiÄŸer topluluk Ã¼yelerine empati gÃ¶stermek

**Kabul edilemez davranÄ±ÅŸ Ã¶rnekleri:**

- âŒ CinselleÅŸtirilmiÅŸ dil veya gÃ¶rsel kullanÄ±mÄ± ve istenmeyen cinsel ilgi
- âŒ Trolleme, hakaret/aÅŸaÄŸÄ±layÄ±cÄ± yorumlar ve kiÅŸisel/politik saldÄ±rÄ±lar
- âŒ AÃ§Ä±k veya Ã¶zel taciz
- âŒ BaÅŸkalarÄ±nÄ±n fiziksel veya elektronik adres gibi Ã¶zel bilgilerini aÃ§Ä±k izin olmadan yayÄ±nlamak
- âŒ Profesyonel ortamda makul olarak uygunsuz sayÄ±labilecek diÄŸer davranÄ±ÅŸlar

## SorumluluklarÄ±mÄ±z

Proje yÃ¶neticileri, kabul edilebilir davranÄ±ÅŸ standartlarÄ±nÄ± netleÅŸtirmekten sorumludur ve uygunsuz davranÄ±ÅŸlara karÅŸÄ± uygun ve adil dÃ¼zeltici Ã¶nlemler almalarÄ± beklenir.

Proje yÃ¶neticileri, bu DavranÄ±ÅŸ KurallarÄ±na uymayan yorumlarÄ±, commit'leri, kodu, wiki dÃ¼zenlemelerini, issue'larÄ± ve diÄŸer katkÄ±larÄ± kaldÄ±rma, dÃ¼zenleme veya reddetme hakkÄ±na sahiptir. Uygunsuz, tehditkar, saldÄ±rgan veya zararlÄ± gÃ¶rÃ¼len davranÄ±ÅŸlarda bulunan katÄ±lÄ±mcÄ±larÄ± geÃ§ici veya kalÄ±cÄ± olarak yasaklayabilirler.

## Kapsam

Bu DavranÄ±ÅŸ KurallarÄ±, bir bireyin projeyi veya topluluÄŸunu temsil ettiÄŸi tÃ¼m proje alanlarÄ±nda ve kamuya aÃ§Ä±k alanlarda geÃ§erlidir. Bir projeyi veya topluluÄŸu temsil etme Ã¶rnekleri arasÄ±nda resmi bir proje e-posta adresi kullanmak, resmi bir sosyal medya hesabÄ± Ã¼zerinden gÃ¶nderi paylaÅŸmak veya Ã§evrimiÃ§i ya da Ã§evrimdÄ±ÅŸÄ± bir etkinlikte atanmÄ±ÅŸ bir temsilci olarak hareket etmek yer alÄ±r.

## Uygulama

Taciz edici, kÃ¶tÃ¼ niyetli veya baÅŸka ÅŸekilde kabul edilemez davranÄ±ÅŸ Ã¶rnekleri, proje ekibiyle **conduct@cyberguard-ai.com** adresinden iletiÅŸime geÃ§ilerek bildirilebilir. TÃ¼m ÅŸikayetler incelenecek ve araÅŸtÄ±rÄ±lacak ve duruma uygun ve gerekli gÃ¶rÃ¼len bir yanÄ±t verilecektir. Proje ekibi, bir olayÄ± bildiren kiÅŸinin gizliliÄŸini korumakla yÃ¼kÃ¼mlÃ¼dÃ¼r.

## YaptÄ±rÄ±m KÄ±lavuzu

Proje yÃ¶neticileri, bu DavranÄ±ÅŸ KurallarÄ±nÄ± ihlal eden herhangi bir davranÄ±ÅŸ iÃ§in uygun sayÄ±lan sonuÃ§larÄ± belirlerken aÅŸaÄŸÄ±daki Topluluk Etkisi KÄ±lavuzunu takip edecektir:

### 1. DÃ¼zeltme

**Topluluk Etkisi**: Uygunsuz dil kullanÄ±mÄ± veya profesyonel olmayan ya da toplulukta hoÅŸ karÅŸÄ±lanmayan diÄŸer davranÄ±ÅŸlar.

**SonuÃ§**: Proje yÃ¶neticilerinden Ã¶zel, yazÄ±lÄ± bir uyarÄ±; ihlal hakkÄ±nda netlik ve davranÄ±ÅŸÄ±n neden uygunsuz olduÄŸuna dair aÃ§Ä±klama saÄŸlanÄ±r. AÃ§Ä±k bir Ã¶zÃ¼r istenebilir.

### 2. UyarÄ±

**Topluluk Etkisi**: Tek bir olay veya bir dizi eylem yoluyla bir ihlal.

**SonuÃ§**: Devam eden davranÄ±ÅŸÄ±n sonuÃ§larÄ±yla ilgili bir uyarÄ±. Belirli bir sÃ¼re boyunca DavranÄ±ÅŸ KurallarÄ±nÄ± uygulayanlarla istenmeyen etkileÅŸim de dahil olmak Ã¼zere, ilgili kiÅŸilerle etkileÅŸim yasaÄŸÄ±. Bu, topluluk alanlarÄ±nÄ±n yanÄ± sÄ±ra sosyal medya gibi harici kanallardan kaÃ§Ä±nmayÄ± iÃ§erir. Bu ÅŸartlarÄ±n ihlali geÃ§ici veya kalÄ±cÄ± bir yasaÄŸa yol aÃ§abilir.

### 3. GeÃ§ici Yasak

**Topluluk Etkisi**: SÃ¼rekli uygunsuz davranÄ±ÅŸ da dahil olmak Ã¼zere, topluluk standartlarÄ±nÄ±n ciddi bir ihlali.

**SonuÃ§**: Belirli bir sÃ¼re boyunca toplulukla her tÃ¼rlÃ¼ etkileÅŸim veya kamusal iletiÅŸimden geÃ§ici bir yasak. Bu sÃ¼re boyunca, DavranÄ±ÅŸ KurallarÄ±nÄ± uygulayan kiÅŸilerle istenmeyen etkileÅŸim de dahil olmak Ã¼zere, ilgili kiÅŸilerle hiÃ§bir kamuya aÃ§Ä±k veya Ã¶zel etkileÅŸime izin verilmez. Bu ÅŸartlarÄ±n ihlali kalÄ±cÄ± bir yasaÄŸa yol aÃ§abilir.

### 4. KalÄ±cÄ± Yasak

**Topluluk Etkisi**: SÃ¼rekli uygunsuz davranÄ±ÅŸ, bir bireyin taciz edilmesi veya bireylerin sÄ±nÄ±flarÄ±na karÅŸÄ± saldÄ±rganlÄ±k veya aÅŸaÄŸÄ±lama da dahil olmak Ã¼zere, topluluk standartlarÄ±nÄ±n ihlal edilme modelini gÃ¶sterme.

**SonuÃ§**: Topluluk iÃ§inde her tÃ¼rlÃ¼ kamuya aÃ§Ä±k etkileÅŸimden kalÄ±cÄ± bir yasak.

## Ä°lham KaynaÄŸÄ±

Bu DavranÄ±ÅŸ KurallarÄ±, **Contributor Covenant** sÃ¼rÃ¼m 2.1'den uyarlanmÄ±ÅŸtÄ±r.
https://www.contributor-covenant.org/version/2/1/code_of_conduct.html

Topluluk Etkisi KÄ±lavuzu, Mozilla'nÄ±n davranÄ±ÅŸ kurallarÄ± uygulama merdiveninden ilham almÄ±ÅŸtÄ±r.
https://github.com/mozilla/diversity

## Sorular

Bu DavranÄ±ÅŸ KurallarÄ± hakkÄ±nda sorularÄ±nÄ±z varsa, lÃ¼tfen **conduct@cyberguard-ai.com** adresinden bizimle iletiÅŸime geÃ§in.

---

**HatÄ±rlatma**: GÃ¼venli, saygÄ±lÄ± ve kapsayÄ±cÄ± bir topluluk oluÅŸturmak hepimizin sorumluluÄŸundadÄ±r. ğŸ’™

---


# CONTRİBUTİNG

# ğŸ¤ CyberGuard AI'ya KatkÄ±da Bulunma

CyberGuard AI'ya katkÄ±da bulunmayÄ± dÃ¼ÅŸÃ¼ndÃ¼ÄŸÃ¼nÃ¼z iÃ§in teÅŸekkÃ¼r ederiz! ğŸ‰

## ğŸ“‹ Ä°Ã§indekiler

- [DavranÄ±ÅŸ KurallarÄ±](#davranÄ±ÅŸ-kurallarÄ±)
- [NasÄ±l KatkÄ±da Bulunabilirim?](#nasÄ±l-katkÄ±da-bulunabilirim)
- [GeliÅŸtirme OrtamÄ± Kurulumu](#geliÅŸtirme-ortamÄ±-kurulumu)
- [Pull Request SÃ¼reci](#pull-request-sÃ¼reci)
- [Kodlama StandartlarÄ±](#kodlama-standartlarÄ±)
- [Commit KurallarÄ±](#commit-kurallarÄ±)
- [Test Yazma](#test-yazma)

---

## ğŸ“œ DavranÄ±ÅŸ KurallarÄ±

Bu proje ve katÄ±lan herkes [DavranÄ±ÅŸ KurallarÄ±](CODE_OF_CONDUCT.md) tarafÄ±ndan yÃ¶netilir. KatÄ±larak bu kurallara uymayÄ± kabul etmiÅŸ sayÄ±lÄ±rsÄ±nÄ±z.

---

## ğŸ¯ NasÄ±l KatkÄ±da Bulunabilirim?

### ğŸ› Hata Bildirimi

Hata bildirmeden Ã¶nce lÃ¼tfen mevcut issue'larÄ± kontrol edin. Hata raporu oluÅŸtururken ÅŸunlarÄ± ekleyin:

- **AÃ§Ä±k baÅŸlÄ±k ve aÃ§Ä±klama**
- **HatayÄ± tekrarlama adÄ±mlarÄ±**
- **Beklenen ve gerÃ§ekleÅŸen davranÄ±ÅŸ**
- **Ekran gÃ¶rÃ¼ntÃ¼leri** (varsa)
- **Ortam detaylarÄ±** (Ä°ÅŸletim sistemi, Python sÃ¼rÃ¼mÃ¼, vb.)

**Hata Raporu Åablonu:**
```markdown
## Hata AÃ§Ä±klamasÄ±
[HatanÄ±n net aÃ§Ä±klamasÄ±]

## Tekrarlama AdÄ±mlarÄ±
1. '...' sayfasÄ±na git
2. '...' butonuna tÄ±kla
3. HatayÄ± gÃ¶r

## Beklenen DavranÄ±ÅŸ
[Ne olmasÄ±nÄ± bekliyordunuz]

## GerÃ§ekleÅŸen DavranÄ±ÅŸ
[Ne oldu]

## Ortam Bilgileri
- Ä°ÅŸletim Sistemi: [Ã¶rn. Ubuntu 22.04]
- Python: [Ã¶rn. 3.10.5]
- Versiyon: [Ã¶rn. v2.0.0]

## Ekran GÃ¶rÃ¼ntÃ¼leri
[Varsa ekleyin]
```

### ğŸ’¡ Ã–zellik Ã–nerme

Ã–zellik Ã¶nerileri memnuniyetle karÅŸÄ±lanÄ±r! LÃ¼tfen ÅŸunlarÄ± ekleyin:

- **AÃ§Ä±k kullanÄ±m senaryosu**
- **DetaylÄ± aÃ§Ä±klama**
- **Mockup veya Ã¶rnekler** (varsa)
- **OlasÄ± implementasyon yaklaÅŸÄ±mÄ±**

**Ã–zellik Ä°steÄŸi Åablonu:**
```markdown
## Ã–zellik AÃ§Ä±klamasÄ±
[Ã–zelliÄŸin net aÃ§Ä±klamasÄ±]

## KullanÄ±m Senaryosu
[Bu Ã¶zellik ne zaman ve neden kullanÄ±lacak?]

## Ã–nerilen Ã‡Ã¶zÃ¼m
[Ã–zelliÄŸin nasÄ±l Ã§alÄ±ÅŸmasÄ±nÄ± Ã¶neriyorsunuz?]

## Alternatifler
[DÃ¼ÅŸÃ¼ndÃ¼ÄŸÃ¼nÃ¼z alternatif Ã§Ã¶zÃ¼mler]

## Ek Bilgiler
[Ekran gÃ¶rÃ¼ntÃ¼leri, mockup'lar, vb.]
```

### ğŸ“ DokÃ¼mantasyon Ä°yileÅŸtirmeleri

DokÃ¼mantasyon her zaman iyileÅŸtirilebilir:

- YazÄ±m hatalarÄ±nÄ± dÃ¼zeltme
- AÃ§Ä±klamalarÄ± netleÅŸtirme
- Ã–rnekler ekleme
- TÃ¼rkÃ§e/Ä°ngilizce Ã§eviri geliÅŸtirmeleri

---

## ğŸ’» GeliÅŸtirme OrtamÄ± Kurulumu

### 1. Repository'yi Fork Edin

```bash
# GitHub'da "Fork" butonuna tÄ±klayÄ±n
# Sonra klonlayÄ±n:
git clone https://github.com/KULLANICI_ADINIZ/cyberguard-ai.git
cd cyberguard-ai
```

### 2. Upstream Remote Ekleyin

```bash
git remote add upstream https://github.com/cyberguard-ai/cyberguard-ai.git
```

### 3. Sanal Ortam OluÅŸturun

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

### 4. BaÄŸÄ±mlÄ±lÄ±klarÄ± YÃ¼kleyin

```bash
# Gerekli paketler
pip install -r requirements.txt

# GeliÅŸtirme paketleri
pip install -r requirements-dev.txt
```

### 5. Pre-commit Hook'larÄ± Kurun

```bash
pre-commit install
```

---

## ğŸ”„ Pull Request SÃ¼reci

### 1. Branch OluÅŸturun

```bash
# Feature iÃ§in
git checkout -b feature/yeni-ozellik-adi

# Bug fix iÃ§in
git checkout -b bugfix/hata-aciklamasi

# DokÃ¼mantasyon iÃ§in
git checkout -b docs/dokuman-aciklamasi
```

### 2. DeÄŸiÅŸikliklerinizi YapÄ±n

- KÃ¼Ã§Ã¼k, odaklanmÄ±ÅŸ deÄŸiÅŸiklikler yapÄ±n
- Her commit tek bir konuya odaklanmalÄ±
- Kod standartlarÄ±na uyun

### 3. Test Edin

```bash
# TÃ¼m testleri Ã§alÄ±ÅŸtÄ±rÄ±n
pytest

# Coverage kontrolÃ¼
pytest --cov=src tests/

# Linting
flake8 src/
black --check src/
```

### 4. Commit Edin

```bash
git add .
git commit -m "feat: yeni Ã¶zellik eklendi"
```

### 5. Push Edin

```bash
git push origin feature/yeni-ozellik-adi
```

### 6. Pull Request AÃ§Ä±n

- GitHub'da repository'nize gidin
- "Pull Request" butonuna tÄ±klayÄ±n
- DeÄŸiÅŸikliklerinizi aÃ§Ä±klayÄ±n
- Ä°lgili issue'larÄ± baÄŸlayÄ±n

**PR Åablonu:**
```markdown
## AÃ§Ä±klama
[DeÄŸiÅŸikliklerinizin kÄ±sa aÃ§Ä±klamasÄ±]

## DeÄŸiÅŸiklik Tipi
- [ ] ğŸ› Bug fix
- [ ] âœ¨ Yeni Ã¶zellik
- [ ] ğŸ“ DokÃ¼mantasyon
- [ ] ğŸ¨ Stil/formatting
- [ ] â™»ï¸ Refactoring
- [ ] ğŸ”§ KonfigÃ¼rasyon

## BaÄŸlantÄ±lÄ± Issue'lar
Fixes #(issue numarasÄ±)

## Test Edilen Senaryolar
- [ ] Test senaryosu 1
- [ ] Test senaryosu 2

## Checklist
- [ ] Kod kodlama standartlarÄ±na uygun
- [ ] Testler yazÄ±ldÄ± ve geÃ§iyor
- [ ] DokÃ¼mantasyon gÃ¼ncellendi
- [ ] CHANGELOG.md gÃ¼ncellendi
```

---

## ğŸ“ Kodlama StandartlarÄ±

### Python Stil KÄ±lavuzu

**PEP 8 StandartlarÄ±na uyun:**

```python
# âœ… Ä°YÄ°
def calculate_risk_score(vulnerability_data: dict) -> float:
    """
    Zafiyet verilerinden risk skoru hesaplar.
    
    Args:
        vulnerability_data: Zafiyet bilgilerini iÃ§eren sÃ¶zlÃ¼k
        
    Returns:
        0-10 arasÄ± risk skoru
    """
    severity = vulnerability_data.get('severity', 0)
    exploitability = vulnerability_data.get('exploitability', 0)
    return (severity * 0.6) + (exploitability * 0.4)

# âŒ KÃ–TÃœ
def calc(d):
    s=d.get('severity',0)
    e=d.get('exploitability',0)
    return s*0.6+e*0.4
```

### Genel Kurallar

1. **Ä°simlendirme:**
    - `snake_case` fonksiyonlar ve deÄŸiÅŸkenler iÃ§in
    - `PascalCase` sÄ±nÄ±flar iÃ§in
    - `UPPER_CASE` sabitler iÃ§in

2. **Docstring:**
    - Her fonksiyon ve sÄ±nÄ±f iÃ§in docstring yazÄ±n
    - Google style veya NumPy style kullanÄ±n

3. **Type Hints:**
    - MÃ¼mkÃ¼n olduÄŸunca type hint kullanÄ±n
   ```python
   def process_log(log_file: str) -> List[dict]:
       pass
   ```

4. **Imports:**
   ```python
   # Standart kÃ¼tÃ¼phane
   import os
   import sys
   
   # ÃœÃ§Ã¼ncÃ¼ parti
   import numpy as np
   import pandas as pd
   
   # Yerel
   from src.models import AIModel
   from src.utils import logger
   ```

### Code Formatting

```bash
# Black ile otomatik formatlama
black src/

# isort ile import sÄ±ralama
isort src/

# flake8 ile lint kontrolÃ¼
flake8 src/
```

---

## ğŸ“ Commit KurallarÄ±

**Conventional Commits** formatÄ±nÄ± kullanÄ±n:

### Commit Mesaj FormatÄ±

```
<tip>(<kapsam>): <kÄ±sa aÃ§Ä±klama>

[opsiyonel detaylÄ± aÃ§Ä±klama]

[opsiyonel footer]
```

### Commit Tipleri

| Tip | AÃ§Ä±klama | Ã–rnek |
|-----|----------|-------|
| `feat` | Yeni Ã¶zellik | `feat(chatbot): NLP modeli eklendi` |
| `fix` | Hata dÃ¼zeltme | `fix(scanner): port tarama hatasÄ± dÃ¼zeltildi` |
| `docs` | DokÃ¼mantasyon | `docs(readme): kurulum adÄ±mlarÄ± gÃ¼ncellendi` |
| `style` | Kod formatÄ± | `style: black ile formatlama yapÄ±ldÄ±` |
| `refactor` | Kod iyileÅŸtirme | `refactor(api): endpoint yapÄ±sÄ± dÃ¼zenlendi` |
| `test` | Test ekleme | `test(scanner): unit testler eklendi` |
| `chore` | Genel iÅŸler | `chore: dependencies gÃ¼ncellendi` |
| `perf` | Performans | `perf(ml): model inference hÄ±zlandÄ±rÄ±ldÄ±` |

### Ã–rnekler

```bash
# Yeni Ã¶zellik
git commit -m "feat(chatbot): Ã§oklu dil desteÄŸi eklendi"

# Hata dÃ¼zeltme
git commit -m "fix(database): baÄŸlantÄ± timeout sorunu Ã§Ã¶zÃ¼ldÃ¼"

# DokÃ¼mantasyon
git commit -m "docs(api): endpoint Ã¶rnekleri eklendi"

# DetaylÄ± commit
git commit -m "feat(scanner): deep scan modu eklendi

- CVE veritabanÄ± entegrasyonu
- DetaylÄ± port analizi
- PDF rapor oluÅŸturma

Closes #123"
```

---

## ğŸ§ª Test Yazma

### Test YapÄ±sÄ±

```
tests/
â”œâ”€â”€ unit/              # Birim testler
â”‚   â”œâ”€â”€ test_chatbot.py
â”‚   â”œâ”€â”€ test_scanner.py
â”‚   â””â”€â”€ test_models.py
â”œâ”€â”€ integration/       # Entegrasyon testler
â”‚   â”œâ”€â”€ test_api.py
â”‚   â””â”€â”€ test_database.py
â””â”€â”€ e2e/              # End-to-end testler
    â””â”€â”€ test_workflows.py
```

### Test Yazma KurallarÄ±

**1. Her fonksiyon iÃ§in test yazÄ±n:**

```python
# src/scanner.py
def scan_port(ip: str, port: int) -> bool:
    """Port'un aÃ§Ä±k olup olmadÄ±ÄŸÄ±nÄ± kontrol eder."""
    # implementasyon
    pass

# tests/unit/test_scanner.py
def test_scan_port_open():
    """AÃ§Ä±k port doÄŸru tespit edilmeli."""
    result = scan_port("127.0.0.1", 80)
    assert result is True

def test_scan_port_closed():
    """KapalÄ± port doÄŸru tespit edilmeli."""
    result = scan_port("127.0.0.1", 9999)
    assert result is False

def test_scan_port_invalid_ip():
    """GeÃ§ersiz IP ile hata fÄ±rlatmalÄ±."""
    with pytest.raises(ValueError):
        scan_port("invalid", 80)
```

**2. Fixture kullanÄ±n:**

```python
@pytest.fixture
def sample_vulnerability():
    return {
        'cve_id': 'CVE-2024-1234',
        'severity': 9.8,
        'description': 'Test vulnerability'
    }

def test_process_vulnerability(sample_vulnerability):
    result = process_vulnerability(sample_vulnerability)
    assert result['risk_level'] == 'critical'
```

**3. Mock kullanÄ±n:**

```python
from unittest.mock import Mock, patch

@patch('src.scanner.socket.socket')
def test_scan_with_mock(mock_socket):
    mock_socket.return_value.connect_ex.return_value = 0
    result = scan_port("192.168.1.1", 22)
    assert result is True
```

### Test Ã‡alÄ±ÅŸtÄ±rma

```bash
# TÃ¼m testler
pytest

# Belirli bir dosya
pytest tests/unit/test_scanner.py

# Belirli bir test
pytest tests/unit/test_scanner.py::test_scan_port_open

# Coverage ile
pytest --cov=src --cov-report=html

# Verbose mode
pytest -v

# Sadece failed testler
pytest --lf
```

### Coverage Hedefi

- **Minimum %80 coverage** gereklidir
- Kritik modÃ¼ller iÃ§in **%90+** hedefleyin
- Coverage raporunu kontrol edin: `htmlcov/index.html`

---

## ğŸ” Code Review SÃ¼reci

### Review Beklerken

1. âœ… TÃ¼m testlerin geÃ§tiÄŸinden emin olun
2. âœ… CI/CD pipeline'Ä±nÄ±n baÅŸarÄ±lÄ± olduÄŸunu kontrol edin
3. âœ… Ã‡akÄ±ÅŸmalarÄ± Ã§Ã¶zÃ¼n
4. âœ… Review yorumlarÄ±na hÄ±zlÄ±ca yanÄ±t verin

### Review Yaparken

**Kontrol Edilecekler:**

- [ ] Kod anlaÅŸÄ±lÄ±r ve bakÄ±mÄ± kolay mÄ±?
- [ ] Testler yeterli mi?
- [ ] DokÃ¼mantasyon gÃ¼ncel mi?
- [ ] GÃ¼venlik aÃ§Ä±klarÄ± var mÄ±?
- [ ] Performance etkileri dÃ¼ÅŸÃ¼nÃ¼lmÃ¼ÅŸ mÃ¼?
- [ ] Error handling yeterli mi?

**YapÄ±cÄ± Geri Bildirim:**

```markdown
# âŒ KÃ¶tÃ¼
Bu kod berbat.

# âœ… Ä°yi
Bu fonksiyonda error handling eksik gÃ¶rÃ¼nÃ¼yor. 
`try-except` bloÄŸu ekleyerek daha robust hale getirebiliriz.
Ã–rnek: [link to example]
```

---

## ğŸ·ï¸ Issue ve PR Etiketleri

### Issue Etiketleri

| Etiket | AÃ§Ä±klama |
|--------|----------|
| `bug` ğŸ› | Bir ÅŸeyler Ã§alÄ±ÅŸmÄ±yor |
| `enhancement` âœ¨ | Yeni Ã¶zellik veya istek |
| `documentation` ğŸ“ | DokÃ¼mantasyon iyileÅŸtirmesi |
| `good first issue` ğŸ‘¶ | Yeni katkÄ±cÄ±lar iÃ§in uygun |
| `help wanted` ğŸ†˜ | Ekstra dikkat gerekiyor |
| `priority: high` ğŸ”´ | YÃ¼ksek Ã¶ncelikli |
| `priority: low` ğŸŸ¢ | DÃ¼ÅŸÃ¼k Ã¶ncelikli |
| `wontfix` â›” | Ãœzerinde Ã§alÄ±ÅŸÄ±lmayacak |

### PR Etiketleri

| Etiket | AÃ§Ä±klama |
|--------|----------|
| `WIP` ğŸš§ | Work in progress |
| `ready for review` ğŸ‘€ | Review iÃ§in hazÄ±r |
| `needs work` ğŸ”§ | DeÄŸiÅŸiklik gerekiyor |
| `approved` âœ… | OnaylandÄ± |

---

## ğŸ“ Ä°letiÅŸim ve Sorular

### Soru Sormadan Ã–nce

1. ğŸ“– [DokÃ¼mantasyonu](docs/) okudunuz mu?
2. ğŸ” [Mevcut issue'larda](https://github.com/cyberguard-ai/issues) aradÄ±nÄ±z mÄ±?
3. ğŸ’¬ [Discussions](https://github.com/cyberguard-ai/discussions) bÃ¶lÃ¼mÃ¼nÃ¼ kontrol ettiniz mi?

### Ä°letiÅŸim KanallarÄ±

- ğŸ’¬ **Discord**: [discord.gg/cyberguard](https://discord.gg/cyberguard)
- ğŸ“§ **Email**: contribute@cyberguard-ai.com
- ğŸ¦ **Twitter**: [@cyberguard_ai](https://twitter.com/cyberguard_ai)

---

## ğŸ‰ Ä°lk KatkÄ±nÄ±zÄ± YapÄ±n!

Yeni baÅŸlÄ±yorsanÄ±z:

1. `good first issue` etiketli issue'lara bakÄ±n
2. KÃ¼Ã§Ã¼k bir dÃ¼zeltme ile baÅŸlayÄ±n (typo, dokÃ¼mantasyon)
3. Topluluktan yardÄ±m istemekten Ã§ekinmeyin!

---

## ğŸ™ TeÅŸekkÃ¼rler!

Her katkÄ±, bÃ¼yÃ¼k ya da kÃ¼Ã§Ã¼k, Ã§ok deÄŸerlidir. CyberGuard AI'yÄ± daha iyi hale getirmeye yardÄ±mcÄ± olduÄŸunuz iÃ§in teÅŸekkÃ¼r ederiz! ğŸ’™

---

**Not:** Bu kÄ±lavuz sÃ¼rekli geliÅŸtirilmektedir. Ã–nerileriniz varsa lÃ¼tfen issue aÃ§Ä±n!

---


# DATASETS

# ğŸ“Š Datasets DokÃ¼mantasyonu

CyberGuard AI'da kullanÄ±lan veri setleri hakkÄ±nda detaylÄ± bilgi

---

## ğŸ“‹ Ä°Ã§indekiler

- [Genel BakÄ±ÅŸ](#genel-bakÄ±ÅŸ)
- [NSL-KDD Dataset](#nsl-kdd-dataset)
- [CICIDS2017 Dataset](#cicids2017-dataset)
- [BoT-IoT Dataset](#bot-iot-dataset)
- [Veri Ã–n Ä°ÅŸleme](#veri-Ã¶n-iÅŸleme)
- [Feature Engineering](#feature-engineering)

---

## ğŸŒŸ Genel BakÄ±ÅŸ

### Desteklenen Veri Setleri

| Dataset | KayÄ±t SayÄ±sÄ± | Ã–zellik SayÄ±sÄ± | SÄ±nÄ±f SayÄ±sÄ± | YÄ±l |
|---------|--------------|----------------|--------------|-----|
| NSL-KDD | 148,517 | 41 | 5 | 2009 |
| CICIDS2017 | 2,830,743 | 78 | 15 | 2017 |
| BoT-IoT | 73,370,443 | 43 | 11 | 2019 |

### Veri Seti KarÅŸÄ±laÅŸtÄ±rmasÄ±

```
NSL-KDD     â–ˆâ–ˆâ–ˆâ–ˆâ–‘â–‘â–‘â–‘â–‘â–‘â–‘â–‘â–‘â–‘â–‘â–‘â–‘â–‘â–‘â–‘ 148K kayÄ±t
CICIDS2017  â–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–‘â–‘â–‘â–‘â–‘â–‘â–‘â–‘â–‘â–‘â–‘â–‘ 2.8M kayÄ±t  
BoT-IoT     â–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆ 73M kayÄ±t
```

---

## ğŸ” NSL-KDD Dataset

### Genel Bilgiler

| Ã–zellik | DeÄŸer |
|---------|-------|
| **Kaynak** | University of New Brunswick (UNB) |
| **YÄ±l** | 2009 |
| **Orijinal** | KDD'99 Dataset (iyileÅŸtirilmiÅŸ) |
| **Boyut** | ~130 MB |
| **Ä°ndirme** | [NSL-KDD Dataset](https://www.unb.ca/cic/datasets/nsl.html) |

### SaldÄ±rÄ± TÃ¼rleri

| Kategori | SaldÄ±rÄ± TÃ¼rleri | Ã–rnek |
|----------|-----------------|-------|
| **DoS** | back, land, neptune, pod, smurf, teardrop | Denial of Service |
| **Probe** | ipsweep, nmap, portsweep, satan | Network scanning |
| **R2L** | ftp_write, guess_passwd, imap, multihop | Remote to Local |
| **U2R** | buffer_overflow, loadmodule, perl, rootkit | User to Root |

### Ã–zellikler (41 Feature)

**Temel Ã–zellikler:**

```
duration, protocol_type, service, flag, src_bytes, dst_bytes,
land, wrong_fragment, urgent
```

**Ä°Ã§erik Ã–zellikleri:**

```
hot, num_failed_logins, logged_in, num_compromised, root_shell,
su_attempted, num_root, num_file_creations, num_shells,
num_access_files, num_outbound_cmds, is_host_login, is_guest_login
```

**Trafik Ã–zellikleri:**

```
count, srv_count, serror_rate, srv_serror_rate, rerror_rate,
srv_rerror_rate, same_srv_rate, diff_srv_rate, srv_diff_host_rate
```

**Host Ã–zellikleri:**

```
dst_host_count, dst_host_srv_count, dst_host_same_srv_rate,
dst_host_diff_srv_rate, dst_host_same_src_port_rate,
dst_host_srv_diff_host_rate, dst_host_serror_rate,
dst_host_srv_serror_rate, dst_host_rerror_rate, dst_host_srv_rerror_rate
```

### SÄ±nÄ±f DaÄŸÄ±lÄ±mÄ±

```
Normal      â–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–‘â–‘ 67,343 (45.3%)
DoS         â–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–‘â–‘â–‘â–‘â–‘â–‘â–‘ 45,927 (30.9%)
Probe       â–ˆâ–ˆâ–ˆâ–ˆâ–‘â–‘â–‘â–‘â–‘â–‘â–‘â–‘â–‘â–‘â–‘â–‘â–‘â–‘â–‘â–‘ 11,656 (7.8%)
R2L         â–ˆâ–ˆâ–ˆâ–‘â–‘â–‘â–‘â–‘â–‘â–‘â–‘â–‘â–‘â–‘â–‘â–‘â–‘â–‘â–‘â–‘ 995 (0.7%)
U2R         â–‘â–‘â–‘â–‘â–‘â–‘â–‘â–‘â–‘â–‘â–‘â–‘â–‘â–‘â–‘â–‘â–‘â–‘â–‘â–‘ 52 (0.03%)
```

---

## ğŸŒ CICIDS2017 Dataset

### Genel Bilgiler

| Ã–zellik | DeÄŸer |
|---------|-------|
| **Kaynak** | Canadian Institute for Cybersecurity |
| **YÄ±l** | 2017 |
| **SÃ¼re** | 5 iÅŸ gÃ¼nÃ¼ |
| **Boyut** | ~8 GB |
| **Ä°ndirme** | [CICIDS2017 Dataset](https://www.unb.ca/cic/datasets/ids-2017.html) |

### SaldÄ±rÄ± TÃ¼rleri (15 SÄ±nÄ±f)

| GÃ¼n | SaldÄ±rÄ± TÃ¼rÃ¼ | AÃ§Ä±klama |
|-----|-------------|----------|
| Monday | Normal | Sadece normal trafik |
| Tuesday | FTP-Patator, SSH-Patator | Brute force saldÄ±rÄ±larÄ± |
| Wednesday | DoS Slowloris, DoS Slowhttptest, DoS Hulk, DoS GoldenEye, Heartbleed | DoS saldÄ±rÄ±larÄ± |
| Thursday | Web Attack (XSS, SQL Injection, Brute Force), Infiltration | Web saldÄ±rÄ±larÄ± |
| Friday | Botnet, Port Scan, DDoS | Distributed saldÄ±rÄ±lar |

### Ã–zellikler (78 Feature)

**Flow Ã–zellikleri:**

```
Flow Duration, Total Fwd Packets, Total Backward Packets,
Total Length of Fwd Packets, Total Length of Bwd Packets,
Fwd Packet Length Max/Min/Mean/Std, Bwd Packet Length Max/Min/Mean/Std
```

**Zaman Ã–zellikleri:**

```
Flow Bytes/s, Flow Packets/s, Flow IAT Mean/Std/Max/Min,
Fwd IAT Total/Mean/Std/Max/Min, Bwd IAT Total/Mean/Std/Max/Min
```

**Flag Ã–zellikleri:**

```
Fwd PSH Flags, Bwd PSH Flags, Fwd URG Flags, Bwd URG Flags,
Fwd Header Length, Bwd Header Length
```

**Paket Ã–zellikleri:**

```
Fwd Packets/s, Bwd Packets/s, Min Packet Length, Max Packet Length,
Packet Length Mean/Std/Variance, FIN Flag Count, SYN Flag Count,
RST Flag Count, PSH Flag Count, ACK Flag Count, URG Flag Count
```

### SÄ±nÄ±f DaÄŸÄ±lÄ±mÄ±

```
BENIGN               â–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆ 2,273,097 (80.3%)
DDoS                 â–ˆâ–ˆâ–ˆâ–‘â–‘â–‘â–‘â–‘â–‘â–‘â–‘â–‘â–‘â–‘â–‘â–‘â–‘â–‘â–‘â–‘ 128,027 (4.5%)
PortScan             â–ˆâ–ˆâ–ˆâ–‘â–‘â–‘â–‘â–‘â–‘â–‘â–‘â–‘â–‘â–‘â–‘â–‘â–‘â–‘â–‘â–‘ 158,930 (5.6%)
DoS Hulk             â–ˆâ–ˆâ–‘â–‘â–‘â–‘â–‘â–‘â–‘â–‘â–‘â–‘â–‘â–‘â–‘â–‘â–‘â–‘â–‘â–‘ 231,073 (8.2%)
DoS GoldenEye        â–‘â–‘â–‘â–‘â–‘â–‘â–‘â–‘â–‘â–‘â–‘â–‘â–‘â–‘â–‘â–‘â–‘â–‘â–‘â–‘ 10,293 (0.4%)
FTP-Patator          â–‘â–‘â–‘â–‘â–‘â–‘â–‘â–‘â–‘â–‘â–‘â–‘â–‘â–‘â–‘â–‘â–‘â–‘â–‘â–‘ 7,938 (0.3%)
SSH-Patator          â–‘â–‘â–‘â–‘â–‘â–‘â–‘â–‘â–‘â–‘â–‘â–‘â–‘â–‘â–‘â–‘â–‘â–‘â–‘â–‘ 5,897 (0.2%)
DoS Slowloris        â–‘â–‘â–‘â–‘â–‘â–‘â–‘â–‘â–‘â–‘â–‘â–‘â–‘â–‘â–‘â–‘â–‘â–‘â–‘â–‘ 5,796 (0.2%)
...
```

---

## ğŸ¤– BoT-IoT Dataset

### Genel Bilgiler

| Ã–zellik | DeÄŸer |
|---------|-------|
| **Kaynak** | UNSW Sydney |
| **YÄ±l** | 2019 |
| **Ortam** | IoT Network Simulation |
| **Boyut** | ~16 GB |
| **Ä°ndirme** | [BoT-IoT Dataset](https://research.unsw.edu.au/projects/bot-iot-dataset) |

### SaldÄ±rÄ± TÃ¼rleri

| Kategori | Alt TÃ¼rler | AÃ§Ä±klama |
|----------|------------|----------|
| **DDoS** | UDP, TCP, HTTP | Distributed DoS |
| **DoS** | UDP, TCP, HTTP | Denial of Service |
| **Reconnaissance** | OS, Service | KeÅŸif saldÄ±rÄ±larÄ± |
| **Theft** | Data, Keylogging | Veri Ã§alma |
| **Normal** | - | Normal IoT trafiÄŸi |

### Ã–zellikler (43 Feature)

```
pkSeqID, stime, flgs, flgs_number, proto, proto_number,
saddr, sport, daddr, dport, pkts, bytes, state, state_number,
ltime, seq, dur, mean, stddev, sum, min, max, spkts, dpkts,
sbytes, dbytes, rate, srate, drate, TnBPSrcIP, TnBPDstIP,
TnP_PSrcIP, TnP_PDstIP, TnP_PerProto, TnP_Per_Dport, AR_P_Proto_P_SrcIP,
AR_P_Proto_P_DstIP, N_IN_Conn_P_DstIP, N_IN_Conn_P_SrcIP,
AR_P_Proto_P_Sport, AR_P_Proto_P_Dport, Pkts_P_State_P_Protocol_P_DestIP,
Pkts_P_State_P_Protocol_P_SrcIP
```

### SÄ±nÄ±f DaÄŸÄ±lÄ±mÄ±

```
DDoS                 â–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆ 56,844,535 (77.5%)
DoS                  â–ˆâ–ˆâ–ˆâ–ˆâ–‘â–‘â–‘â–‘â–‘â–‘â–‘â–‘â–‘â–‘â–‘â–‘â–‘â–‘â–‘â–‘ 12,315,997 (16.8%)
Normal               â–ˆâ–‘â–‘â–‘â–‘â–‘â–‘â–‘â–‘â–‘â–‘â–‘â–‘â–‘â–‘â–‘â–‘â–‘â–‘â–‘ 477,116 (0.7%)
Reconnaissance       â–ˆâ–‘â–‘â–‘â–‘â–‘â–‘â–‘â–‘â–‘â–‘â–‘â–‘â–‘â–‘â–‘â–‘â–‘â–‘â–‘ 2,652,191 (3.6%)
Theft                â–‘â–‘â–‘â–‘â–‘â–‘â–‘â–‘â–‘â–‘â–‘â–‘â–‘â–‘â–‘â–‘â–‘â–‘â–‘â–‘ 1,080,604 (1.5%)
```

---

## ğŸ”§ Veri Ã–n Ä°ÅŸleme

### 1. Veri Temizleme

```python
import pandas as pd
import numpy as np

# NaN ve Infinity deÄŸerleri temizle
df = df.replace([np.inf, -np.inf], np.nan)
df = df.dropna()

# Duplicate kayÄ±tlarÄ± kaldÄ±r
df = df.drop_duplicates()

# Outlier temizleme (IQR method)
Q1 = df.quantile(0.25)
Q3 = df.quantile(0.75)
IQR = Q3 - Q1
df = df[~((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))).any(axis=1)]
```

### 2. Feature Encoding

```python
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

# Label Encoding (ordinal)
le = LabelEncoder()
df['protocol_type'] = le.fit_transform(df['protocol_type'])

# One-Hot Encoding (categorical)
df = pd.get_dummies(df, columns=['service', 'flag'])
```

### 3. Feature Scaling

```python
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# StandardScaler (mean=0, std=1)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# MinMaxScaler (0-1 range)
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
```

### 4. Class Balancing

```python
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

# SMOTE (Synthetic Minority Over-sampling)
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Random Under-sampling (majority class)
rus = RandomUnderSampler(random_state=42)
X_resampled, y_resampled = rus.fit_resample(X, y)
```

---

## ğŸ¯ Feature Engineering

### 1. Temporal Features

```python
# Zaman bazlÄ± Ã¶zellikler
df['hour'] = pd.to_datetime(df['timestamp']).dt.hour
df['day_of_week'] = pd.to_datetime(df['timestamp']).dt.dayofweek
df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
```

### 2. Statistical Features

```python
# Grup bazlÄ± istatistikler
df['src_ip_count'] = df.groupby('source_ip')['source_ip'].transform('count')
df['dst_port_mean_bytes'] = df.groupby('dst_port')['bytes'].transform('mean')
```

### 3. Rolling Window Features

```python
# Sliding window istatistikleri
df['rolling_mean_bytes'] = df['bytes'].rolling(window=100).mean()
df['rolling_std_bytes'] = df['bytes'].rolling(window=100).std()
```

### 4. Feature Selection

```python
from sklearn.feature_selection import SelectKBest, mutual_info_classif

# Mutual Information
selector = SelectKBest(mutual_info_classif, k=50)
X_selected = selector.fit_transform(X, y)

# Feature importance from Random Forest
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=100)
rf.fit(X, y)
importances = rf.feature_importances_
```

---

## ğŸ“ Veri Seti DosyalarÄ±

```
data/
â”œâ”€â”€ nsl_kdd/
â”‚   â”œâ”€â”€ KDDTrain+.txt
â”‚   â”œâ”€â”€ KDDTest+.txt
â”‚   â””â”€â”€ processed/
â”œâ”€â”€ cicids2017/
â”‚   â”œâ”€â”€ Monday-WorkingHours.pcap_ISCX.csv
â”‚   â”œâ”€â”€ Tuesday-WorkingHours.pcap_ISCX.csv
â”‚   â”œâ”€â”€ Wednesday-workingHours.pcap_ISCX.csv
â”‚   â”œâ”€â”€ Thursday-WorkingHours.pcap_ISCX.csv
â”‚   â”œâ”€â”€ Friday-WorkingHours.pcap_ISCX.csv
â”‚   â””â”€â”€ processed/
â””â”€â”€ bot_iot/
    â”œâ”€â”€ UNSW_2018_IoT_Botnet_Dataset_*.csv
    â””â”€â”€ processed/
```

---

## ğŸ“ Referanslar

- [NSL-KDD Dataset Analysis](https://ieeexplore.ieee.org/document/5356528)
- [CICIDS2017: A Realistic Cyber Defense Dataset](https://www.scitepress.org/Papers/2018/66398/66398.pdf)
- [BoT-IoT: Building Automation Attack Dataset](https://ieeexplore.ieee.org/document/8717639)


---


# DEPLOYMENT

# ğŸš€ Deployment Guide

CyberGuard AI Deployment DokÃ¼mantasyonu

---

## ğŸ“‹ Ä°Ã§indekiler

- [Genel BakÄ±ÅŸ](#genel-bakÄ±ÅŸ)
- [Local Deployment](#local-deployment)
- [Streamlit Cloud](#streamlit-cloud)
- [Docker Deployment](#docker-deployment)
- [AWS Deployment](#aws-deployment)
- [Heroku Deployment](#heroku-deployment)
- [Production Checklist](#production-checklist)
- [Monitoring](#monitoring)
- [Troubleshooting](#troubleshooting)

---

## ğŸŒŸ Genel BakÄ±ÅŸ

CyberGuard AI'Ä± farklÄ± ortamlarda deploy edebilirsiniz:

| Platform | Maliyet | Kolay | Performans | Ã–nerilen |
|----------|---------|-------|------------|----------|
| Local | Ãœcretsiz | â­â­â­â­â­ | Orta | Dev |
| Streamlit Cloud | Ãœcretsiz | â­â­â­â­â­ | Ä°yi | Demo |
| Docker | DÃ¼ÅŸÃ¼k | â­â­â­â­ | Ä°yi | Test |
| AWS | Orta-YÃ¼ksek | â­â­â­ | MÃ¼kemmel | Production |
| Heroku | Orta | â­â­â­â­ | Ä°yi | MVP |

---

## ğŸ’» Local Deployment

### Gereksinimler

- Python 3.10+
- 8GB+ RAM
- 5GB+ disk space

### Kurulum

```bash
# 1. Repository'yi klonla
git clone https://github.com/yourusername/CyberGuard_AI.git
cd CyberGuard_AI

# 2. Virtual environment oluÅŸtur
python -m venv venv

# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate

# 3. Paketleri kur
pip install -r requirements.txt

# 4. .env dosyasÄ± oluÅŸtur
echo "GOOGLE_API_KEY=your_api_key_here" > .env

# 5. Mock veri oluÅŸtur (opsiyonel)
python src/utils/mock_data_generator.py

# 6. Model eÄŸit
python train_model.py

# 7. Ã‡alÄ±ÅŸtÄ±r
cd app
streamlit run main.py
```

### Port YapÄ±landÄ±rmasÄ±

```bash
# FarklÄ± port kullan
streamlit run main.py --server.port 8080

# Network'e aÃ§
streamlit run main.py --server.address 0.0.0.0
```

---

## â˜ï¸ Streamlit Cloud Deployment

### Avantajlar

- âœ… Ãœcretsiz (public apps)
- âœ… Otomatik HTTPS
- âœ… GitHub entegrasyonu
- âœ… Kolay gÃ¼ncelleme

### AdÄ±m 1: GitHub'a Push

```bash
git add .
git commit -m "Ready for deployment"
git push origin main
```

### AdÄ±m 2: Streamlit Cloud'a BaÄŸlan

1. [share.streamlit.io](https://share.streamlit.io) adresine git
2. GitHub ile giriÅŸ yap
3. "New app" tÄ±kla
4. Repository seÃ§: `yourusername/CyberGuard_AI`
5. Main file path: `app/main.py`

### AdÄ±m 3: Secrets Ekle

Dashboard â†’ App settings â†’ Secrets

```toml
# .streamlit/secrets.toml
GOOGLE_API_KEY = "your_api_key_here"
```

### AdÄ±m 4: Deploy

"Deploy!" butonuna tÄ±kla ve bekle (2-5 dakika)

### Config DosyasÄ±

`.streamlit/config.toml` oluÅŸtur:

```toml
[theme]
primaryColor = "#667eea"
backgroundColor = "#0e1117"
secondaryBackgroundColor = "#262730"
textColor = "#fafafa"
font = "sans serif"

[server]
headless = true
port = 8501
enableCORS = false
enableXsrfProtection = true

[browser]
gatherUsageStats = false
```

---

## ğŸ³ Docker Deployment

### Dockerfile

```dockerfile
# Dockerfile
FROM python:3.10-slim

# Ã‡alÄ±ÅŸma dizini
WORKDIR /app

# Sistem paketleri
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

# Python baÄŸÄ±mlÄ±lÄ±klarÄ±
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Uygulama dosyalarÄ±
COPY . .

# Port
EXPOSE 8501

# Health check
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

# BaÅŸlat
ENTRYPOINT ["streamlit", "run", "app/main.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

### docker-compose.yml

```yaml
version: '3.8'

services:
  cyberguard-app:
    build: .
    container_name: cyberguard_ai
    ports:
      - "8501:8501"
    environment:
      - GOOGLE_API_KEY=${GOOGLE_API_KEY}
    volumes:
      - ./cyberguard.db:/app/cyberguard.db
      - ./models:/app/models
    restart: unless-stopped
    networks:
      - cyberguard-network

networks:
  cyberguard-network:
    driver: bridge
```

### .dockerignore

```
venv/
__pycache__/
*.pyc
*.pyo
*.pyd
.Python
.env
.git
.gitignore
.vscode
.idea
*.log
temp_*
test_*
```

### Build & Run

```bash
# Build
docker build -t cyberguard-ai .

# Run
docker run -p 8501:8501 \
  -e GOOGLE_API_KEY=your_key \
  -v $(pwd)/cyberguard.db:/app/cyberguard.db \
  cyberguard-ai

# Docker Compose ile
docker-compose up -d

# LoglarÄ± izle
docker-compose logs -f

# Durdur
docker-compose down
```

---

## â˜ï¸ AWS Deployment

### Architecture

```
Internet â†’ Route 53 â†’ CloudFront â†’ ALB â†’ ECS (Fargate) â†’ RDS
                                           â†“
                                          S3 (models)
```

### 1. EC2 Instance (Basit)

```bash
# 1. EC2 instance oluÅŸtur (t2.medium, Ubuntu 22.04)

# 2. SSH ile baÄŸlan
ssh -i your-key.pem ubuntu@your-ec2-ip

# 3. Kurulum
sudo apt update && sudo apt upgrade -y
sudo apt install python3-pip python3-venv -y

# 4. Uygulama deploy
git clone https://github.com/yourusername/CyberGuard_AI.git
cd CyberGuard_AI
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 5. .env oluÅŸtur
nano .env
# GOOGLE_API_KEY=your_key

# 6. Systemd service oluÅŸtur
sudo nano /etc/systemd/system/cyberguard.service
```

**cyberguard.service:**

```ini
[Unit]
Description=CyberGuard AI
After=network.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/home/ubuntu/CyberGuard_AI
Environment="PATH=/home/ubuntu/CyberGuard_AI/venv/bin"
ExecStart=/home/ubuntu/CyberGuard_AI/venv/bin/streamlit run app/main.py --server.port 8501 --server.address 0.0.0.0
Restart=always

[Install]
WantedBy=multi-user.target
```

```bash
# 7. Servisi baÅŸlat
sudo systemctl daemon-reload
sudo systemctl enable cyberguard
sudo systemctl start cyberguard

# 8. Security group'ta 8501 portunu aÃ§
```

### 2. ECS Fargate (Production)

**task-definition.json:**

```json
{
  "family": "cyberguard-task",
  "containerDefinitions": [
    {
      "name": "cyberguard-container",
      "image": "your-account.dkr.ecr.region.amazonaws.com/cyberguard:latest",
      "portMappings": [
        {
          "containerPort": 8501,
          "protocol": "tcp"
        }
      ],
      "environment": [
        {
          "name": "GOOGLE_API_KEY",
          "value": "your_key_here"
        }
      ],
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/cyberguard",
          "awslogs-region": "us-east-1",
          "awslogs-stream-prefix": "ecs"
        }
      }
    }
  ],
  "requiresCompatibilities": ["FARGATE"],
  "networkMode": "awsvpc",
  "cpu": "1024",
  "memory": "2048"
}
```

**Deploy:**

```bash
# 1. ECR'a push
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin your-account.dkr.ecr.us-east-1.amazonaws.com

docker build -t cyberguard .
docker tag cyberguard:latest your-account.dkr.ecr.us-east-1.amazonaws.com/cyberguard:latest
docker push your-account.dkr.ecr.us-east-1.amazonaws.com/cyberguard:latest

# 2. ECS task oluÅŸtur
aws ecs register-task-definition --cli-input-json file://task-definition.json

# 3. Service oluÅŸtur
aws ecs create-service \
  --cluster cyberguard-cluster \
  --service-name cyberguard-service \
  --task-definition cyberguard-task \
  --desired-count 2 \
  --launch-type FARGATE \
  --network-configuration "awsvpcConfiguration={subnets=[subnet-xxx],securityGroups=[sg-xxx],assignPublicIp=ENABLED}"
```

### 3. S3 + CloudFront (Static Assets)

```bash
# Models ve static dosyalarÄ± S3'e yÃ¼kle
aws s3 cp models/ s3://cyberguard-models/ --recursive

# CloudFront distribution oluÅŸtur
aws cloudfront create-distribution --origin-domain-name cyberguard-models.s3.amazonaws.com
```

---

## ğŸŒ Heroku Deployment

### Procfile

```
web: streamlit run app/main.py --server.port=$PORT --server.address=0.0.0.0
```

### runtime.txt

```
python-3.10.12
```

### Deploy

```bash
# 1. Heroku CLI kur
# https://devcenter.heroku.com/articles/heroku-cli

# 2. Login
heroku login

# 3. App oluÅŸtur
heroku create cyberguard-ai

# 4. Config vars ekle
heroku config:set GOOGLE_API_KEY=your_key_here

# 5. Deploy
git push heroku main

# 6. AÃ§
heroku open

# 7. LoglarÄ± izle
heroku logs --tail
```

### Buildpack (Opsiyonel)

```bash
heroku buildpacks:set heroku/python
```

---

## âœ… Production Checklist

### Security

- [ ] API keys `.env` dosyasÄ±nda
- [ ] `.env` gitignore'da
- [ ] HTTPS kullanÄ±mÄ±
- [ ] Rate limiting
- [ ] Input validation
- [ ] SQL injection korumasÄ±
- [ ] XSS korumasÄ±

### Performance

- [ ] Database indexing
- [ ] Caching (@st.cache_resource)
- [ ] Lazy loading
- [ ] Image optimization
- [ ] Gzip compression
- [ ] CDN kullanÄ±mÄ±

### Monitoring

- [ ] Error logging
- [ ] Performance monitoring
- [ ] Uptime monitoring
- [ ] Alert sistemi
- [ ] Backup stratejisi

### Documentation

- [ ] README.md gÃ¼ncel
- [ ] API dokÃ¼mantasyonu
- [ ] Deployment guide
- [ ] User guide
- [ ] Changelog

---

## ğŸ“Š Monitoring

### Logs

```bash
# Streamlit logs
tail -f ~/.streamlit/logs/*.log

# Docker logs
docker logs -f cyberguard_ai

# AWS CloudWatch
aws logs tail /ecs/cyberguard --follow
```

### Uptime Monitoring

**UptimeRobot** (Ãœcretsiz):

```
https://uptimerobot.com
Monitor Type: HTTP(s)
URL: https://your-app-url.com
```

### Application Monitoring

```python
# src/utils/monitoring.py
import time
from functools import wraps

def monitor_performance(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        duration = time.time() - start
        
        logger.info(f"{func.__name__} took {duration:.2f}s")
        return result
    return wrapper
```

---

## ğŸ”§ Troubleshooting

### Common Issues

#### 1. Port Already in Use

```bash
# Port'u kullanÄ±mdan kaldÄ±r
# Windows
netstat -ano | findstr :8501
taskkill /PID <PID> /F

# Mac/Linux
lsof -ti:8501 | xargs kill -9
```

#### 2. Module Not Found

```bash
# Virtual environment aktif mi?
which python  # venv iÃ§inde olmalÄ±

# Paketleri yeniden kur
pip install -r requirements.txt --force-reinstall
```

#### 3. Database Locked

```python
# Timeout artÄ±r
import sqlite3
conn = sqlite3.connect('cyberguard.db', timeout=30)
```

#### 4. Memory Error

```bash
# Streamlit memory limit artÄ±r
streamlit run app/main.py --server.maxUploadSize=1000
```

#### 5. Streamlit Cloud Secrets

```toml
# .streamlit/secrets.toml oluÅŸtur
# Sonra Streamlit Cloud dashboard'dan ekle
```

---

## ğŸ”„ CI/CD Pipeline (GitHub Actions)

**.github/workflows/deploy.yml:**

```yaml
name: Deploy to Production

on:
  push:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      
      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: '3.10'
      
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
      
      - name: Run tests
        run: |
          pytest tests/
  
  deploy:
    needs: test
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      
      - name: Deploy to Streamlit Cloud
        run: |
          # Streamlit Cloud auto-deploys on push
          echo "Deployed to Streamlit Cloud"
      
      # Ya da Docker
      - name: Build and push Docker
        run: |
          docker build -t cyberguard:${{ github.sha }} .
          docker push your-registry/cyberguard:${{ github.sha }}
```

---

## ğŸ“ˆ Scaling

### Vertical Scaling

```bash
# Daha gÃ¼Ã§lÃ¼ instance
# AWS: t2.medium â†’ t2.xlarge
# Heroku: Standard-1X â†’ Performance-M
```

### Horizontal Scaling

```yaml
# docker-compose.yml
services:
  cyberguard:
    deploy:
      replicas: 3  # 3 instance
    
  nginx:
    image: nginx
    ports:
      - "80:80"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
```

**nginx.conf:**

```nginx
upstream cyberguard {
    server cyberguard_1:8501;
    server cyberguard_2:8501;
    server cyberguard_3:8501;
}

server {
    listen 80;
    
    location / {
        proxy_pass http://cyberguard;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
    }
}
```

---

## ğŸ’° Cost Estimation

### Free Tier

- **Local**: Ãœcretsiz
- **Streamlit Cloud**: Ãœcretsiz (public apps)

### Paid Options

| Platform | Monthly Cost | Specs |
|----------|--------------|-------|
| Heroku Standard | $25-50 | 512MB-1GB RAM |
| AWS EC2 t2.medium | $30-40 | 4GB RAM, 2 vCPU |
| AWS Fargate | $50-100 | 2GB RAM, 1 vCPU |
| DigitalOcean | $12-24 | 2-4GB RAM |

---

## ğŸš¨ Backup & Recovery

### Database Backup

```bash
# Otomatik backup scripti
#!/bin/bash
DATE=$(date +%Y%m%d_%H%M%S)
cp cyberguard.db backups/cyberguard_$DATE.db

# Eski backup'larÄ± sil (30 gÃ¼nden eski)
find backups/ -name "*.db" -mtime +30 -delete
```

### Cron Job

```bash
# GÃ¼nlÃ¼k backup (her gÃ¼n 03:00)
0 3 * * * /path/to/backup.sh
```

### S3'e Yedekleme

```bash
aws s3 sync backups/ s3://cyberguard-backups/
```

---

## ğŸ“ Support

Deployment ile ilgili sorularÄ±nÄ±z iÃ§in:

- ğŸ“§ Email: devops@cyberguardai.com
- ğŸ’¬ Discord: [discord.gg/cyberguardai](https://discord.gg/cyberguardai)
- ğŸ“– Docs: [docs.cyberguardai.com/deployment](https://docs.cyberguardai.com/deployment)

---

[â¬†ï¸ Back to Top](#-deployment-guide)

---


# FAQ

# â“ SÄ±kÃ§a Sorulan Sorular (FAQ)

CyberGuard AI hakkÄ±nda en Ã§ok sorulan sorular ve cevaplarÄ±

---

## ğŸ“‹ Ä°Ã§indekiler

- [Genel Sorular](#genel-sorular)
- [Kurulum](#kurulum)
- [KullanÄ±m](#kullanÄ±m)
- [ML/AI](#mlai)
- [API](#api)
- [GÃ¼venlik](#gÃ¼venlik)
- [Performans](#performans)
- [Lisans ve Destek](#lisans-ve-destek)

---

## ğŸŒŸ Genel Sorular

### CyberGuard AI nedir?

CyberGuard AI, yapay zeka destekli siber gÃ¼venlik platformudur. SSA-LSTMIDS modeli ile aÄŸ trafiÄŸindeki saldÄ±rÄ±larÄ± %99+ doÄŸrulukla tespit eder.

### Hangi saldÄ±rÄ± tÃ¼rlerini tespit edebilir?

- DDoS (Distributed Denial of Service)
- Port Scanning
- Brute Force
- SQL Injection
- XSS (Cross-Site Scripting)
- Malware
- Botnet aktivitesi
- Ve 15+ diÄŸer saldÄ±rÄ± tÃ¼rÃ¼

### Hangi veri setleri Ã¼zerinde eÄŸitildi?

| Dataset | KayÄ±t | Accuracy |
|---------|-------|----------|
| NSL-KDD | 148K | 99.36% |
| CICIDS2017 | 2.8M | 99.88% |
| BoT-IoT | 73M | 99.99% |

### Ãœcretsiz mi?

Community edition Ã¼cretsiz ve aÃ§Ä±k kaynak. Enterprise Ã¶zellikleri iÃ§in lisans gereklidir.

---

## ğŸ”§ Kurulum

### Minimum sistem gereksinimleri neler?

| BileÅŸen | Minimum | Ã–nerilen |
|---------|---------|----------|
| CPU | 4 cores | 8+ cores |
| RAM | 8 GB | 16+ GB |
| Disk | 50 GB SSD | 100+ GB SSD |
| GPU | - | NVIDIA CUDA |

### Hangi iÅŸletim sistemlerinde Ã§alÄ±ÅŸÄ±r?

- Windows 10/11, Windows Server 2019+
- Ubuntu 20.04+, CentOS 8+
- macOS 11+
- Docker (herhangi bir platform)

### Python versiyonu?

Python 3.9+ gereklidir. Python 3.11 Ã¶nerilir.

### Kurulum ne kadar sÃ¼rer?

- Tam kurulum: 10-15 dakika
- Docker: 5 dakika
- Model indirme: 5-10 dakika (opsiyonel)

### Kurulum hatasÄ± alÄ±yorum, ne yapmalÄ±yÄ±m?

1. Python versiyonunu kontrol edin: `python --version`
2. Virtual environment aktif mi: `which python`
3. BaÄŸÄ±mlÄ±lÄ±klarÄ± yeniden yÃ¼kleyin: `pip install -r requirements.txt`
4. DetaylÄ± log: `pip install -r requirements.txt -v`

Bkz: [Troubleshooting](troubleshooting.md)

---

## ğŸ’» KullanÄ±m

### Backend'i nasÄ±l baÅŸlatÄ±rÄ±m?

```bash
cd app
python -m uvicorn main:app --reload
# http://localhost:8000
```

### Frontend'i nasÄ±l baÅŸlatÄ±rÄ±m?

```bash
cd frontend
npm run dev
# http://localhost:5173
```

### API dokÃ¼mantasyonuna nasÄ±l eriÅŸirim?

Backend Ã§alÄ±ÅŸÄ±rken: `http://localhost:8000/api/docs`

### VarsayÄ±lan kullanÄ±cÄ± bilgileri nedir?

```
Username: admin
Password: admin123
```

âš ï¸ Ä°lk giriÅŸte ÅŸifreyi deÄŸiÅŸtirin!

### Dashboard'da veriler neden boÅŸ gÃ¶rÃ¼nÃ¼yor?

1. Database migration Ã§alÄ±ÅŸtÄ±rÄ±n
2. Mock data oluÅŸturun: `python scripts/generate_mock_data.py`
3. API baÄŸlantÄ±sÄ±nÄ± kontrol edin

---

## ğŸ§  ML/AI

### Hangi ML modelleri kullanÄ±lÄ±yor?

| Model | TÃ¼r | Accuracy |
|-------|-----|----------|
| SSA-LSTMIDS | Deep Learning | 99.88% |
| BiLSTM | Deep Learning | 99.12% |
| Random Forest | Ensemble | 97.45% |
| XGBoost | Ensemble | 97.21% |

### Model eÄŸitimi ne kadar sÃ¼rer?

| Dataset | GPU | CPU |
|---------|-----|-----|
| NSL-KDD | 30 min | 2 hours |
| CICIDS2017 | 2 hours | 8 hours |
| BoT-IoT | 4 hours | 16 hours |

### GPU olmadan Ã§alÄ±ÅŸÄ±r mÄ±?

Evet, ama eÄŸitim Ã§ok daha yavaÅŸ olur. Inference CPU'da sorunsuz Ã§alÄ±ÅŸÄ±r.

### Kendi modelimi eÄŸitebilir miyim?

Evet! Bkz: [Model Training Guide](model_training_guide.md)

```python
python scripts/train_custom_model.py --dataset /path/to/data.csv
```

### XAI (AÃ§Ä±klanabilir AI) nedir?

Model kararlarÄ±nÄ± aÃ§Ä±klamak iÃ§in SHAP ve LIME kullanÄ±yoruz. Bu sayede modelin neden belirli bir tahminde bulunduÄŸunu anlayabilirsiniz.

Bkz: [XAI Documentation](xai.md)

---

## ğŸ”Œ API

### KaÃ§ endpoint var?

150+ endpoint mevcut. Bkz: [API Endpoints Full](api_endpoints_full.md)

### Rate limit nedir?

| Plan | Limit |
|------|-------|
| Community | 100 req/dakika |
| Professional | 1000 req/dakika |
| Enterprise | SÄ±nÄ±rsÄ±z |

### API key nasÄ±l oluÅŸtururum?

```bash
# Web UI
Settings â†’ API Keys â†’ Create New Key

# API
POST /api/keys
{"name": "My API Key", "permissions": ["read", "write"]}
```

### Hangi response formatÄ± kullanÄ±lÄ±yor?

JSON formatÄ±nda standart response:

```json
{
  "success": true,
  "data": {...},
  "message": "Ä°ÅŸlem baÅŸarÄ±lÄ±"
}
```

---

## ğŸ” GÃ¼venlik

### Veriler ÅŸifreleniyor mu?

Evet, AES-256 encryption kullanÄ±lÄ±yor. Transit'te TLS 1.3.

### MFA destekleniyor mu?

Evet, TOTP (Google Authenticator vb.) desteklenir.

### GDPR/KVKK uyumlu mu?

TasarÄ±m gereÄŸi uyumlu. KiÅŸisel veri minimum tutulur.

### GÃ¼venlik aÃ§Ä±ÄŸÄ± bulursam ne yapmalÄ±yÄ±m?

LÃ¼tfen `security@cyberguard-ai.com` adresine bildirin. Bkz: [Security Policy](SECURITY_POLICY.md)

---

## âš¡ Performans

### Ne kadar trafik iÅŸleyebilir?

- Single node: 10K req/s
- Cluster: 100K+ req/s

### Bellek kullanÄ±mÄ± ne kadar?

- Backend: 500MB-2GB
- Frontend: 100-300MB
- Model inference: 1-4GB

### YavaÅŸ Ã§alÄ±ÅŸÄ±yor, ne yapmalÄ±yÄ±m?

1. Database indekslerini kontrol edin
2. Redis cache aktif mi?
3. Model warmup yapÄ±n
4. Resource limitlerini artÄ±rÄ±n

Bkz: [Performance Tuning](performance_tuning.md)

---

## ğŸ“œ Lisans ve Destek

### Lisans tÃ¼rÃ¼ nedir?

MIT License - Ticari kullanÄ±ma aÃ§Ä±k.

### Destek nasÄ±l alabilirim?

| Kanal | SÃ¼re |
|-------|------|
| GitHub Issues | 24-48 saat |
| Email | 24-48 saat |
| Discord | CanlÄ± |
| Enterprise | SLA |

### KatkÄ±da bulunabilir miyim?

Evet! Bkz: [Contributing](contributing.md)

---

## ğŸ”— Daha Fazla Kaynak

- [Kurulum Rehberi](installation.md)
- [KullanÄ±cÄ± Rehberi](user_guide.md)
- [API Reference](api_reference.md)
- [Troubleshooting](troubleshooting.md)


---


# FEDERATED_LEARNİNG

# ğŸ”— Federated Learning DokÃ¼mantasyonu

DaÄŸÄ±tÄ±k makine Ã¶ÄŸrenmesi ve gizlilik koruyan eÄŸitim

---

## ğŸ“‹ Ä°Ã§indekiler

- [Genel BakÄ±ÅŸ](#genel-bakÄ±ÅŸ)
- [Mimari](#mimari)
- [API Endpoints](#api-endpoints)
- [Aggregation YÃ¶ntemleri](#aggregation-yÃ¶ntemleri)
- [Gizlilik Ã–zellikleri](#gizlilik-Ã¶zellikleri)

---

## ğŸŒŸ Genel BakÄ±ÅŸ

Federated Learning, verileri merkezi bir sunucuya gÃ¶ndermeden, cihazlar Ã¼zerinde model eÄŸitimi yapÄ±lmasÄ±nÄ± saÄŸlar.

### Avantajlar

- ğŸ”’ **Gizlilik**: Veriler cihazda kalÄ±r
- ğŸŒ **DaÄŸÄ±tÄ±k**: Merkezi sunucu gereksiz
- ğŸ“Š **Ã–lÃ§eklenebilir**: Binlerce client destekler
- âš¡ **Verimli**: Sadece model gÃ¼ncellemeleri iletilir

---

## ğŸ—ï¸ Mimari

```
â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
â”‚  Central Server â”‚
â”‚   (Aggregator)  â”‚
â””â”€â”€â”€â”€â”€â”€â”€â”€â”¬â”€â”€â”€â”€â”€â”€â”€â”€â”˜
         â”‚
    â”Œâ”€â”€â”€â”€â”´â”€â”€â”€â”€â”
    â”‚         â”‚
â”Œâ”€â”€â”€â”´â”€â”€â”€â” â”Œâ”€â”€â”€â”´â”€â”€â”€â”
â”‚Client1â”‚ â”‚Client2â”‚ ... ClientN
â””â”€â”€â”€â”€â”€â”€â”€â”˜ â””â”€â”€â”€â”€â”€â”€â”€â”˜
```

### EÄŸitim DÃ¶ngÃ¼sÃ¼

1. Server global modeli client'lara daÄŸÄ±tÄ±r
2. Her client kendi verileriyle local eÄŸitim yapar
3. Client'lar model gÃ¼ncellemelerini server'a gÃ¶nderir
4. Server gÃ¼ncellemeleri aggregate eder
5. Yeni global model oluÅŸturulur
6. Tekrar 1'den baÅŸla

---

## ğŸ”Œ API Endpoints

### GET /api/federated/status

Federated learning sistem durumu

### POST /api/federated/clients

Yeni client ekle

```json
{
  "name": "Edge Device 1",
  "data_size": 5000,
  "compute_power": "high",
  "location": "TR"
}
```

### POST /api/federated/start

Federated training baÅŸlat

```json
{
  "model_id": "best_cicids2017",
  "num_rounds": 10,
  "min_clients": 3,
  "aggregation_method": "fedavg",
  "differential_privacy": true,
  "epsilon": 1.0
}
```

### GET /api/federated/aggregation

Aggregation metodlarÄ±nÄ± listele

### GET /api/federated/privacy

Gizlilik Ã¶zelliklerini listele

---

## ğŸ”„ Aggregation YÃ¶ntemleri

### 1. FedAvg (Federated Averaging)

- En basit yÃ¶ntem
- TÃ¼m client aÄŸÄ±rlÄ±klarÄ±nÄ±n ortalamasÄ±
- IID data varsayÄ±mÄ±

### 2. FedProx

- Non-IID data iÃ§in optimize
- Proximal term ile stabilite
- Heterojen sistemler iÃ§in uygun

### 3. SCAFFOLD

- Variance reduction
- Daha hÄ±zlÄ± convergence
- Daha yÃ¼ksek communication cost

---

## ğŸ”’ Gizlilik Ã–zellikleri

### Differential Privacy

- Gradientlere noise ekleme
- Îµ (epsilon) parametresi ile kontrol
- Trade-off: privacy vs accuracy

### Secure Aggregation

- Kriptografik aggregation
- Server bile bireysel gÃ¼ncellemeleri gÃ¶remez
- MPC (Multi-Party Computation)

### Homomorphic Encryption

- Åifreli veri Ã¼zerinde hesaplama
- En yÃ¼ksek gÃ¼venlik seviyesi
- YÃ¼ksek computational cost

---

## ğŸ’» KullanÄ±m

### Client Ekleme

```python
response = requests.post("/api/federated/clients", json={
    "name": "Factory Sensor 1",
    "data_size": 10000,
    "compute_power": "medium"
})
client_id = response.json()["data"]["client_id"]
```

### EÄŸitim BaÅŸlatma

```python
response = requests.post("/api/federated/start", json={
    "model_id": "ids_model",
    "num_rounds": 20,
    "min_clients": 5,
    "differential_privacy": True
})

final_accuracy = response.json()["data"]["final_global_accuracy"]
```

---

## ğŸ“ˆ SonuÃ§ Metrikleri

- **Global Accuracy**: Aggregate modelin doÄŸruluÄŸu
- **Client Accuracy**: Her client'Ä±n local doÄŸruluÄŸu
- **Communication Cost**: Iletilen veri miktarÄ±
- **Training Time**: Round baÅŸÄ±na sÃ¼re
- **Privacy Budget**: Harcanan Îµ miktarÄ±

---

## ğŸ“ Referanslar

- [Communication-Efficient Learning](https://arxiv.org/abs/1602.05629)
- [Federated Learning at Scale](https://arxiv.org/abs/1902.01046)
- [Advances in Federated Learning](https://arxiv.org/abs/1912.04977)


---


# GİTHUB_UPLOAD

# ğŸ“¤ GitHub YÃ¼kleme Rehberi

Bu rehber, CyberGuard AI projesini GitHub'a yÃ¼klemek iÃ§in adÄ±m adÄ±m talimatlar iÃ§erir.

---

## âš ï¸ Ã–nemli: BÃ¼yÃ¼k Dosya SorunlarÄ±

GitHub'Ä±n dosya limitleri:

- **Tek dosya:** Maksimum 100MB (sert limit)
- **Toplam repo:** Ã–nerilen < 1GB, maksimum 5GB
- **Push:** Tek push'ta maksimum 2GB

### Projemizdeki Potansiyel BÃ¼yÃ¼k Dosyalar

| Dosya/KlasÃ¶r | Tahmini Boyut | Ã‡Ã¶zÃ¼m |
|--------------|---------------|-------|
| `.venv/` | 500MB+ | âŒ .gitignore'a ekle |
| `node_modules/` | 300MB+ | âŒ .gitignore'a ekle |
| `data/` (datasets) | 100MB-6GB | âš ï¸ Git LFS veya dÄ±ÅŸ link |
| `models/*.h5` | 50-500MB | âš ï¸ Git LFS |
| `__pycache__/` | 10MB+ | âŒ .gitignore'a ekle |
| `.pdf` dosyalar | 6MB+ | âœ… OK |

---

## ğŸ“‹ AdÄ±m AdÄ±m Plan

### AdÄ±m 1: .gitignore KontrolÃ¼

Mevcut `.gitignore` dosyasÄ±nÄ± kontrol et ve eksikleri ekle:

```gitignore
# Python
__pycache__/
*.py[cod]
*.so
.Python
.venv/
venv/
ENV/

# Node
node_modules/
npm-debug.log

# IDE
.idea/
.vscode/
*.swp

# OS
.DS_Store
Thumbs.db

# Env
.env
.env.local

# Data (bÃ¼yÃ¼k dosyalar)
data/raw/
data/CICIDS2017/
*.csv.gz
*.parquet

# Models (opsiyonel - Git LFS kullan)
# models/*.h5
# models/*.keras

# Logs
logs/
*.log

# Uploads
uploads/*
!uploads/.gitkeep

# Reports (generated)
reports/*
!reports/.gitkeep
```

### AdÄ±m 2: BÃ¼yÃ¼k DosyalarÄ± Tespit Et

```bash
# Windows PowerShell - 100MB'dan bÃ¼yÃ¼k dosyalarÄ± bul
Get-ChildItem -Recurse | Where-Object { $_.Length -gt 100MB } | Select-Object FullName, @{Name="SizeMB";Expression={[math]::Round($_.Length/1MB,2)}}
```

### AdÄ±m 3: Git LFS Kurulumu (BÃ¼yÃ¼k Dosyalar Ä°Ã§in)

EÄŸer model dosyalarÄ± (.h5, .keras) veya bÃ¼yÃ¼k veri setleri varsa:

```bash
# Git LFS kurulumu
git lfs install

# BÃ¼yÃ¼k dosya tÃ¼rlerini track et
git lfs track "*.h5"
git lfs track "*.keras"
git lfs track "*.pkl"
git lfs track "data/*.csv"

# .gitattributes dosyasÄ±nÄ± commit et
git add .gitattributes
```

### AdÄ±m 4: Repository OluÅŸturma

1. [github.com/new](https://github.com/new) adresine git
2. Repository bilgileri:
   - **Name:** `CyberGuard-AI`
   - **Description:** `AI-Powered Cyber Security Platform with LSTM-based IDS`
   - **Visibility:** Public veya Private
   - **Initialize:** âŒ (boÅŸ bÄ±rak, README ekleme)

### AdÄ±m 5: Local Git Kurulumu

```bash
# Proje dizinine git
cd c:\Gelistirme\CyberGuard_AI_Antigravity

# Git baÅŸlat (zaten varsa skip)
git init

# Remote ekle
git remote add origin https://github.com/KULLANICI_ADI/CyberGuard-AI.git

# Ana branch'i ayarla
git branch -M main
```

### AdÄ±m 6: Ä°lk Commit

```bash
# TÃ¼m dosyalarÄ± ekle
git add .

# Commit
git commit -m "ğŸš€ Initial commit: CyberGuard AI - Full Platform"

# Push
git push -u origin main
```

---

## ğŸ”§ Sorun Giderme

### Problem: "File too large" hatasÄ±

```bash
# BÃ¼yÃ¼k dosyayÄ± git geÃ§miÅŸinden sil
git filter-branch --force --index-filter \
  "git rm --cached --ignore-unmatch PATH/TO/LARGE/FILE" \
  --prune-empty --tag-name-filter cat -- --all

# Veya BFG Repo-Cleaner kullan (daha hÄ±zlÄ±)
java -jar bfg.jar --strip-blobs-bigger-than 100M
```

### Problem: Push Ã§ok yavaÅŸ

```bash
# Daha kÃ¼Ã§Ã¼k parÃ§alar halinde push
git push --progress
```

### Problem: Git LFS quota aÅŸÄ±ldÄ±

GitHub Free: 1GB storage, 1GB/ay bandwidth

- Ã‡Ã¶zÃ¼m 1: External storage (S3, Google Drive)
- Ã‡Ã¶zÃ¼m 2: GitHub Pro/Team upgrade
- Ã‡Ã¶zÃ¼m 3: Model dosyalarÄ±nÄ± Hugging Face Hub'a yÃ¼kle

---

## ğŸ“ Ã–nerilen Dosya YapÄ±sÄ±

```
CyberGuard-AI/
â”œâ”€â”€ README.md              # âœ… Proje tanÄ±tÄ±mÄ±
â”œâ”€â”€ LICENSE                # âœ… MIT License
â”œâ”€â”€ .gitignore             # âœ… Ignore rules
â”œâ”€â”€ .gitattributes         # âœ… LFS rules (varsa)
â”œâ”€â”€ requirements.txt       # âœ… Python deps
â”œâ”€â”€ package.json           # âœ… Node deps (frontend iÃ§in)
â”‚
â”œâ”€â”€ app/                   # âœ… Backend
â”œâ”€â”€ frontend/              # âœ… Frontend (node_modules hariÃ§)
â”œâ”€â”€ src/                   # âœ… ML models
â”œâ”€â”€ docs/                  # âœ… Documentation
â”œâ”€â”€ tests/                 # âœ… Test files
â”‚
â”œâ”€â”€ data/                  # âš ï¸ Sadece sample data
â”‚   â””â”€â”€ sample/
â”œâ”€â”€ models/                # âš ï¸ Sadece kÃ¼Ã§Ã¼k modeller
â”‚   â””â”€â”€ .gitkeep
â””â”€â”€ notebooks/             # âœ… Jupyter notebooks
```

---

## ğŸš€ HÄ±zlÄ± BaÅŸlangÄ±Ã§ Scripti

AÅŸaÄŸÄ±daki PowerShell scriptini Ã§alÄ±ÅŸtÄ±r:

```powershell
# 1. BÃ¼yÃ¼k dosyalarÄ± kontrol et
Write-Host "=== BÃ¼yÃ¼k Dosyalar (>50MB) ===" -ForegroundColor Yellow
Get-ChildItem -Recurse -File | Where-Object { $_.Length -gt 50MB } | 
    Select-Object @{N='Size(MB)';E={[math]::Round($_.Length/1MB,2)}}, FullName

# 2. Toplam boyut
Write-Host "`n=== Toplam Proje Boyutu ===" -ForegroundColor Yellow
$size = (Get-ChildItem -Recurse | Measure-Object -Property Length -Sum).Sum / 1GB
Write-Host ("Toplam: {0:N2} GB" -f $size)

# 3. HariÃ§ tutulacak klasÃ¶rler
Write-Host "`n=== HariÃ§ Tutulacaklar ===" -ForegroundColor Yellow
@(".venv", "node_modules", "__pycache__", "data/raw") | ForEach-Object {
    if (Test-Path $_) {
        $s = (Get-ChildItem $_ -Recurse | Measure-Object -Property Length -Sum).Sum / 1MB
        Write-Host ("{0}: {1:N0} MB" -f $_, $s)
    }
}
```

---

## âœ… Checklist

- [ ] `.gitignore` gÃ¼ncel mi?
- [ ] BÃ¼yÃ¼k dosyalar (>100MB) tespit edildi mi?
- [ ] Git LFS gerekli mi?
- [ ] `.env` dosyasÄ± .gitignore'da mÄ±?
- [ ] `node_modules/` .gitignore'da mÄ±?
- [ ] `.venv/` .gitignore'da mÄ±?
- [ ] README.md hazÄ±r mÄ±?
- [ ] LICENSE dosyasÄ± var mÄ±?

---

## ğŸ“ Alternatifler

### BÃ¼yÃ¼k Dosyalar Ä°Ã§in

1. **Hugging Face Hub** - ML modelleri iÃ§in ideal
2. **Google Drive** - Datasets iÃ§in link paylaÅŸÄ±mÄ±
3. **AWS S3** - Production iÃ§in
4. **DVC** (Data Version Control) - ML pipelines iÃ§in

### Release Ä°Ã§in

GitHub Releases ile bÃ¼yÃ¼k dosyalarÄ± (100MB'a kadar) yÃ¼kleyebilirsin:

1. GitHub'da Release oluÅŸtur
2. Assets bÃ¶lÃ¼mÃ¼ne dosya yÃ¼kle
3. README'de link ver


---


# GİTHUB_UPLOAD_GUİDE

# ğŸ“¤ CyberGuard AI - GitHub'a YÃ¼kleme Rehberi

Bu rehber, bÃ¼yÃ¼k dosyalarÄ± olan projeyi GitHub'a nasÄ±l yÃ¼kleyeceÄŸinizi aÃ§Ä±klar.

---

## âš ï¸ Ã–nemli: GitHub SÄ±nÄ±rlarÄ±

| SÄ±nÄ±r | DeÄŸer |
| ----- | ----- |
| Tek dosya maksimum | **100 MB** |
| Repo toplam boyut (Ã¶nerilir) | **1 GB** |
| Repo sert limit | **5 GB** |
| Push limit | **2 GB** |

---

## ğŸ“Š Projenizin Durumu

BÃ¼yÃ¼k dosyalarÄ±nÄ±z:

- `src/database/cyberguard.db` - **~5 GB** (Ã§ok bÃ¼yÃ¼k!)
- `models/*.keras` - **~150 MB** toplam
- `data/raw/` - **~500 MB+** CSV dosyalarÄ±

---

## âœ… YÃ¶ntem 1: BÃ¼yÃ¼k DosyalarÄ± HariÃ§ Tut (Ã–nerilen)

`.gitignore` zaten ayarlandÄ±. Åu dosyalar otomatik hariÃ§ tutulacak:

```
âœ“ *.keras       # ML modelleri
âœ“ *.h5          # Eski modeller
âœ“ *.db          # VeritabanlarÄ±
âœ“ data/raw/     # Ham veri setleri
âœ“ .venv/        # Python sanal ortam
âœ“ node_modules/ # Node paketleri
```

### AdÄ±mlar

```bash
# 1. Git'i baÅŸlat (zaten yapÄ±lmÄ±ÅŸsa atla)
git init

# 2. TÃ¼m dosyalarÄ± ekle (.gitignore'a gÃ¶re filtrelenir)
git add .

# 3. Commit yap
git commit -m "Initial commit: CyberGuard AI v3.1"

# 4. Remote ekle
git remote add origin https://github.com/KULLANICI/CyberGuard_AI.git

# 5. Push et
git push -u origin main
```

---

## ğŸ”„ YÃ¶ntem 2: Git LFS (Large File Storage)

EÄŸer modelleri de yÃ¼klemek istiyorsan:

### Kurulum

```bash
# 1. Git LFS yÃ¼kle
# Windows: https://git-lfs.com adresinden indir
# veya
winget install GitHub.GitLFS

# 2. LFS'i aktifleÅŸtir
git lfs install

# 3. BÃ¼yÃ¼k dosya tÃ¼rlerini takip et
git lfs track "*.keras"
git lfs track "*.h5"
git lfs track "*.db"

# 4. .gitattributes'u ekle
git add .gitattributes

# 5. Normal commit ve push
git add .
git commit -m "Add LFS tracking"
git push
```

### LFS Limitleri

- GitHub Free: **1 GB storage**, **1 GB/ay bandwidth**
- GitHub Pro: **2 GB storage**, **2 GB/ay bandwidth**

---

## ğŸ—‚ï¸ YÃ¶ntem 3: AyrÄ± Repo (Modeller iÃ§in)

BÃ¼yÃ¼k dosyalarÄ± ayrÄ± bir repo'da tut:

### Ana Repo (kod)

```
CyberGuard_AI/
â”œâ”€â”€ app/
â”œâ”€â”€ frontend/
â”œâ”€â”€ src/
â”œâ”€â”€ docs/
â””â”€â”€ README.md
```

### Model Repo (bÃ¼yÃ¼k dosyalar)

```
CyberGuard_AI_Models/
â”œâ”€â”€ production/
â”œâ”€â”€ archived/
â””â”€â”€ README.md
```

### KullanÄ±cÄ±lara

```markdown
## Model DosyalarÄ±

EÄŸitilmiÅŸ modeller ayrÄ± repoda:
https://github.com/KULLANICI/CyberGuard_AI_Models

Veya Google Drive:
https://drive.google.com/...
```

---

## ğŸ“¦ YÃ¶ntem 4: Releases ile DaÄŸÄ±tÄ±m

BÃ¼yÃ¼k dosyalarÄ± GitHub Releases'a yÃ¼kle:

```bash
# 1. Modelleri zipple
Compress-Archive -Path models\production\* -DestinationPath models_v3.1.zip

# 2. GitHub CLI ile release oluÅŸtur
gh release create v3.1.0 models_v3.1.zip --title "v3.1 - Models"
```

### Release Limiti

- Tek dosya: **2 GB**
- Toplam: **SÄ±nÄ±rsÄ±z**

---

## ğŸš€ HÄ±zlÄ± BaÅŸlangÄ±Ã§ (Ã–nerilen)

```powershell
# 1. Proje klasÃ¶rÃ¼ne git
cd C:\Gelistirme\CyberGuard_AI_Antigravity

# 2. Git durumunu kontrol et
git status

# 3. Yeni deÄŸiÅŸiklikleri ekle
git add .

# 4. Commit yap
git commit -m "v3.1.0: Globe3D ML integration, tests, docs update"

# 5. Push et
git push origin main
```

---

## ğŸ” YÃ¼kleme Ã–ncesi Kontrol

```powershell
# Repo boyutunu kontrol et
git count-objects -vH

# BÃ¼yÃ¼k dosyalarÄ± bul
git rev-list --objects --all | git cat-file --batch-check='%(objectname) %(objectsize) %(rest)' | sort -k2 -n -r | head -20
```

---

## â“ SÄ±k Sorunlar

### "File too large" hatasÄ±

```bash
# DosyayÄ± git history'den temizle
git filter-branch --force --index-filter "git rm --cached --ignore-unmatch DOSYA_ADI" --prune-empty --tag-name-filter cat -- --all

# Daha modern yÃ¶ntem (BFG Repo Cleaner)
bfg --strip-blobs-bigger-than 100M
```

### Push Ã§ok yavaÅŸ

- `.gitignore` kontrol et
- `git lfs` kullan
- Push'u parÃ§ala: `git push origin main --force`

---

## ğŸ“‹ Checklist

YÃ¼klemeden Ã¶nce:

- [ ] `.gitignore` gÃ¼ncel
- [ ] `data/raw/` hariÃ§ tutuldu
- [ ] `src/database/cyberguard.db` hariÃ§ tutuldu
- [ ] `models/*.keras` hariÃ§ tutuldu (veya LFS)
- [ ] `.venv/` hariÃ§ tutuldu
- [ ] `node_modules/` hariÃ§ tutuldu
- [ ] `.env` hariÃ§ tutuldu (gÃ¼venlik!)

---

**Åimdi hazÄ±rsÄ±n! ğŸš€**


---


# GLOSSARY

# ğŸ“š Terimler SÃ¶zlÃ¼ÄŸÃ¼ (Glossary)

CyberGuard AI'da kullanÄ±lan terimler ve aÃ§Ä±klamalarÄ±

---

## A

### Accuracy

Model tahminlerinin doÄŸru olma oranÄ±. `(TP + TN) / (TP + TN + FP + FN)`

### Adversarial Attack

ML modellerini kandÄ±rmak iÃ§in tasarlanmÄ±ÅŸ manipÃ¼le edilmiÅŸ girdiler.

### AES

Advanced Encryption Standard. Simetrik ÅŸifreleme algoritmasÄ±.

### API (Application Programming Interface)

YazÄ±lÄ±mlar arasÄ± iletiÅŸim protokolÃ¼.

### AUC-ROC

Area Under ROC Curve. Model performans metriÄŸi.

### AutoML

Automated Machine Learning. Otomatik model seÃ§imi ve hiperparametre optimizasyonu.

---

## B

### Batch Size

Model eÄŸitiminde bir iterasyonda iÅŸlenen Ã¶rnek sayÄ±sÄ±.

### BiLSTM

Bidirectional LSTM. Ä°ki yÃ¶nlÃ¼ LSTM aÄŸÄ±.

### Botnet

SaldÄ±rgan kontrolÃ¼ndeki zombi bilgisayar aÄŸÄ±.

### Brute Force

TÃ¼m olasÄ± kombinasyonlarÄ± deneyerek ÅŸifre kÄ±rma yÃ¶ntemi.

---

## C

### C&W Attack

Carlini & Wagner. GÃ¼Ã§lÃ¼ adversarial saldÄ±rÄ± yÃ¶ntemi.

### CICIDS2017

Canadian Institute for Cybersecurity Intrusion Detection Dataset.

### CNN

Convolutional Neural Network. GÃ¶rÃ¼ntÃ¼ iÅŸlemede kullanÄ±lan derin Ã¶ÄŸrenme modeli.

### CORS

Cross-Origin Resource Sharing. Web gÃ¼venlik mekanizmasÄ±.

### Cross-Validation

Model performansÄ±nÄ± deÄŸerlendirmek iÃ§in veriyi bÃ¶lÃ¼mlere ayÄ±rma.

### CVE

Common Vulnerabilities and Exposures. GÃ¼venlik aÃ§Ä±klarÄ± veritabanÄ±.

### CVSS

Common Vulnerability Scoring System. Zafiyet derecelendirme sistemi (0-10).

---

## D

### DDoS

Distributed Denial of Service. DaÄŸÄ±tÄ±k hizmet engelleme saldÄ±rÄ±sÄ±.

### Deep Learning

Derin Ã¶ÄŸrenme. Ã‡ok katmanlÄ± sinir aÄŸlarÄ±.

### Differential Privacy

Gizlilik koruyan veri analizi tekniÄŸi.

### DoS

Denial of Service. Hizmet engelleme saldÄ±rÄ±sÄ±.

### Dropout

Overfitting'i Ã¶nlemek iÃ§in rastgele nÃ¶ronlarÄ± devre dÄ±ÅŸÄ± bÄ±rakma.

### Drift Detection

Model performansÄ±nÄ±n zamanla dÃ¼ÅŸÃ¼ÅŸÃ¼nÃ¼ tespit etme.

---

## E

### Epoch

TÃ¼m eÄŸitim verisinin bir kez iÅŸlenmesi.

### Ensemble

Birden fazla modelin birleÅŸtirilmesi.

### Epsilon (Îµ)

Adversarial saldÄ±rÄ±larda perturbation miktarÄ±.

---

## F

### F1-Score

Precision ve Recall'Ä±n harmonik ortalamasÄ±.

### False Negative (FN)

YanlÄ±ÅŸlÄ±kla normal olarak sÄ±nÄ±flandÄ±rÄ±lan saldÄ±rÄ±.

### False Positive (FP)

YanlÄ±ÅŸlÄ±kla saldÄ±rÄ± olarak sÄ±nÄ±flandÄ±rÄ±lan normal trafik.

### Feature

Model girdisi olarak kullanÄ±lan Ã¶zellik.

### Feature Engineering

Ham veriden anlamlÄ± Ã¶zellikler Ã§Ä±karma.

### Federated Learning

Veriyi merkeze toplamadan daÄŸÄ±tÄ±k model eÄŸitimi.

### FGSM

Fast Gradient Sign Method. HÄ±zlÄ± adversarial saldÄ±rÄ±.

---

## G

### GDPR

General Data Protection Regulation. AB veri koruma yasasÄ±.

### Gradient

KayÄ±p fonksiyonunun parametrelere gÃ¶re tÃ¼revi.

### GRU

Gated Recurrent Unit. LSTM'e alternatif RNN hÃ¼cresi.

---

## H

### Hiperparametre

Model eÄŸitimi Ã¶ncesi belirlenen parametreler (learning rate, epochs, vb.)

### Honeypot

SaldÄ±rganlarÄ± tespit etmek iÃ§in kurulan sahte sistemler.

### HSM

Hardware Security Module. Kriptografik iÅŸlemler iÃ§in gÃ¼venli donanÄ±m.

---

## I

### IDS

Intrusion Detection System. SaldÄ±rÄ± tespit sistemi.

### IoC

Indicators of Compromise. SaldÄ±rÄ± gÃ¶stergeleri.

### IPS

Intrusion Prevention System. SaldÄ±rÄ± Ã¶nleme sistemi.

---

## J

### JWT

JSON Web Token. Kimlik doÄŸrulama tokenÄ±.

### JSMA

Jacobian-based Saliency Map Attack.

---

## K

### Keras

TensorFlow Ã¼zerine kurulu yÃ¼ksek seviye deep learning kÃ¼tÃ¼phanesi.

### KVKK

KiÅŸisel Verilerin KorunmasÄ± Kanunu.

---

## L

### L2 Distance

Euclidean mesafe. VektÃ¶rler arasÄ± uzaklÄ±k Ã¶lÃ§Ã¼mÃ¼.

### Learning Rate

Model aÄŸÄ±rlÄ±klarÄ±nÄ±n gÃ¼ncelleme hÄ±zÄ±.

### LIME

Local Interpretable Model-agnostic Explanations. XAI yÃ¶ntemi.

### LSTM

Long Short-Term Memory. Uzun vadeli baÄŸÄ±mlÄ±lÄ±klarÄ± Ã¶ÄŸrenebilen RNN.

---

## M

### Malware

ZararlÄ± yazÄ±lÄ±m.

### MFA

Multi-Factor Authentication. Ã‡ok faktÃ¶rlÃ¼ kimlik doÄŸrulama.

### MITRE ATT&CK

SaldÄ±rÄ± taktik ve tekniklerinin framework'Ã¼.

---

## N

### NAS

Neural Architecture Search. Otomatik model mimarisi keÅŸfi.

### NLP

Natural Language Processing. DoÄŸal dil iÅŸleme.

### NSL-KDD

Network Security Laboratory KDD Dataset.

---

## O

### One-Hot Encoding

Kategorik deÄŸiÅŸkenleri binary vektÃ¶rlere dÃ¶nÃ¼ÅŸtÃ¼rme.

### Overfitting

Modelin eÄŸitim verisine aÅŸÄ±rÄ± uyum saÄŸlamasÄ±.

---

## P

### PCAP

Packet Capture. AÄŸ paketlerini kaydetme formatÄ±.

### PGD

Projected Gradient Descent. Ä°teratif adversarial saldÄ±rÄ±.

### Port Scanning

AÃ§Ä±k portlarÄ± tespit etmek iÃ§in aÄŸ taramasÄ±.

### Precision

TP / (TP + FP). Pozitif tahminlerin doÄŸruluÄŸu.

---

## R

### R2L

Remote to Local. Uzaktan yerel eriÅŸim saldÄ±rÄ±sÄ±.

### Random Forest

Karar aÄŸacÄ± ensemble yÃ¶ntemi.

### Recall

TP / (TP + FN). GerÃ§ek pozitiflerin bulunma oranÄ±.

### Reinforcement Learning

Ã–dÃ¼l/ceza ile Ã¶ÄŸrenme paradigmasÄ±.

### REST API

REpresentational State Transfer. Web API mimarisi.

### RNN

Recurrent Neural Network. Tekrarlayan sinir aÄŸÄ±.

### Robustness

Modelin saldÄ±rÄ±lara dayanÄ±klÄ±lÄ±ÄŸÄ±.

---

## S

### Scaler

Verileri normalize eden dÃ¶nÃ¼ÅŸtÃ¼rÃ¼cÃ¼.

### SHAP

SHapley Additive exPlanations. XAI yÃ¶ntemi.

### SIEM

Security Information and Event Management.

### SMOTE

Synthetic Minority Over-sampling Technique.

### SOC

Security Operations Center.

### SQL Injection

SQL komutlarÄ± enjekte ederek veritabanÄ± saldÄ±rÄ±sÄ±.

### SSA

Sparrow Search Algorithm. Metaheuristik optimizasyon.

### STIX/TAXII

Threat intelligence paylaÅŸÄ±m standartlarÄ±.

---

## T

### TensorFlow

Google'Ä±n aÃ§Ä±k kaynak ML framework'Ã¼.

### Threat Intelligence

Tehdit istihbaratÄ±.

### TLS

Transport Layer Security. GÃ¼venli iletiÅŸim protokolÃ¼.

### Token

Kimlik doÄŸrulama jetonu.

### Transformer

Self-attention mekanizmasÄ± kullanan model mimarisi.

### True Negative (TN)

DoÄŸru ÅŸekilde normal olarak sÄ±nÄ±flandÄ±rÄ±lan trafik.

### True Positive (TP)

DoÄŸru ÅŸekilde saldÄ±rÄ± olarak sÄ±nÄ±flandÄ±rÄ±lan trafik.

---

## U

### U2R

User to Root. Yetki yÃ¼kseltme saldÄ±rÄ±sÄ±.

### Underfitting

Modelin veriyi yeterince Ã¶ÄŸrenememesi.

---

## V

### Validation Set

Model hiperparametrelerini ayarlamak iÃ§in kullanÄ±lan veri.

### Vectorization

Metin/kategorik veriyi sayÄ±sal vektÃ¶rlere dÃ¶nÃ¼ÅŸtÃ¼rme.

### Vulnerability

GÃ¼venlik aÃ§Ä±ÄŸÄ±.

---

## W

### WebSocket

Ã‡ift yÃ¶nlÃ¼ gerÃ§ek zamanlÄ± iletiÅŸim protokolÃ¼.

---

## X

### XAI

Explainable AI. AÃ§Ä±klanabilir yapay zeka.

### XGBoost

eXtreme Gradient Boosting. Gradient boosting algoritmasÄ±.

### XSS

Cross-Site Scripting. Web saldÄ±rÄ± tÃ¼rÃ¼.

---

## Z

### Zero-Day

HenÃ¼z yamasÄ± olmayan gÃ¼venlik aÃ§Ä±ÄŸÄ±.

### Zero Trust

GÃ¼venli aÄŸ mimarisi yaklaÅŸÄ±mÄ±.

### ZTNA

Zero Trust Network Access.


---


# İNSTALLATİON

# ğŸš€ Installation Guide

CyberGuard AI kurulum ve yapÄ±landÄ±rma rehberi

---

## ğŸ“‹ Ä°Ã§indekiler

- [Gereksinimler](#gereksinimler)
- [HÄ±zlÄ± Kurulum](#hÄ±zlÄ±-kurulum)
- [Manuel Kurulum](#manuel-kurulum)
- [Docker ile Kurulum](#docker-ile-kurulum)
- [KonfigÃ¼rasyon](#konfigÃ¼rasyon)
- [DoÄŸrulama](#doÄŸrulama)
- [Sorun Giderme](#sorun-giderme)

---

## ğŸ’» Gereksinimler

### Sistem Gereksinimleri

| BileÅŸen | Minimum | Ã–nerilen |
|---------|---------|----------|
| **CPU** | 4 cores | 8+ cores |
| **RAM** | 8 GB | 16+ GB |
| **Disk** | 50 GB SSD | 100+ GB SSD |
| **GPU** | - | NVIDIA (CUDA 11+) |
| **OS** | Windows 10, Ubuntu 20.04, macOS 11 | Ubuntu 22.04 |

### YazÄ±lÄ±m Gereksinimleri

| YazÄ±lÄ±m | Min Versiyon | Ä°ndirme |
|---------|--------------|---------|
| **Python** | 3.9+ | [python.org](https://python.org) |
| **Node.js** | 18+ | [nodejs.org](https://nodejs.org) |
| **Git** | 2.30+ | [git-scm.com](https://git-scm.com) |
| **PostgreSQL** | 14+ | [postgresql.org](https://postgresql.org) |

---

## âš¡ HÄ±zlÄ± Kurulum

### Windows (PowerShell)

```powershell
# 1. Repository'yi klonla
git clone https://github.com/salihoglueyup/CyberGuard_AI.git
cd CyberGuard_AI

# 2. Otomatik kurulum scripti
.\scripts\install.ps1

# 3. Servisleri baÅŸlat
.\start-servers.bat
```

### Linux/macOS (Bash)

```bash
# 1. Repository'yi klonla
git clone https://github.com/salihoglueyup/CyberGuard_AI.git
cd CyberGuard_AI

# 2. Otomatik kurulum scripti
chmod +x scripts/install.sh
./scripts/install.sh

# 3. Servisleri baÅŸlat
./start-servers.sh
```

---

## ğŸ”§ Manuel Kurulum

### AdÄ±m 1: Repository'yi Klonla

```bash
git clone https://github.com/salihoglueyup/CyberGuard_AI.git
cd CyberGuard_AI
```

### AdÄ±m 2: Python Virtual Environment

```bash
# Virtual environment oluÅŸtur
python -m venv venv

# Aktive et
# Windows:
.\venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate
```

### AdÄ±m 3: Python BaÄŸÄ±mlÄ±lÄ±klarÄ±

```bash
# Temel baÄŸÄ±mlÄ±lÄ±klar
pip install --upgrade pip
pip install -r requirements.txt

# GPU desteÄŸi iÃ§in (opsiyonel)
pip install tensorflow-gpu==2.15.0
```

### AdÄ±m 4: Frontend BaÄŸÄ±mlÄ±lÄ±klarÄ±

```bash
cd frontend
npm install
cd ..
```

### AdÄ±m 5: Environment Variables

```bash
# .env dosyasÄ± oluÅŸtur
cp .env.example .env

# DÃ¼zenle
nano .env  # veya herhangi bir editor
```

**.env dosyasÄ±:**

```env
# API Keys
GOOGLE_API_KEY=your_gemini_api_key_here

# Database
DATABASE_URL=postgresql://user:password@localhost:5432/cyberguard
REDIS_URL=redis://localhost:6379

# Security
SECRET_KEY=your_secret_key_here
JWT_SECRET=your_jwt_secret_here

# Server
HOST=localhost
PORT=8000
FRONTEND_PORT=5173
DEBUG=True
```

### AdÄ±m 6: VeritabanÄ± Kurulumu

```bash
# PostgreSQL'e baÄŸlan
psql -U postgres

# Database oluÅŸtur
CREATE DATABASE cyberguard;
\q

# Migration Ã§alÄ±ÅŸtÄ±r
python -m alembic upgrade head
```

### AdÄ±m 7: Model Ä°ndirme (Opsiyonel)

```bash
# Pre-trained modelleri indir
python scripts/download_models.py

# veya manuel
gdown https://drive.google.com/uc?id=YOUR_MODEL_ID -O models/production/
```

---

## ğŸ³ Docker ile Kurulum

### Docker Compose (Ã–nerilen)

```bash
# Docker Compose ile baÅŸlat
docker-compose up -d

# LoglarÄ± gÃ¶rÃ¼ntÃ¼le
docker-compose logs -f
```

### docker-compose.yml

```yaml
version: '3.8'

services:
  backend:
    build: ./app
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=postgresql://postgres:password@db:5432/cyberguard
      - REDIS_URL=redis://redis:6379
    depends_on:
      - db
      - redis
    volumes:
      - ./models:/app/models
      - ./data:/app/data

  frontend:
    build: ./frontend
    ports:
      - "5173:5173"
    depends_on:
      - backend

  db:
    image: postgres:14
    environment:
      - POSTGRES_DB=cyberguard
      - POSTGRES_USER=postgres
      - POSTGRES_PASSWORD=password
    volumes:
      - pgdata:/var/lib/postgresql/data

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"

volumes:
  pgdata:
```

### Tek Container

```bash
# Backend
docker build -t cyberguard-backend ./app
docker run -p 8000:8000 cyberguard-backend

# Frontend
docker build -t cyberguard-frontend ./frontend
docker run -p 5173:5173 cyberguard-frontend
```

---

## âš™ï¸ KonfigÃ¼rasyon

### config/config.yaml

```yaml
# Genel ayarlar
general:
  project_name: "CyberGuard AI"
  version: "2.0.0"
  environment: "development"  # development, staging, production
  debug: true
  language: "tr"
  timezone: "Europe/Istanbul"

# VeritabanÄ±
database:
  type: "postgresql"
  host: "localhost"
  port: 5432
  name: "cyberguard"
  user: "postgres"
  password: "${DB_PASSWORD}"
  pool_size: 20

# Redis
redis:
  host: "localhost"
  port: 6379
  db: 0
  password: null

# ML Modeller
models:
  path: "./models"
  default_model: "best_cicids2017"
  auto_load: true
  gpu_memory_limit: 0.5  # GPU bellek limiti (0-1)

# API
api:
  host: "0.0.0.0"
  port: 8000
  workers: 4
  rate_limit: 100  # requests per minute
  cors_origins:
    - "http://localhost:5173"
    - "http://localhost:3000"

# Gemini AI
gemini:
  api_key: "${GOOGLE_API_KEY}"
  model: "gemini-pro"
  max_tokens: 8192
  temperature: 0.7

# Logging
logging:
  level: "INFO"  # DEBUG, INFO, WARNING, ERROR
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
  file: "./logs/app.log"
  max_size: "100MB"
  backup_count: 5

# GÃ¼venlik
security:
  secret_key: "${SECRET_KEY}"
  jwt_algorithm: "HS256"
  jwt_expiry: 3600  # seconds
  password_min_length: 8
  mfa_enabled: false
  rate_limit_enabled: true
```

---

## âœ… DoÄŸrulama

### Backend Test

```bash
# Backend'i baÅŸlat
cd app
python -m uvicorn main:app --reload

# SaÄŸlÄ±k kontrolÃ¼
curl http://localhost:8000/
# Beklenen: {"message": "ğŸ›¡ï¸ CyberGuard AI API", "version": "2.0.0", ...}

# API Docs
# TarayÄ±cÄ±da aÃ§: http://localhost:8000/api/docs
```

### Frontend Test

```bash
# Frontend'i baÅŸlat
cd frontend
npm run dev

# TarayÄ±cÄ±da aÃ§: http://localhost:5173
```

### Model Test

```python
# Python test
from src.models.predictor import AttackPredictor

predictor = AttackPredictor()
predictor.load_models()
print("Models loaded successfully!")
```

### Tam Sistem Testi

```bash
# Test suite Ã§alÄ±ÅŸtÄ±r
pytest tests/ -v

# Coverage raporu
pytest tests/ --cov=app --cov-report=html
```

---

## ğŸ”¥ Sorun Giderme

### YaygÄ±n Hatalar

#### 1. ModuleNotFoundError

```bash
# Ã‡Ã¶zÃ¼m: Virtual environment aktif deÄŸil
source venv/bin/activate  # Linux
.\venv\Scripts\activate   # Windows
pip install -r requirements.txt
```

#### 2. Port Zaten KullanÄ±mda

```bash
# Port'u kullanan iÅŸlemi bul
# Windows:
netstat -ano | findstr :8000
taskkill /PID <PID> /F

# Linux:
lsof -i :8000
kill -9 <PID>
```

#### 3. CUDA/GPU HatasÄ±

```bash
# GPU olmadan Ã§alÄ±ÅŸtÄ±r
CUDA_VISIBLE_DEVICES="" python app/main.py

# veya config'de
TF_FORCE_GPU_ALLOW_GROWTH=true
```

#### 4. Database BaÄŸlantÄ± HatasÄ±

```bash
# PostgreSQL Ã§alÄ±ÅŸÄ±yor mu?
# Windows:
pg_isready

# Linux:
sudo systemctl status postgresql

# BaÄŸlantÄ± testi
psql -U postgres -h localhost -d cyberguard
```

#### 5. npm install HatasÄ±

```bash
# Cache temizle
npm cache clean --force
rm -rf node_modules package-lock.json
npm install
```

### Log DosyalarÄ±

```
logs/
â”œâ”€â”€ app.log          # Uygulama loglarÄ±
â”œâ”€â”€ error.log        # Hata loglarÄ±
â”œâ”€â”€ access.log       # EriÅŸim loglarÄ±
â””â”€â”€ model.log        # Model loglarÄ±
```

---

## ğŸš€ Sonraki AdÄ±mlar

1. **BaÅŸlangÄ±Ã§ KÄ±lavuzu**: [User Guide](user_guide.md)
2. **API DokÃ¼mantasyonu**: [API Reference](api_reference.md)
3. **Model EÄŸitimi**: [ML Models](ml_models.md)
4. **Deployment**: [Deployment Guide](deployment.md)

---

## ğŸ“ Destek

SorunlarÄ±nÄ±z iÃ§in:

- ğŸ“– [Documentation](https://docs.cyberguard-ai.com)
- ğŸ› [GitHub Issues](https://github.com/salihoglueyup/CyberGuard_AI/issues)
- ğŸ’¬ [Discord](https://discord.gg/cyberguard)


---


# KULLANIM_KILAVUZU

# ğŸ›¡ï¸ CyberGuard AI - TÃ¼rkÃ§e KullanÄ±m KÄ±lavuzu

> **Versiyon:** 2.0  
> **GÃ¼ncelleme:** Ocak 2026  
> **Platform:** Windows / Linux / macOS

---

## ğŸ“‹ Ä°Ã§indekiler

1. [BaÅŸlarken](#-baÅŸlarken)
2. [Sistem Gereksinimleri](#-sistem-gereksinimleri)
3. [Kurulum](#-kurulum)
4. [ModÃ¼l AÃ§Ä±klamalarÄ±](#-modÃ¼l-aÃ§Ä±klamalarÄ±)
5. [KullanÄ±m SenaryolarÄ±](#-kullanÄ±m-senaryolarÄ±)
6. [SÄ±k Sorulan Sorular](#-sÄ±k-sorulan-sorular)

---

## ğŸš€ BaÅŸlarken

CyberGuard AI, yapay zeka destekli bir siber gÃ¼venlik platformudur. AÄŸ trafiÄŸini izler, tehditleri tespit eder ve otomatik yanÄ±t mekanizmalarÄ± sunar.

### HÄ±zlÄ± BaÅŸlangÄ±Ã§

```bash
# 1. Backend'i baÅŸlat
cd app
python -m uvicorn main:app --reload --host 0.0.0.0 --port 8000

# 2. Frontend'i baÅŸlat (yeni terminal)
cd frontend
npm run dev
```

**EriÅŸim Adresleri:**

- ğŸ–¥ï¸ Frontend: <http://localhost:5173>
- ğŸ”Œ Backend API: <http://localhost:8000>
- ğŸ“š API Docs: <http://localhost:8000/api/docs>

---

## ğŸ’» Sistem Gereksinimleri

| BileÅŸen  | Minimum    | Ã–nerilen    |
| -------- | ---------- | ----------- |
| RAM      | 4 GB       | 8+ GB       |
| CPU      | 2 Ã§ekirdek | 4+ Ã§ekirdek |
| Disk     | 5 GB       | 20+ GB      |
| Python   | 3.9+       | 3.11+       |
| Node.js  | 18+        | 20+         |

### Gerekli YazÄ±lÄ±mlar

- Python 3.9+
- Node.js 18+
- Git
- (Ä°steÄŸe baÄŸlÄ±) Docker Desktop

---

## ğŸ“¦ Kurulum

### 1. Projeyi Ä°ndirin

```bash
git clone https://github.com/your-repo/CyberGuard_AI.git
cd CyberGuard_AI
```

### 2. Python BaÄŸÄ±mlÄ±lÄ±klarÄ±nÄ± YÃ¼kleyin

```bash
pip install -r requirements.txt
```

### 3. Frontend BaÄŸÄ±mlÄ±lÄ±klarÄ±nÄ± YÃ¼kleyin

```bash
cd frontend
npm install
```

### 4. Ortam DeÄŸiÅŸkenlerini AyarlayÄ±n

`.env` dosyasÄ± oluÅŸturun:

```env
# API AnahtarlarÄ± (opsiyonel)
GROQ_API_KEY=your_groq_key
VIRUSTOTAL_API_KEY=your_vt_key
OPENAI_API_KEY=your_openai_key

# VeritabanÄ±
DATABASE_URL=sqlite:///./cyberguard.db
```

---

## ğŸ“Š ModÃ¼l AÃ§Ä±klamalarÄ±

### ğŸ  Dashboard (Ana Sayfa)

**AmaÃ§:** Genel gÃ¼venlik durumunu tek bakÄ±ÅŸta gÃ¶rme

**Ã–zellikler:**

- CanlÄ± tehdit sayÄ±sÄ±
- Son 24 saat istatistikleri
- Sistem durumu gÃ¶stergeleri
- HÄ±zlÄ± eriÅŸim kÄ±sayollarÄ±

**NasÄ±l KullanÄ±lÄ±r:**

1. <http://localhost:5173> adresine gidin
2. Dashboard otomatik olarak yÃ¼klenir
3. Ä°statistikler gerÃ§ek zamanlÄ± gÃ¼ncellenir

---

### ğŸŒ Attack Map (SaldÄ±rÄ± HaritasÄ±)

**AmaÃ§:** DÃ¼nya genelindeki saldÄ±rÄ±larÄ± gÃ¶rselleÅŸtirme

**Ã–zellikler:**

- 2D/3D harita gÃ¶rÃ¼nÃ¼mÃ¼
- GerÃ§ek zamanlÄ± saldÄ±rÄ± akÄ±ÅŸÄ±
- Ãœlke bazlÄ± istatistikler
- Tehdit seviyesi renk kodlamasÄ±

**NasÄ±l KullanÄ±lÄ±r:**

1. Sol menÃ¼den "SaldÄ±rÄ± HaritasÄ±" seÃ§in
2. SaÄŸ Ã¼stten 2D/3D moduna geÃ§in
3. Ãœlkelere tÄ±klayarak detay gÃ¶rÃ¼n
4. "CanlÄ± GÃ¼ncelle" ile gerÃ§ek zamanlÄ± izleyin

---

### ğŸ” Malware Scanner (ZararlÄ± TarayÄ±cÄ±)

**AmaÃ§:** DosyalarÄ± zararlÄ± yazÄ±lÄ±mlara karÅŸÄ± tarama

**Ã–zellikler:**

- Dosya yÃ¼kleme ve tarama
- Hash tabanlÄ± analiz
- VirusTotal entegrasyonu
- Statik analiz sonuÃ§larÄ±

**NasÄ±l KullanÄ±lÄ±r:**

1. "TarayÄ±cÄ±" sayfasÄ±na gidin
2. DosyayÄ± sÃ¼rÃ¼kle-bÄ±rak veya seÃ§
3. "Tara" butonuna tÄ±klayÄ±n
4. SonuÃ§larÄ± inceleyin

---

### ğŸŒ Network Monitor (AÄŸ Ä°zleme)

**AmaÃ§:** AÄŸ trafiÄŸini gerÃ§ek zamanlÄ± izleme

**Ã–zellikler:**

- Aktif baÄŸlantÄ±lar listesi
- Bandwidth kullanÄ±mÄ±
- Interface detaylarÄ±
- Anomali tespiti

**NasÄ±l KullanÄ±lÄ±r:**

1. "AÄŸ" menÃ¼sÃ¼ne gidin
2. Aktif interface'leri gÃ¶rÃ¼n
3. Ä°ndirme/yÃ¼kleme hÄ±zlarÄ±nÄ± izleyin
4. ÅÃ¼pheli baÄŸlantÄ±larÄ± filtreleyin

---

### ğŸ¤– AI Assistant (Yapay Zeka Asistan)

**AmaÃ§:** GÃ¼venlik sorularÄ±na AI destekli yanÄ±t

**Ã–zellikler:**

- DoÄŸal dil iÅŸleme
- GÃ¼venlik Ã¶nerileri
- Log analizi
- Tehdit aÃ§Ä±klamalarÄ±

**NasÄ±l KullanÄ±lÄ±r:**

1. "AI Asistan" sayfasÄ±na gidin
2. Sorunuzu yazÄ±n (Ã¶rn: "Bu IP zararlÄ± mÄ±?")
3. Enter tuÅŸuna basÄ±n
4. AI yanÄ±tÄ±nÄ± okuyun

**Ã–rnek Sorular:**

- "192.168.1.100 IP adresi hakkÄ±nda bilgi ver"
- "DDoS saldÄ±rÄ±sÄ±na karÅŸÄ± ne yapmalÄ±yÄ±m?"
- "Log dosyasÄ±ndaki bu hatayÄ± aÃ§Ä±kla"

---

### ğŸ“Š ML Models (Makine Ã–ÄŸrenimi)

**AmaÃ§:** Tehdit tespiti iÃ§in ML modellerini yÃ¶netme

**Ã–zellikler:**

- Model eÄŸitimi
- Performans metrikleri
- Model karÅŸÄ±laÅŸtÄ±rma
- Tahmin yapma

**NasÄ±l KullanÄ±lÄ±r:**

1. "ML Modeller" sayfasÄ±na gidin
2. Mevcut modelleri inceleyin
3. "EÄŸit" ile yeni model oluÅŸturun
4. "Test Et" ile performans Ã¶lÃ§Ã¼n

---

### ğŸ¯ Threat Hunting (Tehdit AvcÄ±lÄ±ÄŸÄ±)

**AmaÃ§:** Proaktif tehdit araÅŸtÄ±rmasÄ±

**Ã–zellikler:**

- Sorgu tabanlÄ± arama
- HazÄ±r ÅŸablonlar
- IOC arama
- SoruÅŸturma yÃ¶netimi

**NasÄ±l KullanÄ±lÄ±r:**

1. "Tehdit AvcÄ±lÄ±ÄŸÄ±" sayfasÄ±na gidin
2. Sorgu yazÄ±n veya ÅŸablon seÃ§in
3. Zaman aralÄ±ÄŸÄ± belirleyin
4. "Hunt BaÅŸlat" tÄ±klayÄ±n
5. SonuÃ§larÄ± inceleyin

**Ã–rnek Sorgular:**

```sql
# Brute force tespiti
failed login | authentication failure

# Veri sÄ±zÄ±ntÄ±sÄ±
upload | POST | large transfer

# ZararlÄ± aktivite
malware | virus | trojan
```

---

### ğŸ” Security Hub (GÃ¼venlik Merkezi)

**AmaÃ§:** Genel gÃ¼venlik durumu ve uyumluluk

**Ã–zellikler:**

- GÃ¼venlik skoru (A-F)
- Uyumluluk kontrolleri
- AÄŸ topolojisi
- Bal kÃ¼pÃ¼ izleme

**NasÄ±l KullanÄ±lÄ±r:**

1. "GÃ¼venlik Merkezi" sayfasÄ±na gidin
2. Genel skoru inceleyin
3. Sekmelerde detaylara bakÄ±n
4. Ã–nerileri uygulayÄ±n

---

### ğŸ“¦ Container Security (Konteyner GÃ¼venliÄŸi)

**AmaÃ§:** Docker konteyner ve imajlarÄ±nÄ± tarama

**Ã–zellikler:**

- Container listesi
- Ä°maj gÃ¼venlik taramasÄ±
- AÃ§Ä±klÄ±k tespiti
- CVE raporlama

**Ã–n KoÅŸul:** Docker Desktop Ã§alÄ±ÅŸÄ±yor olmalÄ±

**NasÄ±l KullanÄ±lÄ±r:**

1. Docker Desktop'Ä± baÅŸlatÄ±n
2. "Container GÃ¼venlik" sayfasÄ±na gidin
3. Ä°maj adÄ± girin ve "Tara" tÄ±klayÄ±n
4. GÃ¼venlik aÃ§Ä±klarÄ±nÄ± inceleyin

---

### ğŸ”— SIEM Integration (SIEM Entegrasyonu)

**AmaÃ§:** Harici SIEM sistemlerine baÄŸlanma

**Desteklenen Platformlar:**

- Splunk Enterprise
- Elastic SIEM
- IBM QRadar
- Microsoft Sentinel
- Wazuh

**NasÄ±l KullanÄ±lÄ±r:**

1. "SIEM" sayfasÄ±na gidin
2. Platform seÃ§in
3. BaÄŸlantÄ± bilgilerini girin
4. "BaÄŸlan" tÄ±klayÄ±n
5. Event forwarding kurallarÄ± oluÅŸturun

---

### ğŸ§ª Sandbox (Kum Havuzu)

**AmaÃ§:** ÅÃ¼pheli dosyalarÄ± izole ortamda analiz

**Ã–zellikler:**

- Dosya yÃ¼kleme
- Statik analiz
- VirusTotal entegrasyonu
- Risk skorlama

**NasÄ±l KullanÄ±lÄ±r:**

1. "Sandbox" sayfasÄ±na gidin
2. Dosya yÃ¼kleyin
3. Analiz sonuÃ§larÄ±nÄ± bekleyin
4. Tehdit raporunu inceleyin

---

### â›“ï¸ Blockchain Audit (DeÄŸiÅŸmez KayÄ±t)

**AmaÃ§:** GÃ¼venlik olaylarÄ±nÄ±n deÄŸiÅŸtirilemez kaydÄ±

**Ã–zellikler:**

- Olay zinciri
- Hash doÄŸrulama
- Arama
- BÃ¼tÃ¼nlÃ¼k kontrolÃ¼

**NasÄ±l KullanÄ±lÄ±r:**

1. "Blockchain" sayfasÄ±na gidin
2. Son bloklarÄ± inceleyin
3. "DoÄŸrula" ile bÃ¼tÃ¼nlÃ¼k kontrolÃ¼ yapÄ±n
4. Arama ile geÃ§miÅŸ olaylarÄ± bulun

---

## ğŸ“š KullanÄ±m SenaryolarÄ±

### Senaryo 1: GÃ¼nlÃ¼k GÃ¼venlik KontrolÃ¼

```bash
1. Dashboard'u aÃ§Ä±n â†’ Genel durumu kontrol edin
2. Attack Map'e bakÄ±n â†’ Aktif tehditleri gÃ¶rÃ¼n
3. Network Monitor â†’ ÅÃ¼pheli baÄŸlantÄ±larÄ± kontrol edin
4. Security Hub â†’ GÃ¼venlik skorunuzu gÃ¶rÃ¼n
```

### Senaryo 2: ÅÃ¼pheli Dosya Analizi

```bash
1. Sandbox'a gidin
2. DosyayÄ± yÃ¼kleyin
3. Analiz sonucunu bekleyin
4. Risk skoru yÃ¼ksekse:
   - AI Assistant'a sorun
   - Threat Hunting yapÄ±n
```

### Senaryo 3: Olay AraÅŸtÄ±rmasÄ±

```bash
1. Threat Hunting sayfasÄ±na gidin
2. Åablon seÃ§in veya sorgu yazÄ±n
3. EÅŸleÅŸmeleri inceleyin
4. Blockchain'de ilgili loglarÄ± doÄŸrulayÄ±n
5. Rapor oluÅŸturun
```

### Senaryo 4: SIEM Entegrasyonu

```bash
1. SIEM sayfasÄ±na gidin
2. Platformunuzu seÃ§in (Splunk vb.)
3. API bilgilerini girin
4. BaÄŸlantÄ±yÄ± test edin
5. Forwarding kurallarÄ±nÄ± aktifleÅŸtirin
```

---

## â“ SÄ±k Sorulan Sorular

### Backend baÅŸlamÄ±yor?

```bash
# Port kullanÄ±mda olabilir
netstat -ano | findstr :8000
# FarklÄ± port kullanÄ±n
uvicorn main:app --port 8001
```

### Frontend hatasÄ± alÄ±yorum?

```bash
# Node modules'Ã¼ temizleyin
rm -rf node_modules
npm install
npm run dev
```

### AI Assistant yanÄ±t vermiyor?

- `.env` dosyasÄ±nda `GROQ_API_KEY` veya `OPENAI_API_KEY` olduÄŸundan emin olun
- API limitlerinizi kontrol edin

### Docker baÄŸlantÄ±sÄ± yok?

- Docker Desktop'Ä±n Ã§alÄ±ÅŸtÄ±ÄŸÄ±ndan emin olun
- WSL2 entegrasyonunu kontrol edin

### 404 hatasÄ± alÄ±yorum?

- Backend'in Ã§alÄ±ÅŸtÄ±ÄŸÄ±ndan emin olun
- `http://localhost:8000/api/docs` eriÅŸilebilir mi kontrol edin

---

## ğŸ“ Destek

**Hata Bildirimi:** GitHub Issues  
**DokÃ¼mantasyon:** `/docs` klasÃ¶rÃ¼  
**API Referans:** <http://localhost:8000/api/docs>

---

## ğŸ” GÃ¼venlik Ä°puÃ§larÄ±

1. âœ… API anahtarlarÄ±nÄ± `.env` dosyasÄ±nda saklayÄ±n
2. âœ… `.env` dosyasÄ±nÄ± git'e eklemeyin
3. âœ… GÃ¼Ã§lÃ¼ parolalar kullanÄ±n
4. âœ… DÃ¼zenli gÃ¼ncelleme yapÄ±n
5. âœ… Log dosyalarÄ±nÄ± dÃ¼zenli inceleyin

---

**ğŸ›¡ï¸ CyberGuard AI ile gÃ¼vende kalÄ±n!**


---


# LICENSE

# ğŸ“œ License

MIT License

Copyright (c) 2024-2026 CyberGuard AI Team

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

---

## ğŸ“‹ Ã–zet

| Ä°zin | Durum |
|------|-------|
| âœ… Ticari KullanÄ±m | Ä°zin verildi |
| âœ… Modifikasyon | Ä°zin verildi |
| âœ… DaÄŸÄ±tÄ±m | Ä°zin verildi |
| âœ… Ã–zel KullanÄ±m | Ä°zin verildi |
| âŒ Sorumluluk | Kabul edilmez |
| âŒ Garanti | Verilmez |

---

## ğŸ”— ÃœÃ§Ã¼ncÃ¼ Parti LisanslarÄ±

Bu proje aÅŸaÄŸÄ±daki aÃ§Ä±k kaynak kÃ¼tÃ¼phaneleri kullanÄ±r:

| KÃ¼tÃ¼phane | Lisans |
|-----------|--------|
| TensorFlow | Apache 2.0 |
| FastAPI | MIT |
| React | MIT |
| Scikit-learn | BSD-3-Clause |
| Pandas | BSD-3-Clause |
| NumPy | BSD-3-Clause |

TÃ¼m baÄŸÄ±mlÄ±lÄ±klarÄ±n lisanslarÄ± MIT lisansÄ± ile uyumludur.

---

## ğŸ“ Ä°letiÅŸim

Lisans sorularÄ±nÄ±z iÃ§in: <legal@cyberguard-ai.com>


---


# ML_MODELS

# ğŸ§  Machine Learning Models DokÃ¼mantasyonu

CyberGuard AI'da kullanÄ±lan tÃ¼m makine Ã¶ÄŸrenmesi modelleri

---

## ğŸ“‹ Ä°Ã§indekiler

- [Model Mimarisi](#model-mimarisi)
- [SSA-LSTMIDS (Ana Model)](#ssa-lstmids-ana-model)
- [Desteklenen Modeller](#desteklenen-modeller)
- [Model PerformanslarÄ±](#model-performanslarÄ±)
- [Model EÄŸitimi](#model-eÄŸitimi)
- [Model KullanÄ±mÄ±](#model-kullanÄ±mÄ±)

---

## ğŸ—ï¸ Model Mimarisi

### SSA-LSTMIDS Mimarisi

```
â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
â”‚                      INPUT LAYER                             â”‚
â”‚                    (78 features)                             â”‚
â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¬â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜
                          â”‚
â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â–¼â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
â”‚                   CONV1D BLOCK 1                             â”‚
â”‚  Conv1D(30, kernel=3) â†’ BatchNorm â†’ ReLU â†’ MaxPool(2)       â”‚
â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¬â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜
                          â”‚
â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â–¼â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
â”‚                   CONV1D BLOCK 2                             â”‚
â”‚  Conv1D(60, kernel=3) â†’ BatchNorm â†’ ReLU â†’ MaxPool(2)       â”‚
â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¬â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜
                          â”‚
â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â–¼â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
â”‚                     LSTM LAYER                               â”‚
â”‚           LSTM(120 units, return_sequences=True)             â”‚
â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¬â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜
                          â”‚
â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â–¼â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
â”‚                   ATTENTION LAYER                            â”‚
â”‚            MultiHeadAttention(num_heads=4)                   â”‚
â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¬â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜
                          â”‚
â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â–¼â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
â”‚                    DENSE LAYERS                              â”‚
â”‚      Dense(512) â†’ Dropout(0.2) â†’ Dense(256) â†’ Dropout(0.2)  â”‚
â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¬â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜
                          â”‚
â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â–¼â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
â”‚                   OUTPUT LAYER                               â”‚
â”‚              Dense(num_classes, softmax)                     â”‚
â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜
```

---

## ğŸ¯ SSA-LSTMIDS (Ana Model)

### Genel Bilgiler

| Ã–zellik | DeÄŸer |
|---------|-------|
| **Model AdÄ±** | SSA-LSTMIDS (Sparrow Search Algorithm - LSTM IDS) |
| **Kaynak Makale** | "An optimized LSTM-based deep learning model for anomaly network intrusion detection" |
| **YayÄ±n** | Scientific Reports, 2025 |
| **Optimizasyon** | SSA (Sparrow Search Algorithm) |

### SSA Optimizasyonu

SSA (Sparrow Search Algorithm), serÃ§elerin yiyecek arama davranÄ±ÅŸÄ±ndan ilham alan metaheuristik bir optimizasyon algoritmasÄ±dÄ±r.

**Optimize Edilen Hiperparametreler:**

- Conv1D filter sayÄ±sÄ± (30)
- LSTM unit sayÄ±sÄ± (120)
- Dense layer units (512)
- Dropout oranÄ± (0.2)
- Epoch sayÄ±sÄ± (300)
- Batch size (120)

### Performans SonuÃ§larÄ±

| Dataset | Accuracy | Precision | Recall | F1-Score |
|---------|----------|-----------|--------|----------|
| **NSL-KDD** | 99.36% | 99.37% | 99.36% | 99.36% |
| **CICIDS2017** | 99.88% | 99.89% | 99.88% | 99.88% |
| **BoT-IoT** | 99.99% | 99.99% | 99.99% | 99.99% |

---

## ğŸ“š Desteklenen Modeller

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

- **KullanÄ±m**: Temporal pattern recognition
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

- **KullanÄ±m**: Forward + backward context
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

- **KullanÄ±m**: Feature extraction + sequence learning
- **Accuracy**: ~98-99.5%

#### Transformer

```python
# Attention-based model
inputs = keras.layers.Input(shape=(timesteps, features))
x = keras.layers.MultiHeadAttention(num_heads=4, key_dim=64)(inputs, inputs)
x = keras.layers.GlobalAveragePooling1D()(x)
outputs = keras.layers.Dense(num_classes, activation='softmax')(x)
```

- **KullanÄ±m**: Self-attention mechanisms
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

- **KullanÄ±m**: Baseline, hÄ±zlÄ± inference
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

- **KullanÄ±m**: Gradient boosting
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

- **KullanÄ±m**: Linear/non-linear classification
- **Accuracy**: ~90-94%

---

## ğŸ“Š Model PerformanslarÄ±

### Dataset BazÄ±nda KarÅŸÄ±laÅŸtÄ±rma

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

## ğŸ”§ Model EÄŸitimi

### EÄŸitim Scripti

```bash
# Full training pipeline
python scripts/train_cicids_full_ssa.py

# Specific dataset
python scripts/train_nsl_kdd.py
python scripts/train_botiot.py

# Fine-tuning
python scripts/finetune_deep_ssa.py
```

### EÄŸitim Parametreleri

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

## ğŸ’» Model KullanÄ±mÄ±

### Python API

```python
from src.models.predictor import AttackPredictor

# Model yÃ¼kle
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

## ğŸ“ Model DosyalarÄ±

```
models/
â”œâ”€â”€ production/
â”‚   â”œâ”€â”€ best_cicids2017_model.h5
â”‚   â”œâ”€â”€ best_nslkdd_model.h5
â”‚   â””â”€â”€ best_botiot_model.h5
â”œâ”€â”€ experimental/
â”‚   â”œâ”€â”€ transformer_v1.h5
â”‚   â””â”€â”€ bilstm_attention.h5
â”œâ”€â”€ archived/
â”‚   â””â”€â”€ old_models/
â”œâ”€â”€ scalers/
â”‚   â”œâ”€â”€ cicids2017_scaler.pkl
â”‚   â”œâ”€â”€ nslkdd_scaler.pkl
â”‚   â””â”€â”€ botiot_scaler.pkl
â””â”€â”€ model_registry.json
```

---

## ğŸ“ Referanslar

- [An optimized LSTM-based deep learning model](https://doi.org/10.1038/s41598-025-85248-z)
- [NSL-KDD Dataset](https://www.unb.ca/cic/datasets/nsl.html)
- [CICIDS2017 Dataset](https://www.unb.ca/cic/datasets/ids-2017.html)
- [BoT-IoT Dataset](https://research.unsw.edu.au/projects/bot-iot-dataset)


---


# MONİTORİNG

# ğŸ“Š Monitoring Guide

CyberGuard AI sistem izleme ve alerting rehberi

---

## ğŸ“‹ Ä°Ã§indekiler

- [Prometheus & Grafana](#prometheus--grafana)
- [Log YÃ¶netimi](#log-yÃ¶netimi)
- [Alerting](#alerting)
- [Health Checks](#health-checks)
- [Dashboard](#dashboard)

---

## ğŸ“ˆ Prometheus & Grafana

### Kurulum

```yaml
# docker-compose.monitoring.yml
version: '3.8'

services:
  prometheus:
    image: prom/prometheus
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
    ports:
      - "9090:9090"

  grafana:
    image: grafana/grafana
    ports:
      - "3000:3000"
    volumes:
      - grafana-data:/var/lib/grafana

volumes:
  grafana-data:
```

### Prometheus Config

```yaml
# prometheus.yml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'cyberguard-api'
    static_configs:
      - targets: ['api:8000']

  - job_name: 'node-exporter'
    static_configs:
      - targets: ['node-exporter:9100']
```

### FastAPI Metrics

```python
# app/metrics.py
from prometheus_client import Counter, Histogram, Gauge

REQUEST_COUNT = Counter(
    'http_requests_total',
    'Total HTTP requests',
    ['method', 'endpoint', 'status']
)

REQUEST_LATENCY = Histogram(
    'http_request_duration_seconds',
    'HTTP request latency',
    ['method', 'endpoint']
)

ACTIVE_CONNECTIONS = Gauge(
    'active_connections',
    'Active WebSocket connections'
)

MODEL_INFERENCE_TIME = Histogram(
    'model_inference_seconds',
    'Model inference time'
)
```

---

## ğŸ“ Log YÃ¶netimi

### Log FormatÄ±

```python
# app/logging_config.py
LOGGING_CONFIG = {
    "version": 1,
    "formatters": {
        "default": {
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        },
        "json": {
            "class": "pythonjsonlogger.jsonlogger.JsonFormatter",
            "format": "%(asctime)s %(name)s %(levelname)s %(message)s"
        }
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "default"
        },
        "file": {
            "class": "logging.handlers.RotatingFileHandler",
            "filename": "logs/app.log",
            "maxBytes": 10485760,
            "backupCount": 5,
            "formatter": "json"
        }
    },
    "root": {
        "level": "INFO",
        "handlers": ["console", "file"]
    }
}
```

### ELK Stack

```yaml
# docker-compose.logging.yml
services:
  elasticsearch:
    image: elasticsearch:8.6.0
    environment:
      - discovery.type=single-node
    ports:
      - "9200:9200"

  logstash:
    image: logstash:8.6.0
    volumes:
      - ./logstash.conf:/usr/share/logstash/pipeline/logstash.conf

  kibana:
    image: kibana:8.6.0
    ports:
      - "5601:5601"
```

---

## ğŸ”” Alerting

### Alert Rules

```yaml
# alerts.yml
groups:
  - name: cyberguard
    rules:
    - alert: HighErrorRate
      expr: rate(http_requests_total{status=~"5.."}[5m]) > 0.1
      for: 5m
      labels:
        severity: critical
      annotations:
        summary: High error rate detected
    
    - alert: SlowResponse
      expr: histogram_quantile(0.95, http_request_duration_seconds_bucket) > 1
      for: 10m
      labels:
        severity: warning
      annotations:
        summary: API response time is slow
    
    - alert: HighMemoryUsage
      expr: process_resident_memory_bytes > 2e9
      for: 5m
      labels:
        severity: warning
```

### Slack Entegrasyonu

```python
import requests

def send_slack_alert(message, severity="warning"):
    webhook_url = os.getenv("SLACK_WEBHOOK_URL")
    color = "#ff0000" if severity == "critical" else "#ffcc00"
    
    payload = {
        "attachments": [{
            "color": color,
            "title": f"CyberGuard Alert ({severity})",
            "text": message
        }]
    }
    
    requests.post(webhook_url, json=payload)
```

---

## ğŸ¥ Health Checks

### Endpoints

```python
# app/api/routes/health.py
from fastapi import APIRouter

router = APIRouter()

@router.get("/health")
async def health():
    return {"status": "healthy"}

@router.get("/health/ready")
async def readiness():
    # Check DB, Redis, Model
    checks = {
        "database": check_db(),
        "redis": check_redis(),
        "model": check_model()
    }
    
    status = "ready" if all(checks.values()) else "not_ready"
    return {"status": status, "checks": checks}

@router.get("/health/live")
async def liveness():
    return {"status": "alive"}
```

### Docker Health Check

```dockerfile
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1
```

---

## ğŸ“Š Dashboard Metrikleri

### Temel Metrikler

| Metrik | AÃ§Ä±klama | Alert EÅŸik |
|--------|----------|------------|
| Request Rate | req/s | > 1000 |
| Error Rate | % | > 1% |
| Latency P95 | ms | > 500ms |
| CPU Usage | % | > 80% |
| Memory Usage | GB | > 4GB |
| DB Connections | count | > 90% pool |

### Grafana Panel'leri

1. **Request Overview**
   - Total requests
   - Requests by endpoint
   - Error rate

2. **Performance**
   - Response time histogram
   - P50, P95, P99 latencies

3. **System**
   - CPU, Memory, Disk
   - Network I/O

4. **ML Model**
   - Inference count
   - Inference latency
   - Prediction distribution


---


# PERFORMANCE_TUNİNG

# âš¡ Performance Tuning Guide

CyberGuard AI performans optimizasyonu rehberi

---

## ğŸ“‹ Ä°Ã§indekiler

- [Genel BakÄ±ÅŸ](#genel-bakÄ±ÅŸ)
- [Backend Optimizasyonu](#backend-optimizasyonu)
- [Database Optimizasyonu](#database-optimizasyonu)
- [Model Optimizasyonu](#model-optimizasyonu)
- [Frontend Optimizasyonu](#frontend-optimizasyonu)
- [Caching](#caching)
- [Scaling](#scaling)

---

## ğŸ¯ Genel BakÄ±ÅŸ

### Performans Hedefleri

| Metrik | Hedef | Kritik |
|--------|-------|--------|
| API Response (P95) | < 200ms | < 500ms |
| Model Inference | < 50ms | < 100ms |
| Page Load | < 2s | < 5s |
| Throughput | 1000 req/s | 500 req/s |

---

## ğŸ–¥ï¸ Backend Optimizasyonu

### Async Endpoints

```python
# âŒ YavaÅŸ - Senkron
@app.get("/attacks")
def get_attacks():
    return db.query(Attack).all()

# âœ… HÄ±zlÄ± - Asenkron
@app.get("/attacks")
async def get_attacks():
    return await db.execute(select(Attack)).all()
```

### Connection Pooling

```python
# SQLAlchemy pool ayarlarÄ±
engine = create_engine(
    DATABASE_URL,
    pool_size=20,
    max_overflow=10,
    pool_pre_ping=True,
    pool_recycle=3600
)
```

### Worker Configuration

```bash
# Uvicorn workers
uvicorn main:app --workers 4 --loop uvloop

# Gunicorn
gunicorn main:app -w 4 -k uvicorn.workers.UvicornWorker
```

---

## ğŸ—„ï¸ Database Optimizasyonu

### Indexler

```sql
-- SÄ±k sorgulanan kolonlara index
CREATE INDEX idx_attacks_type ON attacks(attack_type);
CREATE INDEX idx_attacks_created ON attacks(created_at DESC);
CREATE INDEX idx_attacks_severity ON attacks(severity);

-- Composite index
CREATE INDEX idx_attacks_type_created ON attacks(attack_type, created_at);
```

### Query Optimizasyonu

```python
# âŒ N+1 Query
attacks = db.query(Attack).all()
for attack in attacks:
    print(attack.user.name)  # Her seferinde sorgu

# âœ… Eager Loading
attacks = db.query(Attack).options(joinedload(Attack.user)).all()

# âŒ SELECT *
SELECT * FROM attacks

# âœ… Sadece gerekli kolonlar
SELECT id, attack_type, severity FROM attacks
```

### Pagination

```python
# Offset pagination (bÃ¼yÃ¼k tablolarda yavaÅŸ)
attacks = db.query(Attack).offset(1000).limit(20).all()

# Cursor pagination (daha hÄ±zlÄ±)
attacks = db.query(Attack)\
    .filter(Attack.id > last_id)\
    .limit(20).all()
```

---

## ğŸ§  Model Optimizasyonu

### Model Warmup

```python
# BaÅŸlangÄ±Ã§ta model'i Ä±sÄ±t
@app.on_event("startup")
async def warmup():
    predictor.load_models()
    # Dummy prediction
    predictor.predict_single([0.0] * 78)
```

### Batch Prediction

```python
# âŒ Tek tek
for sample in samples:
    results.append(model.predict(sample))

# âœ… Batch
results = model.predict(np.array(samples), batch_size=64)
```

### Model Quantization

```python
import tensorflow as tf

# Float16 quantization
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float16]
tflite_model = converter.convert()
```

### GPU Memory

```python
# Memory growth
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

# Hard limit
tf.config.set_logical_device_configuration(
    gpus[0],
    [tf.config.LogicalDeviceConfiguration(memory_limit=4096)]
)
```

---

## ğŸ¨ Frontend Optimizasyonu

### Code Splitting

```javascript
// Lazy loading
const Dashboard = React.lazy(() => import('./pages/Dashboard'));
const Prediction = React.lazy(() => import('./pages/Prediction'));

function App() {
  return (
    <Suspense fallback={<Spinner />}>
      <Routes>
        <Route path="/dashboard" element={<Dashboard />} />
      </Routes>
    </Suspense>
  );
}
```

### Bundle Size

```bash
# Analyze
npm run build -- --analyze

# Vite rollup options
export default defineConfig({
  build: {
    rollupOptions: {
      output: {
        manualChunks: {
          vendor: ['react', 'react-dom'],
          charts: ['recharts']
        }
      }
    }
  }
})
```

### Image Optimization

```jsx
// WebP format + lazy loading
<img 
  src="image.webp" 
  loading="lazy"
  decoding="async"
/>
```

---

## ğŸ’¾ Caching

### Redis Cache

```python
from functools import lru_cache
import redis

redis_client = redis.Redis(host='localhost', port=6379)

async def get_dashboard_stats():
    # Cache check
    cached = redis_client.get('dashboard_stats')
    if cached:
        return json.loads(cached)
    
    # Compute
    stats = await compute_heavy_stats()
    
    # Cache (5 dakika)
    redis_client.setex('dashboard_stats', 300, json.dumps(stats))
    return stats
```

### LRU Cache

```python
from functools import lru_cache

@lru_cache(maxsize=1000)
def get_attack_type_name(encoded):
    return ATTACK_TYPES.get(encoded, "Unknown")
```

### HTTP Cache Headers

```python
from fastapi import Response

@app.get("/api/static-data")
async def static_data(response: Response):
    response.headers["Cache-Control"] = "public, max-age=3600"
    return {"data": "..."}
```

---

## ğŸ“ˆ Scaling

### Horizontal Scaling

```yaml
# docker-compose.yml
services:
  api:
    image: cyberguard/api
    deploy:
      replicas: 4
    
  nginx:
    image: nginx
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
```

### Load Balancing

```nginx
# nginx.conf
upstream api {
    least_conn;
    server api1:8000;
    server api2:8000;
    server api3:8000;
}

server {
    location /api {
        proxy_pass http://api;
    }
}
```

### Kubernetes HPA

```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: cyberguard-api-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: cyberguard-api
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
```

---

## ğŸ“Š Profiling

### Python Profiling

```bash
# cProfile
python -m cProfile -o output.prof app/main.py

# Visualize
pip install snakeviz
snakeviz output.prof
```

### Memory Profiling

```python
from memory_profiler import profile

@profile
def heavy_function():
    data = [i for i in range(10**7)]
    return sum(data)
```


---


# QUICK_START

# âš¡ CyberGuard AI - HÄ±zlÄ± BaÅŸlangÄ±Ã§ (5 Dakika)

> Bu rehber ile 5 dakikada CyberGuard AI'Ä± Ã§alÄ±ÅŸtÄ±rabilirsiniz.

---

## ğŸ“‹ Ã–n Gereksinimler

- âœ… Python 3.9+ kurulu
- âœ… Node.js 18+ kurulu
- âœ… Git kurulu

---

## ğŸš€ AdÄ±m 1: Projeyi Ä°ndirin

```bash
git clone https://github.com/salihoglueyup/CyberGuard_AI.git
cd CyberGuard_AI
```

---

## ğŸ AdÄ±m 2: Python BaÄŸÄ±mlÄ±lÄ±klarÄ±

```bash
pip install -r requirements.txt
```

---

## ğŸ“¦ AdÄ±m 3: Frontend BaÄŸÄ±mlÄ±lÄ±klarÄ±

```bash
cd frontend
npm install
cd ..
```

---

## ğŸ”‘ AdÄ±m 4: Ortam DeÄŸiÅŸkenleri (Opsiyonel)

`.env` dosyasÄ± oluÅŸturun:

```bash
# Windows
copy .env.example .env

# Linux/Mac
cp .env.example .env
```

AI Asistan iÃ§in API anahtarÄ± ekleyin:

```env
GROQ_API_KEY=your_groq_api_key
```

> ğŸ’¡ **Ä°pucu:** Ãœcretsiz Groq API anahtarÄ± almak iÃ§in: <https://console.groq.com>

---

## â–¶ï¸ AdÄ±m 5: BaÅŸlatÄ±n

### Kolay Yol (Windows)

```bash
run.bat
```

### Manuel Yol

**Terminal 1 - Backend:**

```bash
cd app
python -m uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

**Terminal 2 - Frontend:**

```bash
cd frontend
npm run dev
```

---

## ğŸŒ EriÅŸim Adresleri

| Servis | URL |
| ------ | --- |
| ğŸ–¥ï¸ Frontend | <http://localhost:5173> |
| ğŸ”Œ Backend API | <http://localhost:8000> |
| ğŸ“š API Docs | <http://localhost:8000/api/docs> |
| ğŸ“– ReDoc | <http://localhost:8000/api/redoc> |

---

## âœ… BaÅŸarÄ±lÄ± Kurulum KontrolÃ¼

1. TarayÄ±cÄ±da <http://localhost:5173> aÃ§Ä±n
2. Dashboard yÃ¼klenirse âœ…
3. Sol menÃ¼den "Attack Map" seÃ§in
4. 3D Globe gÃ¶rÃ¼ntÃ¼lenirse âœ…

---

## ğŸ”§ Sorun Giderme

### Port kullanÄ±mda hatasÄ±

```bash
# Windows - 8000 portunu kullanan processi bul
netstat -ano | findstr :8000

# FarklÄ± port kullan
uvicorn main:app --port 8001
```

### npm hatasÄ±

```bash
# Node modules'Ã¼ temizle
rm -rf node_modules
npm cache clean --force
npm install
```

### Backend baÅŸlamÄ±yor

```bash
# Eksik paketleri kontrol et
pip install -r requirements.txt --upgrade
```

---

## ğŸ“š Sonraki AdÄ±mlar

- ğŸ“– [KullanÄ±m KÄ±lavuzu](KULLANIM_KILAVUZU.md) - DetaylÄ± kullanÄ±m
- ğŸ”Œ [API Ã–rnekleri](API_EXAMPLES.md) - API kullanÄ±mÄ±
- ğŸŒ [WebSocket Rehberi](WEBSOCKET_GUIDE.md) - GerÃ§ek zamanlÄ± veri

---

**ğŸ›¡ï¸ Haydi baÅŸlayalÄ±m!**


---


# RELEASE_NOTES

# ğŸ“‹ Release Notes

CyberGuard AI sÃ¼rÃ¼m notlarÄ±

---

## ğŸš€ v3.0.0 - Mega Update (2026-01-10)

### ğŸ‰ Highlights

Bu bÃ¼yÃ¼k gÃ¼ncelleme ile CyberGuard AI, orijinal akademik makalenin kapsamÄ±nÄ±n Ã§ok Ã¶tesine geÃ§erek **25+ yeni Ã¶zellik** ile tam kapsamlÄ± bir siber gÃ¼venlik platformuna dÃ¶nÃ¼ÅŸmÃ¼ÅŸtÃ¼r.

### âœ¨ Yeni Ã–zellikler

#### API'ler (17 Yeni ModÃ¼l)

- **XAI (Explainable AI)**: SHAP ve LIME ile model aÃ§Ä±klamalarÄ±
- **Adversarial Testing**: Model gÃ¼venlik testleri
- **Federated Learning**: DaÄŸÄ±tÄ±k model eÄŸitimi
- **AutoML**: Otomatik model seÃ§imi ve optimizasyonu
- **Threat Intelligence**: IP/Domain/Hash reputation
- **Email Alerts**: Otomatik bildirim sistemi
- **PDF Reports**: Profesyonel rapor oluÅŸturma
- **Model Comparison**: Model benchmark ve leaderboard
- **Anomaly Detection**: Anomali tespit algoritmalarÄ±
- **Security Advanced**: PCAP analizi, Honeypot, Compliance
- **Vulnerability Scanner**: Port tarama, CVE kontrolÃ¼
- **Log Analyzer**: ML ile log analizi
- **Incidents**: Olay timeline ve user behavior
- **API Keys**: API anahtar yÃ¶netimi

#### Frontend (5 Yeni Sayfa)

- XAI Explainer (`/xai`)
- Security Hub (`/security-hub`)
- AutoML Pipeline (`/automl`)
- Vulnerability Scanner (`/vuln-scanner`)
- Incident Timeline (`/incidents`)

#### DokÃ¼mantasyon (14 Yeni Dosya)

- faq.md, troubleshooting.md, glossary.md
- api_endpoints_full.md, testing.md, ci_cd.md
- monitoring.md, backup_recovery.md
- performance_tuning.md, LICENSE.md
- SECURITY_POLICY.md, release_notes.md
- ml_models.md, datasets.md

### ğŸ“Š Ä°statistikler

| Metrik | DeÄŸer |
|--------|-------|
| Yeni API Endpoint | 80+ |
| Toplam Endpoint | 150+ |
| Yeni Frontend Sayfa | 5 |
| Yeni Docs DosyasÄ± | 14 |
| Makalede Olmayan Ã–zellik | 25+ |

### ğŸ”§ Ä°yileÅŸtirmeler

- Dosya yapÄ±sÄ± reorganize edildi
  - scripts/ â†’ training/, optimization/, data/, utils/, archived/
  - models/ â†’ production/, experimental/, archived/
  - docs/ yazÄ±m hatalarÄ± dÃ¼zeltildi

### ğŸ“ DokÃ¼mantasyon

- TÃ¼m yeni Ã¶zellikler belgelendi
- API endpoint listesi gÃ¼ncellendi
- Changelog v3.0.0 iÃ§in gÃ¼ncellendi

---

## ğŸš€ v2.0.0 (2025-01-15)

### âœ¨ Yeni Ã–zellikler

- AI-Powered Chatbot
- Gemini AI entegrasyonu
- Real-time threat monitoring
- PDF ve Excel export
- MFA desteÄŸi
- Enhanced dashboard

### ğŸ”§ Ä°yileÅŸtirmeler

- Model accuracy %95+ â†’ %99+
- API response time %40 iyileÅŸtirildi
- UI/UX tamamen yenilendi

### ğŸ› DÃ¼zeltmeler

- Port tarama timeout sorunu
- Database connection pool sÄ±zÄ±ntÄ±sÄ±
- Memory leak

---

## ğŸš€ v1.5.0 (2024-10-20)

### âœ¨ Yeni Ã–zellikler

- ML-based threat detection
- Random Forest classifier
- Scheduled scans
- Email notifications
- Slack integration

### ğŸ”§ Ä°yileÅŸtirmeler

- Scanner performance %30 artÄ±rÄ±ldÄ±
- False positive rate azaltÄ±ldÄ±

---

## ğŸš€ v1.0.0 (2024-06-01)

### Ä°lk Stable SÃ¼rÃ¼m

- Port scanning
- Vulnerability detection
- CVE database integration
- Web dashboard
- REST API
- PostgreSQL support

---

## ğŸ“… Upgrade Guide

### v2.x â†’ v3.0

```bash
# 1. Backup
./scripts/backup_all.sh

# 2. Pull latest
git pull origin main

# 3. Install dependencies
pip install -r requirements.txt
cd frontend && npm install

# 4. Run migrations
alembic upgrade head

# 5. Restart
./start-servers.sh
```

### Breaking Changes

- API v1 endpoints kaldÄ±rÄ±ldÄ±
- `config.yaml` formatÄ± deÄŸiÅŸti
- Model dosya yapÄ±sÄ± deÄŸiÅŸti

---

## ğŸ“ Destek

- GitHub Issues
- Discord: discord.gg/cyberguard
- Email: <support@cyberguard-ai.com>


---


# ROADMAP

# ğŸ—ºï¸ CyberGuard AI Roadmap

Bu dokÃ¼manda CyberGuard AI'nÄ±n gelecekteki geliÅŸtirme planlarÄ±nÄ±, hedeflerini ve kilometre taÅŸlarÄ±nÄ± bulabilirsiniz.

## ğŸ“Š Genel BakÄ±ÅŸ

```
2025 Q1 â–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–‘â–‘â–‘â–‘â–‘â–‘â–‘â–‘ 60% TamamlandÄ±
2025 Q2 â–ˆâ–ˆâ–ˆâ–ˆâ–‘â–‘â–‘â–‘â–‘â–‘â–‘â–‘â–‘â–‘â–‘â–‘â–‘â–‘â–‘â–‘ 20% Planlama
2025 Q3 â–‘â–‘â–‘â–‘â–‘â–‘â–‘â–‘â–‘â–‘â–‘â–‘â–‘â–‘â–‘â–‘â–‘â–‘â–‘â–‘  0% Planlama
2025 Q4 â–‘â–‘â–‘â–‘â–‘â–‘â–‘â–‘â–‘â–‘â–‘â–‘â–‘â–‘â–‘â–‘â–‘â–‘â–‘â–‘  0% Planlama
```

---

## ğŸ¯ Vizyon

**2025 Sonu Hedefi**: EndÃ¼stri lideri AI-powered siber gÃ¼venlik platformu olmak

**Temel Hedefler**:
- ğŸ¤– En geliÅŸmiÅŸ AI/ML altyapÄ±sÄ±
- ğŸŒ KÃ¼resel kullanÄ±cÄ± tabanÄ± (100K+ users)
- ğŸ† %99.9 uptime guarantee
- ğŸ”’ Zero-day threat detection

---

## ğŸ“… 2025 Q1 (Ocak - Mart) - 60% âœ…

### âœ… Tamamlanan Ã–zellikler

- [x] AI Chatbot v2.0 entegrasyonu
- [x] Transformer-based NLP model
- [x] Multi-language support (TR/EN)
- [x] Real-time threat monitoring
- [x] PDF/Excel report generation
- [x] API v2.0 launch

### ğŸš§ Devam Eden

- [ ] Mobile app (iOS/Android) - %70
- [ ] Advanced ML models - %50
- [ ] Cloud integration - %40

### ğŸ’¡ Q1 Yeni Ã–zellikler

**1. Enhanced Chatbot**
- **Durum**: âœ… TamamlandÄ±
- **Ã–zellikler**:
    - Context-aware conversations
    - File upload & analysis
    - Code execution capability
    - Multi-turn dialogues

**2. Threat Intelligence**
- **Durum**: ğŸš§ Devam ediyor (%80)
- **Ã–zellikler**:
    - Live threat feeds
    - CVE database integration
    - IoC (Indicators of Compromise) tracking
    - MITRE ATT&CK framework mapping

**3. Automated Response**
- **Durum**: ğŸ“‹ Planlama
- **Hedef Tarih**: Mart 2025
- **Ã–zellikler**:
    - Auto-remediation
    - Playbook execution
    - Incident response automation

---

## ğŸ“… 2025 Q2 (Nisan - Haziran)

### ğŸ¯ Ana Hedefler

1. **Mobile Platform Launch** ğŸ“±
2. **Enterprise Features** ğŸ¢
3. **Advanced Analytics** ğŸ“Š

### Planlanan Ã–zellikler

#### 1. Mobile Applications

**iOS App** ğŸ
- **BaÅŸlangÄ±Ã§**: Nisan 2025
- **Beta Release**: MayÄ±s 2025
- **Production**: Haziran 2025
- **Ã–zellikler**:
    - Real-time notifications
    - Quick scan capability
    - Dashboard viewing
    - Touch ID / Face ID support

**Android App** ğŸ¤–
- **BaÅŸlangÄ±Ã§**: Nisan 2025
- **Beta Release**: MayÄ±s 2025
- **Production**: Haziran 2025
- **Ã–zellikler**:
    - Material Design 3
    - Offline mode
    - Widget support
    - Biometric authentication

#### 2. Enterprise Features

**Multi-Tenancy** ğŸ¢
- Organization management
- Role-based access control (RBAC)
- Team collaboration tools
- Audit logging

**SSO Integration** ğŸ”
- SAML 2.0
- OAuth 2.0
- LDAP/Active Directory
- Google Workspace
- Microsoft Azure AD

**Compliance Reporting** ğŸ“‹
- ISO 27001 reports
- PCI DSS compliance
- GDPR compliance
- SOC 2 Type II reports

#### 3. Advanced Analytics

**Predictive Analytics** ğŸ”®
- Threat forecasting
- Risk trend analysis
- Anomaly prediction
- Resource optimization

**Custom Dashboards** ğŸ“Š
- Drag-and-drop builder
- Widget marketplace
- Custom KPIs
- Real-time data visualization

---

## ğŸ“… 2025 Q3 (Temmuz - EylÃ¼l)

### ğŸ¯ Ana Hedefler

1. **AI/ML Enhancement** ğŸ§ 
2. **Cloud-Native Architecture** â˜ï¸
3. **Threat Hunting Platform** ğŸ¯

### Planlanan Ã–zellikler

#### 1. AI/ML Enhancements

**Advanced Models** ğŸ¤–
- **Deep Learning**:
    - CNN for image analysis
    - RNN for time-series
    - GAN for synthetic data

- **Reinforcement Learning**:
    - Auto-tuning algorithms
    - Adaptive defense mechanisms

- **Federated Learning**:
    - Privacy-preserving ML
    - Distributed training

**AutoML Platform** ğŸ”¬
- Automated model selection
- Hyperparameter tuning
- Model deployment pipeline
- A/B testing framework

#### 2. Cloud-Native Architecture

**Multi-Cloud Support** â˜ï¸
- AWS integration
- Azure integration
- Google Cloud integration
- Hybrid cloud support

**Kubernetes Integration** âš“
- Helm charts
- Auto-scaling
- Service mesh (Istio)
- GitOps (ArgoCD)

**Serverless Functions** âš¡
- AWS Lambda
- Azure Functions
- Google Cloud Functions
- Event-driven architecture

#### 3. Threat Hunting Platform

**Advanced Search** ğŸ”
- Query language (similar to KQL)
- Full-text search
- Time-based queries
- Correlation engine

**Investigation Tools** ğŸ•µï¸
- Timeline analysis
- Relationship mapping
- IOC extraction
- Evidence collection

**Playbooks** ğŸ“š
- Pre-built playbooks
- Custom playbook creator
- Workflow automation
- Integration marketplace

---

## ğŸ“… 2025 Q4 (Ekim - AralÄ±k)

### ğŸ¯ Ana Hedefler

1. **Global Expansion** ğŸŒ
2. **Performance Optimization** âš¡
3. **Security Hardening** ğŸ”’

### Planlanan Ã–zellikler

#### 1. Global Expansion

**Internationalization (i18n)** ğŸŒ
- 10+ language support
- RTL (Right-to-Left) support
- Localized content
- Regional compliance

**Regional Data Centers** ğŸ—ºï¸
- Europe (Frankfurt)
- Asia (Singapore)
- Americas (Virginia)
- Data residency compliance

#### 2. Performance Optimization

**System Performance** âš¡
- Sub-second query response
- 10x faster scanning
- Real-time processing
- Edge computing support

**ML Model Optimization** ğŸš€
- Model quantization
- Pruning techniques
- GPU acceleration
- Distributed inference

#### 3. Security Hardening

**Zero Trust Architecture** ğŸ›¡ï¸
- Microsegmentation
- Identity verification
- Least privilege access
- Continuous monitoring

**Advanced Encryption** ğŸ”
- Homomorphic encryption
- Quantum-safe cryptography
- Hardware security modules (HSM)
- Key management service

---

## ğŸ”® 2026 Vizyonu

### Uzun Vadeli Hedefler

**Artificial General Intelligence (AGI) Integration** ğŸ¤–
- Fully autonomous security operations
- Self-healing systems
- Cognitive threat hunting
- Natural language security policies

**Quantum Computing Ready** âš›ï¸
- Quantum-resistant algorithms
- Quantum ML models
- Post-quantum cryptography

**Edge AI** ğŸ“¡
- On-device ML inference
- Offline threat detection
- Low-latency processing
- IoT security

**Blockchain Integration** â›“ï¸
- Immutable audit logs
- Decentralized threat intelligence
- Smart contract automation

---

## ğŸ“Š Feature Requests & Community Votes

En Ã§ok talep edilen Ã¶zellikler (GitHub issues'dan):

| # | Ã–zellik | Oylar | Durum | Hedef |
|---|---------|-------|-------|-------|
| 1 | Container Security Scanning | 342 | ğŸš§ In Progress | Q2 2025 |
| 2 | GraphQL API | 287 | ğŸ“‹ Planned | Q3 2025 |
| 3 | Dark Web Monitoring | 256 | ğŸ“‹ Planned | Q4 2025 |
| 4 | Blockchain Analysis | 198 | ğŸ’­ Under Review | TBD |
| 5 | AI-powered Code Review | 187 | ğŸ“‹ Planned | Q3 2025 |
| 6 | Threat Intelligence Sharing | 165 | ğŸš§ In Progress | Q2 2025 |
| 7 | Video Conference Security | 143 | ğŸ’­ Under Review | TBD |
| 8 | Social Engineering Detection | 128 | ğŸ“‹ Planned | Q4 2025 |
| 9 | Cloud Cost Optimization | 112 | ğŸ’­ Under Review | 2026 |
| 10 | AR/VR Security Visualization | 98 | ğŸ’­ Research | 2026+ |

**Oy Vermek Ä°Ã§in**: [Feature Request](https://github.com/cyberguard-ai/issues/new?template=feature_request.md)

---

## ğŸ—ï¸ Technical Debt & Infrastructure

### Q1-Q2 2025

- [ ] **Database Migration**: PostgreSQL â†’ PostgreSQL 15
- [ ] **Redis Cluster**: Single â†’ Multi-node
- [ ] **Message Queue**: RabbitMQ â†’ Apache Kafka
- [ ] **Monitoring**: Prometheus + Grafana stack
- [ ] **Logging**: ELK Stack upgrade

### Q3-Q4 2025

- [ ] **Service Mesh**: Istio implementation
- [ ] **Observability**: OpenTelemetry integration
- [ ] **Chaos Engineering**: Gremlin/Chaos Monkey
- [ ] **API Gateway**: Kong/Tyk implementation
- [ ] **CDN**: CloudFlare Enterprise

---

## ğŸ“ Research & Development

### Active R&D Projects

**1. Adversarial ML** ğŸ¥Š
- Adversarial attack detection
- Robust model training
- Defensive distillation
- **Timeline**: Q2-Q3 2025

**2. Explainable AI (XAI)** ğŸ”
- SHAP values integration
- LIME explanations
- Model interpretability
- **Timeline**: Q2-Q4 2025

**3. Privacy-Preserving ML** ğŸ”’
- Differential privacy
- Secure multi-party computation
- Federated learning
- **Timeline**: Q3 2025 - Q1 2026

**4. Neuromorphic Computing** ğŸ§ 
- Spiking neural networks
- Energy-efficient AI
- Real-time processing
- **Timeline**: Research phase

---

## ğŸ’° Investment Areas

### 2025 Budget Allocation

```
Development     â–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–‘â–‘â–‘â–‘ 40%
Infrastructure  â–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–‘â–‘â–‘â–‘â–‘â–‘â–‘â–‘â–‘â–‘â–‘â–‘ 20%
Research        â–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–‘â–‘â–‘â–‘â–‘â–‘â–‘â–‘â–‘â–‘â–‘â–‘â–‘â–‘ 15%
Marketing       â–ˆâ–ˆâ–ˆâ–ˆâ–‘â–‘â–‘â–‘â–‘â–‘â–‘â–‘â–‘â–‘â–‘â–‘â–‘â–‘â–‘â–‘ 10%
Support         â–ˆâ–ˆâ–ˆâ–ˆâ–‘â–‘â–‘â–‘â–‘â–‘â–‘â–‘â–‘â–‘â–‘â–‘â–‘â–‘â–‘â–‘ 10%
Operations      â–ˆâ–ˆâ–‘â–‘â–‘â–‘â–‘â–‘â–‘â–‘â–‘â–‘â–‘â–‘â–‘â–‘â–‘â–‘â–‘â–‘  5%
```

---

## ğŸ“ˆ Success Metrics (KPIs)

### 2025 Targets

| Metrik | Mevcut | Q2 Hedef | Q4 Hedef |
|--------|--------|----------|----------|
| Aktif KullanÄ±cÄ± | 10K | 50K | 100K |
| API Calls/Day | 1M | 5M | 10M |
| Uptime | 99.5% | 99.9% | 99.95% |
| MTTR (dakika) | 15 | 10 | 5 |
| False Positive | 5% | 3% | 1% |
| Detection Accuracy | 92% | 95% | 98% |
| Customer Satisfaction | 4.2/5 | 4.5/5 | 4.8/5 |

---

## ğŸ¤ Partnership & Integration

### Planned Integrations

**Security Tools** ğŸ”§
- [ ] Splunk
- [ ] Elastic Security
- [ ] Microsoft Sentinel
- [ ] IBM QRadar
- [ ] Palo Alto Networks

**Cloud Providers** â˜ï¸
- [x] AWS
- [ ] Azure
- [ ] Google Cloud
- [ ] Oracle Cloud
- [ ] DigitalOcean

**Ticketing Systems** ğŸ«
- [ ] Jira
- [ ] ServiceNow
- [ ] Zendesk
- [ ] PagerDuty
- [ ] Opsgenie

**Communication** ğŸ’¬
- [x] Slack
- [ ] Microsoft Teams
- [ ] Discord
- [ ] Telegram
- [ ] Email (SMTP)

---

## ğŸ® Community & Events

### 2025 Events

**Q1**
- âœ… CyberGuard AI v2.0 Launch (Ocak)
- ğŸ“… Webinar: "AI in Cybersecurity" (Mart)

**Q2**
- ğŸ“… RSA Conference 2025 (Nisan)
- ğŸ“… Black Hat USA 2025 (Haziran)
- ğŸ“… Community Meetup - Ä°stanbul (MayÄ±s)

**Q3**
- ğŸ“… DEF CON 33 (AÄŸustos)
- ğŸ“… Annual Developer Conference (EylÃ¼l)

**Q4**
- ğŸ“… CyberGuard Summit 2025 (Ekim)
- ğŸ“… Year-end Hackathon (AralÄ±k)

---

## ğŸ“‹ Deprecation Schedule

### Planlanan KaldÄ±rmalar

| Ã–zellik | Deprecation | Removal | Alternatif |
|---------|-------------|---------|------------|
| API v1 | Q2 2025 | Q4 2025 | API v2 |
| Legacy Scanner | Q3 2025 | Q1 2026 | New Scanner Engine |
| Old Dashboard | Q2 2025 | Q3 2025 | Dashboard v2 |

---

## ğŸ’¬ Feedback & Suggestions

Roadmap hakkÄ±nda geri bildirimlerinizi paylaÅŸÄ±n:

- ğŸ’¬ [GitHub Discussions](https://github.com/cyberguard-ai/discussions)
- ğŸ“§ Email: roadmap@cyberguard-ai.com
- ğŸ¦ Twitter: @cyberguard_ai
- ğŸ’¬ Discord: discord.gg/cyberguard

---

## ğŸ”„ Roadmap Updates

Bu roadmap dÃ¼zenli olarak gÃ¼ncellenir:

- **HaftalÄ±k**: Progress updates
- **AylÄ±k**: Feature status changes
- **ÃœÃ§ AylÄ±k**: Major roadmap revisions

**Son GÃ¼ncelleme**: 2025-01-15  
**Sonraki Ä°nceleme**: 2025-04-01

---

## âš ï¸ Ã–nemli Notlar

- Roadmap, planlanan Ã¶zellikleri gÃ¶sterir ancak deÄŸiÅŸebilir
- Tarihler tahmindir ve garantili deÄŸildir
- Community feedback'e gÃ¶re Ã¶ncelikler deÄŸiÅŸebilir
- Enterprise mÃ¼ÅŸterileri iÃ§in Ã¶zel roadmap mevcut

---

**ğŸš€ GeleceÄŸi birlikte inÅŸa edelim!**

KatkÄ±da bulunmak iÃ§in [CONTRIBUTING.md](CONTRIBUTING.md) dosyasÄ±na bakÄ±n.

---


# SECURITY_POLICY

# ğŸ”’ Security Policy

CyberGuard AI gÃ¼venlik politikasÄ± ve aÃ§Ä±k bildirimi

---

## ğŸ“‹ Ä°Ã§indekiler

- [Desteklenen SÃ¼rÃ¼mler](#desteklenen-sÃ¼rÃ¼mler)
- [GÃ¼venlik AÃ§Ä±ÄŸÄ± Bildirimi](#gÃ¼venlik-aÃ§Ä±ÄŸÄ±-bildirimi)
- [Responsible Disclosure](#responsible-disclosure)
- [GÃ¼venlik Ã–nlemleri](#gÃ¼venlik-Ã¶nlemleri)
- [Bug Bounty](#bug-bounty)

---

## âœ… Desteklenen SÃ¼rÃ¼mler

| SÃ¼rÃ¼m | Destek |
|-------|--------|
| 3.x.x | âœ… Aktif destek |
| 2.x.x | âœ… GÃ¼venlik gÃ¼ncellemeleri |
| 1.x.x | âŒ Destek sona erdi |
| < 1.0 | âŒ Desteklenmiyor |

---

## ğŸ” GÃ¼venlik AÃ§Ä±ÄŸÄ± Bildirimi

### NasÄ±l Bildirilir?

âš ï¸ **Ã–NEMLÄ°**: GÃ¼venlik aÃ§Ä±klarÄ±nÄ± **PUBLIC** olarak bildirmeyin!

1. **Email**: <security@cyberguard-ai.com>
2. **GPG Key**: [Public Key](https://cyberguard-ai.com/security.gpg)
3. **HackerOne**: hackerone.com/cyberguard

### Bildirimde BulunmasÄ± Gerekenler

```
Konu: [SECURITY] <KÄ±sa aÃ§Ä±klama>

1. AÃ§Ä±ÄŸÄ±n TÃ¼rÃ¼: (XSS, SQL Injection, vb.)
2. Etkilenen BileÅŸen: (API, Frontend, Model, vb.)
3. Etkilenen SÃ¼rÃ¼m: 
4. AdÄ±m AdÄ±m Reproduce:
   1. ...
   2. ...
5. Beklenen DavranÄ±ÅŸ:
6. GerÃ§ekleÅŸen DavranÄ±ÅŸ:
7. Proof of Concept: (varsa)
8. Ã–nerilen DÃ¼zeltme: (varsa)
```

### YanÄ±t SÃ¼resi

| AÅŸama | SÃ¼re |
|-------|------|
| Ä°lk YanÄ±t | 24 saat |
| DeÄŸerlendirme | 72 saat |
| Fix (Critical) | 7 gÃ¼n |
| Fix (High) | 30 gÃ¼n |
| Fix (Medium) | 60 gÃ¼n |

---

## ğŸ“œ Responsible Disclosure

### Kurallar

1. âœ… Sadece kendi test sistemlerinizi kullanÄ±n
2. âœ… Verileri modifiye etmeyin veya silmeyin
3. âœ… Hizmet kesintisi yapmayÄ±n
4. âœ… BulduÄŸunuzu bize bildirin, baÅŸkalarÄ±na deÄŸil
5. âœ… Patch yayÄ±nlanana kadar bekleyin
6. âŒ ÃœÃ§Ã¼ncÃ¼ taraf verilere eriÅŸmeyin
7. âŒ DDoS veya brute force yapmayÄ±n

### Safe Harbor

Ä°yi niyetli gÃ¼venlik araÅŸtÄ±rmacÄ±larÄ±na karÅŸÄ± **yasal iÅŸlem baÅŸlatmayÄ±z**.

---

## ğŸ›¡ï¸ GÃ¼venlik Ã–nlemleri

### Uygulanan

| Ã–nlem | AÃ§Ä±klama |
|-------|----------|
| âœ… TLS 1.3 | TÃ¼m iletiÅŸimde |
| âœ… AES-256 | Veri ÅŸifreleme |
| âœ… JWT + Refresh | Kimlik doÄŸrulama |
| âœ… Rate Limiting | DoS korumasÄ± |
| âœ… Input Validation | Pydantic models |
| âœ… CORS | Origin kontrolÃ¼ |
| âœ… SQL Parameterization | Injection korumasÄ± |
| âœ… XSS Protection | CSP headers |
| âœ… CSRF Tokens | Form gÃ¼venliÄŸi |
| âœ… Dependency Scanning | Snyk/Dependabot |

### Planlanan

- [ ] Hardware Security Module (HSM)
- [ ] Zero Trust Architecture
- [ ] Quantum-resistant encryption

---

## ğŸ’° Bug Bounty

### Scope

**In Scope:**

- api.cyberguard-ai.com
- app.cyberguard-ai.com
- CyberGuard AI GitHub repo

**Out of Scope:**

- Third-party services
- Physical attacks
- Social engineering

### Ã–dÃ¼ller

| Severity | Ã–dÃ¼l |
|----------|------|
| Critical (9.0-10.0) | $1,000 - $5,000 |
| High (7.0-8.9) | $500 - $1,000 |
| Medium (4.0-6.9) | $100 - $500 |
| Low (0.1-3.9) | Hall of Fame |

### Hall of Fame

GÃ¼venlik aÃ§Ä±ÄŸÄ± bildiren araÅŸtÄ±rmacÄ±lar (izinleriyle):

- ğŸ† [Ä°sim] - Critical XSS (2025)
- ğŸ¥ˆ [Ä°sim] - IDOR (2025)

---

## ğŸ“ Ä°letiÅŸim

- **Security Email**: <security@cyberguard-ai.com>
- **GPG Key ID**: 0x1234567890ABCDEF
- **Response Time**: 24 saat iÃ§inde

---

## ğŸ“… Son GÃ¼ncelleme

2026-01-10


---


# SECURİTY

# ğŸ”’ GÃ¼venlik PolitikasÄ±

## ğŸ“‹ Ä°Ã§indekiler

- [Desteklenen Versiyonlar](#desteklenen-versiyonlar)
- [GÃ¼venlik AÃ§Ä±ÄŸÄ± Bildirimi](#gÃ¼venlik-aÃ§Ä±ÄŸÄ±-bildirimi)
- [GÃ¼venlik GÃ¼ncellemeleri](#gÃ¼venlik-gÃ¼ncellemeleri)
- [GÃ¼venlik En Ä°yi UygulamalarÄ±](#gÃ¼venlik-en-iyi-uygulamalarÄ±)
- [GÃ¼venlik Denetimi](#gÃ¼venlik-denetimi)

---

## ğŸ›¡ï¸ Desteklenen Versiyonlar

AÅŸaÄŸÄ±daki CyberGuard AI versiyonlarÄ± iÃ§in gÃ¼venlik gÃ¼ncellemeleri saÄŸlanmaktadÄ±r:

| Versiyon | Destek Durumu | Destek BitiÅŸ Tarihi |
|----------|---------------|---------------------|
| 2.0.x    | âœ… Tam Destek | 2026-01-15 |
| 1.5.x    | âœ… GÃ¼venlik Yamalar | 2025-06-20 |
| 1.0.x    | âš ï¸ Kritik Yamalar | 2025-01-01 |
| < 1.0    | âŒ Desteklenmiyor | - |

### Versiyon Destek PolitikasÄ±

- **Tam Destek**: TÃ¼m gÃ¼venlik ve bug fix'ler
- **GÃ¼venlik Yamalar**: Sadece kritik gÃ¼venlik yamalarÄ±
- **Kritik Yamalar**: Sadece kritik gÃ¼venlik aÃ§Ä±klarÄ±
- **Desteklenmiyor**: HiÃ§bir gÃ¼venlik gÃ¼ncellemesi yok

**Ã–nemli**: GÃ¼venlik iÃ§in her zaman en son stabil versiyonu kullanÄ±n!

---

## ğŸš¨ GÃ¼venlik AÃ§Ä±ÄŸÄ± Bildirimi

### Rapor Etme SÃ¼reci

Bir gÃ¼venlik aÃ§Ä±ÄŸÄ± bulduysanÄ±z, lÃ¼tfen **sorumlu bir ÅŸekilde bildirin**.

#### 1. ğŸ“§ Ã–zel Bildirim (Tercih Edilen)

GÃ¼venlik aÃ§Ä±klarÄ±nÄ± **ASLA** public issue'larda bildirmeyin!

**Email**: security@cyberguard-ai.com

**Åablon**:
```
Konu: [SECURITY] KÄ±sa AÃ§Ä±klama

# GÃ¼venlik AÃ§Ä±ÄŸÄ± Raporu

## Ã–zet
[AÃ§Ä±ÄŸÄ±n kÄ±sa aÃ§Ä±klamasÄ±]

## Etkilenen Versiyon(lar)
[Ã–rn: v2.0.0, v1.5.3]

## Zafiyet TÃ¼rÃ¼
[Ã–rn: SQL Injection, XSS, RCE, vb.]

## CVSS Skoru (varsa)
[Ã–rn: 9.8 - Critical]

## DetaylÄ± AÃ§Ä±klama
[Teknik detaylar]

## Tekrarlama AdÄ±mlarÄ± (PoC)
1. [AdÄ±m 1]
2. [AdÄ±m 2]
3. [AdÄ±m 3]

## Etki Analizi
[Bu aÃ§Ä±ÄŸÄ±n potansiyel etkileri]

## Ã–nerilen Ã‡Ã¶zÃ¼m
[Varsa Ã§Ã¶zÃ¼m Ã¶neriniz]

## Ek Bilgiler
- Ä°letiÅŸim: [Email/Twitter/LinkedIn]
- Disclosure Preference: [Koordineli, Public, vb.]
```

#### 2. ğŸ” PGP Encrypted Email (Hassas Durumlar)

Ã‡ok kritik aÃ§Ä±klar iÃ§in PGP ÅŸifreli email kullanÄ±n:

```
PGP Public Key Fingerprint:
1234 5678 90AB CDEF 1234 5678 90AB CDEF 1234 5678

PGP Key: https://keybase.io/cyberguard_ai
```

#### 3. ğŸ’¬ Bug Bounty Platform

KayÄ±tlÄ± gÃ¼venlik araÅŸtÄ±rmacÄ±larÄ± iÃ§in:
- **HackerOne**: https://hackerone.com/cyberguard-ai
- **Bugcrowd**: https://bugcrowd.com/cyberguard-ai

### YanÄ±t SÃ¼resi

| AÅŸama | SÃ¼re |
|-------|------|
| Ä°lk YanÄ±t | 24-48 saat |
| Ä°nceleme | 3-5 iÅŸ gÃ¼nÃ¼ |
| DÃ¼zeltme Tahmini | 7-30 gÃ¼n (kritiklik gÃ¶re) |
| Public Disclosure | 90 gÃ¼n (koordineli) |

### GÃ¼venlik AÃ§Ä±ÄŸÄ± Kritiklik Seviyeleri

**Critical (9.0-10.0)** ğŸ”´
- Remote Code Execution (RCE)
- Authentication Bypass
- SQL Injection (kritik)
- **SLA**: 24 saat iÃ§inde yama

**High (7.0-8.9)** ğŸŸ 
- Privilege Escalation
- Sensitive Data Exposure
- XSS (stored)
- **SLA**: 7 gÃ¼n iÃ§inde yama

**Medium (4.0-6.9)** ğŸŸ¡
- CSRF
- XSS (reflected)
- Information Disclosure
- **SLA**: 30 gÃ¼n iÃ§inde yama

**Low (0.1-3.9)** ğŸŸ¢
- Minor information leaks
- Best practice violations
- **SLA**: Bir sonraki release

---
## ğŸ“¢ GÃ¼venlik GÃ¼ncellemeleri

### Security Advisory AboneliÄŸi

GÃ¼venlik gÃ¼ncellemelerinden haberdar olmak iÃ§in:

1. **GitHub Watch**: "Security alerts only" seÃ§eneÄŸini aktif edin
2. **Mailing List**: security-announce@cyberguard-ai.com
3. **RSS Feed**: https://cyberguard-ai.com/security/feed
4. **Twitter**: @cyberguard_security

### GÃ¼venlik DuyurularÄ±

TÃ¼m gÃ¼venlik yamalarÄ± aÅŸaÄŸÄ±daki kanallarda duyurulur:

- ğŸ“§ Email: security-announce@cyberguard-ai.com
- ğŸ¦ Twitter: @cyberguard_security
- ğŸ“° Blog: https://blog.cyberguard-ai.com/security
- ğŸ“¢ GitHub Security Advisories

### CVE NumaralarÄ±

Ciddi gÃ¼venlik aÃ§Ä±klarÄ± iÃ§in CVE (Common Vulnerabilities and Exposures) numarasÄ± alÄ±nÄ±r ve ÅŸu platformlarda yayÄ±nlanÄ±r:

- NIST National Vulnerability Database
- MITRE CVE List
- GitHub Security Advisories

---

## ğŸ› ï¸ GÃ¼venlik En Ä°yi UygulamalarÄ±

### Kurulum GÃ¼venliÄŸi

**1. GÃ¼venli KonfigÃ¼rasyon**

```bash
# âŒ ASLA production'da default ÅŸifreler kullanmayÄ±n!
# âŒ KÃ–TÃœ
DB_PASSWORD=admin123
API_KEY=default_key

# âœ… Ä°YÄ°
DB_PASSWORD=$(openssl rand -base64 32)
API_KEY=$(uuidgen)
```

**2. Environment Variables**

```bash
# .env dosyasÄ±nÄ± ASLA commit etmeyin!
# .gitignore'a ekleyin
echo ".env" >> .gitignore

# .env.example kullanÄ±n
cp .env.example .env
# DeÄŸerleri gÃ¼ncelleyin
```

**3. HTTPS KullanÄ±mÄ±**

```yaml
# config/security.yaml
server:
  ssl:
    enabled: true
    cert: /path/to/cert.pem
    key: /path/to/key.pem
    min_version: TLSv1.3
```

**4. Firewall KurallarÄ±**

```bash
# Sadece gerekli portlarÄ± aÃ§Ä±n
ufw allow 443/tcp  # HTTPS
ufw allow 22/tcp   # SSH (IP whitelist ile)
ufw enable
```

### Uygulama GÃ¼venliÄŸi

**1. Input Validation**

```python
# âœ… Ä°YÄ°: Her input'u validate edin
from pydantic import BaseModel, validator

class ScanRequest(BaseModel):
    target: str
    
    @validator('target')
    def validate_target(cls, v):
        if not is_valid_ip(v) and not is_valid_domain(v):
            raise ValueError('Invalid target')
        return v
```

**2. SQL Injection Protection**

```python
# âŒ KÃ–TÃœ: String concatenation
query = f"SELECT * FROM users WHERE id = {user_id}"

# âœ… Ä°YÄ°: Parameterized queries
query = "SELECT * FROM users WHERE id = %s"
cursor.execute(query, (user_id,))
```

**3. XSS Protection**

```python
# âœ… Output encoding
from markupsafe import escape

user_input = escape(user_input)
```

**4. Authentication**

```python
# âœ… GÃ¼Ã§lÃ¼ ÅŸifre politikasÄ±
from passlib.hash import argon2

# Argon2 kullanÄ±n (bcrypt'ten daha gÃ¼venli)
hashed = argon2.hash(password)
```

**5. Rate Limiting**

```python
# âœ… API rate limiting
from flask_limiter import Limiter

limiter = Limiter(
    app,
    default_limits=["100 per hour", "10 per minute"]
)
```

### Database GÃ¼venliÄŸi

```sql
-- âœ… Minimum privilege principle
CREATE USER 'cyberguard_app'@'localhost' 
IDENTIFIED BY 'secure_password';

GRANT SELECT, INSERT, UPDATE 
ON cyberguard.* 
TO 'cyberguard_app'@'localhost';

-- Database encryption at rest
ALTER TABLE sensitive_data 
ENCRYPTION='Y';
```

### Logging ve Monitoring

```python
# âœ… GÃ¼venlik olaylarÄ±nÄ± logla
import logging

logger = logging.getLogger('security')

# Failed login attempts
logger.warning(f"Failed login: {username} from {ip}")

# Successful privilege escalation
logger.critical(f"Privilege escalation: {user} -> admin")

# ASLA hassas bilgileri loglama!
# âŒ KÃ–TÃœ
logger.info(f"Password: {password}")

# âœ… Ä°YÄ°
logger.info(f"Password changed for user: {user_id}")
```

---

## ğŸ” GÃ¼venlik Denetimi

### Otomatik GÃ¼venlik TaramalarÄ±

**1. Dependency Scanning**

```bash
# Python dependencies
pip-audit

# GitHub Dependabot
# .github/dependabot.yml
version: 2
updates:
  - package-ecosystem: "pip"
    directory: "/"
    schedule:
      interval: "daily"
```

**2. SAST (Static Application Security Testing)**

```bash
# Bandit - Python security linter
bandit -r src/

# SonarQube
sonar-scanner
```

**3. DAST (Dynamic Application Security Testing)**

```bash
# OWASP ZAP
zap-cli quick-scan https://cyberguard-ai.com

# Burp Suite Professional
```

**4. Container Security**

```bash
# Trivy - Container vulnerability scanner
trivy image cyberguard-ai:latest

# Snyk
snyk container test cyberguard-ai:latest
```

### Manuel GÃ¼venlik Testleri

**Periyodik Denetimler:**

- ğŸ“… **HaftalÄ±k**: Dependency updates
- ğŸ“… **AylÄ±k**: Vulnerability scanning
- ğŸ“… **ÃœÃ§ AylÄ±k**: Penetration testing
- ğŸ“… **YÄ±llÄ±k**: Full security audit

### Security Checklist

- [ ] TÃ¼m dependencies gÃ¼ncel mi?
- [ ] Known vulnerabilities var mÄ±?
- [ ] SSL/TLS doÄŸru yapÄ±landÄ±rÄ±lmÄ±ÅŸ mÄ±?
- [ ] Authentication gÃ¼Ã§lÃ¼ mÃ¼?
- [ ] Logging ve monitoring aktif mi?
- [ ] Backup stratejisi var mÄ±?
- [ ] Incident response planÄ± hazÄ±r mÄ±?
- [ ] Security training yapÄ±ldÄ± mÄ±?

---

## ğŸ“Š GÃ¼venlik Metrikleri

GÃ¼venlik durumumuzu ÅŸu metriklerle takip ediyoruz:

| Metrik | Hedef | Mevcut |
|--------|-------|--------|
| Mean Time to Detect (MTTD) | < 1 saat | 45 dakika |
| Mean Time to Respond (MTTR) | < 4 saat | 3.5 saat |
| Vulnerability Backlog | < 10 | 5 |
| Security Test Coverage | > 80% | 85% |
| False Positive Rate | < 5% | 3% |

---

## ğŸ“ GÃ¼venlik EÄŸitimi

TÃ¼m geliÅŸtiricilerin tamamlamasÄ± gereken:

1. **OWASP Top 10** (yÄ±llÄ±k)
2. **Secure Coding Practices** (yÄ±llÄ±k)
3. **Security Awareness Training** (6 ayda bir)
4. **Incident Response Training** (yÄ±llÄ±k)

---

## ğŸ“ Ä°letiÅŸim

### GÃ¼venlik Ekibi

- ğŸ“§ **Genel**: security@cyberguard-ai.com
- ğŸš¨ **Acil**: security-urgent@cyberguard-ai.com
- ğŸ” **PGP Key**: https://keybase.io/cyberguard_security

### Ã‡alÄ±ÅŸma Saatleri

- **Ä°ÅŸ GÃ¼nleri**: 09:00 - 18:00 (UTC+3)
- **Acil Durumlar**: 7/24 on-call team

---

## ğŸ“š Kaynaklar

### Standartlar ve Frameworks

- [OWASP Top 10](https://owasp.org/www-project-top-ten/)
- [CWE Top 25](https://cwe.mitre.org/top25/)
- [NIST Cybersecurity Framework](https://www.nist.gov/cyberframework)
- [ISO 27001](https://www.iso.org/isoiec-27001-information-security.html)

### GÃ¼venlik AraÃ§larÄ±

- [Bandit](https://github.com/PyCQA/bandit) - Python security linter
- [OWASP ZAP](https://www.zaproxy.org/) - Web app security scanner
- [Trivy](https://github.com/aquasecurity/trivy) - Container scanner
- [SonarQube](https://www.sonarqube.org/) - Code quality & security

---

## âš–ï¸ Yasal UyarÄ±

CyberGuard AI, sorumlu gÃ¼venlik araÅŸtÄ±rmalarÄ±nÄ± destekler ve aÅŸaÄŸÄ±daki koÅŸullarda yasal iÅŸlem baÅŸlatmayacaÄŸÄ±nÄ± taahhÃ¼t eder:

- âœ… AÃ§Ä±k, sorumlu ÅŸekilde bildirildiÄŸinde
- âœ… Test, belirlenen kapsamda yapÄ±ldÄ±ÄŸÄ±nda
- âœ… Veri Ã§alÄ±nmadÄ±ÄŸÄ±nda veya tahrip edilmediÄŸinde
- âœ… DoS/DDoS saldÄ±rÄ±sÄ± yapÄ±lmadÄ±ÄŸÄ±nda

---

**Son GÃ¼ncelleme**: 2025-01-15  
**Versiyon**: 2.0  
**Sonraki Ä°nceleme**: 2025-07-15

---

**ğŸ”’ GÃ¼venlik, hepimizin sorumluluÄŸudur. Birlikte daha gÃ¼venli bir dijital dÃ¼nya oluÅŸturalÄ±m!**

---


# SECURİTY_HUB

# ğŸ›¡ï¸ Security Hub DokÃ¼mantasyonu

KapsamlÄ± gÃ¼venlik izleme ve analiz merkezi

---

## ğŸ“‹ Ä°Ã§indekiler

- [Security Score](#security-score)
- [Honeypot](#honeypot)
- [Compliance](#compliance)
- [Network Topology](#network-topology)
- [Threat Heatmap](#threat-heatmap)
- [Attack Replay](#attack-replay)
- [Vulnerability Scanner](#vulnerability-scanner)

---

## ğŸ“Š Security Score

### Genel BakÄ±ÅŸ

Sistemin genel gÃ¼venlik durumunu 0-100 arasÄ± bir skor olarak hesaplar.

### API Endpoint

```
GET /api/security/score
```

### BileÅŸenler

| BileÅŸen | AÄŸÄ±rlÄ±k |
|---------|---------|
| Network Security | 25% |
| Endpoint Protection | 20% |
| Application Security | 20% |
| Data Protection | 15% |
| Access Control | 20% |

### Derece Sistemi

- **A (90-100)**: Excellent
- **B (80-89)**: Good
- **C (70-79)**: Fair
- **D (60-69)**: Poor
- **F (0-59)**: Critical

---

## ğŸ¯ Honeypot

Sahte servisler ile saldÄ±rganlarÄ± tespit etme sistemi.

### Desteklenen Honeypot TÃ¼rleri

| TÃ¼r | Port | AÃ§Ä±klama |
|-----|------|----------|
| SSH | 22 | SSH brute force tespiti |
| HTTP | 80 | Web saldÄ±rÄ± tespiti |
| FTP | 21 | Dosya transfer saldÄ±rÄ±larÄ± |
| RDP | 3389 | Remote desktop saldÄ±rÄ±larÄ± |

### API Endpoint

```
GET /api/security/honeypot
```

### Metrikler

- Yakalanan saldÄ±rÄ± sayÄ±sÄ±
- Unique saldÄ±rgan IP'ler
- En son saldÄ±rÄ± zamanÄ±
- Yakalanan credential'lar

---

## âœ… Compliance

GÃ¼venlik standartlarÄ±na uyumluluk durumu.

### Desteklenen Standartlar

- **GDPR**: EU veri koruma
- **HIPAA**: SaÄŸlÄ±k verisi gÃ¼venliÄŸi
- **PCI-DSS**: Ã–deme kartÄ± gÃ¼venliÄŸi
- **ISO 27001**: Bilgi gÃ¼venliÄŸi yÃ¶netimi
- **NIST**: Siber gÃ¼venlik Ã§erÃ§evesi
- **SOC 2**: Servis organizasyonu kontrolÃ¼
- **KVKK**: KiÅŸisel verilerin korunmasÄ±

### API Endpoint

```
GET /api/security/compliance
```

---

## ğŸŒ Network Topology

AÄŸ yapÄ±sÄ±nÄ±n gÃ¶rselleÅŸtirilmesi.

### API Endpoint

```
GET /api/security/topology
```

### Response Format

```json
{
  "nodes": [
    {"id": "router-main", "type": "router", "label": "Main Router"}
  ],
  "edges": [
    {"from": "router-main", "to": "firewall", "status": "active"}
  ]
}
```

### Desteklenen Cihaz TÃ¼rleri

- Router
- Firewall
- Switch
- Server
- Workstation

---

## ğŸ—ºï¸ Threat Heatmap

CoÄŸrafi tehdit daÄŸÄ±lÄ±mÄ±.

### API Endpoint

```
GET /api/security/heatmap
```

### Ã–zellikler

- Ãœlke bazlÄ± saldÄ±rÄ± sayÄ±sÄ±
- YoÄŸunluk gÃ¶sterimi
- Top saldÄ±rÄ± tÃ¼rleri
- Trend analizi

---

## â±ï¸ Attack Replay

GeÃ§miÅŸ saldÄ±rÄ±larÄ± yeniden oynatma ve analiz.

### API Endpoint

```
GET /api/security/attack-replay
```

### Ã–zellikler

- SaldÄ±rÄ± timeline
- Paket analizi
- SaldÄ±rÄ± aÅŸamalarÄ±
- EÄŸitim amaÃ§lÄ± replay

---

## ğŸ” Vulnerability Scanner

Port tarama ve CVE kontrolÃ¼.

### API Endpoints

```
POST /api/vuln/scan
POST /api/vuln/port-scan
GET /api/vuln/cve/{cve_id}
GET /api/vuln/history
```

### Tarama TÃ¼rleri

| TÃ¼r | AÃ§Ä±klama |
|-----|----------|
| Quick | HÄ±zlÄ±, temel portlar |
| Full | TÃ¼m portlar |
| Deep | DetaylÄ± analiz |

### Tespit Edilenler

- AÃ§Ä±k portlar
- Servis versiyonlarÄ±
- Bilinen CVE'ler
- GÃ¼venlik aÃ§Ä±klarÄ±

---

## ğŸ’» KullanÄ±m

### Security Score Alma

```python
response = requests.get("/api/security/score")
score = response.json()["data"]
print(f"Score: {score['overall_score']} ({score['grade']})")
```

### Vulnerability Scan

```python
response = requests.post("/api/vuln/scan", json={
    "target": "192.168.1.100",
    "scan_type": "full"
})
vulns = response.json()["data"]["vulnerabilities"]
```

---

## ğŸ“ Referanslar

- [OWASP Testing Guide](https://owasp.org/www-project-web-security-testing-guide/)
- [NIST Cybersecurity Framework](https://www.nist.gov/cyberframework)
- [CIS Controls](https://www.cisecurity.org/controls)


---


# TESTİNG

# ğŸ§ª Testing Guide

CyberGuard AI test stratejisi ve komutlarÄ±

---

## ğŸ“‹ Ä°Ã§indekiler

- [Test TÃ¼rleri](#test-tÃ¼rleri)
- [Kurulum](#kurulum)
- [Unit Tests](#unit-tests)
- [Integration Tests](#integration-tests)
- [E2E Tests](#e2e-tests)
- [ML Model Tests](#ml-model-tests)
- [Performance Tests](#performance-tests)
- [CI/CD Entegrasyonu](#cicd-entegrasyonu)

---

## ğŸ¯ Test TÃ¼rleri

| TÃ¼r | Kapsam | AraÃ§ | SÃ¼re |
|-----|--------|------|------|
| Unit | Fonksiyon | pytest | Saniye |
| Integration | ModÃ¼l | pytest | Dakika |
| E2E | Sistem | Cypress/Playwright | Dakika |
| ML | Model | pytest + sklearn | Dakika |
| Performance | Load | Locust/k6 | Dakika |

---

## ğŸ”§ Kurulum

```bash
# Test baÄŸÄ±mlÄ±lÄ±klarÄ±
pip install pytest pytest-cov pytest-asyncio httpx

# Frontend testleri
cd frontend
npm install -D vitest @testing-library/react
```

---

## ğŸ”¬ Unit Tests

### Ã‡alÄ±ÅŸtÄ±rma

```bash
# TÃ¼m unit testler
pytest tests/unit/ -v

# Coverage ile
pytest tests/unit/ --cov=app --cov-report=html

# Belirli dosya
pytest tests/unit/test_predictor.py -v

# Belirli test
pytest tests/unit/test_predictor.py::test_model_load -v
```

### Ã–rnek Test

```python
# tests/unit/test_predictor.py
import pytest
from src.models.predictor import AttackPredictor

class TestAttackPredictor:
    
    @pytest.fixture
    def predictor(self):
        return AttackPredictor()
    
    def test_model_load(self, predictor):
        """Model yÃ¼kleme testi"""
        predictor.load_models()
        assert predictor.model is not None
    
    def test_predict_single(self, predictor):
        """Tek tahmin testi"""
        predictor.load_models()
        features = [0.1] * 78  # 78 feature
        result = predictor.predict_single(features)
        
        assert 'predicted_type' in result
        assert 'confidence' in result
        assert 0 <= result['confidence'] <= 1
    
    def test_invalid_input(self, predictor):
        """GeÃ§ersiz girdi testi"""
        predictor.load_models()
        
        with pytest.raises(ValueError):
            predictor.predict_single([0.1] * 10)  # Eksik feature
```

---

## ğŸ”— Integration Tests

### Ã‡alÄ±ÅŸtÄ±rma

```bash
pytest tests/integration/ -v
```

### API Test Ã–rneÄŸi

```python
# tests/integration/test_api.py
import pytest
from httpx import AsyncClient
from app.main import app

@pytest.fixture
async def client():
    async with AsyncClient(app=app, base_url="http://test") as client:
        yield client

@pytest.mark.asyncio
async def test_health_check(client):
    response = await client.get("/")
    assert response.status_code == 200
    assert response.json()["message"] == "ğŸ›¡ï¸ CyberGuard AI API"

@pytest.mark.asyncio
async def test_dashboard(client):
    response = await client.get("/api/dashboard")
    assert response.status_code == 200
    assert response.json()["success"] == True

@pytest.mark.asyncio
async def test_prediction(client):
    response = await client.post("/api/prediction/predict", json={
        "features": [0.1] * 78,
        "model_id": "best_cicids2017"
    })
    assert response.status_code == 200
    assert "predicted_type" in response.json()["data"]
```

---

## ğŸŒ E2E Tests

### Playwright Kurulum

```bash
npm install -D @playwright/test
npx playwright install
```

### E2E Test Ã–rneÄŸi

```typescript
// tests/e2e/dashboard.spec.ts
import { test, expect } from '@playwright/test';

test.describe('Dashboard', () => {
    test('should load dashboard', async ({ page }) => {
        await page.goto('http://localhost:5173/dashboard');
        await expect(page.locator('h1')).toContainText('Dashboard');
    });
    
    test('should show attack statistics', async ({ page }) => {
        await page.goto('http://localhost:5173/dashboard');
        await expect(page.locator('.stat-card')).toHaveCount(4);
    });
    
    test('should navigate to prediction', async ({ page }) => {
        await page.goto('http://localhost:5173/dashboard');
        await page.click('text=Prediction');
        await expect(page).toHaveURL(/.*prediction/);
    });
});
```

### Ã‡alÄ±ÅŸtÄ±rma

```bash
npx playwright test
npx playwright test --ui
npx playwright test --headed
```

---

## ğŸ§  ML Model Tests

```python
# tests/ml/test_model_performance.py
import pytest
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from src.models.predictor import AttackPredictor

class TestModelPerformance:
    
    @pytest.fixture
    def test_data(self):
        # Test verisi yÃ¼kle
        X_test = np.load("data/test/X_test.npy")
        y_test = np.load("data/test/y_test.npy")
        return X_test, y_test
    
    def test_accuracy_threshold(self, test_data):
        """Accuracy %95 Ã¼stÃ¼nde olmalÄ±"""
        X_test, y_test = test_data
        predictor = AttackPredictor()
        predictor.load_models()
        
        y_pred = predictor.predict_batch(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        assert accuracy >= 0.95, f"Accuracy {accuracy:.2%} < 95%"
    
    def test_f1_score(self, test_data):
        """F1-Score %90 Ã¼stÃ¼nde olmalÄ±"""
        X_test, y_test = test_data
        predictor = AttackPredictor()
        predictor.load_models()
        
        y_pred = predictor.predict_batch(X_test)
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        assert f1 >= 0.90
    
    def test_inference_time(self, test_data):
        """Inference 100ms altÄ±nda olmalÄ±"""
        import time
        X_test, _ = test_data
        predictor = AttackPredictor()
        predictor.load_models()
        
        start = time.time()
        predictor.predict_single(X_test[0])
        elapsed = time.time() - start
        
        assert elapsed < 0.1, f"Inference {elapsed*1000:.0f}ms > 100ms"
```

---

## âš¡ Performance Tests

### Locust Kurulum

```bash
pip install locust
```

### Locust Test

```python
# tests/performance/locustfile.py
from locust import HttpUser, task, between

class CyberGuardUser(HttpUser):
    wait_time = between(1, 3)
    
    @task(3)
    def get_dashboard(self):
        self.client.get("/api/dashboard")
    
    @task(2)
    def get_attacks(self):
        self.client.get("/api/network/attacks")
    
    @task(1)
    def predict(self):
        self.client.post("/api/prediction/predict", json={
            "features": [0.1] * 78
        })
```

### Ã‡alÄ±ÅŸtÄ±rma

```bash
# Web UI
locust -f tests/performance/locustfile.py

# Headless
locust -f tests/performance/locustfile.py \
    --headless -u 100 -r 10 -t 1m \
    --host http://localhost:8000
```

---

## ğŸ”„ CI/CD Entegrasyonu

### GitHub Actions

```yaml
# .github/workflows/test.yml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'
    
    - name: Install dependencies
      run: |
        pip install -r requirements.txt
        pip install pytest pytest-cov
    
    - name: Run tests
      run: pytest tests/ -v --cov=app
    
    - name: Upload coverage
      uses: codecov/codecov-action@v3
```

---

## ğŸ“Š Coverage Hedefleri

| ModÃ¼l | Hedef |
|-------|-------|
| Models | 90% |
| API Routes | 85% |
| Utils | 80% |
| Frontend | 75% |
| **Toplam** | **80%** |

---

## ğŸš€ Test KomutlarÄ± Ã–zeti

```bash
# TÃ¼m testler
pytest

# Coverage
pytest --cov=app --cov-report=html

# Watch mode
pytest --watch

# Parallel
pytest -n auto

# Failed only
pytest --lf

# Verbose
pytest -v

# Frontend
npm test
npm run test:coverage
```


---


# TROUBLESHOOTİNG

# ğŸ”§ Troubleshooting Guide

CyberGuard AI sorun giderme rehberi

---

## ğŸ“‹ Ä°Ã§indekiler

- [Kurulum SorunlarÄ±](#kurulum-sorunlarÄ±)
- [Backend SorunlarÄ±](#backend-sorunlarÄ±)
- [Frontend SorunlarÄ±](#frontend-sorunlarÄ±)
- [Database SorunlarÄ±](#database-sorunlarÄ±)
- [Model SorunlarÄ±](#model-sorunlarÄ±)
- [API SorunlarÄ±](#api-sorunlarÄ±)
- [Performans SorunlarÄ±](#performans-sorunlarÄ±)

---

## ğŸ”§ Kurulum SorunlarÄ±

### ModuleNotFoundError: No module named 'xxx'

**Sebep**: BaÄŸÄ±mlÄ±lÄ±k eksik veya virtual environment aktif deÄŸil.

**Ã‡Ã¶zÃ¼m:**

```bash
# Virtual environment aktif et
# Windows:
.\venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

# BaÄŸÄ±mlÄ±lÄ±klarÄ± yÃ¼kle
pip install -r requirements.txt

# Tek modÃ¼l
pip install <module_name>
```

### pip install baÅŸarÄ±sÄ±z oluyor

**Sebep**: Network, yetki veya versiyon uyumsuzluÄŸu.

**Ã‡Ã¶zÃ¼m:**

```bash
# pip gÃ¼ncelle
pip install --upgrade pip

# Cache temizle
pip cache purge

# Alternatif mirror
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# Verbose mode
pip install -r requirements.txt -v
```

### npm install baÅŸarÄ±sÄ±z oluyor

**Sebep**: Node versiyonu, network veya cache.

**Ã‡Ã¶zÃ¼m:**

```bash
# Node versiyonu kontrol
node --version  # >= 18.0.0 gerekli

# Cache temizle
npm cache clean --force
rm -rf node_modules package-lock.json
npm install

# Alternative package manager
yarn install   # veya
pnpm install
```

### CUDA/GPU bulunamÄ±yor

**Sebep**: CUDA toolkit kurulu deÄŸil veya sÃ¼rÃ¼m uyumsuz.

**Ã‡Ã¶zÃ¼m:**

```bash
# CUDA kontrol
nvidia-smi
nvcc --version

# CPU modunda Ã§alÄ±ÅŸtÄ±r
CUDA_VISIBLE_DEVICES="" python app/main.py

# TensorFlow GPU
pip install tensorflow[and-cuda]
```

---

## ğŸ–¥ï¸ Backend SorunlarÄ±

### Port zaten kullanÄ±mda (Address already in use)

**Sebep**: BaÅŸka bir iÅŸlem portu kullanÄ±yor.

**Ã‡Ã¶zÃ¼m:**

```bash
# Windows - Port kullanan iÅŸlemi bul
netstat -ano | findstr :8000
# PID'yi bul ve sonlandÄ±r
taskkill /PID <PID> /F

# Linux/macOS
lsof -i :8000
kill -9 <PID>

# Alternatif port kullan
uvicorn main:app --port 8001
```

### uvicorn baÅŸlatÄ±lamÄ±yor

**Sebep**: Import hatasÄ± veya syntax error.

**Ã‡Ã¶zÃ¼m:**

```bash
# Syntax kontrol
python -m py_compile app/main.py

# Import kontrol
python -c "from app.main import app"

# Verbose mode
uvicorn main:app --reload --log-level debug
```

### CORS hatasÄ±

**Sebep**: Frontend origin'i backend'de tanÄ±mlÄ± deÄŸil.

**Ã‡Ã¶zÃ¼m:**

```python
# app/main.py
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

### Internal Server Error (500)

**Sebep**: Backend'de exception.

**Ã‡Ã¶zÃ¼m:**

```bash
# Log kontrol
tail -f logs/app.log

# Debug mode
DEBUG=true python -m uvicorn main:app --reload

# Exception detayÄ±
# Response body'de traceback olacak
```

---

## ğŸ¨ Frontend SorunlarÄ±

### Blank page / Nothing renders

**Sebep**: JavaScript error veya build hatasÄ±.

**Ã‡Ã¶zÃ¼m:**

```bash
# Console kontrol (F12)
# Build yeniden
npm run build
npm run dev

# Cache temizle
rm -rf .vite dist node_modules/.vite
npm run dev
```

### API calls failing

**Sebep**: Backend Ã§alÄ±ÅŸmÄ±yor veya URL yanlÄ±ÅŸ.

**Ã‡Ã¶zÃ¼m:**

```javascript
// services/api.js kontrol
const API_URL = 'http://localhost:8000/api';

// Proxy kullan (vite.config.js)
export default defineConfig({
  server: {
    proxy: {
      '/api': 'http://localhost:8000'
    }
  }
})
```

### Slow page load

**Sebep**: Bundle bÃ¼yÃ¼k veya network yavaÅŸ.

**Ã‡Ã¶zÃ¼m:**

```bash
# Bundle analizi
npm run build -- --analyze

# Lazy loading kullan
const Component = React.lazy(() => import('./Component'));
```

---

## ğŸ—„ï¸ Database SorunlarÄ±

### PostgreSQL baÄŸlantÄ± hatasÄ±

**Sebep**: Servis Ã§alÄ±ÅŸmÄ±yor veya credentials yanlÄ±ÅŸ.

**Ã‡Ã¶zÃ¼m:**

```bash
# Servis kontrol
# Windows:
pg_isready
# Linux:
sudo systemctl status postgresql

# BaÄŸlantÄ± test
psql -U postgres -h localhost -d cyberguard

# .env kontrol
DATABASE_URL=postgresql://user:password@localhost:5432/cyberguard
```

### Migration hatasÄ±

**Sebep**: Schema mismatch veya migration dosyasÄ± eksik.

**Ã‡Ã¶zÃ¼m:**

```bash
# Migration durumu
alembic current
alembic history

# Migration oluÅŸtur
alembic revision --autogenerate -m "description"

# Upgrade
alembic upgrade head

# Rollback
alembic downgrade -1
```

### Database full / Disk space

**Sebep**: Log veya eski veri birikimi.

**Ã‡Ã¶zÃ¼m:**

```sql
-- PostgreSQL vacuum
VACUUM FULL;

-- Eski verileri sil
DELETE FROM attacks WHERE created_at < NOW() - INTERVAL '90 days';

-- Table size kontrol
SELECT pg_size_pretty(pg_total_relation_size('attacks'));
```

---

## ğŸ§  Model SorunlarÄ±

### Model yÃ¼klenemiyor

**Sebep**: Model dosyasÄ± eksik veya corrupt.

**Ã‡Ã¶zÃ¼m:**

```bash
# Model dosyasÄ±nÄ± kontrol
ls -la models/production/

# Yeniden indir
python scripts/download_models.py

# Manuel yÃ¼kle
python -c "from tensorflow import keras; keras.models.load_model('models/production/best_model.h5')"
```

### Out of Memory (OOM)

**Sebep**: Model veya batch size Ã§ok bÃ¼yÃ¼k.

**Ã‡Ã¶zÃ¼m:**

```python
# Batch size kÃ¼Ã§Ã¼lt
model.predict(X, batch_size=32)

# GPU memory limit
import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)

# Veya sabit limit
tf.config.set_logical_device_configuration(
    gpus[0],
    [tf.config.LogicalDeviceConfiguration(memory_limit=4096)]
)
```

### YanlÄ±ÅŸ tahminler

**Sebep**: Veri Ã¶n iÅŸleme uyumsuzluÄŸu veya model drift.

**Ã‡Ã¶zÃ¼m:**

1. AynÄ± scaler kullanÄ±ldÄ±ÄŸÄ±ndan emin ol
2. Feature sÄ±ralamasÄ±nÄ± kontrol et
3. Model versiyonunu kontrol et
4. Drift detection Ã§alÄ±ÅŸtÄ±r

---

## ğŸ”Œ API SorunlarÄ±

### 401 Unauthorized

**Sebep**: Token eksik veya geÃ§ersiz.

**Ã‡Ã¶zÃ¼m:**

```bash
# Token al
curl -X POST http://localhost:8000/api/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username": "admin", "password": "admin123"}'

# Token kullan
curl http://localhost:8000/api/dashboard \
  -H "Authorization: Bearer <token>"
```

### 429 Too Many Requests

**Sebep**: Rate limit aÅŸÄ±ldÄ±.

**Ã‡Ã¶zÃ¼m:**

```bash
# Rate limit bilgisi
curl -I http://localhost:8000/api/attacks
# X-RateLimit-Remaining header'Ä±nÄ± kontrol et

# Bekle veya limit artÄ±r
```

### Timeout

**Sebep**: Ä°ÅŸlem Ã§ok uzun sÃ¼rÃ¼yor.

**Ã‡Ã¶zÃ¼m:**

```bash
# Timeout artÄ±r
curl --max-time 120 http://localhost:8000/api/long-operation

# Background job kullan
POST /api/jobs/start -> {"job_id": "xxx"}
GET /api/jobs/status/xxx -> {"status": "completed"}
```

---

## âš¡ Performans SorunlarÄ±

### YavaÅŸ API response

**Ã‡Ã¶zÃ¼m:**

```python
# Database indexleri
CREATE INDEX idx_attacks_created ON attacks(created_at);
CREATE INDEX idx_attacks_type ON attacks(attack_type);

# Query optimizasyonu
# N+1 query'lerden kaÃ§Ä±n

# Caching
from functools import lru_cache
@lru_cache(maxsize=100)
def get_stats():
    ...
```

### YÃ¼ksek CPU kullanÄ±mÄ±

**Ã‡Ã¶zÃ¼m:**

```bash
# Process kontrol
htop / top

# Model warmup
python -c "from src.models.predictor import AttackPredictor; p = AttackPredictor(); p.load_models()"

# Worker sayÄ±sÄ±
uvicorn main:app --workers 4
```

### YÃ¼ksek memory kullanÄ±mÄ±

**Ã‡Ã¶zÃ¼m:**

```bash
# Memory profiling
pip install memory_profiler
python -m memory_profiler app/main.py

# Garbage collection
import gc
gc.collect()

# Model unload
del model
keras.backend.clear_session()
```

---

## ğŸ“ Daha Fazla YardÄ±m

Sorununuz Ã§Ã¶zÃ¼lmediyse:

1. **GitHub Issues**: github.com/salihoglueyup/CyberGuard_AI/issues
2. **Discord**: discord.gg/cyberguard
3. **Email**: <support@cyberguard-ai.com>

**Log dosyalarÄ±nÄ± paylaÅŸmayÄ± unutmayÄ±n!**


---


# USER_GUİDE

# ğŸ“– User Guide

CyberGuard AI KullanÄ±m KÄ±lavuzu

---

## ğŸ“‹ Ä°Ã§indekiler

- [GiriÅŸ](#giriÅŸ)
- [HÄ±zlÄ± BaÅŸlangÄ±Ã§](#hÄ±zlÄ±-baÅŸlangÄ±Ã§)
- [Temel Ã–zellikler](#temel-Ã¶zellikler)
- [Chatbot KullanÄ±mÄ±](#chatbot-kullanÄ±mÄ±)
- [GÃ¼venlik Analizi](#gÃ¼venlik-analizi)
- [Raporlama](#raporlama)
- [Ayarlar ve KonfigÃ¼rasyon](#ayarlar-ve-konfigÃ¼rasyon)
- [Sorun Giderme](#sorun-giderme)
- [SSS](#sss)

---

## ğŸ¯ GiriÅŸ

CyberGuard AI, yapay zeka destekli siber gÃ¼venlik Ã§Ã¶zÃ¼mÃ¼ sunan kapsamlÄ± bir platformdur. Bu kÄ±lavuz, sistemin tÃ¼m Ã¶zelliklerini etkili bir ÅŸekilde kullanmanÄ±za yardÄ±mcÄ± olacaktÄ±r.

### Hedef Kitle

- ğŸ”’ Siber GÃ¼venlik UzmanlarÄ±
- ğŸ’¼ IT YÃ¶neticileri
- ğŸ›¡ï¸ SOC Analistleri
- ğŸ‘¨â€ğŸ’» Sistem YÃ¶neticileri

---

## ğŸš€ HÄ±zlÄ± BaÅŸlangÄ±Ã§

### Ä°lk Kurulum

1. **Sisteme GiriÅŸ**
   ```bash
   # Web arayÃ¼zÃ¼ne eriÅŸim
   http://localhost:5000
   
   # VarsayÄ±lan kullanÄ±cÄ± bilgileri
   Username: admin
   Password: admin123
   ```

2. **Ä°lk YapÄ±landÄ±rma**
    - Dashboard'a gidin
    - Ayarlar menÃ¼sÃ¼nden temel konfigÃ¼rasyonu yapÄ±n
    - API anahtarlarÄ±nÄ±zÄ± tanÄ±mlayÄ±n

3. **Ä°lk Tarama**
    - "New Scan" butonuna tÄ±klayÄ±n
    - Hedef sistem bilgilerini girin
    - Tarama tipini seÃ§in
    - BaÅŸlat!

### Dashboard Gezintisi

```
â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
â”‚  CyberGuard AI Dashboard            â”‚
â”œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¤
â”‚  ğŸ“Š Statistics                      â”‚
â”‚  â”œâ”€ Active Threats: 0               â”‚
â”‚  â”œâ”€ Total Scans: 0                  â”‚
â”‚  â””â”€ System Health: 100%             â”‚
â”‚                                      â”‚
â”‚  ğŸ¤– AI Chatbot                      â”‚
â”‚  â”œâ”€ Ask security questions          â”‚
â”‚  â””â”€ Get recommendations             â”‚
â”‚                                      â”‚
â”‚  ğŸ” Quick Actions                   â”‚
â”‚  â”œâ”€ New Scan                        â”‚
â”‚  â”œâ”€ View Reports                    â”‚
â”‚  â””â”€ Settings                        â”‚
â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜
```

---

## âš™ï¸ Temel Ã–zellikler

### 1. ğŸ¤– AI-Powered Chatbot

**KullanÄ±m SenaryolarÄ±:**

- â“ GÃ¼venlik sorularÄ± sorma
- ğŸ’¡ Tehdit analizi isteme
- ğŸ” Log analizi yaptÄ±rma
- ğŸ“š Best practice Ã¶nerileri alma

**Ã–rnek Sorgular:**

```
"Bu log dosyasÄ±nÄ± analiz et"
"Port 443'teki trafik normal mi?"
"DDoS saldÄ±rÄ±sÄ±na karÅŸÄ± ne yapmalÄ±yÄ±m?"
"Sistem gÃ¼venliÄŸimi nasÄ±l artÄ±rabilirim?"
```

**Chatbot Ã–zellikleri:**

- ğŸ§  Natural Language Processing
- ğŸ“– Context-aware responses
- ğŸ”„ Multi-turn conversations
- ğŸ“Š Data visualization support

### 2. ğŸ” GÃ¼venlik TaramasÄ±

**Tarama Tipleri:**

1. **Quick Scan**
    - SÃ¼re: ~5 dakika
    - Temel gÃ¼venlik kontrolleri
    - AÃ§Ä±k portlar
    - YaygÄ±n zafiyetler

2. **Deep Scan**
    - SÃ¼re: ~30 dakika
    - KapsamlÄ± gÃ¼venlik analizi
    - CVE taramasÄ±
    - KonfigÃ¼rasyon kontrolleri

3. **Custom Scan**
    - Ã–zelleÅŸtirilebilir parametreler
    - Belirli servislere odaklÄ±
    - Scheduled taramalar

**Tarama BaÅŸlatma:**

```python
# Web UI Ã¼zerinden
1. "New Scan" â†’ "Scan Type" seÃ§
2. Target IP/Domain gir
3. Options ayarla
4. "Start Scan" tÄ±kla

# CLI Ã¼zerinden
python scan.py --type deep --target 192.168.1.1
```

### 3. ğŸ“Š Raporlama ve Analiz

**Rapor Tipleri:**

- ğŸ“„ Executive Summary
- ğŸ”¬ Technical Details
- ğŸ“ˆ Trend Analysis
- ğŸ¯ Risk Assessment

**Rapor OluÅŸturma:**

```bash
# PDF rapor
Generate Report â†’ Select Scan â†’ PDF Export

# Excel rapor
Generate Report â†’ Select Scan â†’ Excel Export

# API Ã¼zerinden
curl -X POST http://localhost:5000/api/reports \
  -H "Content-Type: application/json" \
  -d '{"scan_id": "123", "format": "pdf"}'
```

---

## ğŸ’¬ Chatbot KullanÄ±mÄ±

### Temel KullanÄ±m

1. **Chatbot'u AÃ§ma**
    - Dashboard'dan "AI Assistant" butonuna tÄ±klayÄ±n
    - Veya `Ctrl + Space` kÄ±sayolunu kullanÄ±n

2. **Soru Sorma**
   ```
   User: "Son 24 saatteki gÃ¼venlik olaylarÄ±nÄ± gÃ¶ster"
   Bot: "Son 24 saatte 3 gÃ¼venlik olayÄ± tespit edildi..."
   ```

3. **Dosya YÃ¼kleme**
    - Log dosyalarÄ±nÄ± drag & drop yapÄ±n
    - Chatbot otomatik analiz yapar

### GeliÅŸmiÅŸ Ã–zellikler

**1. Context Management**
```
User: "192.168.1.100 IP adresini analiz et"
Bot: "Analiz ediyorum..."

User: "Bu IP iÃ§in port taramasÄ± yap"  # Context'i hatÄ±rlar
Bot: "Port taramasÄ± baÅŸlatÄ±lÄ±yor..."
```

**2. Multi-modal Inputs**
```
- ğŸ“ Text queries
- ğŸ“ File uploads (logs, configs)
- ğŸ–¼ï¸ Screenshot analysis
- ğŸ“Š Data visualization requests
```

**3. Command Shortcuts**
```
/scan <target>          # Quick scan baÅŸlat
/report <scan_id>       # Rapor gÃ¶ster
/threats                # Aktif tehditleri listele
/help                   # YardÄ±m menÃ¼sÃ¼
```

---

## ğŸ”’ GÃ¼venlik Analizi

### Zafiyet Tespiti

**Desteklenen Zafiyet Tipleri:**

- ğŸ”“ Open Ports
- ğŸ› Software Vulnerabilities (CVE)
- âš™ï¸ Misconfigurations
- ğŸ”‘ Weak Credentials
- ğŸŒ Web Application Flaws

**Zafiyet Skorlama:**

```
Critical (9.0-10.0)  ğŸ”´ - Acil mÃ¼dahale gerekli
High     (7.0-8.9)   ğŸŸ  - YÃ¼ksek Ã¶ncelikli
Medium   (4.0-6.9)   ğŸŸ¡ - Orta Ã¶ncelikli
Low      (0.1-3.9)   ğŸŸ¢ - DÃ¼ÅŸÃ¼k Ã¶ncelikli
```

### Tehdit Ä°zleme

**Real-time Monitoring:**

```python
# Dashboard'dan izleme
Monitoring â†’ Real-time Feed

# GÃ¶rÃ¼ntÃ¼lenecek bilgiler:
- Network traffic anomalies
- Failed login attempts
- Suspicious file changes
- Port scan detections
```

**Alert KonfigÃ¼rasyonu:**

```yaml
# alert_config.yaml
alerts:
  - type: critical_vulnerability
    action: email + slack
    threshold: 8.0
  
  - type: failed_login
    action: email
    threshold: 5 attempts
  
  - type: port_scan
    action: block_ip
    duration: 1h
```

---

## ğŸ“ˆ Raporlama

### Rapor ÅablonlarÄ±

**1. Executive Summary**
- ğŸ‘” YÃ¶netici seviyesi
- ğŸ“Š High-level istatistikler
- ğŸ¯ Ana bulgular
- ğŸ’° Risk analizi

**2. Technical Report**
- ğŸ”§ DetaylÄ± teknik bilgiler
- ğŸ“ CVE detaylarÄ±
- ğŸ› ï¸ Remediation steps
- ğŸ“œ Log Ã¶rnekleri

**3. Compliance Report**
- âœ… Standart uyumluluk (ISO 27001, PCI DSS)
- ğŸ“‹ Kontrol listesi
- ğŸš¦ Uyumluluk durumu

### Ã–zel Rapor OluÅŸturma

```python
# Custom report template
{
  "title": "Quarterly Security Assessment",
  "sections": [
    "executive_summary",
    "vulnerability_overview",
    "threat_analysis",
    "recommendations"
  ],
  "filters": {
    "date_range": "last_90_days",
    "severity": ["high", "critical"]
  }
}
```

---

## âš™ï¸ Ayarlar ve KonfigÃ¼rasyon

### Sistem AyarlarÄ±

**1. Genel Ayarlar**
```yaml
# config/settings.yaml
general:
  language: tr
  timezone: Europe/Istanbul
  theme: dark
  notifications: enabled
```

**2. Tarama AyarlarÄ±**
```yaml
scanning:
  max_concurrent_scans: 5
  timeout: 3600
  retry_failed: true
  auto_schedule: false
```

**3. GÃ¼venlik AyarlarÄ±**
```yaml
security:
  mfa_enabled: true
  session_timeout: 30m
  password_policy: strong
  api_rate_limit: 100/hour
```

### KullanÄ±cÄ± YÃ¶netimi

**Rol TabanlÄ± EriÅŸim:**

| Role | Permissions |
|------|-------------|
| ğŸ‘‘ Admin | Full access |
| ğŸ”§ Analyst | View + Scan |
| ğŸ‘€ Viewer | View only |
| ğŸ¤– API User | API access |

**KullanÄ±cÄ± Ekleme:**
```bash
# Web UI'den
Settings â†’ Users â†’ Add New User

# CLI'den
python manage_users.py add --username john --role analyst
```

---

## ğŸ”§ Sorun Giderme

### YaygÄ±n Sorunlar

**1. Chatbot YanÄ±t Vermiyor**

```bash
# Ã‡Ã¶zÃ¼m 1: Servis restart
systemctl restart cyberguard-chatbot

# Ã‡Ã¶zÃ¼m 2: Log kontrolÃ¼
tail -f logs/chatbot.log

# Ã‡Ã¶zÃ¼m 3: Model cache temizleme
python manage.py clear-cache --component chatbot
```

**2. Tarama BaÅŸlatÄ±lamÄ±yor**

```bash
# Kontrol adÄ±mlarÄ±:
1. Port eriÅŸilebilirliÄŸi: telnet target_ip port
2. Credentials doÄŸruluÄŸu: test_connection.py
3. Resource kullanÄ±mÄ±: top / htop
4. Log analizi: tail -f logs/scanner.log
```

**3. YavaÅŸ Performans**

```python
# Optimizasyon adÄ±mlarÄ±:
1. Database indexing: python manage.py optimize-db
2. Cache temizleme: python manage.py clear-cache
3. Old scan cleanup: python manage.py cleanup --days 30
4. Resource allocation artÄ±rma: config/resources.yaml
```

### Log DosyalarÄ±

```
logs/
â”œâ”€â”€ application.log       # Genel uygulama loglarÄ±
â”œâ”€â”€ chatbot.log          # Chatbot iÅŸlemleri
â”œâ”€â”€ scanner.log          # Tarama iÅŸlemleri
â”œâ”€â”€ api.log              # API istekleri
â”œâ”€â”€ security.log         # GÃ¼venlik olaylarÄ±
â””â”€â”€ error.log            # Hata loglarÄ±
```

**Log Seviyelerini DeÄŸiÅŸtirme:**
```python
# config/logging.yaml
logging:
  level: DEBUG  # DEBUG, INFO, WARNING, ERROR
  rotation: daily
  retention: 30d
```

---

## â“ SSS (SÄ±kÃ§a Sorulan Sorular)

### Genel Sorular

**Q: CyberGuard AI'yÄ± kimler kullanabilir?**
A: Siber gÃ¼venlik uzmanlarÄ±, IT yÃ¶neticileri, SOC analistleri ve sistem yÃ¶neticileri.

**Q: Lisans gerekli mi?**
A: Community edition Ã¼cretsiz, Enterprise Ã¶zellikler iÃ§in lisans gereklidir.

**Q: Hangi iÅŸletim sistemlerinde Ã§alÄ±ÅŸÄ±r?**
A: Linux (Ubuntu 20.04+, CentOS 8+), Windows Server 2019+, macOS 11+

### Teknik Sorular

**Q: API rate limit nedir?**
A: VarsayÄ±lan: 100 istek/saat. Enterprise: SÄ±nÄ±rsÄ±z.

**Q: Maksimum dosya yÃ¼kleme boyutu?**
A: Web UI: 100MB, API: 500MB, Enterprise: 5GB

**Q: KaÃ§ eÅŸzamanlÄ± tarama yapÄ±labilir?**
A: Community: 3, Professional: 10, Enterprise: SÄ±nÄ±rsÄ±z

**Q: Hangi veritabanlarÄ± destekleniyor?**
A: PostgreSQL, MySQL, MongoDB, SQLite

### GÃ¼venlik SorularÄ±

**Q: Veriler nasÄ±l korunuyor?**
A: AES-256 encryption, TLS 1.3, end-to-end encryption

**Q: Multi-factor authentication var mÄ±?**
A: Evet, TOTP ve SMS desteklenir.

**Q: Compliance sertifikalarÄ±?**
A: ISO 27001, SOC 2 Type II, GDPR compliant

---

## ğŸ“ Destek ve Ä°letiÅŸim

### Destek KanallarÄ±

- ğŸ“§ Email: support@cyberguard-ai.com
- ğŸ’¬ Chat: https://chat.cyberguard-ai.com
- ğŸ“š Documentation: https://docs.cyberguard-ai.com
- ğŸ› Bug Reports: https://github.com/cyberguard-ai/issues

### Community

- ğŸ’¼ LinkedIn: @cyberguard-ai
- ğŸ¦ Twitter: @cyberguard_ai
- ğŸ® Discord: discord.gg/cyberguard
- ğŸ“º YouTube: youtube.com/@cyberguard-ai

---

## ğŸ“š Ek Kaynaklar

### Video Tutorials

- ğŸ¥ [Getting Started (10 min)](https://youtube.com/watch?v=xxx)
- ğŸ¥ [Advanced Scanning (15 min)](https://youtube.com/watch?v=yyy)
- ğŸ¥ [Chatbot Best Practices (8 min)](https://youtube.com/watch?v=zzz)

### DokÃ¼mantasyon

- ğŸ“– [API Reference](api_reference.md)
- ğŸ—ï¸ [Architecture Guide](architecture.md)
- ğŸš€ [Deployment Guide](deployment.md)

### Blog YazÄ±larÄ±

- ğŸ“ "10 Tips for Effective Security Scanning"
- ğŸ“ "How AI Improves Threat Detection"
- ğŸ“ "Building a SOC with CyberGuard AI"

---

## ğŸ”„ SÃ¼rÃ¼m GeÃ§miÅŸi

- **v2.0.0** (2025-01) - AI Chatbot entegrasyonu
- **v1.5.0** (2024-10) - ML-based threat detection
- **v1.0.0** (2024-06) - Ä°lk stable sÃ¼rÃ¼m

---

## ğŸ“„ Lisans

Bu yazÄ±lÄ±m MIT lisansÄ± altÄ±nda daÄŸÄ±tÄ±lmaktadÄ±r.

---

**ğŸ‰ CyberGuard AI'yÄ± seÃ§tiÄŸiniz iÃ§in teÅŸekkÃ¼rler!**

*Bu kÄ±lavuz sÃ¼rekli gÃ¼ncellenmektedir. Son sÃ¼rÃ¼m iÃ§in:*
*https://docs.cyberguard-ai.com/user-guide*

---


# WEBSOCKET_GUIDE

# ğŸŒ CyberGuard AI - WebSocket Rehberi

Bu dokÃ¼manda CyberGuard AI'Ä±n WebSocket API'sini kullanarak gerÃ§ek zamanlÄ± veri akÄ±ÅŸÄ±na nasÄ±l baÄŸlanacaÄŸÄ±nÄ±zÄ± Ã¶ÄŸrenebilirsiniz.

---

## ğŸ“‹ Ä°Ã§indekiler

1. [WebSocket Endpoint'leri](#websocket-endpointleri)
2. [BaÄŸlantÄ± Kurma](#baÄŸlantÄ±-kurma)
3. [Mesaj FormatlarÄ±](#mesaj-formatlarÄ±)
4. [Ã–rnek Kodlar](#Ã¶rnek-kodlar)
5. [Hata YÃ¶netimi](#hata-yÃ¶netimi)

---

## ğŸ”Œ WebSocket Endpoint'leri

| Endpoint | AÃ§Ä±klama | Veri Tipi |
| -------- | -------- | --------- |
| `ws://localhost:8000/ws` | Sistem metrikleri | CPU, RAM, Disk |
| `ws://localhost:8000/ws/attacks` | SaldÄ±rÄ± akÄ±ÅŸÄ± | Attack + ML Prediction |
| `ws://localhost:8000/ws/events` | Olay aboneliÄŸi | Ã–zelleÅŸtirilebilir |
| `ws://localhost:8000/ws/security` | GÃ¼venlik metrikleri | Aktif baÄŸlantÄ±lar |

---

## ğŸ”— BaÄŸlantÄ± Kurma

### JavaScript (TarayÄ±cÄ±)

```javascript
const ws = new WebSocket('ws://localhost:8000/ws/attacks');

ws.onopen = () => {
    console.log('âœ… WebSocket baÄŸlantÄ±sÄ± kuruldu');
};

ws.onmessage = (event) => {
    const data = JSON.parse(event.data);
    console.log('Mesaj:', data);
};

ws.onerror = (error) => {
    console.error('âŒ WebSocket hatasÄ±:', error);
};

ws.onclose = () => {
    console.log('ğŸ”Œ WebSocket baÄŸlantÄ±sÄ± kapandÄ±');
};
```

### Python

```python
import asyncio
import websockets
import json

async def connect_to_attacks():
    uri = "ws://localhost:8000/ws/attacks"
    
    async with websockets.connect(uri) as websocket:
        print("âœ… BaÄŸlantÄ± kuruldu")
        
        while True:
            message = await websocket.recv()
            data = json.loads(message)
            print(f"Mesaj: {data}")

asyncio.run(connect_to_attacks())
```

---

## ğŸ“¨ Mesaj FormatlarÄ±

### SaldÄ±rÄ± AkÄ±ÅŸÄ± (`/ws/attacks`)

**BaÄŸlantÄ± MesajÄ±:**

```json
{
    "type": "connected",
    "message": "Connected to attack stream",
    "ml_enabled": true,
    "geoip_enabled": true
}
```

**SaldÄ±rÄ± MesajÄ±:**

```json
{
    "type": "attack",
    "data": {
        "id": "ATK-10042",
        "source": {
            "country": "CN",
            "ip": "185.220.101.1",
            "lat": 35.86,
            "lng": 104.19
        },
        "target": {
            "country": "TR",
            "ip": "192.168.1.100",
            "lat": 39.0,
            "lng": 35.0
        },
        "attack_type": "DDoS",
        "severity": "high",
        "ml_prediction": {
            "is_threat": true,
            "confidence": 0.92,
            "severity": "high",
            "suggested_action": "block"
        }
    },
    "timestamp": "2026-01-13T10:30:00.000Z"
}
```

**Heartbeat:**

```json
{
    "type": "heartbeat"
}
```

### Sistem Metrikleri (`/ws`)

```json
{
    "type": "metrics",
    "data": {
        "cpu_percent": 45.2,
        "memory_percent": 62.5,
        "disk_percent": 35.8,
        "network": {
            "bytes_sent": 1234567890,
            "bytes_recv": 9876543210
        },
        "timestamp": "2026-01-13T10:30:00.000Z"
    }
}
```

---

## ğŸ’» Ã–rnek Kodlar

### React Hook

```javascript
import { useState, useEffect, useRef } from 'react';

function useWebSocket(url) {
    const [messages, setMessages] = useState([]);
    const [connected, setConnected] = useState(false);
    const wsRef = useRef(null);

    useEffect(() => {
        const ws = new WebSocket(url);
        wsRef.current = ws;

        ws.onopen = () => setConnected(true);
        ws.onclose = () => {
            setConnected(false);
            // Auto-reconnect
            setTimeout(() => {
                wsRef.current = new WebSocket(url);
            }, 3000);
        };
        ws.onmessage = (event) => {
            const data = JSON.parse(event.data);
            setMessages(prev => [data, ...prev].slice(0, 100));
        };

        return () => ws.close();
    }, [url]);

    const send = (data) => {
        if (wsRef.current?.readyState === WebSocket.OPEN) {
            wsRef.current.send(JSON.stringify(data));
        }
    };

    return { messages, connected, send };
}

// KullanÄ±m
function AttackMonitor() {
    const { messages, connected } = useWebSocket('ws://localhost:8000/ws/attacks');

    return (
        <div>
            <p>Durum: {connected ? 'ğŸŸ¢ BaÄŸlÄ±' : 'ğŸ”´ BaÄŸlÄ± DeÄŸil'}</p>
            <ul>
                {messages.map((msg, i) => (
                    <li key={i}>{JSON.stringify(msg)}</li>
                ))}
            </ul>
        </div>
    );
}
```

### Python Async Client

```python
import asyncio
import websockets
import json
from datetime import datetime

class AttackMonitor:
    def __init__(self, url="ws://localhost:8000/ws/attacks"):
        self.url = url
        self.attacks = []
        self.connected = False
    
    async def connect(self):
        while True:
            try:
                async with websockets.connect(self.url) as ws:
                    self.connected = True
                    print(f"âœ… [{datetime.now()}] BaÄŸlantÄ± kuruldu")
                    
                    async for message in ws:
                        await self.handle_message(json.loads(message))
                        
            except websockets.exceptions.ConnectionClosed:
                self.connected = False
                print(f"ğŸ”Œ BaÄŸlantÄ± koptu, yeniden baÄŸlanÄ±lÄ±yor...")
                await asyncio.sleep(3)
            except Exception as e:
                print(f"âŒ Hata: {e}")
                await asyncio.sleep(5)
    
    async def handle_message(self, data):
        msg_type = data.get("type")
        
        if msg_type == "attack":
            attack = data.get("data", {})
            self.attacks.append(attack)
            
            # Tehdit analizi
            ml = attack.get("ml_prediction", {})
            if ml.get("is_threat") and ml.get("confidence", 0) > 0.8:
                print(f"âš ï¸ YÃœKSEK TEHDÄ°T!")
                print(f"   Kaynak: {attack.get('source', {}).get('ip')}")
                print(f"   Tip: {attack.get('attack_type')}")
                print(f"   GÃ¼ven: {ml.get('confidence'):.1%}")
        
        elif msg_type == "heartbeat":
            # Ping gÃ¶nder
            pass

# Ã‡alÄ±ÅŸtÄ±r
async def main():
    monitor = AttackMonitor()
    await monitor.connect()

asyncio.run(main())
```

### Node.js Client

```javascript
const WebSocket = require('ws');

class AttackClient {
    constructor(url = 'ws://localhost:8000/ws/attacks') {
        this.url = url;
        this.ws = null;
        this.reconnectInterval = 3000;
    }

    connect() {
        this.ws = new WebSocket(this.url);

        this.ws.on('open', () => {
            console.log('âœ… BaÄŸlantÄ± kuruldu');
        });

        this.ws.on('message', (data) => {
            const message = JSON.parse(data);
            this.handleMessage(message);
        });

        this.ws.on('close', () => {
            console.log('ğŸ”Œ BaÄŸlantÄ± kapandÄ±, yeniden baÄŸlanÄ±lÄ±yor...');
            setTimeout(() => this.connect(), this.reconnectInterval);
        });

        this.ws.on('error', (error) => {
            console.error('âŒ Hata:', error.message);
        });
    }

    handleMessage(message) {
        switch (message.type) {
            case 'attack':
                const attack = message.data;
                const ml = attack.ml_prediction || {};
                
                if (ml.is_threat && ml.confidence > 0.8) {
                    console.log(`âš ï¸ YÃœKSEK TEHDÄ°T: ${attack.source?.ip} -> ${attack.target?.ip}`);
                    console.log(`   Tip: ${attack.attack_type}, GÃ¼ven: ${(ml.confidence * 100).toFixed(0)}%`);
                }
                break;
            
            case 'heartbeat':
                this.ws.send(JSON.stringify({ type: 'ping' }));
                break;
        }
    }
}

const client = new AttackClient();
client.connect();
```

---

## âš ï¸ Hata YÃ¶netimi

### BaÄŸlantÄ± KopmasÄ±

```javascript
const ws = new WebSocket('ws://localhost:8000/ws/attacks');
let reconnectAttempts = 0;
const maxReconnectAttempts = 5;

ws.onclose = () => {
    if (reconnectAttempts < maxReconnectAttempts) {
        reconnectAttempts++;
        const delay = Math.min(1000 * Math.pow(2, reconnectAttempts), 30000);
        console.log(`Yeniden baÄŸlanma denemesi ${reconnectAttempts}/${maxReconnectAttempts} (${delay}ms)`);
        setTimeout(connect, delay);
    } else {
        console.error('Maksimum deneme sayÄ±sÄ±na ulaÅŸÄ±ldÄ±');
    }
};

ws.onopen = () => {
    reconnectAttempts = 0; // BaÅŸarÄ±lÄ± baÄŸlantÄ±da sÄ±fÄ±rla
};
```

### Heartbeat KontrolÃ¼

```javascript
let heartbeatTimeout;

function resetHeartbeat() {
    clearTimeout(heartbeatTimeout);
    heartbeatTimeout = setTimeout(() => {
        console.warn('Heartbeat timeout, baÄŸlantÄ± kontrol ediliyor...');
        ws.close();
    }, 45000); // 45 saniye
}

ws.onmessage = (event) => {
    resetHeartbeat();
    // ... mesaj iÅŸleme
};
```

---

## ğŸ“Š Globe3D Entegrasyonu

Globe3D bileÅŸeni otomatik olarak `/ws/attacks` endpoint'ine baÄŸlanÄ±r:

```javascript
// Globe3D.jsx iÃ§inde
useEffect(() => {
    const ws = new WebSocket('ws://localhost:8000/ws/attacks');
    
    ws.onmessage = (event) => {
        const message = JSON.parse(event.data);
        
        if (message.type === 'attack') {
            // SaldÄ±rÄ±yÄ± haritaya ekle
            setWsAttacks(prev => [message.data, ...prev].slice(0, 50));
            
            // ML tahmini yÃ¼ksekse ses Ã§al
            if (message.data.ml_prediction?.confidence > 0.85) {
                playAlertSound();
            }
        }
    };
    
    return () => ws.close();
}, []);
```

---

## ğŸ”’ GÃ¼venlik NotlarÄ±

1. **Production'da wss:// kullanÄ±n** (SSL/TLS)
2. **Token tabanlÄ± kimlik doÄŸrulama** ekleyin
3. **Rate limiting** uygulayÄ±n
4. **Input validation** yapÄ±n

```javascript
// GÃ¼venli baÄŸlantÄ± Ã¶rneÄŸi
const ws = new WebSocket('wss://your-domain.com/ws/attacks', {
    headers: {
        'Authorization': `Bearer ${token}`
    }
});
```

---

**âš¡ GerÃ§ek zamanlÄ± gÃ¼venlik izleme!**


---


# XAİ

# ğŸ” Explainable AI (XAI) DokÃ¼mantasyonu

CyberGuard AI projesindeki Explainable AI Ã¶zellikleri - DetaylÄ± Rehber

---

## ğŸ“‹ Ä°Ã§indekiler

- [Genel BakÄ±ÅŸ](#genel-bakÄ±ÅŸ)
- [Neden XAI?](#neden-xai)
- [API Endpoints](#api-endpoints)
- [SHAP AÃ§Ä±klamalarÄ±](#shap-aÃ§Ä±klamalarÄ±)
- [LIME AÃ§Ä±klamalarÄ±](#lime-aÃ§Ä±klamalarÄ±)
- [Feature Importance](#feature-importance)
- [GÃ¶rselleÅŸtirmeler](#gÃ¶rselleÅŸtirmeler)
- [KullanÄ±m Ã–rnekleri](#kullanÄ±m-Ã¶rnekleri)
- [Best Practices](#best-practices)

---

## ğŸŒŸ Genel BakÄ±ÅŸ

XAI modÃ¼lÃ¼, makine Ã¶ÄŸrenmesi modellerinin kararlarÄ±nÄ± aÃ§Ä±klamak iÃ§in SHAP (SHapley Additive exPlanations) ve LIME (Local Interpretable Model-agnostic Explanations) yÃ¶ntemlerini kullanÄ±r.

### Desteklenen AÃ§Ä±klama YÃ¶ntemleri

| YÃ¶ntem | TÃ¼r | AÃ§Ä±klama |
|--------|-----|----------|
| **SHAP** | Global + Local | Shapley deÄŸerleri ile aÃ§Ä±klama |
| **LIME** | Local | Lokal yorumlanabilir model |
| **Feature Importance** | Global | Model bazlÄ± Ã¶nem sÄ±ralamasÄ± |
| **Permutation Importance** | Global | PermÃ¼tasyon tabanlÄ± Ã¶nem |

---

## ğŸ¯ Neden XAI?

### Siber GÃ¼venlikte Ã–nem

```
â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
â”‚                    XAI'Ä±n FaydalarÄ±                              â”‚
â”œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¤
â”‚  ğŸ” ÅEFFAFLÄ°K      â”‚ Model kararlarÄ±nÄ±n neden verildiÄŸini anlamaâ”‚
â”‚  ğŸ¤ GÃœVEN          â”‚ KullanÄ±cÄ±larÄ±n AI Ã¶nerilerine gÃ¼venmesi    â”‚
â”‚  ğŸ› DEBUG          â”‚ Model hatalarÄ±nÄ± tespit etmek              â”‚
â”‚  âš–ï¸ COMPLIANCE     â”‚ GDPR, KVKK gibi dÃ¼zenlemelere uyum         â”‚
â”‚  ğŸ“ EÄÄ°TÄ°M         â”‚ GÃ¼venlik analistlerini eÄŸitmek             â”‚
â”‚  âœ… VALÄ°DASYON     â”‚ Model davranÄ±ÅŸÄ±nÄ± doÄŸrulamak               â”‚
â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜
```

### Yasal Gereksinimler

- **GDPR Article 22**: Automated decision-making, including profiling
- **KVKK Madde 11**: KiÅŸinin, kendisiyle ilgili otomatik iÅŸleme dayalÄ± kararlar hakkÄ±nda bilgi edinme hakkÄ±
- **ISO 27001**: Information security management

---

## ğŸ”Œ API Endpoints

### POST /api/xai/explain

Model tahminini aÃ§Ä±kla

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

TÃ¼m modeller iÃ§in ortalama feature importance

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
        "pros": ["Teorik tutarlÄ±lÄ±k", "Global aÃ§Ä±klamalar"],
        "cons": ["YavaÅŸ hesaplama", "YÃ¼ksek bellek"]
      },
      {
        "id": "lime",
        "name": "LIME",
        "description": "Local Interpretable Model-agnostic Explanations",
        "type": "local",
        "pros": ["HÄ±zlÄ±", "Model-agnostik"],
        "cons": ["TutarsÄ±z olabilir", "Sadece lokal"]
      }
    ]
  }
}
```

---

## ğŸ“Š SHAP AÃ§Ä±klamalarÄ±

### Teorik Arka Plan

SHAP, oyun teorisinden gelen Shapley deÄŸerlerini kullanarak her Ã¶zelliÄŸin tahmine katkÄ±sÄ±nÄ± hesaplar.

**Shapley DeÄŸeri FormÃ¼lÃ¼:**

```
Ï†áµ¢ = Î£ [|S|! (n-|S|-1)! / n!] Ã— [f(S âˆª {i}) - f(S)]
```

### SHAP TÃ¼rleri

| TÃ¼r | KullanÄ±m | HÄ±z |
|-----|----------|-----|
| TreeSHAP | Tree-based modeller | âš¡ Ã‡ok HÄ±zlÄ± |
| DeepSHAP | Deep learning | âš¡ HÄ±zlÄ± |
| KernelSHAP | Herhangi model | ğŸ¢ YavaÅŸ |
| LinearSHAP | Lineer modeller | âš¡ Ã‡ok HÄ±zlÄ± |

### Python KullanÄ±mÄ±

```python
import shap

# Model yÃ¼kle
model = load_model("best_cicids2017")

# SHAP explainer oluÅŸtur
explainer = shap.TreeExplainer(model)  # veya DeepExplainer

# AÃ§Ä±klama Ã¼ret
shap_values = explainer.shap_values(X_test)

# Tek Ã¶rnek iÃ§in aÃ§Ä±klama
shap.force_plot(explainer.expected_value, shap_values[0], X_test[0])

# Ã–zet plot
shap.summary_plot(shap_values, X_test)
```

### SHAP GÃ¶rselleri

```
Force Plot (Tek Ã–rnek):
â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
â”‚  Base: 0.12                                                   â”‚
â”‚  â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¬â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¬â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€  â”‚
â”‚  Flow Duration   â”‚  Total Fwd Packets      â”‚  Final: 0.98    â”‚
â”‚  +0.35           â”‚  -0.12                   â”‚                  â”‚
â”‚  â–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ”‚â–’â–’â–’â–’â–’â–’                    â”‚                  â”‚
â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”´â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”´â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜

Summary Plot (TÃ¼m Ã–rnekler):
â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
â”‚  Feature            â”‚ SHAP Value Impact                      â”‚
â”‚  â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¼â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”‚
â”‚  Flow Duration      â”‚ â–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆ High              â”‚
â”‚  Total Fwd Packets  â”‚ â–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–‘â–‘â–‘â–‘â–‘â–‘ Medium            â”‚
â”‚  Fwd Packet Length  â”‚ â–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–‘â–‘â–‘â–‘â–‘â–‘â–‘â–‘â–‘â–‘â–‘ Low               â”‚
â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜
```

---

## ğŸ‹ LIME AÃ§Ä±klamalarÄ±

### NasÄ±l Ã‡alÄ±ÅŸÄ±r?

1. Tahmin noktasÄ± Ã§evresinde perturbation samples oluÅŸtur
2. Her sample iÃ§in orijinal model tahmini al
3. Weighted linear model eÄŸit
4. Linear model katsayÄ±larÄ±nÄ± aÃ§Ä±klama olarak kullan

### Python KullanÄ±mÄ±

```python
from lime import lime_tabular

# LIME explainer oluÅŸtur
explainer = lime_tabular.LimeTabularExplainer(
    X_train,
    feature_names=feature_names,
    class_names=class_names,
    mode='classification'
)

# AÃ§Ä±klama Ã¼ret
explanation = explainer.explain_instance(
    X_test[0],
    model.predict_proba,
    num_features=10
)

# GÃ¶rselle
explanation.show_in_notebook()

# Liste olarak
print(explanation.as_list())
# [('Flow Duration > 1000', 0.25), ('Total Fwd Packets > 500', 0.18), ...]
```

### LIME vs SHAP

| Ã–zellik | SHAP | LIME |
|---------|------|------|
| Teorik TutarlÄ±lÄ±k | âœ… | âŒ |
| HÄ±z | ğŸ¢ | âš¡ |
| Global AÃ§Ä±klama | âœ… | âŒ |
| Model-Agnostik | âœ… | âœ… |
| Stabilite | âœ… | âš ï¸ |
| Bellek KullanÄ±mÄ± | YÃ¼ksek | DÃ¼ÅŸÃ¼k |

---

## ğŸ¯ Feature Importance

### Global Importance

TÃ¼m tahminlerde hangi Ã¶zelliklerin genel olarak Ã¶nemli olduÄŸunu gÃ¶sterir.

```python
# Random Forest feature importance
importance = model.feature_importances_
feature_importance = pd.DataFrame({
    'feature': feature_names,
    'importance': importance
}).sort_values('importance', ascending=False)
```

### Lokal Importance

Tek bir tahmin iÃ§in hangi Ã¶zelliklerin belirleyici olduÄŸunu gÃ¶sterir.

### CyberGuard AI'daki En Ã–nemli Ã–zellikler

| SÄ±ra | Ã–zellik | Ã–nemi | AÃ§Ä±klama |
|------|---------|-------|----------|
| 1 | Flow Duration | 15% | AkÄ±ÅŸ sÃ¼resi |
| 2 | Total Fwd Packets | 12% | Forward paket sayÄ±sÄ± |
| 3 | Fwd Packet Length Mean | 10% | Ortalama forward paket uzunluÄŸu |
| 4 | Bwd Packet Length Mean | 9% | Ortalama backward paket uzunluÄŸu |
| 5 | Flow Bytes/s | 8% | Saniye baÅŸÄ±na byte |

---

## ğŸ“ˆ GÃ¶rselleÅŸtirmeler

### Frontend GÃ¶rselleÅŸtirmeleri

```jsx
// XAIExplainer.jsx'te kullanÄ±m

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

### API ile GÃ¶rsel

```python
import requests
import matplotlib.pyplot as plt

# AÃ§Ä±klama al
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

## ğŸ’» KullanÄ±m Ã–rnekleri

### 1. SaldÄ±rÄ± AÃ§Ä±klamasÄ±

```python
# Bir saldÄ±rÄ± tahmini iÃ§in aÃ§Ä±klama
attack_sample = get_attack_sample("DDoS")

explanation = requests.post("/api/xai/explain", json={
    "model_id": "best_cicids2017",
    "features": attack_sample.tolist(),
    "method": "shap"
}).json()

print(f"Tahmin: {explanation['data']['prediction']}")
print(f"GÃ¼ven: {explanation['data']['confidence']:.2%}")
print("\nÃ–nemli FaktÃ¶rler:")
for f in explanation['data']['explanation']['top_features'][:5]:
    print(f"  {f['feature']}: {f['shap_value']:+.4f}")
```

### 2. Model KarÅŸÄ±laÅŸtÄ±rmasÄ±

```python
# Ä°ki model iÃ§in aynÄ± Ã¶rneÄŸin aÃ§Ä±klamasÄ±
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

### 3. Batch AÃ§Ä±klama

```python
# Birden fazla Ã¶rnek iÃ§in aÃ§Ä±klama
results = []
for sample in samples:
    exp = requests.post("/api/xai/explain", json={
        "model_id": "best_cicids2017",
        "features": sample.tolist(),
        "method": "lime"  # LIME daha hÄ±zlÄ±
    }).json()
    results.append(exp["data"])
```

---

## ğŸ“ Best Practices

### 1. YÃ¶ntem SeÃ§imi

| Senaryo | Ã–nerilen YÃ¶ntem |
|---------|-----------------|
| HÄ±zlÄ± aÃ§Ä±klama | LIME |
| DetaylÄ± analiz | SHAP |
| Tree-based model | TreeSHAP |
| Deep learning | DeepSHAP |
| Global gÃ¶rÃ¼nÃ¼m | SHAP Summary |

### 2. Performans Ä°yileÅŸtirmeleri

```python
# SHAP iÃ§in sample kullan
shap_values = explainer.shap_values(X_test[:100])  # Ä°lk 100 Ã¶rnek

# Background data limitle
explainer = shap.KernelExplainer(
    model.predict, 
    shap.sample(X_train, 100)  # 100 background sample
)
```

### 3. AÃ§Ä±klama Kalitesi

- En az 5-10 Ã¶zellik gÃ¶ster
- Pozitif/negatif katkÄ±larÄ± renklendir
- Ã–zellik deÄŸerlerini de gÃ¶ster
- GÃ¼ven aralÄ±ÄŸÄ± ekle

---

## ğŸ“š Referanslar

- [SHAP Paper](https://arxiv.org/abs/1705.07874) - Lundberg & Lee (2017)
- [LIME Paper](https://arxiv.org/abs/1602.04938) - Ribeiro et al. (2016)
- [Interpretable ML Book](https://christophm.github.io/interpretable-ml-book/)
- [SHAP Documentation](https://shap.readthedocs.io/)
- [LIME Documentation](https://lime-ml.readthedocs.io/)


---

