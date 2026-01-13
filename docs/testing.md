# 🧪 Testing Guide

CyberGuard AI test stratejisi ve komutları

---

## 📋 İçindekiler

- [Test Türleri](#test-türleri)
- [Kurulum](#kurulum)
- [Unit Tests](#unit-tests)
- [Integration Tests](#integration-tests)
- [E2E Tests](#e2e-tests)
- [ML Model Tests](#ml-model-tests)
- [Performance Tests](#performance-tests)
- [CI/CD Entegrasyonu](#cicd-entegrasyonu)

---

## 🎯 Test Türleri

| Tür | Kapsam | Araç | Süre |
|-----|--------|------|------|
| Unit | Fonksiyon | pytest | Saniye |
| Integration | Modül | pytest | Dakika |
| E2E | Sistem | Cypress/Playwright | Dakika |
| ML | Model | pytest + sklearn | Dakika |
| Performance | Load | Locust/k6 | Dakika |

---

## 🔧 Kurulum

```bash
# Test bağımlılıkları
pip install pytest pytest-cov pytest-asyncio httpx

# Frontend testleri
cd frontend
npm install -D vitest @testing-library/react
```

---

## 🔬 Unit Tests

### Çalıştırma

```bash
# Tüm unit testler
pytest tests/unit/ -v

# Coverage ile
pytest tests/unit/ --cov=app --cov-report=html

# Belirli dosya
pytest tests/unit/test_predictor.py -v

# Belirli test
pytest tests/unit/test_predictor.py::test_model_load -v
```

### Örnek Test

```python
# tests/unit/test_predictor.py
import pytest
from src.models.predictor import AttackPredictor

class TestAttackPredictor:
    
    @pytest.fixture
    def predictor(self):
        return AttackPredictor()
    
    def test_model_load(self, predictor):
        """Model yükleme testi"""
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
        """Geçersiz girdi testi"""
        predictor.load_models()
        
        with pytest.raises(ValueError):
            predictor.predict_single([0.1] * 10)  # Eksik feature
```

---

## 🔗 Integration Tests

### Çalıştırma

```bash
pytest tests/integration/ -v
```

### API Test Örneği

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
    assert response.json()["message"] == "🛡️ CyberGuard AI API"

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

## 🌐 E2E Tests

### Playwright Kurulum

```bash
npm install -D @playwright/test
npx playwright install
```

### E2E Test Örneği

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

### Çalıştırma

```bash
npx playwright test
npx playwright test --ui
npx playwright test --headed
```

---

## 🧠 ML Model Tests

```python
# tests/ml/test_model_performance.py
import pytest
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from src.models.predictor import AttackPredictor

class TestModelPerformance:
    
    @pytest.fixture
    def test_data(self):
        # Test verisi yükle
        X_test = np.load("data/test/X_test.npy")
        y_test = np.load("data/test/y_test.npy")
        return X_test, y_test
    
    def test_accuracy_threshold(self, test_data):
        """Accuracy %95 üstünde olmalı"""
        X_test, y_test = test_data
        predictor = AttackPredictor()
        predictor.load_models()
        
        y_pred = predictor.predict_batch(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        assert accuracy >= 0.95, f"Accuracy {accuracy:.2%} < 95%"
    
    def test_f1_score(self, test_data):
        """F1-Score %90 üstünde olmalı"""
        X_test, y_test = test_data
        predictor = AttackPredictor()
        predictor.load_models()
        
        y_pred = predictor.predict_batch(X_test)
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        assert f1 >= 0.90
    
    def test_inference_time(self, test_data):
        """Inference 100ms altında olmalı"""
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

## ⚡ Performance Tests

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

### Çalıştırma

```bash
# Web UI
locust -f tests/performance/locustfile.py

# Headless
locust -f tests/performance/locustfile.py \
    --headless -u 100 -r 10 -t 1m \
    --host http://localhost:8000
```

---

## 🔄 CI/CD Entegrasyonu

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

## 📊 Coverage Hedefleri

| Modül | Hedef |
|-------|-------|
| Models | 90% |
| API Routes | 85% |
| Utils | 80% |
| Frontend | 75% |
| **Toplam** | **80%** |

---

## 🚀 Test Komutları Özeti

```bash
# Tüm testler
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
