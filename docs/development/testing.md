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
# Test ve geliştirici araçlarını yükle (requirements-dev.txt)
pip install -r requirements-dev.txt

# Frontend testleri için bağımlılıklar
# (package.json'da zaten tanımlandı)
cd frontend
npm install
cd ..
```

### Test Bağımlılıkları

`requirements-dev.txt` aşağıdakileri içerir:

```
pytest>=8.0
pytest-asyncio>=0.23
pytest-cov>=5.0
httpx>=0.27
ruff>=0.4         # linter
mypy>=1.10        # tip kontrolü
pre-commit>=3.7   # git hookları
types-requests
```

---

## 🔬 Unit Tests

### Mevcut Test Dosyaları

| Dosya | Test Sayısı | Kapsam |
|-------|-------------|--------|
| `tests/test_auth.py` | 12 | Auth login, logout, token, /me |
| `tests/test_security_monitoring.py` | 15 | Scanner, alerts, incidents, anomaly, network, SIEM, threat-intel |
| `tests/test_ml_endpoints.py` | 22 | Prediction, model yönetimi, training, XAI, AutoML, drift |
| `tests/test_cache.py` | 13 | TTL cache hit/miss, async, invalidation, stats |
| `tests/test_rbac.py` | 17 | RBAC require_role, bcrypt hash, verify_token, load_users, session |
| `tests/test_api.py` | — | Genel API endpoint sağlık testleri |
| `tests/test_api_endpoints.py` | — | Endpoint varlık ve erişilebilirlik testleri |

> **Not:** `tests/test_ml_services.py` ağır ML bağımlılıkları içerdiğinden test koşumundan hariç tutulur (`--ignore=tests/test_ml_services.py`).

### Çalıştırma

```bash
# Tüm unit testler
pytest tests/ -v

# Coverage ile (pyproject.toml konfigürasyonu kullanılır)
# Not: src/ büyük ML eğitim kodu içerir, sadece app/ kapsanır
pytest tests/ --ignore=tests/test_ml_services.py --cov=app --cov-report=term-missing --cov-fail-under=35

# Tek dosya
pytest tests/test_auth.py -v

# Tek test
pytest tests/test_auth.py::TestAuthLogin::test_correct_credentials -v

# Hızlı — ilk hatada dur
pytest tests/ -x
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

## �️ Frontend Vitest Testleri

### Mevcut Test Dosyaları

| Dosya | Kapsam |
|-------|--------|
| `frontend/src/test/components/ProtectedRoute.test.jsx` | Yönlendirme, token kontrol |
| `frontend/src/test/components/NotificationStore.test.jsx` | Zustand bildirim mağazası |
| `frontend/src/test/hooks/useWebSocket.test.jsx` | WebSocket bağlantı altyapısı |
| `frontend/src/test/components/LanguageSwitcher.test.jsx` | TR/EN dil değiştirici |
| `frontend/src/components/__tests__/GlobeHUD.test.jsx` | Three.js Globe bileşeni |
| `frontend/src/components/__tests__/PageWrapper.test.jsx` | Sayfa sarmalayıcı |
| `frontend/src/components/__tests__/Sidebar.test.jsx` | Kenar çubuğu gezinme |

**Toplam: 50 test, 12 test dosyası** — `npm run test -- --run` ile çalışır.

### Çalıştırma

```bash
cd frontend

# Tüm testler (tek seferlik)
npx vitest run

# Watch modu (geliştirme sırasında)
npx vitest

# Coverage
npx vitest run --coverage

# Belirli dosya
npx vitest run src/test/components/ProtectedRoute.test.jsx
```

### Örnek Test

```jsx
// frontend/src/test/components/ProtectedRoute.test.jsx
import { render, screen } from '@testing-library/react';
import { MemoryRouter, Routes, Route } from 'react-router-dom';
import ProtectedRoute from '../../components/ProtectedRoute';

test('kimlik doğrulama yoksa /login yönlendirir', () => {
  sessionStorage.clear();
  render(
    <MemoryRouter initialEntries={['/dashboard']}>
      <Routes>
        <Route element={<ProtectedRoute />}>
          <Route path="/dashboard" element={<div>Dashboard</div>} />
        </Route>
        <Route path="/login" element={<div>Login</div>} />
      </Routes>
    </MemoryRouter>
  );
  expect(screen.getByText('Login')).toBeInTheDocument();
});
```

---

## �🔗 Integration Tests

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

Proje `.github/workflows/ci.yml` dosyasında 3 iş tanımlar:

| İş | Açıklama |
|----|----------|
| `backend` | Python 3.10 + 3.11, ruff lint, pytest + coverage |
| `frontend` | Node 22, lint, vitest, build |
| `docker` | main branch'te docker build doğrulama |

```yaml
# .github/workflows/ci.yml — backend iş özeti
jobs:
  backend:
    strategy:
      matrix:
        python-version: ["3.10", "3.11"]
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python-version }}
      - run: pip install -r requirements.txt -r requirements-dev.txt
      - run: ruff check app/ src/ tests/
      - run: pytest tests/ --cov=app --cov=src --cov-report=xml -x
        env:
          ADMIN_DEFAULT_PASSWORD: test-ci-password-123
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
# Backend — tüm testler
pytest tests/ -v

# Backend — coverage
pytest tests/ --cov=app --cov=src --cov-report=term-missing

# Backend — lint
ruff check app/ src/ tests/

# Backend — tip kontrolü
mypy app/ src/

# Backend — sadece başarısız testler
pytest tests/ --lf

# Frontend — tek seferlik
npx vitest run

# Frontend — watch modu
npx vitest

# Frontend — coverage
npx vitest run --coverage
```
