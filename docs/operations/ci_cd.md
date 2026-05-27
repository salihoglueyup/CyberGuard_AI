# 🔄 CI/CD Pipeline Guide

CyberGuard AI — GitHub Actions CI/CD kurulumu (v2.1)

---

## 📋 İçindekiler

- [Genel Bakış](#genel-bakış)
- [GitHub Actions](#github-actions)
- [Pre-commit Hooks](#pre-commit-hooks)
- [Docker Build](#docker-build)
- [Secrets Management](#secrets-management)
- [Pipeline Metrikleri](#pipeline-metrikleri)

---

## 🌟 Genel Bakış

```
┌──────────────┐    ┌────────────────────┐    ┌───────────────────┐
│  Code Push   │ →  │  GitHub Actions    │ →  │  Docker Build     │
│  (main/PR)   │    │  3 parallel jobs   │    │  (main only)      │
└──────────────┘    └────────────────────┘    └───────────────────┘
                         │
              ┌──────────┼──────────┐
              ▼          ▼          ▼
          backend     frontend    docker
          (py3.10+    (node22,    (buildx,
           3.11,       vitest,    push=false)
           ruff+       build)
           pytest)
```

**Tetikleyiciler:**
- `push` → `main`, `develop`
- `pull_request` → `main`

---

## 🐙 GitHub Actions

### Gerçek Workflow: `.github/workflows/ci.yml`

```yaml
name: CI/CD Pipeline

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

jobs:
  # ─────────────────────────────────────────
  # Python Backend: lint + test
  # ─────────────────────────────────────────
  backend:
    name: Backend (Python ${{ matrix.python-version }})
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ["3.10", "3.11"]
    steps:
      - uses: actions/checkout@v4

      - name: Set up Python ${{ matrix.python-version }}
        uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python-version }}
          cache: pip

      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements.txt
          pip install -r requirements-dev.txt

      - name: Lint with ruff
        run: ruff check app/ src/ tests/

      - name: Run tests with coverage
        env:
          ADMIN_DEFAULT_PASSWORD: test-ci-password-123
          CORS_ORIGINS: http://localhost:5173
        run: pytest tests/ --ignore=tests/test_ml_services.py --cov=app --cov-report=term-missing --cov-report=xml --cov-fail-under=35 -x

      - name: Upload coverage
        uses: codecov/codecov-action@v4
        if: matrix.python-version == '3.11'
        with:
          files: ./coverage.xml
          fail_ci_if_error: false

  # ─────────────────────────────────────────
  # Frontend: lint + test + build
  # ─────────────────────────────────────────
  frontend:
    name: Frontend (Node.js)
    runs-on: ubuntu-latest
    defaults:
      run:
        working-directory: frontend
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-node@v4
        with:
          node-version: 22
          cache: npm
          cache-dependency-path: frontend/package-lock.json
      - run: npm ci
      - run: npm run lint
      - run: npx vitest run
      - run: npm run build

  # ─────────────────────────────────────────
  # Docker Build (main branch only)
  # ─────────────────────────────────────────
  docker:
    needs: [backend, frontend]
    if: github.ref == 'refs/heads/main'
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: docker/setup-buildx-action@v3
      - uses: docker/build-push-action@v5
        with:
          context: ./frontend
          push: false
          tags: cyberguard-frontend:latest
          cache-from: type=gha
          cache-to: type=gha,mode=max
```

### Job Detayları

| Job | Tetikleyici | Matrix | Adımlar |
|-----|-------------|--------|---------|
| `backend` | push + PR | Python 3.10, 3.11 | checkout → pip cache → install → ruff → pytest+cov → codecov |
| `frontend` | push + PR | Node 22 | checkout → npm ci → eslint → vitest → build |
| `docker` | yalnızca `main` push | — | checkout → buildx → build (push=false) |

### Ortam Değişkenleri (CI)

Backend testler için gereken env değerleri:

| Değişken | CI Değeri | Açıklama |
|----------|-----------|----------|
| `ADMIN_DEFAULT_PASSWORD` | `test-ci-password-123` | Test admin şifresi |
| `CORS_ORIGINS` | `http://localhost:5173` | Test CORS |

> Gerçek üretim değerleri GitHub Secrets'ta saklanmalıdır.

---

## 🪝 Pre-commit Hooks

Yerel geliştirmede otomatik lint ve format kontrolü için `.pre-commit-config.yaml`:

```yaml
repos:
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.4.5
    hooks:
      - id: ruff
        args: [--fix]
      - id: ruff-format

  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.6.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-json
```

**Kurulum:**

```bash
pip install pre-commit
pre-commit install
```

---

## 🐳 Docker Build

Frontend container `frontend/Dockerfile` ile build edilir:

```dockerfile
# 1. Build aşaması
FROM node:22-alpine AS builder
WORKDIR /app
COPY package*.json ./
RUN npm ci
COPY . .
RUN npm run build

# 2. Prod aşaması
FROM nginx:alpine
COPY --from=builder /app/dist /usr/share/nginx/html
COPY nginx.conf /etc/nginx/conf.d/default.conf
EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
```

**Yerel Docker çalıştırma:**

```bash
cd frontend
docker compose up --build    # http://localhost:3000
```

**Monitoring stack (ayrı):**

```bash
# Proje kökünde
docker compose -f docker-compose.monitoring.yml up -d
# Prometheus: http://localhost:9090
# Grafana:    http://localhost:3001  (admin / cyberguard2026)
```

---

## 🔐 Secrets Management

### GitHub Secrets

| Secret | Açıklama |
|--------|----------|
| `ADMIN_DEFAULT_PASSWORD` | Production admin şifresi |
| `LLM_API_KEY` | Groq / OpenAI / Claude API anahtarı |
| `CORS_ORIGINS` | İzin verilen origin listesi |

### `.env` (yerel geliştirme)

```env
# Auth
ADMIN_DEFAULT_PASSWORD=strong-local-password

# LLM (opsiyonel)
LLM_PROVIDER=groq
LLM_API_KEY=gsk_...
LLM_MODEL=llama3-8b-8192

# Loglama
LOG_LEVEL=INFO
JSON_CONSOLE_LOG=false

# CORS
CORS_ORIGINS=http://localhost:5173
```

---

## 📊 Pipeline Metrikleri

| Metrik | Hedef |
|--------|-------|
| Backend job süresi | < 3 dk |
| Frontend job süresi | < 2 dk |
| Test kapsamı | > 80% |
| Ruff lint hata sayısı | 0 |
