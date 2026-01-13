# 🔄 CI/CD Pipeline Guide

CyberGuard AI için CI/CD kurulumu

---

## 📋 İçindekiler

- [Genel Bakış](#genel-bakış)
- [GitHub Actions](#github-actions)
- [Docker Build](#docker-build)
- [Deployment](#deployment)
- [Secrets Management](#secrets-management)

---

## 🌟 Genel Bakış

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│    Code     │ -> │    Test     │ -> │    Build    │ -> │   Deploy    │
│    Push     │    │   (pytest)  │    │   (Docker)  │    │  (K8s/VPS)  │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
```

---

## 🐙 GitHub Actions

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

## 🐳 Docker Build

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

## 🚀 Deployment

### Kubernetes (Örnek)

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

## 🔐 Secrets Management

### GitHub Secrets

| Secret | Açıklama |
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

## 📊 Pipeline Metrikleri

| Metrik | Hedef |
|--------|-------|
| Build Time | < 5 min |
| Test Coverage | > 80% |
| Deploy Time | < 2 min |
| Rollback Time | < 1 min |
