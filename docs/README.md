# 📚 CyberGuard AI — Dokümantasyon

CyberGuard AI, yapay zeka destekli siber güvenlik platformu.
Bu klasör tüm proje dokümantasyonunu içerir.

---

## 📂 Klasör Yapısı

```
docs/
├── getting-started/      Kurulum ve ilk adımlar
├── architecture/         Sistem mimarisi ve tasarım kararları
├── api/                  REST API referansı ve WebSocket kılavuzu
├── ml/                   Makine öğrenmesi modelleri ve araçlar
├── security/             Güvenlik politikası ve Security Hub
├── operations/           Dağıtım, izleme, HTTPS, CI/CD
├── development/          Katkı, test ve geliştirme rehberleri
├── reference/            FAQ, changelog, roadmap, sözlük
└── archived/             Eski / kullanımdan kaldırılmış belgeler
```

---

## 🚀 Hızlı Başlangıç

| Amaç | Belge |
|------|-------|
| İlk 5 dakikada çalıştır | [getting-started/QUICK_START.md](getting-started/QUICK_START.md) |
| Tam kurulum rehberi | [getting-started/installation.md](getting-started/installation.md) |
| Türkçe kullanım kılavuzu | [getting-started/KULLANIM_KILAVUZU.md](getting-started/KULLANIM_KILAVUZU.md) |

---

## 🏗️ Mimari ve API

| Amaç | Belge |
|------|-------|
| Sistem mimarisi | [architecture/architecture.md](architecture/architecture.md) |
| Akademik katkılar | [architecture/beyond_paper.md](architecture/beyond_paper.md) |
| REST API referansı | [api/api_reference.md](api/api_reference.md) |
| WebSocket kılavuzu | [api/WEBSOCKET_GUIDE.md](api/WEBSOCKET_GUIDE.md) |

---

## 🧠 Makine Öğrenmesi

| Amaç | Belge |
|------|-------|
| Tüm ML modelleri | [ml/ml_models.md](ml/ml_models.md) |
| Eğitim veri setleri | [ml/datasets.md](ml/datasets.md) |
| AutoML | [ml/automl.md](ml/automl.md) |
| Açıklanabilir AI (XAI) | [ml/xai.md](ml/xai.md) |
| Federe öğrenme | [ml/federated_learning.md](ml/federated_learning.md) |
| Adversarial testing | [ml/adversarial_testing.md](ml/adversarial_testing.md) |

---

## 🔒 Güvenlik

| Amaç | Belge |
|------|-------|
| Güvenlik politikası | [security/security.md](security/security.md) |
| Security Hub özellikleri | [security/security_hub.md](security/security_hub.md) |

---

## ⚙️ Operations

| Amaç | Belge |
|------|-------|
| Dağıtım | [operations/deployment.md](operations/deployment.md) |
| HTTPS / TLS kurulumu | [operations/https_setup.md](operations/https_setup.md) |
| Prometheus + Grafana izleme | [operations/monitoring.md](operations/monitoring.md) |
| CI/CD pipeline | [operations/ci_cd.md](operations/ci_cd.md) |
| Yedekleme ve kurtarma | [operations/backup_recovery.md](operations/backup_recovery.md) |
| Performans ayarları | [operations/performance_tuning.md](operations/performance_tuning.md) |

> **İzleme Stack:** `docker compose -f docker-compose.monitoring.yml up -d`
> → Prometheus: <http://localhost:9090> · Grafana: <http://localhost:3001>

---

## 🛠️ Geliştirme

| Amaç | Belge |
|------|-------|
| Katkı rehberi | [development/contributing.md](development/contributing.md) |
| Test stratejisi | [development/testing.md](development/testing.md) |
| GitHub yükleme | [development/github_upload.md](development/github_upload.md) |
| Davranış kuralları | [development/code_of_conduct.md](development/code_of_conduct.md) |

---

## 📖 Referans

| Amaç | Belge |
|------|-------|
| Sıkça Sorulan Sorular | [reference/faq.md](reference/faq.md) |
| Sorun Giderme | [reference/troubleshooting.md](reference/troubleshooting.md) |
| Değişiklik Günlüğü | [reference/changelog.md](reference/changelog.md) |
| Yol Haritası | [reference/roadmap.md](reference/roadmap.md) |
| Sürüm Notları | [reference/release_notes.md](reference/release_notes.md) |
| Kullanıcı Kılavuzu | [reference/user_guide.md](reference/user_guide.md) |
| Sözlük | [reference/glossary.md](reference/glossary.md) |

---

## 🆕 Son Güncellemeler (v3.3.0 — Nisan 2026)

| Yenilik | İlgili Belge |
|---------|-------------|
| Prometheus + Grafana monitoring stack | [operations/monitoring.md](operations/monitoring.md) |
| Yapılandırılmış JSON loglama (`app/utils/logging.py`) | [operations/monitoring.md](operations/monitoring.md) |
| TTL Cache katmanı (`app/utils/cache.py`) | [reference/faq.md](reference/faq.md#ttl-cache-ne-işe-yarar) |
| LLM Threat Decision Agent | [reference/faq.md](reference/faq.md#llm-threat-agent-nedir) |
| Refresh Token + RBAC | [security/security.md](security/security.md) |
| HTTPS / TLS kurulum rehberi | [operations/https_setup.md](operations/https_setup.md) |
| GitHub Actions CI (backend + frontend + docker) | [operations/ci_cd.md](operations/ci_cd.md) |
| Backend: 122 test, %37.35 coverage — 5 test dosyası | [development/testing.md](development/testing.md) |
| Frontend: 50 test, 12 dosya (Vitest) — 0 ESLint error | [development/testing.md](development/testing.md) |
| `advanced_ml.py` XAI gerçek model yükleme implementasyonu | [ml/xai.md](ml/xai.md) |
| `vulnerability.py` gerçek CVE ID'leri (CVE-2023-36053 vb.) | [security/security_hub.md](security/security_hub.md) |

---

## 🔗 Hızlı Linkler

- **Proje Kökü:** [README.md](../README.md)
- **API Docs (çalışırken):** <http://localhost:8000/api/docs>
- **Prometheus Metrikleri:** <http://localhost:8000/metrics>
- **GitHub:** <https://github.com/salihoglueyup/CyberGuard_AI>
- **Lisans:** [LICENSE.md](LICENSE.md)
