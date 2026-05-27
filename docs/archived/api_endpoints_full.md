# 🔌 API Endpoints - Tam Liste

CyberGuard AI'daki tüm API endpoint'leri

---

## 📊 Genel Bakış

| Kategori | Endpoint Sayısı |
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

## 🔐 Authentication

| Method | Endpoint | Açıklama |
|--------|----------|----------|
| POST | `/api/auth/login` | Kullanıcı girişi |
| POST | `/api/auth/logout` | Çıkış |
| POST | `/api/auth/refresh` | Token yenile |
| POST | `/api/auth/register` | Kayıt (admin) |
| GET | `/api/auth/me` | Mevcut kullanıcı |

---

## 📊 Dashboard

| Method | Endpoint | Açıklama |
|--------|----------|----------|
| GET | `/api/dashboard` | Ana dashboard |
| GET | `/api/dashboard/stats` | İstatistikler |
| GET | `/api/dashboard/threats` | Tehdit özeti |
| GET | `/api/dashboard/timeline` | 24 saat timeline |
| GET | `/api/dashboard/models` | Model durumları |
| GET | `/api/dashboard/system` | Sistem metrikleri |
| GET | `/api/dashboard/recent` | Son aktiviteler |
| GET | `/api/dashboard/quick-actions` | Hızlı eylemler |

---

## 🎯 Prediction

| Method | Endpoint | Açıklama |
|--------|----------|----------|
| POST | `/api/prediction/predict` | Tek tahmin |
| POST | `/api/prediction/bulk` | Toplu tahmin |
| GET | `/api/prediction/models` | Model listesi |
| PUT | `/api/prediction/model` | Aktif model değiştir |
| GET | `/api/prediction/stats` | Tahmin istatistikleri |
| POST | `/api/prediction/realtime` | Gerçek zamanlı tahmin |
| GET | `/api/prediction/history` | Tahmin geçmişi |
| GET | `/api/prediction/confidence` | Güven eşikleri |
| POST | `/api/prediction/validate` | Input validasyon |
| GET | `/api/prediction/classes` | Sınıf listesi |

---

## 🌐 Network

| Method | Endpoint | Açıklama |
|--------|----------|----------|
| GET | `/api/network/attacks` | Saldırı listesi |
| GET | `/api/network/attacks/{id}` | Saldırı detayı |
| GET | `/api/network/stats` | Ağ istatistikleri |
| GET | `/api/network/traffic` | Trafik verileri |
| GET | `/api/network/top-ips` | En aktif IP'ler |
| GET | `/api/network/geo` | Coğrafi dağılım |
| GET | `/api/network/timeline` | Zaman çizelgesi |
| GET | `/api/network/protocols` | Protokol dağılımı |
| GET | `/api/network/ports` | Port istatistikleri |
| POST | `/api/network/analyze` | Trafik analizi |
| GET | `/api/network/flows` | Flow verileri |
| GET | `/api/network/bandwidth` | Bant genişliği |

---

## 📋 Reports

| Method | Endpoint | Açıklama |
|--------|----------|----------|
| POST | `/api/reports/create` | Rapor oluştur |
| GET | `/api/reports/list` | Rapor listesi |
| GET | `/api/reports/{id}` | Rapor detayı |
| GET | `/api/reports/{id}/download` | Rapor indir |
| DELETE | `/api/reports/{id}` | Rapor sil |
| POST | `/api/reports/schedule` | Planla |
| GET | `/api/reports/templates` | Şablonlar |
| POST | `/api/reports/export` | Dışa aktar |

---

## 🤖 Chatbot

| Method | Endpoint | Açıklama |
|--------|----------|----------|
| POST | `/api/chatbot/chat` | Mesaj gönder |
| GET | `/api/chatbot/history` | Geçmiş |
| DELETE | `/api/chatbot/clear` | Geçmişi temizle |
| POST | `/api/chatbot/analyze` | Dosya analizi |
| GET | `/api/chatbot/suggestions` | Öneriler |
| POST | `/api/chatbot/command` | Komut çalıştır |

---

## 🔍 XAI (Explainable AI)

| Method | Endpoint | Açıklama |
|--------|----------|----------|
| POST | `/api/xai/explain` | Model açıklaması |
| GET | `/api/xai/feature-importance/{model_id}` | Feature importance |
| GET | `/api/xai/global-importance` | Global importance |
| GET | `/api/xai/explanation-methods` | Mevcut metodlar |

---

## ⚔️ Adversarial

| Method | Endpoint | Açıklama |
|--------|----------|----------|
| GET | `/api/adversarial/attack-types` | Saldırı türleri |
| POST | `/api/adversarial/test` | Robustness testi |
| POST | `/api/adversarial/simulate` | Saldırı simülasyonu |
| GET | `/api/adversarial/robustness/{model_id}` | Robustness skoru |
| GET | `/api/adversarial/defense-methods` | Savunma yöntemleri |

---

## 🔗 Federated Learning

| Method | Endpoint | Açıklama |
|--------|----------|----------|
| GET | `/api/federated/status` | Sistem durumu |
| POST | `/api/federated/clients` | Client ekle |
| DELETE | `/api/federated/clients/{id}` | Client sil |
| POST | `/api/federated/start` | Eğitim başlat |
| GET | `/api/federated/aggregation` | Aggregation metodları |
| GET | `/api/federated/privacy` | Gizlilik özellikleri |

---

## 🤖 AutoML

| Method | Endpoint | Açıklama |
|--------|----------|----------|
| POST | `/api/automl/start` | Job başlat |
| GET | `/api/automl/status/{job_id}` | Job durumu |
| GET | `/api/automl/algorithms` | Algoritmalar |
| GET | `/api/automl/recommendations` | Öneriler |
| POST | `/api/automl/hyperparameter-search` | HP araması |

---

## 🕵️ Threat Intelligence

| Method | Endpoint | Açıklama |
|--------|----------|----------|
| POST | `/api/threat-intel/check-ip` | IP kontrolü |
| POST | `/api/threat-intel/check-domain` | Domain kontrolü |
| POST | `/api/threat-intel/check-hash` | Hash kontrolü |
| GET | `/api/threat-intel/feeds` | Threat feed'leri |
| GET | `/api/threat-intel/ioc` | IOC listesi |

---

## 📧 Alerts

| Method | Endpoint | Açıklama |
|--------|----------|----------|
| POST | `/api/alerts/send` | Alert gönder |
| GET | `/api/alerts/config` | Konfigürasyon |
| PUT | `/api/alerts/config` | Config güncelle |
| GET | `/api/alerts/history` | Alert geçmişi |
| POST | `/api/alerts/test` | Test gönder |

---

## 🛡️ Security Advanced

| Method | Endpoint | Açıklama |
|--------|----------|----------|
| POST | `/api/security/analyze-pcap` | PCAP analizi |
| GET | `/api/security/score` | Güvenlik skoru |
| GET | `/api/security/honeypot` | Honeypot durumu |
| GET | `/api/security/compliance` | Uyumluluk |
| GET | `/api/security/attack-replay` | Saldırı replay |
| GET | `/api/security/topology` | Ağ topolojisi |
| GET | `/api/security/heatmap` | Tehdit haritası |
| POST | `/api/security/scan-network` | Ağ tarama |
| GET | `/api/security/audit-log` | Audit log |
| GET | `/api/security/risk-scores` | Risk skorları |

---

## 🔍 Vulnerability Scanner

| Method | Endpoint | Açıklama |
|--------|----------|----------|
| POST | `/api/vuln/scan` | Zafiyet taraması |
| GET | `/api/vuln/cve/{cve_id}` | CVE detayı |
| POST | `/api/vuln/port-scan` | Port tarama |
| GET | `/api/vuln/history` | Tarama geçmişi |

---

## 📋 Log Analyzer

| Method | Endpoint | Açıklama |
|--------|----------|----------|
| POST | `/api/logs-analysis/analyze` | Log analizi |
| GET | `/api/logs-analysis/anomalies` | Anomaliler |
| POST | `/api/logs-analysis/upload` | Log yükle |
| GET | `/api/logs-analysis/patterns` | Saldırı pattern'leri |
| GET | `/api/logs-analysis/stats` | İstatistikler |

---

## ⏱️ Incidents

| Method | Endpoint | Açıklama |
|--------|----------|----------|
| GET | `/api/incidents/timeline` | Olay zaman çizelgesi |
| POST | `/api/incidents/add` | Olay ekle |
| GET | `/api/incidents/detail/{id}` | Olay detayı |
| GET | `/api/incidents/behavior/users` | Kullanıcı davranışları |
| GET | `/api/incidents/behavior/anomalies` | Davranış anomalileri |
| GET | `/api/incidents/behavior/user/{id}` | Kullanıcı detayı |

---

## 🔑 API Keys

| Method | Endpoint | Açıklama |
|--------|----------|----------|
| GET | `/api/keys` | Anahtar listesi |
| POST | `/api/keys` | Yeni anahtar |
| PUT | `/api/keys/{key_id}` | Güncelle |
| DELETE | `/api/keys/{key_id}` | Sil |
| GET | `/api/keys/{key_id}/usage` | Kullanım istatistikleri |

---

## ⚙️ Settings

| Method | Endpoint | Açıklama |
|--------|----------|----------|
| GET | `/api/settings/general` | Genel ayarlar |
| PUT | `/api/settings/general` | Ayarları güncelle |
| GET | `/api/settings/notifications` | Bildirim ayarları |
| PUT | `/api/settings/notifications` | Bildirim güncelle |

---

## 📝 Response Format

### Başarılı

```json
{
  "success": true,
  "data": {...},
  "message": "İşlem başarılı"
}
```

### Hata

```json
{
  "success": false,
  "error": "Error type",
  "message": "Hata açıklaması"
}
```

---

## 🔐 Authentication

Tüm endpoint'ler (auth hariç) JWT token gerektirir:

```
Authorization: Bearer <token>
```

---

## ⚡ Rate Limits

| Plan | Limit |
|------|-------|
| Community | 100/dakika |
| Pro | 1000/dakika |
| Enterprise | Unlimited |
