# 🔌 API Reference

CyberGuard AI — FastAPI v2.0 REST API Dokümantasyonu

> **Base URL:** `http://localhost:8000`  
> **Interactive Docs:** `http://localhost:8000/api/docs` (Swagger UI)  
> **Redoc:** `http://localhost:8000/api/redoc`

---

## 📋 İçindekiler

- [Kimlik Doğrulama](#kimlik-doğrulama)
- [Auth API](#auth-api--apıauth)
- [Dashboard API](#dashboard-api--apidashboard)
- [ML Tahmin API](#ml-tahmin-api--apiprediction)
- [Model Yönetim API](#model-yönetim-api--apimodels)
- [Eğitim API](#eğitim-api--apitraining)
- [Saldırı Verileri API](#saldırı-verileri-api--apiattacks)
- [Güvenlik API](#güvenlik-api)
- [Tehdit İstihbaratı API](#tehdit-istihbaratı-api)
- [İzleme API](#izleme-api)
- [Araçlar API](#araçlar-api)
- [WebSocket API](#websocket-api)
- [Hata Kodları](#hata-kodları)

---

## 🔐 Kimlik Doğrulama

Tüm korumalı endpoint'ler `Authorization: Bearer <token>` başlığı gerektirir.

```http
Authorization: Bearer eyJ0eXAiOiJKV1Q...
```

Token, `/api/auth/login` endpoint'inden alınır ve 24 saat geçerlidir.  
IP başına 5 başarısız giriş/60 saniye rate limiting uygulanır.

---

## 🔑 Auth API — `/api/auth`

### POST `/api/auth/login`

Kullanıcı girişi. Token döner.

**Request:**
```json
{
  "username": "admin",
  "password": "your_password"
}
```

**Response 200:**
```json
{
  "token": "abc123...",
  "user": {
    "id": "1",
    "username": "admin",
    "email": "admin@cyberguard.ai",
    "role": "admin"
  }
}
```

**Response 401:** Geçersiz kimlik bilgileri  
**Response 429:** Rate limit aşıldı (5 deneme/60 saniye)

---

### POST `/api/auth/register`

Yeni kullanıcı kaydı (admin yetkisi gerekir).

**Request:**
```json
{
  "username": "newuser",
  "email": "newuser@example.com",
  "password": "SecurePass123!",
  "role": "user"
}
```

**Response 201:** `{ "id": "2", "username": "newuser", ... }`

---

### POST `/api/auth/logout`

Aktif token'ı geçersiz kılar.

**Headers:** `Authorization: Bearer <token>`  
**Response 200:** `{ "message": "Çıkış başarılı" }`

---

### POST `/api/auth/refresh`

Refresh token kullanarak yeni access token alır (7 günlük refresh token süresi içinde).

**Headers:** `Authorization: Bearer <refresh_token>`  
**Response 200:**
```json
{
  "token": "new_access_token...",
  "expires_in": 86400
}
```
**Response 401:** Geçersiz veya süresi dolmuş refresh token

---

### GET `/api/auth/me`

Mevcut kullanıcı bilgilerini döner.

**Response 200:**
```json
{
  "id": "1",
  "username": "admin",
  "email": "admin@cyberguard.ai",
  "role": "admin",
  "created_at": "2026-01-01T00:00:00"
}
```

---

### GET `/api/auth/users`

Tüm kullanıcı listesi (admin yetkisi gerekir).

### POST `/api/auth/change-password`

```json
{ "old_password": "...", "new_password": "..." }
```

---

## 📊 Dashboard API — `/api/dashboard`

| Method | Path | Açıklama |
|--------|------|----------|
| GET | `/api/dashboard/stats` | Genel istatistikler |
| GET | `/api/dashboard/summary` | Özet dashboard verisi |
| GET | `/api/dashboard/hourly-trend` | Saatlik saldırı trendi |
| GET | `/api/dashboard/recent-attacks` | Son saldırılar |
| GET | `/api/dashboard/system/metrics` | CPU, RAM, disk metrikleri |
| GET | `/api/dashboard/model-performance` | Model doğruluk metrikleri |

### GET `/api/dashboard/stats`

**Response 200:**
```json
{
  "total_attacks": 15420,
  "blocked_attacks": 14890,
  "active_threats": 12,
  "models_deployed": 3,
  "uptime_hours": 720
}
```

---

## 🤖 ML Tahmin API — `/api/prediction`

### POST `/api/prediction/single`

Tek ağ trafiği örneği için saldırı tahmini.

**Request:**
```json
{
  "features": [0.1, 0.5, 1024, 80, 6],
  "model_id": "ssa_lstmids_cicids2017"
}
```

**Response 200:**
```json
{
  "prediction": "DDoS",
  "confidence": 0.9987,
  "probabilities": {
    "BENIGN": 0.0013,
    "DDoS": 0.9987
  },
  "model_used": "ssa_lstmids_cicids2017",
  "inference_time_ms": 12.3
}
```

### POST `/api/prediction/batch`

Birden fazla örnek için toplu tahmin.

**Request:**
```json
{
  "samples": [ [0.1, 0.5], [0.2, 0.3] ],
  "model_id": "ssa_lstmids_cicids2017"
}
```

### POST `/api/prediction/batch-file`

CSV dosyası yükleyerek toplu tahmin (multipart/form-data).

| Method | Path | Açıklama |
|--------|------|----------|
| GET | `/api/prediction/models` | Tahmin için kullanılabilir modeller |
| GET | `/api/prediction/stats` | Tahmin istatistikleri |

---

## 🧠 Model Yönetim API — `/api/models`

| Method | Path | Açıklama |
|--------|------|----------|
| GET | `/api/models/` | Tüm modellerin listesi |
| GET | `/api/models/deployed` | Yalnızca aktif modeller |
| GET | `/api/models/stats` | Model istatistikleri |
| GET | `/api/models/{model_id}` | Belirli model detayı |
| POST | `/api/models/{model_id}/predict` | Model ile tahmin yap |
| POST | `/api/models/predict` | Varsayılan model ile tahmin |
| GET | `/api/models/compare/all` | Tüm modelleri karşılaştır |
| DELETE | `/api/models/{model_id}` | Modeli sil |
| POST | `/api/models/{model_id}/deploy` | Modeli aktif et |
| POST | `/api/models/{model_id}/archive` | Modeli arşivle |

---

## 🎓 Eğitim API — `/api/training`

| Method | Path | Açıklama |
|--------|------|----------|
| POST | `/api/training/start` | Model eğitimi başlat |
| GET | `/api/training/status/{session_id}` | Eğitim durumu |
| GET | `/api/training/sessions` | Tüm eğitim oturumları |
| POST | `/api/training/stop/{session_id}` | Eğitimi durdur |
| DELETE | `/api/training/sessions/old` | Eski oturumları temizle |

### POST `/api/training/start`

**Request:**
```json
{
  "model_type": "ssa_lstmids",
  "dataset": "cicids2017",
  "epochs": 50,
  "batch_size": 64,
  "learning_rate": 0.001
}
```

**Response 202:**
```json
{ "session_id": "train_20260424_120000", "status": "started" }
```

---

## ⚔️ Saldırı Verileri API — `/api/attacks`

| Method | Path | Açıklama |
|--------|------|----------|
| GET | `/api/attacks/` | Tüm saldırılar |
| GET | `/api/attacks/stats` | Saldırı istatistikleri |
| GET | `/api/attacks/by-type` | Türe göre gruplandırılmış |
| GET | `/api/attacks/by-severity` | Şiddete göre |
| GET | `/api/attacks/top-ips` | En çok saldıran IP'ler |
| GET | `/api/attacks/timeline` | Zaman serisi verisi |
| GET | `/api/attacks/recent` | Son saldırılar |
| GET | `/api/attacks/search/{query}` | Saldırı araması |

---

## 🛡️ Güvenlik API

### Güvenlik Açığı Tarayıcı — `/api/vuln`

| Method | Path | Açıklama |
|--------|------|----------|
| POST | `/api/vuln/scan` | Yeni tarama başlat |
| GET | `/api/vuln/vulnerabilities` | Tüm açıklar |
| GET | `/api/vuln/vulnerability/{vuln_id}` | Açık detayı |
| PUT | `/api/vuln/vulnerability/{vuln_id}/remediate` | Kapatıldı olarak işaretle |
| GET | `/api/vuln/cve/{cve_id}` | CVE bilgisi |
| GET | `/api/vuln/stats` | Tarama istatistikleri |

### Dosya/URL Tarayıcı — `/api/scanner`

| Method | Path | Açıklama |
|--------|------|----------|
| POST | `/api/scanner/upload` | Dosya yükle ve tara |
| POST | `/api/scanner/scan-url` | URL tara |
| GET | `/api/scanner/results` | Tarama sonuçları |
| GET | `/api/scanner/stats` | İstatistikler |

### Sandbox — `/api/sandbox`

| Method | Path | Açıklama |
|--------|------|----------|
| POST | `/api/sandbox/analyze` | Dosyayı sandbox'ta analiz et |
| GET | `/api/sandbox/analyses` | Tüm analizler |
| GET | `/api/sandbox/analysis/{analysis_id}` | Analiz detayı |

### Zero-Day Tespiti — `/api/zeroday`

| Method | Path | Açıklama |
|--------|------|----------|
| POST | `/api/zeroday/analyze` | Zero-day analizi |
| GET | `/api/zeroday/detections` | Tespitler |
| PUT | `/api/zeroday/detection/{det_id}/resolve` | Çözüldü işaretle |

### Adversarial Test — `/api/adversarial`

| Method | Path | Açıklama |
|--------|------|----------|
| GET | `/api/adversarial/attacks` | Mevcut saldırı tipleri |
| POST | `/api/adversarial/test` | Adversarial test çalıştır |
| GET | `/api/adversarial/tests` | Test geçmişi |

### HSM (Donanımsal Güvenlik Modülü) — `/api/hsm`

| Method | Path | Açıklama |
|--------|------|----------|
| POST | `/api/hsm/keys` | Anahtar oluştur |
| GET | `/api/hsm/keys` | Tüm anahtarlar |
| POST | `/api/hsm/encrypt` | Şifrele |
| POST | `/api/hsm/decrypt` | Şifre çöz |

### GAN Sentezi — `/api/gan`

| Method | Path | Açıklama |
|--------|------|----------|
| GET | `/api/gan/attack-types` | Desteklenen saldırı türleri |
| POST | `/api/gan/generate` | Sentetik saldırı verisi üret |
| GET | `/api/gan/samples` | Üretilmiş örnekler |

### Container Güvenliği — `/api/container`

| Method | Path | Açıklama |
|--------|------|----------|
| GET | `/api/container/containers` | Container listesi |
| POST | `/api/container/scan` | Container tara |
| GET | `/api/container/vulnerabilities` | Açıklar |

---

## 🕵️ Tehdit İstihbaratı API

### Tehdit Analizi — `/api/threat-analysis`

| Method | Path | Açıklama |
|--------|------|----------|
| POST | `/api/threat-analysis/analyze` | Tehdit analizi yap |
| GET | `/api/threat-analysis/summary` | Tehdit özeti |
| GET | `/api/threat-analysis/mitre/tactics` | MITRE ATT&CK taktikleri |
| GET | `/api/threat-analysis/mitre/mapping` | Saldırı-MITRE eşleştirme |
| GET | `/api/threat-analysis/ioc` | IOC listesi |
| POST | `/api/threat-analysis/ioc` | IOC ekle |
| POST | `/api/threat-analysis/ioc/check` | IOC kontrol et |
| GET | `/api/threat-analysis/ip-reputation/{ip}` | IP itibar sorgulama |

### Tehdit İstihbaratı — `/api/threat-intel`

| Method | Path | Açıklama |
|--------|------|----------|
| POST | `/api/threat-intel/lookup` | Genel arama |
| GET | `/api/threat-intel/lookup/ip/{ip}` | IP sorgusu |
| GET | `/api/threat-intel/lookup/hash/{file_hash}` | Hash sorgusu |
| POST | `/api/threat-intel/ioc` | IOC ekle |
| GET | `/api/threat-intel/iocs` | IOC listesi |
| GET | `/api/threat-intel/feeds` | İstihbarat feed'leri |

### Saldırı Haritası — `/api/attack-map`

| Method | Path | Açıklama |
|--------|------|----------|
| GET | `/api/attack-map/live` | Canlı saldırı haritası |
| GET | `/api/attack-map/countries` | Ülke bazlı istatistik |
| GET | `/api/attack-map/hotspots` | Saldırı yoğunluk noktaları |
| GET | `/api/attack-map/top-attackers` | En aktif saldırganlar |

### Dark Web İzleme — `/api/darkweb`

| Method | Path | Açıklama |
|--------|------|----------|
| POST | `/api/darkweb/check` | Dark web sorgusu |
| GET | `/api/darkweb/check/email/{email}` | E-posta sızıntısı kontrolü |
| GET | `/api/darkweb/check/domain/{domain}` | Domain kontrolü |
| GET | `/api/darkweb/breaches` | Sızıntı listesi |
| POST | `/api/darkweb/monitoring` | İzleme başlat |
| GET | `/api/darkweb/alerts` | Dark web uyarıları |

### Tehdit Avı — `/api/threat-hunting`

| Method | Path | Açıklama |
|--------|------|----------|
| POST | `/api/threat-hunting/hunt` | Tehdit avı başlat |
| POST | `/api/threat-hunting/query` | Özel sorgu çalıştır |
| GET | `/api/threat-hunting/templates` | Av şablonları |
| GET | `/api/threat-hunting/history` | Av geçmişi |

### Tuzak Teknolojisi — `/api/deception`

| Method | Path | Açıklama |
|--------|------|----------|
| GET | `/api/deception/honeypots` | Honeypot listesi |
| POST | `/api/deception/honeypots` | Honeypot oluştur |
| GET | `/api/deception/captures` | Yakalanan saldırılar |
| GET | `/api/deception/dashboard` | Deception özeti |

### Saldırı Yüzeyi — `/api/attack-surface`

| Method | Path | Açıklama |
|--------|------|----------|
| GET | `/api/attack-surface/discover` | Yüzey keşfi |
| POST | `/api/attack-surface/assets` | Varlık ekle |
| POST | `/api/attack-surface/scan/{asset_id}` | Varlık taraması |
| GET | `/api/attack-surface/risk-score` | Risk skoru |

---

## 📡 İzleme API

### Gerçek Zamanlı Metrikler — `/api/realtime`

| Method | Path | Açıklama |
|--------|------|----------|
| GET | `/api/realtime/metrics` | Anlık metrikler |
| GET | `/api/realtime/dashboard` | Dashboard verisi |
| GET | `/api/realtime/system` | Sistem metrikleri |
| GET | `/api/realtime/network` | Ağ metrikleri |

### Uyarılar — `/api/alerts`

| Method | Path | Açıklama |
|--------|------|----------|
| GET | `/api/alerts/status` | Uyarı sistemi durumu |
| PUT | `/api/alerts/{alert_id}/acknowledge` | Uyarıyı onayla |
| PUT | `/api/alerts/acknowledge-all` | Tümünü onayla |
| DELETE | `/api/alerts/{alert_id}` | Uyarı sil |
| GET | `/api/alerts/stats` | Uyarı istatistikleri |

### Olaylar — `/api/incidents`

| Method | Path | Açıklama |
|--------|------|----------|
| GET | `/api/incidents/timeline` | Olay zaman çizelgesi |
| GET | `/api/incidents/{incident_id}` | Olay detayı |
| PUT | `/api/incidents/{incident_id}` | Olay güncelle |
| POST | `/api/incidents/{incident_id}/comment` | Yorum ekle |
| GET | `/api/incidents/stats` | Olay istatistikleri |

### Anomali Tespiti — `/api/anomaly`

| Method | Path | Açıklama |
|--------|------|----------|
| POST | `/api/anomaly/detect` | Anomali tespiti |
| GET | `/api/anomaly/anomalies` | Tüm anomaliler |
| GET | `/api/anomaly/stats` | İstatistikler |

### Ağ İzleme — `/api/network`

| Method | Path | Açıklama |
|--------|------|----------|
| GET | `/api/network/interfaces` | Ağ arayüzleri |
| GET | `/api/network/connections` | Aktif bağlantılar |
| GET | `/api/network/topology` | Ağ topolojisi |
| POST | `/api/network/scan` | Ağ taraması |
| GET | `/api/network/bandwidth` | Bant genişliği |

### SIEM Entegrasyonu — `/api/siem`

| Method | Path | Açıklama |
|--------|------|----------|
| GET | `/api/siem/platforms` | SIEM platformları |
| GET | `/api/siem/rules` | SIEM kuralları |
| POST | `/api/siem/connections` | Bağlantı ekle |
| POST | `/api/siem/forward` | Olay ilet |
| GET | `/api/siem/events` | Olaylar |

### Güvenlik — `/api/security`

| Method | Path | Açıklama |
|--------|------|----------|
| POST | `/api/security/analyze-pcap` | PCAP analizi |
| GET | `/api/security/score` | Güvenlik skoru |
| GET | `/api/security/honeypot` | Honeypot özeti |
| GET | `/api/security/compliance` | Uyumluluk durumu |
| GET | `/api/security/heatmap` | Saldırı ısı haritası |

---

## 🔧 Araçlar API

### Chat / LLM — `/api/chat`

| Method | Path | Açıklama |
|--------|------|----------|
| POST | `/api/chat/` | LLM'e mesaj gönder |
| GET | `/api/chat/providers` | Mevcut LLM sağlayıcılar |
| GET | `/api/chat/history` | Konuşma geçmişi |
| POST | `/api/chat/clear` | Geçmişi temizle |
| POST | `/api/chat/stream` | Streaming yanıt (SSE) |

**POST `/api/chat/` Request:**
```json
{
  "message": "CICIDS2017 veri seti hakkında bilgi ver",
  "provider": "groq",
  "use_rag": true
}
```

### IR Playbook'lar — `/api/playbooks`

| Method | Path | Açıklama |
|--------|------|----------|
| GET | `/api/playbooks/{playbook_id}` | Playbook detayı |
| POST | `/api/playbooks/{playbook_id}/execute` | Playbook çalıştır |
| PUT | `/api/playbooks/{playbook_id}` | Playbook güncelle |
| POST | `/api/playbooks/trigger/{trigger_name}` | Otomatik tetikle |

### Blockchain Denetim — `/api/blockchain`

| Method | Path | Açıklama |
|--------|------|----------|
| POST | `/api/blockchain/log` | Olay kaydet |
| GET | `/api/blockchain/chain` | Tam zinciri görüntüle |
| GET | `/api/blockchain/verify` | Zincir bütünlüğü doğrula |
| GET | `/api/blockchain/block/{index}` | Blok detayı |

### STIX/TAXII — `/api/stix-taxii`

| Method | Path | Açıklama |
|--------|------|----------|
| GET | `/api/stix-taxii/taxii2` | TAXII 2.1 kök |
| GET | `/api/stix-taxii/taxii2/default/collections` | STIX koleksiyonları |
| GET | `/api/stix-taxii/stix/objects` | STIX objeleri |
| POST | `/api/stix-taxii/taxii2/default/collections/{collection_id}/objects` | STIX objesi ekle |

### Federated Learning — `/api/federated`

| Method | Path | Açıklama |
|--------|------|----------|
| GET | `/api/federated/nodes` | Katılımcı düğümler |
| POST | `/api/federated/rounds` | Eğitim turu başlat |
| GET | `/api/federated/model` | Global model |

---

## 🔬 ML/AI Gelişmiş API

### XAI (Açıklanabilir AI) — `/api/xai`

| Method | Path | Açıklama |
|--------|------|----------|
| POST | `/api/xai/explain` | SHAP açıklaması |
| POST | `/api/xai/lime-explain` | LIME açıklaması |
| GET | `/api/xai/feature-importance/{model_id}` | Özellik önemi |
| GET | `/api/xai/global-importance` | Küresel özellik önemi |

### AutoML — `/api/automl`

| Method | Path | Açıklama |
|--------|------|----------|
| POST | `/api/automl/jobs` | AutoML işi oluştur |
| GET | `/api/automl/jobs` | Tüm işler |
| POST | `/api/automl/jobs/{job_id}/start` | Başlat |
| GET | `/api/automl/stats` | İstatistikler |

### Drift Tespiti — `/api/drift`

| Method | Path | Açıklama |
|--------|------|----------|
| POST | `/api/drift/check` | Drift kontrol et |
| POST | `/api/drift/baseline/{model_id}` | Baseline ayarla |
| GET | `/api/drift/alerts` | Drift uyarıları |

### Model Karşılaştırma — `/api/comparison`

| Method | Path | Açıklama |
|--------|------|----------|
| GET | `/api/comparison/models` | Karşılaştırılabilir modeller |
| POST | `/api/comparison/benchmark` | Benchmark çalıştır |
| GET | `/api/comparison/leaderboard` | Model sıralaması |

---

## 🌐 WebSocket API

WebSocket bağlantıları gerçek zamanlı veriler için kullanılır.

```javascript
const ws = new WebSocket("ws://localhost:8000/ws/dashboard?token=<TOKEN>");

ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  // { type: "attack_alert", payload: { ... } }
};
```

| Endpoint | Açıklama |
|----------|----------|
| `ws://localhost:8000/ws/dashboard` | Dashboard metrikleri |
| `ws://localhost:8000/ws/attacks` | Gerçek zamanlı saldırılar |
| `ws://localhost:8000/ws/network` | Ağ trafik akışı |
| `ws://localhost:8000/ws/training` | Model eğitim ilerlemesi |
| `ws://localhost:8000/ws/alerts` | Anlık uyarılar |

Daha fazla bilgi için: [WEBSOCKET_GUIDE.md](WEBSOCKET_GUIDE.md)

---

## ❌ Hata Kodları

| HTTP Kodu | Açıklama |
|-----------|----------|
| `400` | Geçersiz istek formatı / eksik parametre |
| `401` | Token eksik veya geçersiz |
| `403` | Yetersiz yetki (rol kontrolü) |
| `404` | Kaynak bulunamadı |
| `422` | Validation hatası (Pydantic) |
| `429` | Rate limit aşıldı |
| `500` | Sunucu hatası |

**Hata yanıt formatı:**
```json
{
  "detail": "Hata açıklaması"
}
```

---

## 📝 Notlar

- Tüm yanıtlar `application/json` formatındadır
- Tarih/saat değerleri ISO 8601 formatındadır (`2026-04-24T12:00:00`)
- `token` değerleri `secrets.token_urlsafe(32)` ile üretilir
- Büyük listeler için `?limit=100&offset=0` query parametreleri desteklenir
- Interactive API dokümantasyonu: `http://localhost:8000/api/docs`

---

[Back to Top](#-api-reference)
