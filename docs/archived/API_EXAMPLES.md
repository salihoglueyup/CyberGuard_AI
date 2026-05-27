# 🔌 CyberGuard AI - API Örnekleri

Bu dokümanda CyberGuard AI API'sini kullanmak için örnek kodlar bulabilirsiniz.

---

## 📋 İçindekiler

1. [Curl Örnekleri](#curl-örnekleri)
2. [Python Örnekleri](#python-örnekleri)
3. [JavaScript Örnekleri](#javascript-örnekleri)
4. [Yaygın Kullanım Senaryoları](#yaygın-kullanım-senaryoları)

---

## 🔧 Curl Örnekleri

### Dashboard Verisi

```bash
curl -X GET "http://localhost:8000/api/dashboard/stats" \
  -H "Content-Type: application/json"
```

### Canlı Saldırılar

```bash
curl -X GET "http://localhost:8000/api/attack-map/live?limit=20" \
  -H "Content-Type: application/json"
```

### Ülke İstatistikleri

```bash
curl -X GET "http://localhost:8000/api/attack-map/countries" \
  -H "Content-Type: application/json"
```

### Ağ Durumu

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
    "message": "Bu IP zararlı mı: 192.168.1.100"
  }'
```

---

## 🐍 Python Örnekleri

### Kurulum

```bash
pip install requests
```

### Temel Kullanım

```python
import requests

BASE_URL = "http://localhost:8000/api"

# Dashboard verisi al
def get_dashboard():
    response = requests.get(f"{BASE_URL}/dashboard/stats")
    return response.json()

# Canlı saldırıları al
def get_live_attacks(limit=50):
    response = requests.get(f"{BASE_URL}/attack-map/live", params={"limit": limit})
    return response.json()

# Ağ durumu
def get_network_status():
    response = requests.get(f"{BASE_URL}/network/status")
    return response.json()

# Kullanım
if __name__ == "__main__":
    print("Dashboard:", get_dashboard())
    print("Attacks:", get_live_attacks(10))
```

### ML Tahmin Örneği

```python
import requests

def predict_threat(data):
    """ML modeli ile tehdit tahmini yap"""
    response = requests.post(
        "http://localhost:8000/api/prediction/predict",
        json={"features": data}
    )
    return response.json()

# Örnek veri
sample_data = {
    "source_ip": "185.220.101.1",
    "target_port": 22,
    "protocol": "TCP",
    "bytes_sent": 1500,
    "duration": 3.5
}

result = predict_threat(sample_data)
print(f"Tehdit Skoru: {result.get('threat_score', 0)}")
print(f"Sınıflandırma: {result.get('classification', 'unknown')}")
```

### Threat Hunting

```python
import requests

def hunt_threats(query, timerange="24h"):
    """Tehdit avlama sorgusu çalıştır"""
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
print(f"Eşleşme sayısı: {len(results.get('data', {}).get('results', []))}")
```

### Sandbox Analizi

```python
import requests

def analyze_file(file_path):
    """Dosyayı sandbox'ta analiz et"""
    with open(file_path, 'rb') as f:
        files = {'file': f}
        response = requests.post(
            "http://localhost:8000/api/sandbox/analyze",
            files=files
        )
    return response.json()

# Örnek kullanım
result = analyze_file("suspicious_file.exe")
print(f"Risk Skoru: {result.get('data', {}).get('risk_score', 0)}")
print(f"Sonuç: {result.get('data', {}).get('verdict', 'unknown')}")
```

---

## 📜 JavaScript Örnekleri

### Fetch API

```javascript
const BASE_URL = 'http://localhost:8000/api';

// Dashboard verisi
async function getDashboard() {
    const response = await fetch(`${BASE_URL}/dashboard/stats`);
    return response.json();
}

// Canlı saldırılar
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

// Kullanım
getDashboard().then(data => console.log('Dashboard:', data));
askAI('DDoS saldırısına karşı ne yapmalıyım?').then(data => console.log('AI:', data));
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

// Saldırılar
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

## 📊 Yaygın Kullanım Senaryoları

### Senaryo 1: Güvenlik Dashboard Oluşturma

```python
import requests
import time

def create_security_dashboard():
    """Güvenlik özeti oluştur"""
    base = "http://localhost:8000/api"
    
    # Verileri topla
    dashboard = requests.get(f"{base}/dashboard/stats").json()
    attacks = requests.get(f"{base}/attack-map/live?limit=10").json()
    network = requests.get(f"{base}/network/status").json()
    security = requests.get(f"{base}/security/score").json()
    
    # Özet oluştur
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
    """Tehdit izleme döngüsü"""
    while True:
        attacks = requests.get(
            "http://localhost:8000/api/attack-map/live?limit=50"
        ).json()
        
        for attack in attacks.get("data", {}).get("attacks", []):
            if attack.get("ml_prediction", {}).get("is_threat"):
                print(f"⚠️ TEHDIT: {attack.get('source', {}).get('ip')} -> {attack.get('target', {}).get('ip')}")
                print(f"   Tip: {attack.get('attack_type')}")
                print(f"   Güven: {attack.get('ml_prediction', {}).get('confidence', 0):.1%}")
        
        time.sleep(interval)

# monitor_threats(30)  # Her 30 saniyede kontrol
```

### Senaryo 3: Rapor Oluşturma

```python
import requests
import json
from datetime import datetime

def generate_report():
    """Günlük güvenlik raporu"""
    base = "http://localhost:8000/api"
    
    report = {
        "title": "Günlük Güvenlik Raporu",
        "date": datetime.now().isoformat(),
        "sections": {}
    }
    
    # Saldırı özeti
    attacks = requests.get(f"{base}/attack-map/stats").json()
    report["sections"]["attacks"] = attacks.get("data", {})
    
    # Ülke dağılımı
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

## 🔗 API Endpoint Listesi

| Kategori | Endpoint | Metod | Açıklama |
| -------- | -------- | ----- | -------- |
| Dashboard | `/dashboard/stats` | GET | Genel istatistikler |
| Attack Map | `/attack-map/live` | GET | Canlı saldırılar |
| Attack Map | `/attack-map/countries` | GET | Ülke bazlı veriler |
| Network | `/network/status` | GET | Ağ durumu |
| Network | `/network/interfaces` | GET | Interface listesi |
| Threat Hunting | `/threat-hunting/query` | POST | Sorgu çalıştır |
| Security | `/security/score` | GET | Güvenlik skoru |
| Chat | `/chat/query` | POST | AI sohbet |
| Sandbox | `/sandbox/analyze` | POST | Dosya analizi |

**Tam liste için:** <http://localhost:8000/api/docs>

---

**🔌 Kolay entegrasyon, güçlü güvenlik!**
