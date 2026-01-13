# 🛡️ Security Hub Dokümantasyonu

Kapsamlı güvenlik izleme ve analiz merkezi

---

## 📋 İçindekiler

- [Security Score](#security-score)
- [Honeypot](#honeypot)
- [Compliance](#compliance)
- [Network Topology](#network-topology)
- [Threat Heatmap](#threat-heatmap)
- [Attack Replay](#attack-replay)
- [Vulnerability Scanner](#vulnerability-scanner)

---

## 📊 Security Score

### Genel Bakış

Sistemin genel güvenlik durumunu 0-100 arası bir skor olarak hesaplar.

### API Endpoint

```
GET /api/security/score
```

### Bileşenler

| Bileşen | Ağırlık |
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

## 🍯 Honeypot

Sahte servisler ile saldırganları tespit etme sistemi.

### Desteklenen Honeypot Türleri

| Tür | Port | Açıklama |
|-----|------|----------|
| SSH | 22 | SSH brute force tespiti |
| HTTP | 80 | Web saldırı tespiti |
| FTP | 21 | Dosya transfer saldırıları |
| RDP | 3389 | Remote desktop saldırıları |

### API Endpoint

```
GET /api/security/honeypot
```

### Metrikler

- Yakalanan saldırı sayısı
- Unique saldırgan IP'ler
- En son saldırı zamanı
- Yakalanan credential'lar

---

## ✅ Compliance

Güvenlik standartlarına uyumluluk durumu.

### Desteklenen Standartlar

- **GDPR**: EU veri koruma
- **HIPAA**: Sağlık verisi güvenliği
- **PCI-DSS**: Ödeme kartı güvenliği
- **ISO 27001**: Bilgi güvenliği yönetimi
- **NIST**: Siber güvenlik çerçevesi
- **SOC 2**: Servis organizasyonu kontrolü
- **KVKK**: Kişisel verilerin korunması

### API Endpoint

```
GET /api/security/compliance
```

---

## 🌐 Network Topology

Ağ yapısının görselleştirilmesi.

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

### Desteklenen Cihaz Türleri

- Router
- Firewall
- Switch
- Server
- Workstation

---

## 🗺️ Threat Heatmap

Coğrafi tehdit dağılımı.

### API Endpoint

```
GET /api/security/heatmap
```

### Özellikler

- Ülke bazlı saldırı sayısı
- Yoğunluk gösterimi
- Top saldırı türleri
- Trend analizi

---

## ⏱️ Attack Replay

Geçmiş saldırıları yeniden oynatma ve analiz.

### API Endpoint

```
GET /api/security/attack-replay
```

### Özellikler

- Saldırı timeline
- Paket analizi
- Saldırı aşamaları
- Eğitim amaçlı replay

---

## 🔍 Vulnerability Scanner

Port tarama ve CVE kontrolü.

### API Endpoints

```
POST /api/vuln/scan
POST /api/vuln/port-scan
GET /api/vuln/cve/{cve_id}
GET /api/vuln/history
```

### Tarama Türleri

| Tür | Açıklama |
|-----|----------|
| Quick | Hızlı, temel portlar |
| Full | Tüm portlar |
| Deep | Detaylı analiz |

### Tespit Edilenler

- Açık portlar
- Servis versiyonları
- Bilinen CVE'ler
- Güvenlik açıkları

---

## 💻 Kullanım

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

## 📝 Referanslar

- [OWASP Testing Guide](https://owasp.org/www-project-web-security-testing-guide/)
- [NIST Cybersecurity Framework](https://www.nist.gov/cyberframework)
- [CIS Controls](https://www.cisecurity.org/controls)
