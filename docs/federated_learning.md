# 🔗 Federated Learning Dokümantasyonu

Dağıtık makine öğrenmesi ve gizlilik koruyan eğitim

---

## 📋 İçindekiler

- [Genel Bakış](#genel-bakış)
- [Mimari](#mimari)
- [API Endpoints](#api-endpoints)
- [Aggregation Yöntemleri](#aggregation-yöntemleri)
- [Gizlilik Özellikleri](#gizlilik-özellikleri)

---

## 🌟 Genel Bakış

Federated Learning, verileri merkezi bir sunucuya göndermeden, cihazlar üzerinde model eğitimi yapılmasını sağlar.

### Avantajlar

- 🔒 **Gizlilik**: Veriler cihazda kalır
- 🌐 **Dağıtık**: Merkezi sunucu gereksiz
- 📊 **Ölçeklenebilir**: Binlerce client destekler
- ⚡ **Verimli**: Sadece model güncellemeleri iletilir

---

## 🏗️ Mimari

```
┌─────────────────┐
│  Central Server │
│   (Aggregator)  │
└────────┬────────┘
         │
    ┌────┴────┐
    │         │
┌───┴───┐ ┌───┴───┐
│Client1│ │Client2│ ... ClientN
└───────┘ └───────┘
```

### Eğitim Döngüsü

1. Server global modeli client'lara dağıtır
2. Her client kendi verileriyle local eğitim yapar
3. Client'lar model güncellemelerini server'a gönderir
4. Server güncellemeleri aggregate eder
5. Yeni global model oluşturulur
6. Tekrar 1'den başla

---

## 🔌 API Endpoints

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

Federated training başlat

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

Aggregation metodlarını listele

### GET /api/federated/privacy

Gizlilik özelliklerini listele

---

## 🔄 Aggregation Yöntemleri

### 1. FedAvg (Federated Averaging)

- En basit yöntem
- Tüm client ağırlıklarının ortalaması
- IID data varsayımı

### 2. FedProx

- Non-IID data için optimize
- Proximal term ile stabilite
- Heterojen sistemler için uygun

### 3. SCAFFOLD

- Variance reduction
- Daha hızlı convergence
- Daha yüksek communication cost

---

## 🔒 Gizlilik Özellikleri

### Differential Privacy

- Gradientlere noise ekleme
- ε (epsilon) parametresi ile kontrol
- Trade-off: privacy vs accuracy

### Secure Aggregation

- Kriptografik aggregation
- Server bile bireysel güncellemeleri göremez
- MPC (Multi-Party Computation)

### Homomorphic Encryption

- Şifreli veri üzerinde hesaplama
- En yüksek güvenlik seviyesi
- Yüksek computational cost

---

## 💻 Kullanım

### Client Ekleme

```python
response = requests.post("/api/federated/clients", json={
    "name": "Factory Sensor 1",
    "data_size": 10000,
    "compute_power": "medium"
})
client_id = response.json()["data"]["client_id"]
```

### Eğitim Başlatma

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

## 📈 Sonuç Metrikleri

- **Global Accuracy**: Aggregate modelin doğruluğu
- **Client Accuracy**: Her client'ın local doğruluğu
- **Communication Cost**: Iletilen veri miktarı
- **Training Time**: Round başına süre
- **Privacy Budget**: Harcanan ε miktarı

---

## 📝 Referanslar

- [Communication-Efficient Learning](https://arxiv.org/abs/1602.05629)
- [Federated Learning at Scale](https://arxiv.org/abs/1902.01046)
- [Advances in Federated Learning](https://arxiv.org/abs/1912.04977)
