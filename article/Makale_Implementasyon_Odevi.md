# SSA-LSTMIDS Makale Implementasyonu - Ödev Raporu

**Hazırlayan:** Eyüp Salih OĞLU  
**Tarih:** Ocak 2026  
**Referans Makale:** An optimized LSTM-based deep learning model for anomaly network intrusion detection (Scientific Reports, 2025)

---

## 1. MAKALE ÖZETİ

Referans makale, siber saldırı tespiti için SSA (Sparrow Search Algorithm) ile optimize edilmiş LSTM tabanlı bir derin öğrenme modeli önermektedir. Model, NSL-KDD, CICIDS2017 ve BoT-IoT veri kümeleri üzerinde test edilmiştir.

---

## 2. PYTHON İMPLEMENTASYONU

### 2.1 Model Mimarisi (ssa_lstmids.py)

**[EKRAN GÖRÜNTÜSÜ 1: Model Build Fonksiyonu]**

```python
# SSA-LSTMIDS Model Mimarisi
# Kaynak: src/network_detection/models/ssa_lstmids.py

inputs = layers.Input(shape=self.input_shape, name="input")

# Conv1D Layer - Feature extraction
x = layers.Conv1D(
    filters=30,      # Makale parametresi
    kernel_size=5,   # Makale parametresi
    padding="same",
    activation="relu",
    name="conv1d",
)(inputs)

# MaxPooling
x = layers.MaxPooling1D(pool_size=2, name="maxpool")(x)

# LSTM Layer - Temporal pattern learning
x = layers.LSTM(
    units=120,       # Makale parametresi
    activation="tanh",
    recurrent_activation="sigmoid",
    return_sequences=False,
    name="lstm",
)(x)

# Dense Layer
x = layers.Dense(units=512, activation="relu", name="dense")(x)

# Dropout
x = layers.Dropout(rate=0.2, name="dropout")(x)

# Output Layer
outputs = layers.Dense(units=num_classes, activation="softmax", name="output")(x)
```

### 2.2 SSA Optimizer (ssa.py)

**[EKRAN GÖRÜNTÜSÜ 2: SSA Optimizer]**

SSA (Sparrow Search Algorithm), hiperparametreleri optimize etmek için kullanılmıştır:

| Parametre | Aralık | Optimal Değer |
|-----------|--------|---------------|
| conv_filters | [16, 64] | 30 |
| lstm_units | [32, 256] | 120 |
| dropout_rate | [0.1, 0.5] | 0.2 |
| learning_rate | [0.0001, 0.01] | 0.001 |
| batch_size | [32, 256] | 120 |

---

## 3. SONUÇLAR

### 3.1 Veri Kümesi Bazlı Sonuçlar

**[GÖRSEL 1: Accuracy Karşılaştırma Bar Chart]**

| Veri Kümesi | Makale Accuracy | Bizim Accuracy | Fark |
|-------------|-----------------|----------------|------|
| NSL-KDD | %99.36 | %94.76 | -4.60% |
| CICIDS2017 | %99.88 | %99.78 | -0.10% |
| BoT-IoT | %99.99 | %99.97 | -0.02% |

### 3.2 F1-Score Karşılaştırması

**[GÖRSEL 2: F1-Score Karşılaştırma Bar Chart]**

| Veri Kümesi | Makale F1 | Bizim F1 | Fark |
|-------------|-----------|----------|------|
| NSL-KDD | %99.36 | %94.39 | -4.97% |
| CICIDS2017 | %99.88 | %99.75 | -0.13% |
| BoT-IoT | %99.99 | %99.97 | -0.02% |

---

## 4. KARŞILAŞTIRMA ANALİZİ

### 4.1 Genel Değerlendirme

1. **CICIDS2017 ve BoT-IoT:** Sonuçlarımız makale ile neredeyse aynı (%99.7+)
2. **NSL-KDD:** Makaleye göre ~%5 düşük performans gözlemlenmiştir

### 4.2 Farklılık Nedenleri

- **Veri ön işleme:** Farklı normalizasyon yöntemleri
- **Veri bölümleme:** Train/test split oranları
- **Random seed:** Rastgelelik faktörü
- **Donanım:** GPU/CPU farklılıkları

### 4.3 Sonuç

Implementasyonumuz, referans makaledeki SSA-LSTMIDS mimarisini başarıyla yeniden oluşturmuştur. CICIDS2017 ve BoT-IoT veri kümelerinde makale ile **yaklaşık aynı sonuçlar** elde edilmiştir.

---

## 5. EKRAN GÖRÜNTÜLERİ

### 5.1 Model Eğitim Çıktısı

**[EKRAN GÖRÜNTÜSÜ: Model eğitim log'u]**

### 5.2 Model Summary

**[EKRAN GÖRÜNTÜSÜ: model.summary() çıktısı]**

### 5.3 Sonuç JSON Dosyası

**[EKRAN GÖRÜNTÜSÜ: ssa_lstmids_results.json]**

---

## KAYNAKÇA

[1] Scientific Reports. (2025). An optimized LSTM-based deep learning model for anomaly network intrusion detection. Scientific Reports, 15, 1554. <https://doi.org/10.1038/s41598-025-85248-z>

---

**© 2026 CyberGuard AI**
