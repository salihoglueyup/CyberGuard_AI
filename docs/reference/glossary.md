# 📚 Terimler Sözlüğü (Glossary)

CyberGuard AI'da kullanılan terimler ve açıklamaları

---

## A

### Accuracy

Model tahminlerinin doğru olma oranı. `(TP + TN) / (TP + TN + FP + FN)`

### Adversarial Attack

ML modellerini kandırmak için tasarlanmış manipüle edilmiş girdiler.

### AES

Advanced Encryption Standard. Simetrik şifreleme algoritması.

### API (Application Programming Interface)

Yazılımlar arası iletişim protokolü.

### AUC-ROC

Area Under ROC Curve. Model performans metriği.

### AutoML

Automated Machine Learning. Otomatik model seçimi ve hiperparametre optimizasyonu.

---

## B

### Batch Size

Model eğitiminde bir iterasyonda işlenen örnek sayısı.

### BiLSTM

Bidirectional LSTM. İki yönlü LSTM ağı.

### bcrypt

Güçlü parola hash fonksiyonu. Adaptif maliyet faktörü ile kaba kuvvet saldırılarına dayanıklı. CyberGuard'da kullanıcı şifreleri için kullanılır.

### Botnet

Saldırgan kontrolündeki zombi bilgisayar ağı.

### Brute Force

Tüm olası kombinasyonları deneyerek şifre kırma yöntemi.

---

## C

### C&W Attack

Carlini & Wagner. Güçlü adversarial saldırı yöntemi.

### CICIDS2017

Canadian Institute for Cybersecurity Intrusion Detection Dataset.

### CNN

Convolutional Neural Network. Görüntü işlemede kullanılan derin öğrenme modeli.

### CORS

Cross-Origin Resource Sharing. Web güvenlik mekanizması.

### Cross-Validation

Model performansını değerlendirmek için veriyi bölümlere ayırma.

### CVE

Common Vulnerabilities and Exposures. Güvenlik açıkları veritabanı.

### CVSS

Common Vulnerability Scoring System. Zafiyet derecelendirme sistemi (0-10).

---

## D

### DDoS

Distributed Denial of Service. Dağıtık hizmet engelleme saldırısı.

### Deep Learning

Derin öğrenme. Çok katmanlı sinir ağları.

### Differential Privacy

Gizlilik koruyan veri analizi tekniği.

### DoS

Denial of Service. Hizmet engelleme saldırısı.

### Dropout

Overfitting'i önlemek için rastgele nöronları devre dışı bırakma.

### Drift Detection

Model performansının zamanla düşüşünü tespit etme.

---

## E

### Epoch

Tüm eğitim verisinin bir kez işlenmesi.

### Ensemble

Birden fazla modelin birleştirilmesi.

### Epsilon (ε)

Adversarial saldırılarda perturbation miktarı.

---

## F

### F1-Score

Precision ve Recall'ın harmonik ortalaması.

### False Negative (FN)

Yanlışlıkla normal olarak sınıflandırılan saldırı.

### False Positive (FP)

Yanlışlıkla saldırı olarak sınıflandırılan normal trafik.

### Feature

Model girdisi olarak kullanılan özellik.

### Feature Engineering

Ham veriden anlamlı özellikler çıkarma.

### Federated Learning

Veriyi merkeze toplamadan dağıtık model eğitimi.

### FGSM

Fast Gradient Sign Method. Hızlı adversarial saldırı.

---

## G

### GDPR

General Data Protection Regulation. AB veri koruma yasası.

### Grafana

Prometheus ve diğer veri kaynaklarından alınan metrikleri görselleştiren açık kaynaklı izleme aracı. CyberGuard'da port 3001'de çalışır; istek oranı, gecikme, CPU/bellek panelleri bulunur.

### Gradient

Kayıp fonksiyonunun parametrelere göre türevi.

### GRU

Gated Recurrent Unit. LSTM'e alternatif RNN hücresi.

---

## H

### Hiperparametre

Model eğitimi öncesi belirlenen parametreler (learning rate, epochs, vb.)

### Honeypot

Saldırganları tespit etmek için kurulan sahte sistemler.

### HSM

Hardware Security Module. Kriptografik işlemler için güvenli donanım.

---

## I

### IDS

Intrusion Detection System. Saldırı tespit sistemi.

### IoC

Indicators of Compromise. Saldırı göstergeleri.

### IPS

Intrusion Prevention System. Saldırı önleme sistemi.

---

## J

### JWT

JSON Web Token. Kimlik doğrulama tokenı.

### JSON Logging (Yapılandırılmış Loglama)

Log kayıtlarını JSON formatında üretme. Her kayıt `timestamp`, `level`, `request_id`, `method`, `path`, `status_code`, `duration_ms` alanlarını içerir. CyberGuard'da `app/utils/logging.py` + `RotatingFileHandler` ile uygulanmıştır.

### JSMA

Jacobian-based Saliency Map Attack.

---

## K

### Keras

TensorFlow üzerine kurulu yüksek seviye deep learning kütüphanesi.

### KVKK

Kişisel Verilerin Korunması Kanunu.

---

## L

### L2 Distance

Euclidean mesafe. Vektörler arası uzaklık ölçümü.

### Learning Rate

Model ağırlıklarının güncelleme hızı.

### LIME

Local Interpretable Model-agnostic Explanations. XAI yöntemi.

### LLM (Large Language Model)

Geniş metin korpusları üzerinde eğitilmiş büyük dil modeli (GPT-4, Claude, LLaMA vb.). CyberGuard'da `ThreatDecisionAgent` tehdit analizi için Groq/OpenAI/Ollama LLM'lerini kullanır.

### LLM Agent

LLM tabанlı özerk karar verici. CyberGuard'daki `ThreatDecisionAgent` tehdit olayı alır, şiddet belirler, oyun planı üretir ve `data/incidents.json`'a kaydeder. LLM yapılandırılmamışsa kural tabanlı yedeğe düşer.

### LSTM

Long Short-Term Memory. Uzun vadeli bağımlılıkları öğrenebilen RNN.

---

## M

### Malware

Zararlı yazılım.

### MFA

Multi-Factor Authentication. Çok faktörlü kimlik doğrulama.

### MITRE ATT&CK

Saldırı taktik ve tekniklerinin framework'ü.

---

## N

### NAS

Neural Architecture Search. Otomatik model mimarisi keşfi.

### NLP

Natural Language Processing. Doğal dil işleme.

### NSL-KDD

Network Security Laboratory KDD Dataset.

---

## O

### One-Hot Encoding

Kategorik değişkenleri binary vektörlere dönüştürme.

### Overfitting

Modelin eğitim verisine aşırı uyum sağlaması.

---

## P

### PCAP

Packet Capture. Ağ paketlerini kaydetme formatı.

### PGD

Projected Gradient Descent. İteratif adversarial saldırı.

### Prometheus

Zaman serisi tabanlı açık kaynaklı izleme sistemi. CyberGuard'da `prometheus-fastapi-instrumentator` ile `/metrics` endpoint'i üretilir; port 9090'da çalışır. Scrape aralığı: 15s.

### Port Scanning

Açık portları tespit etmek için ağ taraması.

### Precision

TP / (TP + FP). Pozitif tahminlerin doğruluğu.

---

## R

### R2L

Remote to Local. Uzaktan yerel erişim saldırısı.

### Random Forest

Karar ağacı ensemble yöntemi.

### Rate Limiting

IP başına istek hızını sınırlama. CyberGuard'da `slowapi` kütüphanesi ile uygulanır; kaba kuvvet saldırılarına karşı koruma sağlar.

### RBAC (Role-Based Access Control)

Rol tabanlı erişim denetimi. Kullanıcılara roller atanarak erişim yetkisi verilir. CyberGuard'da `require_role(*roles)` fabrika fonksiyonu ile uygulanır; roller: `admin`, `analyst`, `viewer`.

### Recall

TP / (TP + FN). Gerçek pozitiflerin bulunma oranı.

### Refresh Token

Access token'in süresi dolduğunda yeni token almak için kullanılan uzun ömürlü token. CyberGuard'da 7 gün TTL ile `POST /auth/refresh` endpoint'i üzerinden yenilenir.

### Reinforcement Learning

Ödül/ceza ile öğrenme paradigması.

### REST API

REpresentational State Transfer. Web API mimarisi.

### RNN

Recurrent Neural Network. Tekrarlayan sinir ağı.

### Robustness

Modelin saldırılara dayanıklılığı.

---

## S

### Ruff

Rust ile yazılmış hızlı Python linter ve formatter. CyberGuard CI pipeline'da `flake8`/`black`/`isort` yerine kullanılır. `.pre-commit-config.yaml` ile yerel geliştirmede de aktif.

### Scaler

Verileri normalize eden dönüştürücü.

### SHAP

SHapley Additive exPlanations. XAI yöntemi.

### SIEM

Security Information and Event Management.

### SMOTE

Synthetic Minority Over-sampling Technique.

### SOC

Security Operations Center.

### SQL Injection

SQL komutları enjekte ederek veritabanı saldırısı.

### SSA

Sparrow Search Algorithm. Metaheuristik optimizasyon.

### STIX/TAXII

Threat intelligence paylaşım standartları.

---

## T

### TensorFlow

Google'ın açık kaynak ML framework'ü.

### Threat Intelligence

Tehdit istihbaratı.

### TLS

Transport Layer Security. Güvenli iletişim protokolü.

### Token

Kimlik doğrulama jetonu.

### Token

Kimlik doğrulama jetonu. Bkz. **Refresh Token**.

### TTL Cache (Time-To-Live Cache)

Belirli süre sonra otomatik olarak geçersiz olan önbelleğe alma mekanizması. CyberGuard'da `app/utils/cache.py` içinde `@ttl_cache(ttl=N)` dekoratörü ile uygulanır. Thread-safe, in-process; Redis gerektirmez.

### Transformer

Self-attention mekanizması kullanan model mimarisi.

### True Negative (TN)

Doğru şekilde normal olarak sınıflandırılan trafik.

### True Positive (TP)

Doğru şekilde saldırı olarak sınıflandırılan trafik.

---

## U

### U2R

User to Root. Yetki yükseltme saldırısı.

### Underfitting

Modelin veriyi yeterince öğrenememesi.

---

## V

### Validation Set

Model hiperparametrelerini ayarlamak için kullanılan veri.

### Vectorization

Metin/kategorik veriyi sayısal vektörlere dönüştürme.

### Vulnerability

Güvenlik açığı.

---

## W

### WebSocket

Çift yönlü gerçek zamanlı iletişim protokolü.

---

## X

### XAI

Explainable AI. Açıklanabilir yapay zeka.

### XGBoost

eXtreme Gradient Boosting. Gradient boosting algoritması.

### XSS

Cross-Site Scripting. Web saldırı türü.

---

## Z

### Zero-Day

Henüz yaması olmayan güvenlik açığı.

### Zero Trust

Güvenli ağ mimarisi yaklaşımı.

### ZTNA

Zero Trust Network Access.
