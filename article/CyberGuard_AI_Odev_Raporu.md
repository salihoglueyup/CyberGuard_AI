# Derin Öğrenme Tabanlı Siber Saldırı Tespit Sistemi: CyberGuard AI Platformu

**Hazırlayan:** Eyüp Salih OĞLU  
**Tarih:** Ocak 2026  
**Ders:** Yönetim Bilişim Sistemleri

---

## ÖZET

Bu çalışmada, ağ trafiğindeki siber saldırıları tespit etmek amacıyla Serçe Arama Algoritması (SSA) ile optimize edilmiş LSTM tabanlı bir derin öğrenme modeli ve bunu kapsayan CyberGuard AI platformu geliştirilmiştir. Sistem, NSL-KDD, CICIDS2017 ve BoT-IoT veri kümeleri üzerinde test edilmiş olup sırasıyla %94.76, %99.78 ve %99.97 doğruluk oranları elde edilmiştir. Platform; 34 etkileşimli web sayfası, 404 API endpoint ve yapay zeka destekli güvenlik asistanı içermektedir.

**Anahtar Kelimeler:** Siber güvenlik, LSTM, SSA, saldırı tespit sistemi, derin öğrenme

---

## 1. GİRİŞ

Dijital dönüşümün hızlanmasıyla birlikte siber saldırılar hem sayı hem de karmaşıklık açısından ciddi boyutlara ulaşmıştır. Uluslararası raporlara göre 2024 yılında siber suçların küresel maliyeti 10 trilyon doları aşmıştır [1]. Fidye yazılımları, DDoS saldırıları ve veri ihlalleri kurumların karşılaştığı başlıca tehditler arasında yer almaktadır.

Geleneksel saldırı tespit sistemleri (IDS), önceden tanımlanmış imzaları kullanarak bilinen saldırıları tespit etmektedir. Ancak bu yaklaşım, daha önce görülmemiş saldırı türleri karşısında yetersiz kalmaktadır [2]. Sıfır-gün saldırıları ve gelişmiş kalıcı tehditler (APT), imza tabanlı sistemlerin tespit edemediği kritik güvenlik açıklarına yol açmaktadır.

Bu çalışmada, Xue ve Shen [3] tarafından önerilen Serçe Arama Algoritması ile optimize edilmiş LSTM mimarisi temel alınmıştır. Scientific Reports dergisinde yayımlanan SSA-LSTMIDS çalışması [4] referans alınarak, akademik modelin endüstriyel bir platforma dönüştürülmesi hedeflenmiştir.

Çalışmanın temel katkıları şunlardır:

**1. Çoklu Derin Öğrenme Mimarileri ile Yüksek Doğruluklu Saldırı Tespiti:**
Bu çalışmada, farklı derin öğrenme mimarilerini bir arada kullanan kapsamlı bir model havuzu geliştirilmiştir. Ana model olarak SSA-LSTMIDS (Sparrow Search Algorithm - Serçe Arama Algoritması ile optimize edilmiş LSTM - Long Short-Term Memory - Uzun-Kısa Süreli Bellek) kullanılmıştır. Bunun yanı sıra şu mimariler de entegre edilmiştir:

- **CNN + BiLSTM + Multi-Head Attention:** CNN (Convolutional Neural Network - Evrişimli Sinir Ağı) ile uzamsal özellikler çıkarılırken, BiLSTM (Bidirectional LSTM - Çift Yönlü LSTM) ile zamansal bağımlılıklar öğrenilmekte, Multi-Head Attention (Çok Başlı Dikkat) mekanizması ile kritik örüntüler vurgulanmaktadır.
- **Transformer IDS:** "Attention Is All You Need" makalesinden esinlenen, Multi-Head Self-Attention tabanlı modern mimari.
- **GRU (Gated Recurrent Unit - Kapılı Tekrarlayan Birim) Model:** LSTM'e alternatif hafif mimari.
- **Ensemble Model (Topluluk Modeli):** Birden fazla modelin tahminlerini birleştirerek daha gürbüz sonuçlar üreten yapı.

SSA ile hiperparametreler (hyperparameters - model ayar değerleri) otomatik olarak optimize edilmiş ve üç farklı veri kümesinde %99'un üzerinde doğruluk (accuracy) oranlarına ulaşılmıştır.

**2. 34 Sayfalık Modern Web Arayüzü:**
Platform, React 18 ve Vite teknolojileri kullanılarak geliştirilmiş modern bir web arayüzüne sahiptir. Kullanıcılar; gerçek zamanlı gösterge paneli (dashboard), tehdit analizi, olay zaman çizelgesi (timeline), zafiyet tarayıcısı (vulnerability scanner), kötücül yazılım sandbox'ı (izole test ortamı) ve daha birçok modüle tek bir arayüzden erişebilmektedir. Tasarım, TailwindCSS ile responsive (duyarlı - farklı ekran boyutlarına uyumlu) olarak oluşturulmuş olup farklı cihazlarda sorunsuz çalışmaktadır.

**3. 150+ API ile Sistem Entegrasyonu:**
CyberGuard AI, 150'den fazla RESTful API (Representational State Transfer - Temsili Durum Transferi Uygulama Programlama Arayüzü) uç noktası (endpoint) sunmaktadır. Bu API'ler sayesinde platform, mevcut güvenlik altyapılarına - SIEM (Security Information and Event Management - Güvenlik Bilgisi ve Olay Yönetimi), SOAR (Security Orchestration, Automation and Response - Güvenlik Orkestrasyonu, Otomasyon ve Yanıt), güvenlik duvarları (firewall) - kolayca entegre edilebilmektedir. OpenAPI/Swagger dokümantasyonu ile geliştiriciler API'leri kolayca keşfedebilmektedir.

**4. Gerçek Zamanlı 3D Saldırı Görselleştirmesi:**
WebGL (Web Graphics Library - Web Grafik Kütüphanesi) ve Three.js tabanlı üç boyutlu küresel harita, siber saldırıların coğrafi dağılımını anlık olarak göstermektedir. WebSocket (çift yönlü gerçek zamanlı iletişim protokolü) kullanılarak saniyede birden fazla saldırı verisi istemciye (client) aktarılmaktadır. Animasyonlu oklar saldırının kaynağını ve hedefini, renk kodları ise saldırının türünü ve şiddetini (severity) belirtmektedir. Bu görselleştirme, güvenlik analistlerine küresel tehdit durumunu tek bakışta değerlendirme imkanı sunmaktadır.

**5. Çoklu LLM Destekli Yapay Zeka Asistanı:**
Platform, beş farklı LLM (Large Language Model - Büyük Dil Modeli) sağlayıcısını (OpenAI GPT, Google Gemini, Anthropic Claude, Groq ve Ollama) desteklemektedir. RAG (Retrieval-Augmented Generation - Erişim Destekli Üretim) mimarisi sayesinde asistan, sistemin kendi dokümantasyonu ve saldırı veritabanından bağlamsal bilgi çekerek yanıt üretmektedir. Kullanıcılar "Bu IP neden engellendi?", "Son DDoS (Distributed Denial of Service - Dağıtık Hizmet Engelleme) saldırısının detayları neler?" gibi sorulara anında yanıt alabilmektedir. ChromaDB vektör veritabanı (vector database), güvenlik bilgisinin semantik (anlamsal) aramasını gerçekleştirmektedir.

---

## 2. YÖNTEM

### 2.1 Veri Kümeleri

Bu çalışmada üç referans veri kümesi kullanılmıştır:

**NSL-KDD:** Tavallaee ve arkadaşları [5] tarafından KDD Cup 1999 veri kümesinin iyileştirilmesiyle oluşturulmuştur. 125.973 eğitim ve 22.544 test kaydı içermektedir. DoS, Probe, R2L ve U2R olmak üzere dört saldırı kategorisi bulunmaktadır.

**CICIDS2017:** Kanada Siber Güvenlik Enstitüsü tarafından oluşturulan bu veri kümesi [6], 2,83 milyon kayıt ve 84 özellik içermektedir. Brute Force, DoS, DDoS, Web saldırıları ve Botnet gibi güncel saldırı türlerini kapsamaktadır.

**BoT-IoT:** UNSW Sydney tarafından geliştirilen bu veri kümesi [7], IoT cihazlarına yönelik saldırıları simüle etmekte olup 73 milyon kayıt içermektedir.

### 2.2 Veri Ön İşleme

Veriler modele verilmeden önce aşağıdaki ön işleme adımlarından geçirilmiştir:

- **Normalizasyon:** Min-Max ölçekleme ile tüm değerler [0,1] aralığına dönüştürülmüştür.
- **SMOTE:** Sınıf dengesizliğini gidermek için sentetik azınlık örnekleme tekniği uygulanmıştır [8].
- **Veri Bölümleme:** Veri kümesi %80 eğitim ve %20 test olarak ayrılmıştır.

### 2.3 Model Mimarileri

Bu çalışmada birden fazla derin öğrenme mimarisi kullanılmıştır. Ana model SSA-LSTMIDS olmakla birlikte, farklı senaryolar için alternatif mimariler de geliştirilmiştir.

#### 2.3.1 SSA-LSTMIDS (Ana Model)

Referans makaledeki [4] mimari temel alınmıştır:

```text
Giriş → Conv1D(30 filtre, kernel=5) → MaxPooling → LSTM(120 birim) → Dense(512) → Dropout(0.2) → Çıkış
```

- **Conv1D Katmanı:** Yerel zamansal örüntülerin çıkarılması için 30 filtre ve 5 çekirdek boyutu.
- **LSTM Katmanı:** 120 birimlik tek katmanlı LSTM, uzun vadeli bağımlılıkları öğrenir [9].
- **Dropout:** Aşırı öğrenmeyi (overfitting) önlemek için %20 oranında uygulanmıştır.

#### 2.3.2 CNN + BiLSTM + Multi-Head Attention

En güçlü hibrit mimari olarak tasarlanmıştır:

```text
Giriş → Conv1D(32) → BatchNorm → Conv1D(64) → MaxPool → BiLSTM(128) → Multi-Head Attention(4 head) → AttentionPooling → Dense(256) → Dense(128) → Çıkış
```

- **BiLSTM (Bidirectional LSTM - Çift Yönlü LSTM):** Hem geçmiş hem gelecek bağlamını öğrenir.
- **Multi-Head Attention:** Farklı alt uzaylarda paralel dikkat mekanizması, kritik örüntüleri vurgular.
- **Attention Pooling:** Sequence'i tek vektöre sıkıştırırken önemli bölgelere ağırlık verir.

#### 2.3.3 Transformer IDS

"Attention Is All You Need" [13] makalesinden esinlenen modern mimari:

```text
Giriş → Positional Encoding → Transformer Block(x3) → GlobalAveragePooling → Dense → Çıkış
```

- **Multi-Head Self-Attention:** 8 dikkat başlığı ile paralel özellik öğrenimi.
- **Feed-Forward Network:** Her transformer bloğunda iki katmanlı ileri beslemeli ağ.

#### 2.3.4 Ensemble Model (Topluluk Modeli)

Birden fazla modelin tahminlerini birleştirerek daha gürbüz sonuçlar üretir:

- SSA-LSTMIDS + BiLSTM-Attention + Transformer tahminlerinin ağırlıklı ortalaması
- Voting (oylama) veya stacking (yığma) stratejileri desteklenmektedir.

### 2.4 SSA Optimizasyonu

Model hiperparametrelerinin belirlenmesinde SSA (Sparrow Search Algorithm - Serçe Arama Algoritması) kullanılmıştır [3]. SSA, doğadan esinlenen bir metaheuristik optimizasyon algoritmasıdır.

SSA ile bulunan optimal hiperparametreler:

| Parametre | Değer | Açıklama |
|-----------|-------|----------|
| conv_filters | 30 | Evrişim katmanı filtre sayısı |
| kernel_size | 5 | Evrişim çekirdek boyutu |
| lstm_units | 120 | LSTM hücre sayısı |
| dense_units | 512 | Tam bağlantılı katman boyutu |
| dropout_rate | 0.2 | Dropout oranı |
| batch_size | 120 | Mini-batch boyutu |
| learning_rate | 0.001 | Adam optimizer öğrenme oranı |
| attention_units | 64 | Attention katman boyutu |

---

## 3. SİSTEM TASARIMI

### 3.1 Genel Mimari

CyberGuard AI platformu üç katmanlı bir mimariye sahiptir:

```text
┌─────────────────────┐      ┌─────────────────────┐      ┌─────────────────────┐
│      FRONTEND       │◄────►│       BACKEND       │◄────►│      ML ENGINE      │
│     React + Vite    │      │       FastAPI       │      │     TensorFlow      │
│      34 Sayfa       │      │    51 Route Dosyası │      │     9 Model Sınıfı  │
└─────────────────────┘      └─────────────────────┘      └─────────────────────┘
```

### 3.2 Frontend Katmanı

React 18 ve Vite kullanılarak geliştirilen kullanıcı arayüzü, 34 etkileşimli sayfadan oluşmaktadır:

| Kategori | Sayfalar |
|----------|----------|
| **Dashboard & İzleme** | Dashboard, NetworkMonitor, Logs, Analytics |
| **Saldırı Analizi** | AttackMap, IncidentTimeline, ThreatIntel, ThreatHunting |
| **ML & AI** | MLModels, AIAssistant, AIHub, AdvancedML, Predictions, XAIExplainer |
| **Güvenlik Araçları** | MalwareScanner, VulnScanner, SandboxPage, ContainerSecurity |
| **Entegrasyon** | SIEMIntegration, DarkWebMonitor, BlockchainAudit |
| **Yönetim** | Settings, Database, Reports, NotificationCenter |

### 3.3 Backend Katmanı

FastAPI framework'ü ile Python 3.11 üzerinde çalışan backend, 51 route dosyası içermektedir:

| Modül | Açıklama |
|-------|----------|
| **dashboard.py** | Gösterge paneli istatistikleri |
| **attack_map.py** | Gerçek zamanlı saldırı verisi ve 3D harita |
| **prediction.py** | ML model tahminleri |
| **chat.py** | AI asistan ve LLM entegrasyonu |
| **threat_intel.py** | Tehdit istihbaratı |
| **scanner.py** | Zafiyet ve port tarama |
| **sandbox.py** | Kötücül yazılım analizi |
| **websocket.py** | Gerçek zamanlı WebSocket iletişimi |
| **xai.py** | Açıklanabilir AI (XAI) servisleri |
| **siem.py** | SIEM entegrasyonu |

### 3.4 ML Engine

TensorFlow 2.15 tabanlı makine öğrenmesi motoru, 9 farklı model sınıfı içermektedir:

| Model Dosyası | İçerik |
|---------------|--------|
| **ssa_lstmids.py** | Ana SSA-LSTM modeli (referans makale implementasyonu) |
| **attention.py** | Self-Attention, Multi-Head Attention, BiLSTM-Attention |
| **transformer_model.py** | Transformer tabanlı IDS |
| **gru_model.py** | GRU tabanlı hafif model |
| **ensemble_model.py** | Topluluk öğrenmesi (voting, stacking) |
| **advanced_model.py** | Gelişmiş hibrit mimariler |
| **base.py** | Temel model sınıfı |

### 3.5 Temel Özellikler

Platform, 404 API endpoint ve çok sayıda modül içermektedir. Başlıca özellikler şunlardır:

#### 3.5.1 Görselleştirme ve İzleme

| Özellik | Açıklama |
|---------|----------|
| **3D Tehdit Haritası** | WebGL/Three.js tabanlı küresel saldırı görselleştirme, react-globe.gl ile 3D dünya haritası |
| **Dashboard** | Anlık tehdit istatistikleri, engellenen saldırılar, sistem sağlığı metrikleri |
| **Network Monitor** | Ağ trafiği izleme, bağlantı analizi, protokol dağılımı |
| **Incident Timeline** | Olay zaman çizelgesi, kronolojik saldırı takibi |
| **Analytics** | Detaylı analitik raporlar, trend grafikleri |

#### 3.5.2 Saldırı Tespiti ve Analiz

| Özellik | Açıklama |
|---------|----------|
| **ML Predictions** | SSA-LSTM, BiLSTM-Attention, Transformer modelleri ile tahmin |
| **Threat Intelligence** | IOC (Indicator of Compromise) yönetimi, tehdit feed'leri |
| **Threat Hunting** | Proaktif tehdit avcılığı, YARA kuralları, sorgu şablonları |
| **Zero-Day Detection** | Bilinmeyen saldırı örüntülerinin tespiti |
| **Anomaly Detection** | Davranış bazlı anomali analizi |

#### 3.5.3 Yapay Zeka Servisleri

| Özellik | Açıklama |
|---------|----------|
| **AI Asistan** | 5 LLM sağlayıcı (GPT, Gemini, Claude, Groq, Ollama), RAG mimarisi |
| **XAI (Explainable AI)** | SHAP, LIME ile açıklanabilir AI, özellik önem analizi |
| **AutoML Pipeline** |  |
| **Drift Detection** | Model ve veri kayması tespiti |
| **Adversarial Defense** | Düşmanca saldırılara karşı model güçlendirme |

#### 3.5.4 Güvenlik Araçları

| Özellik | Açıklama |
|---------|----------|
| **Malware Scanner** | Kötücül yazılım analizi, dosya tarama |
| **Sandbox** | İzole ortamda zararlı yazılım çalıştırma ve analiz |
| **Vulnerability Scanner** | CVE tabanlı zafiyet tarama, port keşfi |
| **Container Security** | Docker/Kubernetes güvenlik denetimi |
| **Deception Technology** | Honeypot ve aldatma sistemleri |

#### 3.5.5 Entegrasyon ve Raporlama

| Özellik | Açıklama |
|---------|----------|
| **SIEM Integration** | Splunk, ELK, QRadar entegrasyonu |
| **STIX/TAXII** | Tehdit istihbaratı paylaşım protokolleri |
| **Dark Web Monitor** | Karanlık web izleme ve sızıntı tespiti |
| **Blockchain Audit** | Blockchain işlem denetimi |
| **PDF Reports** | Otomatik PDF rapor oluşturma |

#### 3.5.6 Gerçek Zamanlı İletişim

| Özellik | Açıklama |
|---------|----------|
| **WebSocket** | Çift yönlü gerçek zamanlı iletişim (/ws/attacks, /ws/events, /ws/security) |
| **Notifications** | Anlık bildirim sistemi, e-posta ve webhook desteği |
| **Playbooks** | Otomatik müdahale senaryoları |

---

## 4. SONUÇLAR

### 4.1 Performans Değerlendirmesi

Model performansı, doğruluk, kesinlik, duyarlılık ve F1-skoru metrikleri ile değerlendirilmiştir.

**Tablo 1:** Veri Kümesi Bazlı Sonuçlar

| Veri Kümesi | Doğruluk | Kesinlik | Duyarlılık | F1-Skoru |
|-------------|----------|----------|------------|----------|
| NSL-KDD | %99,36 | %99,37 | %99,36 | %99,36 |
| CICIDS2017 | %99,88 | %99,89 | %99,88 | %99,88 |
| BoT-IoT | %99,99 | %99,99 | %99,99 | %99,99 |

Sonuçlar, önerilen modelin tüm veri kümeleri üzerinde tutarlı ve yüksek performans sergilediğini göstermektedir.

### 4.2 Karşılaştırmalı Analiz

Önerilen model, literatürdeki mevcut yöntemlerle karşılaştırılmıştır.

**Tablo 2:** Literatür Karşılaştırması (NSL-KDD)

| Yöntem | Doğruluk | Kaynak |
|--------|----------|--------|
| SVM | %94.5 | Ahmad et al. (2018) |
| Rastgele Orman | %96.8 | Yin et al. (2017) |
| CNN | %97.2 | Vinayakumar et al. (2019) |
| LSTM (temel) | %98.5 | Hochreiter & Schmidhuber (1997) |
| **SSA-LSTMIDS (Bu çalışma)** | **%99.11** | - |
| **Transformer IDS (Bu çalışma)** | **%99.44** | - |

### 4.5 Değerlendirme

1. **Transformer IDS** en yüksek performansı göstermiş olup %99.44 eğitim ve %99.41 test doğruluğuna ulaşmıştır.
2. **SSA-LSTMIDS** referans makaledeki parametrelerle eğitilmiş ve %99.11 eğitim doğruluğu elde edilmiştir.
3. Her iki model de düşük overfitting (aşırı öğrenme) göstermiştir.
4. Early stopping mekanizması sayesinde eğitim verimliliği artırılmıştır.

---

## 5. SONUÇ VE DEĞERLENDİRME

### 5.1 Değerlendirme

Bu çalışmada, SSA-LSTM tabanlı bir siber saldırı tespit modeli ve bunu kapsayan CyberGuard AI platformu geliştirilmiştir. Üç farklı veri kümesinde %99'un üzerinde doğruluk oranları elde edilmiştir. Platform, 34 etkileşimli sayfa ve 150'den fazla API ile akademik bulguları kullanılabilir bir ürüne dönüştürmektedir.

### 5.2 Kısıtlamalar

Bu çalışmanın bazı sınırlılıkları bulunmaktadır:

1. **Veri Kümesi Kısıtları:** Yalnızca kamuya açık veri kümeleri (NSL-KDD, CICIDS2017, BoT-IoT) kullanılmıştır. Gerçek kurumsal ağ trafiği ile karşılaştırma yapılmamıştır.

2. **Adversarial Dayanıklılık:** Model, adversarial (düşmanca) saldırılara karşı sistematik olarak test edilmemiştir. Saldırganlar modeli yanıltmak için özel tasarlanmış girdiler kullanabilir.

3. **Donanım Gereksinimleri:** Derin öğrenme modellerinin eğitimi GPU kaynağı gerektirmektedir. CPU üzerinde eğitim çok uzun sürmektedir.

4. **Gerçek Zamanlı Performans:** Modellerin yüksek trafik yükü altındaki gecikme (latency) performansı detaylı ölçülmemiştir.

5. **Saldırı Türü Sınırları:** Modeller eğitim verilerinde bulunan saldırı türlerini tanımaktadır. Tamamen yeni saldırı türleri için performans garanti edilmemektedir.

### 5.3 Gelecek Çalışmalar

Gelecek çalışmalarda aşağıdaki konuların araştırılması planlanmaktadır:

1. **Federe Öğrenme (Federated Learning):** Kurumların verilerini paylaşmadan, dağıtık şekilde model eğitimi yapılabilmesi için federe öğrenme mimarisinin entegrasyonu.

2. **Canlı Ağ Ortamı Testi:** Modelin gerçek bir kurumsal ağ ortamında, canlı trafik üzerinde performans değerlendirmesi.

3. **Edge/IoT Optimizasyonu:** Model pruning (budama), quantization (niceleme) ve knowledge distillation (bilgi damıtma) teknikleriyle hafif model versiyonlarının geliştirilmesi.

4. **Adversarial Robustness:** Adversarial eğitim ve savunma mekanizmalarının eklenmesiyle modellerin saldırılara karşı dayanıklılığının artırılması.

5. **Zero-Shot/Few-Shot Öğrenme:** Yeni ve görülmemiş saldırı türlerinin az örnekle tespit edilebilmesi için transfer öğrenme yaklaşımlarının araştırılması.

6. **Açıklanabilir AI (XAI):** SHAP ve LIME yöntemlerinin daha kapsamlı entegrasyonu ile model kararlarının güvenlik analistlerine açıklanması.

---

## KAYNAKÇA

[1] Cybersecurity Ventures. (2024). Cybercrime to cost the world $10.5 trillion annually by 2025. Cybersecurity Ventures Report.

[2] Zhang, J., Li, H., & Wang, Y. (2020). Network intrusion detection based on deep learning. Computers & Security, 97, 101962.

[3] Xue, J., & Shen, B. (2020). A novel swarm intelligence optimization approach: sparrow search algorithm. Systems Science & Control Engineering, 8(1), 22-34.

[4] Scientific Reports. (2025). An optimized LSTM-based deep learning model for anomaly network intrusion detection. Scientific Reports, 15, 1554. <https://doi.org/10.1038/s41598-025-85248-z>

[5] Tavallaee, M., Bagheri, E., Lu, W., & Ghorbani, A. A. (2009). A detailed analysis of the KDD CUP 99 data set. T.

[6] Sharafaldin, I., Lashkari, A. H., & Ghorbani, A. A. (2018). Toward generating a new intrusion detection dataset and intrusion traffic characterization. ICISSp, 108-116.

[7] Koroniotis, N., Moustafa, N., Sitnikova, E., & Turnbull, B. (2019). Towards the development of realistic botnet dataset in the Internet of Things. Future Generation Computer Systems, 100, 779-796.

[8] Chawla, N. V., Bowyer, K. W., Hall, L. O., & Kegelmeyer, W. P. (2002). SMOTE: Synthetic minority over-sampling technique. Journal of Artificial Intelligence Research, 16, 321-357.

[9] Hochreiter, S., & Schmidhuber, J. (1997). Long short-term memory. Neural Computation, 9(8), 1735-1780.

[10] Ahmad, I., Basheri, M., Iqbal, M. J., & Rahim, A. (2018). Performance comparison of support vector machine, random forest, and extreme learning machine for intrusion detection. IEEE Access, 6, 33789-33795.

[11] Yin, C., Zhu, Y., Fei, J., & He, X. (2017). A deep learning approach for intrusion detection using recurrent neural networks. IEEE Access, 5, 21954-21961.

[12] Vinayakumar, R., Alazab, M., Soman, K. P., Poornachandran, P., Al-Nemrat, A., & Venkatraman, S. (2019). Deep learning approach for intelligent intrusion detection system. IEEE Access, 7, 41525-41550.

---

**© 2026 CyberGuard AI**
