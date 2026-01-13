# 📊 Datasets Dokümantasyonu

CyberGuard AI'da kullanılan veri setleri hakkında detaylı bilgi

---

## 📋 İçindekiler

- [Genel Bakış](#genel-bakış)
- [NSL-KDD Dataset](#nsl-kdd-dataset)
- [CICIDS2017 Dataset](#cicids2017-dataset)
- [BoT-IoT Dataset](#bot-iot-dataset)
- [Veri Ön İşleme](#veri-ön-işleme)
- [Feature Engineering](#feature-engineering)

---

## 🌟 Genel Bakış

### Desteklenen Veri Setleri

| Dataset | Kayıt Sayısı | Özellik Sayısı | Sınıf Sayısı | Yıl |
|---------|--------------|----------------|--------------|-----|
| NSL-KDD | 148,517 | 41 | 5 | 2009 |
| CICIDS2017 | 2,830,743 | 78 | 15 | 2017 |
| BoT-IoT | 73,370,443 | 43 | 11 | 2019 |

### Veri Seti Karşılaştırması

```
NSL-KDD     ████░░░░░░░░░░░░░░░░ 148K kayıt
CICIDS2017  ████████░░░░░░░░░░░░ 2.8M kayıt  
BoT-IoT     ████████████████████ 73M kayıt
```

---

## 🔐 NSL-KDD Dataset

### Genel Bilgiler

| Özellik | Değer |
|---------|-------|
| **Kaynak** | University of New Brunswick (UNB) |
| **Yıl** | 2009 |
| **Orijinal** | KDD'99 Dataset (iyileştirilmiş) |
| **Boyut** | ~130 MB |
| **İndirme** | [NSL-KDD Dataset](https://www.unb.ca/cic/datasets/nsl.html) |

### Saldırı Türleri

| Kategori | Saldırı Türleri | Örnek |
|----------|-----------------|-------|
| **DoS** | back, land, neptune, pod, smurf, teardrop | Denial of Service |
| **Probe** | ipsweep, nmap, portsweep, satan | Network scanning |
| **R2L** | ftp_write, guess_passwd, imap, multihop | Remote to Local |
| **U2R** | buffer_overflow, loadmodule, perl, rootkit | User to Root |

### Özellikler (41 Feature)

**Temel Özellikler:**

```
duration, protocol_type, service, flag, src_bytes, dst_bytes,
land, wrong_fragment, urgent
```

**İçerik Özellikleri:**

```
hot, num_failed_logins, logged_in, num_compromised, root_shell,
su_attempted, num_root, num_file_creations, num_shells,
num_access_files, num_outbound_cmds, is_host_login, is_guest_login
```

**Trafik Özellikleri:**

```
count, srv_count, serror_rate, srv_serror_rate, rerror_rate,
srv_rerror_rate, same_srv_rate, diff_srv_rate, srv_diff_host_rate
```

**Host Özellikleri:**

```
dst_host_count, dst_host_srv_count, dst_host_same_srv_rate,
dst_host_diff_srv_rate, dst_host_same_src_port_rate,
dst_host_srv_diff_host_rate, dst_host_serror_rate,
dst_host_srv_serror_rate, dst_host_rerror_rate, dst_host_srv_rerror_rate
```

### Sınıf Dağılımı

```
Normal      ██████████████████░░ 67,343 (45.3%)
DoS         █████████████░░░░░░░ 45,927 (30.9%)
Probe       ████░░░░░░░░░░░░░░░░ 11,656 (7.8%)
R2L         ███░░░░░░░░░░░░░░░░░ 995 (0.7%)
U2R         ░░░░░░░░░░░░░░░░░░░░ 52 (0.03%)
```

---

## 🌐 CICIDS2017 Dataset

### Genel Bilgiler

| Özellik | Değer |
|---------|-------|
| **Kaynak** | Canadian Institute for Cybersecurity |
| **Yıl** | 2017 |
| **Süre** | 5 iş günü |
| **Boyut** | ~8 GB |
| **İndirme** | [CICIDS2017 Dataset](https://www.unb.ca/cic/datasets/ids-2017.html) |

### Saldırı Türleri (15 Sınıf)

| Gün | Saldırı Türü | Açıklama |
|-----|-------------|----------|
| Monday | Normal | Sadece normal trafik |
| Tuesday | FTP-Patator, SSH-Patator | Brute force saldırıları |
| Wednesday | DoS Slowloris, DoS Slowhttptest, DoS Hulk, DoS GoldenEye, Heartbleed | DoS saldırıları |
| Thursday | Web Attack (XSS, SQL Injection, Brute Force), Infiltration | Web saldırıları |
| Friday | Botnet, Port Scan, DDoS | Distributed saldırılar |

### Özellikler (78 Feature)

**Flow Özellikleri:**

```
Flow Duration, Total Fwd Packets, Total Backward Packets,
Total Length of Fwd Packets, Total Length of Bwd Packets,
Fwd Packet Length Max/Min/Mean/Std, Bwd Packet Length Max/Min/Mean/Std
```

**Zaman Özellikleri:**

```
Flow Bytes/s, Flow Packets/s, Flow IAT Mean/Std/Max/Min,
Fwd IAT Total/Mean/Std/Max/Min, Bwd IAT Total/Mean/Std/Max/Min
```

**Flag Özellikleri:**

```
Fwd PSH Flags, Bwd PSH Flags, Fwd URG Flags, Bwd URG Flags,
Fwd Header Length, Bwd Header Length
```

**Paket Özellikleri:**

```
Fwd Packets/s, Bwd Packets/s, Min Packet Length, Max Packet Length,
Packet Length Mean/Std/Variance, FIN Flag Count, SYN Flag Count,
RST Flag Count, PSH Flag Count, ACK Flag Count, URG Flag Count
```

### Sınıf Dağılımı

```
BENIGN               ████████████████████ 2,273,097 (80.3%)
DDoS                 ███░░░░░░░░░░░░░░░░░ 128,027 (4.5%)
PortScan             ███░░░░░░░░░░░░░░░░░ 158,930 (5.6%)
DoS Hulk             ██░░░░░░░░░░░░░░░░░░ 231,073 (8.2%)
DoS GoldenEye        ░░░░░░░░░░░░░░░░░░░░ 10,293 (0.4%)
FTP-Patator          ░░░░░░░░░░░░░░░░░░░░ 7,938 (0.3%)
SSH-Patator          ░░░░░░░░░░░░░░░░░░░░ 5,897 (0.2%)
DoS Slowloris        ░░░░░░░░░░░░░░░░░░░░ 5,796 (0.2%)
...
```

---

## 🤖 BoT-IoT Dataset

### Genel Bilgiler

| Özellik | Değer |
|---------|-------|
| **Kaynak** | UNSW Sydney |
| **Yıl** | 2019 |
| **Ortam** | IoT Network Simulation |
| **Boyut** | ~16 GB |
| **İndirme** | [BoT-IoT Dataset](https://research.unsw.edu.au/projects/bot-iot-dataset) |

### Saldırı Türleri

| Kategori | Alt Türler | Açıklama |
|----------|------------|----------|
| **DDoS** | UDP, TCP, HTTP | Distributed DoS |
| **DoS** | UDP, TCP, HTTP | Denial of Service |
| **Reconnaissance** | OS, Service | Keşif saldırıları |
| **Theft** | Data, Keylogging | Veri çalma |
| **Normal** | - | Normal IoT trafiği |

### Özellikler (43 Feature)

```
pkSeqID, stime, flgs, flgs_number, proto, proto_number,
saddr, sport, daddr, dport, pkts, bytes, state, state_number,
ltime, seq, dur, mean, stddev, sum, min, max, spkts, dpkts,
sbytes, dbytes, rate, srate, drate, TnBPSrcIP, TnBPDstIP,
TnP_PSrcIP, TnP_PDstIP, TnP_PerProto, TnP_Per_Dport, AR_P_Proto_P_SrcIP,
AR_P_Proto_P_DstIP, N_IN_Conn_P_DstIP, N_IN_Conn_P_SrcIP,
AR_P_Proto_P_Sport, AR_P_Proto_P_Dport, Pkts_P_State_P_Protocol_P_DestIP,
Pkts_P_State_P_Protocol_P_SrcIP
```

### Sınıf Dağılımı

```
DDoS                 ████████████████████ 56,844,535 (77.5%)
DoS                  ████░░░░░░░░░░░░░░░░ 12,315,997 (16.8%)
Normal               █░░░░░░░░░░░░░░░░░░░ 477,116 (0.7%)
Reconnaissance       █░░░░░░░░░░░░░░░░░░░ 2,652,191 (3.6%)
Theft                ░░░░░░░░░░░░░░░░░░░░ 1,080,604 (1.5%)
```

---

## 🔧 Veri Ön İşleme

### 1. Veri Temizleme

```python
import pandas as pd
import numpy as np

# NaN ve Infinity değerleri temizle
df = df.replace([np.inf, -np.inf], np.nan)
df = df.dropna()

# Duplicate kayıtları kaldır
df = df.drop_duplicates()

# Outlier temizleme (IQR method)
Q1 = df.quantile(0.25)
Q3 = df.quantile(0.75)
IQR = Q3 - Q1
df = df[~((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))).any(axis=1)]
```

### 2. Feature Encoding

```python
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

# Label Encoding (ordinal)
le = LabelEncoder()
df['protocol_type'] = le.fit_transform(df['protocol_type'])

# One-Hot Encoding (categorical)
df = pd.get_dummies(df, columns=['service', 'flag'])
```

### 3. Feature Scaling

```python
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# StandardScaler (mean=0, std=1)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# MinMaxScaler (0-1 range)
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
```

### 4. Class Balancing

```python
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

# SMOTE (Synthetic Minority Over-sampling)
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Random Under-sampling (majority class)
rus = RandomUnderSampler(random_state=42)
X_resampled, y_resampled = rus.fit_resample(X, y)
```

---

## 🎯 Feature Engineering

### 1. Temporal Features

```python
# Zaman bazlı özellikler
df['hour'] = pd.to_datetime(df['timestamp']).dt.hour
df['day_of_week'] = pd.to_datetime(df['timestamp']).dt.dayofweek
df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
```

### 2. Statistical Features

```python
# Grup bazlı istatistikler
df['src_ip_count'] = df.groupby('source_ip')['source_ip'].transform('count')
df['dst_port_mean_bytes'] = df.groupby('dst_port')['bytes'].transform('mean')
```

### 3. Rolling Window Features

```python
# Sliding window istatistikleri
df['rolling_mean_bytes'] = df['bytes'].rolling(window=100).mean()
df['rolling_std_bytes'] = df['bytes'].rolling(window=100).std()
```

### 4. Feature Selection

```python
from sklearn.feature_selection import SelectKBest, mutual_info_classif

# Mutual Information
selector = SelectKBest(mutual_info_classif, k=50)
X_selected = selector.fit_transform(X, y)

# Feature importance from Random Forest
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=100)
rf.fit(X, y)
importances = rf.feature_importances_
```

---

## 📁 Veri Seti Dosyaları

```
data/
├── nsl_kdd/
│   ├── KDDTrain+.txt
│   ├── KDDTest+.txt
│   └── processed/
├── cicids2017/
│   ├── Monday-WorkingHours.pcap_ISCX.csv
│   ├── Tuesday-WorkingHours.pcap_ISCX.csv
│   ├── Wednesday-workingHours.pcap_ISCX.csv
│   ├── Thursday-WorkingHours.pcap_ISCX.csv
│   ├── Friday-WorkingHours.pcap_ISCX.csv
│   └── processed/
└── bot_iot/
    ├── UNSW_2018_IoT_Botnet_Dataset_*.csv
    └── processed/
```

---

## 📝 Referanslar

- [NSL-KDD Dataset Analysis](https://ieeexplore.ieee.org/document/5356528)
- [CICIDS2017: A Realistic Cyber Defense Dataset](https://www.scitepress.org/Papers/2018/66398/66398.pdf)
- [BoT-IoT: Building Automation Attack Dataset](https://ieeexplore.ieee.org/document/8717639)
