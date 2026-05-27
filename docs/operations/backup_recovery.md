# 💾 Backup & Recovery Guide

CyberGuard AI yedekleme ve kurtarma rehberi

---

## 📋 İçindekiler

- [Veri Kaynakları](#veri-kaynakları)
- [Yedekleme Stratejisi](#yedekleme-stratejisi)
- [SQLite Backup](#sqlite-backup)
- [JSON Veri Dosyaları Backup](#json-veri-dosyaları-backup)
- [Model Backup](#model-backup)
- [Log Backup](#log-backup)
- [Disaster Recovery](#disaster-recovery)
- [Checklist](#checklist)

---

## 🗂️ Veri Kaynakları

CyberGuard AI üç tür veri saklama kullanır (PostgreSQL/Redis yoktur):

| Kaynak | Konum | İçerik |
|--------|-------|---------|
| **SQLite** | `src/database/cyberguard.db` | İlişkisel veriler |
| **ChromaDB** | `src/database/chroma/` | Vektör embedding'leri |
| **JSON dosyaları** | `data/` | Kullanıcılar, oturumlar, olaylar, yapılandırma |
| **Keras modeller** | `model_artifacts/*.keras` | Eğitilmiş ML modelleri |
| **Loglar** | `logs/app/cyberguard.log` | Uygulama logları (rotating) |

---

## 🎯 Yedekleme Stratejisi

### 3-2-1 Kuralı

- **3** kopya (orijinal + 2 yedek)
- **2** farklı ortam (local + harici disk/bulut)
- **1** off-site yedek

### Yedekleme Sıklığı

| Veri Türü | Sıklık | Saklama |
|-----------|--------|---------|
| SQLite DB | Günlük | 30 gün |
| JSON veri dosyaları | Günlük | 30 gün |
| ChromaDB | Haftalık | 4 hafta |
| Model artifacts | Her eğitimde | 10 versiyon |
| Loglar | Otomatik (rotating) | 5 dosya × 10MB |
| Konfigürasyon | Her değişiklikte | Git history |

---

## 🗄️ SQLite Backup

### Manuel Yedekleme

```bash
# Online backup (veri bütünlüğü korunur)
sqlite3 src/database/cyberguard.db ".backup backups/db/cyberguard_$(date +%Y%m%d_%H%M%S).db"

# Alternatif: dump SQL olarak
sqlite3 src/database/cyberguard.db .dump > backups/db/cyberguard_$(date +%Y%m%d).sql

# Sıkıştırılmış
sqlite3 src/database/cyberguard.db .dump | gzip > backups/db/cyberguard_$(date +%Y%m%d).sql.gz
```

### Otomatik Yedekleme Scripti

```python
# scripts/backup_db.py
import shutil
import subprocess
from datetime import datetime
from pathlib import Path

BACKUP_DIR = Path("backups/db")
DB_PATH = Path("src/database/cyberguard.db")
KEEP_DAYS = 30

def backup_sqlite():
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dest = BACKUP_DIR / f"cyberguard_{timestamp}.db"

    # sqlite3 .backup komutu — online backup, kilitleme gerektirmez
    subprocess.run(
        ["sqlite3", str(DB_PATH), f".backup {dest}"],
        check=True
    )
    print(f"✅ SQLite backup: {dest}")

    # Eski yedekleri temizle
    cutoff = datetime.now().timestamp() - (KEEP_DAYS * 86400)
    for f in BACKUP_DIR.glob("*.db"):
        if f.stat().st_mtime < cutoff:
            f.unlink()
            print(f"🗑️  Silindi: {f}")

if __name__ == "__main__":
    backup_sqlite()
```

### Restore

```bash
# Backup'ı geri yükle
cp backups/db/cyberguard_20260424_030000.db src/database/cyberguard.db

# SQL dump'tan geri yükle
sqlite3 src/database/cyberguard_new.db < backups/db/cyberguard_20260424.sql
```

### Windows Görev Zamanlayıcı

```batch
@echo off
:: scripts/backup_db.bat
python scripts/backup_db.py
```

```
Görev Zamanlayıcı → Temel Görev Oluştur
  Tetikleyici: Günlük 03:00
  Eylem: scripts/backup_db.bat
```

---

## 📂 JSON Veri Dosyaları Backup

`data/` klasörü operasyonel durumu saklar:

```
data/
├── users.json           # Kullanıcı hesapları (bcrypt hash'li şifreler)
├── sessions.json        # Aktif oturumlar
├── honeypots.json       # Honeypot yapılandırması
├── honeypot_captures.json  # Yakalanan saldırılar
├── siem_config.json     # SIEM yapılandırması
├── threat_hunts.json    # Tehdit avı kayıtları
├── incidents.json       # LLM Agent olay kayıtları
└── gan_generated.json   # GAN ile üretilen örnek trafik
```

### Yedekleme

```bash
# Tüm data/ klasörünü sıkıştır
tar -czf backups/data/data_$(date +%Y%m%d_%H%M%S).tar.gz data/

# PowerShell
Compress-Archive -Path data\ -DestinationPath backups\data\data_$(Get-Date -Format 'yyyyMMdd_HHmmss').zip
```

### Restore

```bash
tar -xzf backups/data/data_20260424_030000.tar.gz
# veya seçici olarak:
tar -xzf backups/data/data_20260424.tar.gz data/users.json
```

> ⚠️ `data/users.json` hassas veri içerir. Yedekler güvenli konumda saklanmalı.

---

## 🧠 Model Backup

### Model Versioning

Eğitilmiş modeller `model_artifacts/` klasöründe timestamp'li isimle saklanır:

```
model_artifacts/
├── ssa_lstmids_cicids2017_20260105_202005.keras
├── best_cicids2017.keras           ← üretim modeli
├── best_cicids_full.keras
├── best_bot-iot.keras
├── model_registry.json             ← versiyon kaydı
└── comparison_results.json         ← benchmark sonuçları
```

### Yedekleme

```bash
# Sadece üretim modellerini yedekle
cp model_artifacts/best_*.keras backups/models/

# Tüm model artifacts
tar -czf backups/models/model_artifacts_$(date +%Y%m%d).tar.gz model_artifacts/
```

### Model Registry

`model_artifacts/model_registry.json` dosyasını takip edin:

```bash
# Registry'yi git ile takip et
git add model_artifacts/model_registry.json
git commit -m "chore: model registry update"
```

### Büyük Model Dosyaları

`.keras` dosyaları 50-500MB olabilir. GitHub'a göndermeden önce kontrol edin:

```bash
# Büyük dosyaları listele
Get-ChildItem model_artifacts\ | Where-Object {$_.Length -gt 50MB} | Select-Object Name, @{N='MB';E={[math]::Round($_.Length/1MB,1)}}
```

> Büyük model dosyaları için Git LFS veya harici depolama (OneDrive, Google Drive) kullanın.

---

## 📋 Log Backup

Loglar `app/utils/logging.py`'deki `RotatingFileHandler` tarafından otomatik yönetilir:

```
logs/app/
├── cyberguard.log        ← aktif log (max 10MB)
├── cyberguard.log.1      ← 1. rotate
├── cyberguard.log.2      ← 2. rotate
├── cyberguard.log.3
├── cyberguard.log.4
└── cyberguard.log.5      ← en eski (5 × 10MB = max 50MB)
```

Log ayarlarını `LOG_LEVEL` ve dosyayı `.env` ile yapılandırın:

```env
LOG_LEVEL=INFO
JSON_CONSOLE_LOG=false
```

Eski logları arşivlemek için:

```bash
# Günlük log arşivi
Compress-Archive -Path logs\app\cyberguard.log.* -DestinationPath backups\logs\logs_$(Get-Date -Format 'yyyyMMdd').zip
```

---

## 🔄 Disaster Recovery

### Recovery Sırası

1. **Durumu Değerlendir**

   ```bash
   # API sağlık durumu
   curl http://localhost:8000/health

   # Veri dosyalarını kontrol et
   python -c "import json; json.load(open('data/users.json'))"
   sqlite3 src/database/cyberguard.db "PRAGMA integrity_check;"
   ```

2. **Veritabanını Kurtar**

   ```bash
   # SQLite backup'ı geri yükle
   cp backups/db/cyberguard_YYYYMMDD.db src/database/cyberguard.db
   ```

3. **JSON Veri Dosyalarını Kurtar**

   ```bash
   tar -xzf backups/data/data_YYYYMMDD.tar.gz
   ```

4. **Modelleri Kurtar**

   ```bash
   cp backups/models/best_*.keras model_artifacts/
   ```

5. **Servisi Yeniden Başlat**

   ```bash
   uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
   ```

6. **Doğrula**

   ```bash
   curl http://localhost:8000/api/docs
   curl http://localhost:8000/health
   ```

### RTO & RPO

| Sistem | RTO | RPO |
|--------|-----|-----|
| API | 5 dk | 0 (stateless) |
| SQLite DB | 10 dk | 1 gün (günlük backup) |
| JSON dosyaları | 5 dk | 1 gün |
| ML Modeller | 15 dk | Eğitim tarihine kadar |

---

## 📋 Checklist

- [ ] `scripts/backup_db.py` zamanlanmış mı? (günlük 03:00)
- [ ] `backups/` klasörü `.gitignore`'da mı?
- [ ] `data/users.json` yedekleri şifreli ortamda mı?
- [ ] Model artifacts'ın son versiyonu yedeklendi mi?
- [ ] Restore test edildi mi?
- [ ] `model_artifacts/model_registry.json` git'te güncel mi?

---

## 🎯 Yedekleme Stratejisi

### 3-2-1 Kuralı

- **3** kopya (orijinal + 2 yedek)
- **2** farklı ortam (local + cloud)
- **1** off-site yedek

### Yedekleme Sıklığı

| Veri Türü | Sıklık | Retention |
|-----------|--------|-----------|
| Database | Günlük | 30 gün |
| Config | Haftalık | 90 gün |
| Models | Her eğitimde | 10 versiyon |
| Logs | Günlük | 7 gün |

---

## 🗄️ Database Backup

### PostgreSQL Backup

```bash
# Full backup
pg_dump -U postgres -h localhost cyberguard > backup_$(date +%Y%m%d).sql

# Compressed
pg_dump -U postgres cyberguard | gzip > backup_$(date +%Y%m%d).sql.gz

# Custom format (parallel restore)
pg_dump -U postgres -Fc cyberguard > backup.dump
```

### Automated Backup Script

```bash
#!/bin/bash
# scripts/backup_db.sh

DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="/backups/postgres"
DB_NAME="cyberguard"

# Create backup
pg_dump -U postgres -Fc $DB_NAME > $BACKUP_DIR/backup_$DATE.dump

# Upload to S3
aws s3 cp $BACKUP_DIR/backup_$DATE.dump s3://cyberguard-backups/db/

# Cleanup old backups (keep 30 days)
find $BACKUP_DIR -name "*.dump" -mtime +30 -delete
```

### Cron Job

```bash
# Günlük 03:00'te backup
0 3 * * * /opt/cyberguard/scripts/backup_db.sh
```

### Restore

```bash
# SQL restore
psql -U postgres cyberguard < backup.sql

# Custom format
pg_restore -U postgres -d cyberguard backup.dump
```

---

## 🧠 Model Backup

### Model Versioning

```python
# scripts/backup_models.py
import shutil
from datetime import datetime

def backup_model(model_name):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    src = f"models/production/{model_name}.h5"
    dst = f"backups/models/{model_name}_{timestamp}.h5"
    shutil.copy(src, dst)
    
    # Upload to cloud
    upload_to_s3(dst, f"s3://cyberguard-backups/models/")
```

### Model Registry

```json
// models/model_registry.json
{
  "best_cicids2017": {
    "version": "2.0.0",
    "created_at": "2026-01-10",
    "accuracy": 0.9988,
    "path": "production/best_cicids2017.h5",
    "backups": [
      "backups/best_cicids2017_20260109.h5",
      "backups/best_cicids2017_20260108.h5"
    ]
  }
}
```

---

## 🔄 Disaster Recovery

### RTO & RPO

| Sistem | RTO | RPO |
|--------|-----|-----|
| API | 15 min | 1 hour |
| Database | 30 min | 1 hour |
| Models | 1 hour | 24 hours |

### Recovery Steps

1. **Assess Damage**

   ```bash
   docker-compose ps
   docker-compose logs
   ```

2. **Restore Database**

   ```bash
   # Latest backup
   aws s3 cp s3://cyberguard-backups/db/latest.dump .
   pg_restore -U postgres -d cyberguard latest.dump
   ```

3. **Restore Models**

   ```bash
   aws s3 sync s3://cyberguard-backups/models/ models/production/
   ```

4. **Restart Services**

   ```bash
   docker-compose down
   docker-compose up -d
   ```

5. **Verify**

   ```bash
   curl http://localhost:8000/health
   ```

### Failover

```bash
# Secondary server'a geç
./scripts/failover.sh secondary

# DNS güncelle
aws route53 change-resource-record-sets ...
```

---

## 📋 Checklist

- [ ] Günlük DB backup çalışıyor mu?
- [ ] S3'e upload başarılı mı?
- [ ] Model versiyonlama aktif mi?
- [ ] Recovery test edildi mi?
- [ ] Dokümantasyon güncel mi?
