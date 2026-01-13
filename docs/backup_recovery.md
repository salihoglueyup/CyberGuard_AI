# 💾 Backup & Recovery Guide

CyberGuard AI yedekleme ve kurtarma rehberi

---

## 📋 İçindekiler

- [Yedekleme Stratejisi](#yedekleme-stratejisi)
- [Database Backup](#database-backup)
- [Model Backup](#model-backup)
- [Disaster Recovery](#disaster-recovery)

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
