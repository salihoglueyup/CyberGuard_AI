# 📤 GitHub Yükleme Rehberi

Bu rehber, CyberGuard AI projesini GitHub'a yüklemek için adım adım talimatlar içerir.

---

## ⚠️ Önemli: Büyük Dosya Sorunları

GitHub'ın dosya limitleri:

- **Tek dosya:** Maksimum 100MB (sert limit)
- **Toplam repo:** Önerilen < 1GB, maksimum 5GB
- **Push:** Tek push'ta maksimum 2GB

### Projemizdeki Potansiyel Büyük Dosyalar

| Dosya/Klasör | Tahmini Boyut | Çözüm |
|--------------|---------------|-------|
| `.venv/` | 500MB+ | ❌ .gitignore'a ekle |
| `node_modules/` | 300MB+ | ❌ .gitignore'a ekle |
| `data/` (datasets) | 100MB-6GB | ⚠️ Git LFS veya dış link |
| `model_artifacts/*.keras` | 50-500MB | ⚠️ Git LFS |
| `__pycache__/` | 10MB+ | ❌ .gitignore'a ekle |
| `.pdf` dosyalar | 6MB+ | ✅ OK |

---

## 📋 Adım Adım Plan

### Adım 1: .gitignore Kontrolü

Mevcut `.gitignore` dosyasını kontrol et ve eksikleri ekle:

```gitignore
# Python
__pycache__/
*.py[cod]
*.so
.Python
.venv/
venv/
ENV/

# Node
node_modules/
npm-debug.log

# IDE
.idea/
.vscode/
*.swp

# OS
.DS_Store
Thumbs.db

# Env
.env
.env.local

# Data (büyük dosyalar)
data/raw/
data/CICIDS2017/
*.csv.gz
*.parquet

# Models (opsiyonel - Git LFS kullan)
# model_artifacts/*.keras

# Logs
logs/
*.log

# Uploads
uploads/*
!uploads/.gitkeep

# Reports (generated)
reports/*
!reports/.gitkeep
```

### Adım 2: Büyük Dosyaları Tespit Et

```bash
# Windows PowerShell - 100MB'dan büyük dosyaları bul
Get-ChildItem -Recurse | Where-Object { $_.Length -gt 100MB } | Select-Object FullName, @{Name="SizeMB";Expression={[math]::Round($_.Length/1MB,2)}}
```

### Adım 3: Git LFS Kurulumu (Büyük Dosyalar İçin)

Eğer model dosyaları (.h5, .keras) veya büyük veri setleri varsa:

```bash
# Git LFS kurulumu
git lfs install

# Keras model dosyalarını track et
git lfs track "model_artifacts/*.keras"
git lfs track "model_artifacts/*.json"  # model_registry.json büyüyebilir
git lfs track "*.pkl"
git lfs track "data/*.csv"

# .gitattributes dosyasını commit et
git add .gitattributes
```

### Adım 4: Repository Oluşturma

1. [github.com/new](https://github.com/new) adresine git
2. Repository bilgileri:
   - **Name:** `CyberGuard-AI`
   - **Description:** `AI-Powered Cyber Security Platform with LSTM-based IDS`
   - **Visibility:** Public veya Private
   - **Initialize:** ❌ (boş bırak, README ekleme)

### Adım 5: Local Git Kurulumu

```bash
# Proje dizinine git
cd c:\Gelistirme\CyberGuard_AI_Antigravity

# Git başlat (zaten varsa skip)
git init

# Remote ekle
git remote add origin https://github.com/salihoglueyup/CyberGuard_AI.git

# Ana branch'i ayarla
git branch -M main
```

### Adım 6: İlk Commit

```bash
# Tüm dosyaları ekle
git add .

# Commit
git commit -m "🚀 Initial commit: CyberGuard AI - Full Platform"

# Push
git push -u origin main
```

---

## 🔧 Sorun Giderme

### Problem: "File too large" hatası

```bash
# Büyük dosyayı git geçmişinden sil
git filter-branch --force --index-filter \
  "git rm --cached --ignore-unmatch PATH/TO/LARGE/FILE" \
  --prune-empty --tag-name-filter cat -- --all

# Veya BFG Repo-Cleaner kullan (daha hızlı)
java -jar bfg.jar --strip-blobs-bigger-than 100M
```

### Problem: Push çok yavaş

```bash
# Daha küçük parçalar halinde push
git push --progress
```

### Problem: Git LFS quota aşıldı

GitHub Free: 1GB storage, 1GB/ay bandwidth

- Çözüm 1: External storage (S3, Google Drive)
- Çözüm 2: GitHub Pro/Team upgrade
- Çözüm 3: Model dosyalarını Hugging Face Hub'a yükle

---

## 📁 Önerilen Dosya Yapısı

```
CyberGuard-AI/
├── README.md              # ✅ Proje tanıtımı
├── LICENSE                # ✅ MIT License
├── .gitignore             # ✅ Ignore rules
├── .gitattributes         # ✅ LFS rules (varsa)
├── requirements.txt       # ✅ Python deps
├── package.json           # ✅ Node deps (frontend için)
│
├── app/                   # ✅ Backend
├── frontend/              # ✅ Frontend (node_modules hariç)
├── src/                   # ✅ ML models
├── docs/                  # ✅ Documentation
├── tests/                 # ✅ Test files
│
├── data/                  # ⚠️ Sadece sample data
│   └── sample/
├── model_artifacts/        # ⚠️ Git LFS önerilir (*.keras dosyaları 50-500MB)
│   ├── best_cicids2017.keras
│   ├── best_bot-iot.keras
│   ├── best_cicids_full.keras
│   └── model_registry.json
└── notebooks/             # ✅ Jupyter notebooks
```

---

## 🚀 Hızlı Başlangıç Scripti

Aşağıdaki PowerShell scriptini çalıştır:

```powershell
# 1. Büyük dosyaları kontrol et
Write-Host "=== Büyük Dosyalar (>50MB) ===" -ForegroundColor Yellow
Get-ChildItem -Recurse -File | Where-Object { $_.Length -gt 50MB } | 
    Select-Object @{N='Size(MB)';E={[math]::Round($_.Length/1MB,2)}}, FullName

# 2. Toplam boyut
Write-Host "`n=== Toplam Proje Boyutu ===" -ForegroundColor Yellow
$size = (Get-ChildItem -Recurse | Measure-Object -Property Length -Sum).Sum / 1GB
Write-Host ("Toplam: {0:N2} GB" -f $size)

# 3. Hariç tutulacak klasörler
Write-Host "`n=== Hariç Tutulacaklar ===" -ForegroundColor Yellow
@(".venv", "node_modules", "__pycache__", "data/raw") | ForEach-Object {
    if (Test-Path $_) {
        $s = (Get-ChildItem $_ -Recurse | Measure-Object -Property Length -Sum).Sum / 1MB
        Write-Host ("{0}: {1:N0} MB" -f $_, $s)
    }
}
```

---

## ✅ Checklist

- [ ] `.gitignore` güncel mi?
- [ ] Büyük dosyalar (>100MB) tespit edildi mi?
- [ ] Git LFS gerekli mi?
- [ ] `.env` dosyası .gitignore'da mı?
- [ ] `node_modules/` .gitignore'da mı?
- [ ] `.venv/` .gitignore'da mı?
- [ ] README.md hazır mı?
- [ ] LICENSE dosyası var mı?

---

## 📞 Alternatifler

### Büyük Dosyalar İçin

1. **Hugging Face Hub** - ML modelleri için ideal
2. **Google Drive** - Datasets için link paylaşımı
3. **AWS S3** - Production için
4. **DVC** (Data Version Control) - ML pipelines için

### Release İçin

GitHub Releases ile büyük dosyaları (100MB'a kadar) yükleyebilirsin:

1. GitHub'da Release oluştur
2. Assets bölümüne dosya yükle
3. README'de link ver
