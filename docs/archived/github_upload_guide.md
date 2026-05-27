# 📤 CyberGuard AI - GitHub'a Yükleme Rehberi

Bu rehber, büyük dosyaları olan projeyi GitHub'a nasıl yükleyeceğinizi açıklar.

---

## ⚠️ Önemli: GitHub Sınırları

| Sınır | Değer |
| ----- | ----- |
| Tek dosya maksimum | **100 MB** |
| Repo toplam boyut (önerilir) | **1 GB** |
| Repo sert limit | **5 GB** |
| Push limit | **2 GB** |

---

## 📊 Projenizin Durumu

Büyük dosyalarınız:

- `src/database/cyberguard.db` - **~5 GB** (çok büyük!)
- `models/*.keras` - **~150 MB** toplam
- `data/raw/` - **~500 MB+** CSV dosyaları

---

## ✅ Yöntem 1: Büyük Dosyaları Hariç Tut (Önerilen)

`.gitignore` zaten ayarlandı. Şu dosyalar otomatik hariç tutulacak:

```
✓ *.keras       # ML modelleri
✓ *.h5          # Eski modeller
✓ *.db          # Veritabanları
✓ data/raw/     # Ham veri setleri
✓ .venv/        # Python sanal ortam
✓ node_modules/ # Node paketleri
```

### Adımlar

```bash
# 1. Git'i başlat (zaten yapılmışsa atla)
git init

# 2. Tüm dosyaları ekle (.gitignore'a göre filtrelenir)
git add .

# 3. Commit yap
git commit -m "Initial commit: CyberGuard AI v3.1"

# 4. Remote ekle
git remote add origin https://github.com/KULLANICI/CyberGuard_AI.git

# 5. Push et
git push -u origin main
```

---

## 🔄 Yöntem 2: Git LFS (Large File Storage)

Eğer modelleri de yüklemek istiyorsan:

### Kurulum

```bash
# 1. Git LFS yükle
# Windows: https://git-lfs.com adresinden indir
# veya
winget install GitHub.GitLFS

# 2. LFS'i aktifleştir
git lfs install

# 3. Büyük dosya türlerini takip et
git lfs track "*.keras"
git lfs track "*.h5"
git lfs track "*.db"

# 4. .gitattributes'u ekle
git add .gitattributes

# 5. Normal commit ve push
git add .
git commit -m "Add LFS tracking"
git push
```

### LFS Limitleri

- GitHub Free: **1 GB storage**, **1 GB/ay bandwidth**
- GitHub Pro: **2 GB storage**, **2 GB/ay bandwidth**

---

## 🗂️ Yöntem 3: Ayrı Repo (Modeller için)

Büyük dosyaları ayrı bir repo'da tut:

### Ana Repo (kod)

```
CyberGuard_AI/
├── app/
├── frontend/
├── src/
├── docs/
└── README.md
```

### Model Repo (büyük dosyalar)

```
CyberGuard_AI_Models/
├── production/
├── archived/
└── README.md
```

### Kullanıcılara

```markdown
## Model Dosyaları

Eğitilmiş modeller ayrı repoda:
https://github.com/KULLANICI/CyberGuard_AI_Models

Veya Google Drive:
https://drive.google.com/...
```

---

## 📦 Yöntem 4: Releases ile Dağıtım

Büyük dosyaları GitHub Releases'a yükle:

```bash
# 1. Modelleri zipple
Compress-Archive -Path models\production\* -DestinationPath models_v3.1.zip

# 2. GitHub CLI ile release oluştur
gh release create v3.1.0 models_v3.1.zip --title "v3.1 - Models"
```

### Release Limiti

- Tek dosya: **2 GB**
- Toplam: **Sınırsız**

---

## 🚀 Hızlı Başlangıç (Önerilen)

```powershell
# 1. Proje klasörüne git
cd C:\Gelistirme\CyberGuard_AI_Antigravity

# 2. Git durumunu kontrol et
git status

# 3. Yeni değişiklikleri ekle
git add .

# 4. Commit yap
git commit -m "v3.1.0: Globe3D ML integration, tests, docs update"

# 5. Push et
git push origin main
```

---

## 🔍 Yükleme Öncesi Kontrol

```powershell
# Repo boyutunu kontrol et
git count-objects -vH

# Büyük dosyaları bul
git rev-list --objects --all | git cat-file --batch-check='%(objectname) %(objectsize) %(rest)' | sort -k2 -n -r | head -20
```

---

## ❓ Sık Sorunlar

### "File too large" hatası

```bash
# Dosyayı git history'den temizle
git filter-branch --force --index-filter "git rm --cached --ignore-unmatch DOSYA_ADI" --prune-empty --tag-name-filter cat -- --all

# Daha modern yöntem (BFG Repo Cleaner)
bfg --strip-blobs-bigger-than 100M
```

### Push çok yavaş

- `.gitignore` kontrol et
- `git lfs` kullan
- Push'u parçala: `git push origin main --force`

---

## 📋 Checklist

Yüklemeden önce:

- [ ] `.gitignore` güncel
- [ ] `data/raw/` hariç tutuldu
- [ ] `src/database/cyberguard.db` hariç tutuldu
- [ ] `models/*.keras` hariç tutuldu (veya LFS)
- [ ] `.venv/` hariç tutuldu
- [ ] `node_modules/` hariç tutuldu
- [ ] `.env` hariç tutuldu (güvenlik!)

---

**Şimdi hazırsın! 🚀**
