# 🤝 CyberGuard AI'ya Katkıda Bulunma

CyberGuard AI'ya katkıda bulunmayı düşündüğünüz için teşekkür ederiz! 🎉

## 📋 İçindekiler

- [Davranış Kuralları](#davranış-kuralları)
- [Nasıl Katkıda Bulunabilirim?](#nasıl-katkıda-bulunabilirim)
- [Geliştirme Ortamı Kurulumu](#geliştirme-ortamı-kurulumu)
- [Pull Request Süreci](#pull-request-süreci)
- [Kodlama Standartları](#kodlama-standartları)
- [Commit Kuralları](#commit-kuralları)
- [Test Yazma](#test-yazma)

---

## 📜 Davranış Kuralları

Bu proje ve katılan herkes [Davranış Kuralları](CODE_OF_CONDUCT.md) tarafından yönetilir. Katılarak bu kurallara uymayı kabul etmiş sayılırsınız.

---

## 🎯 Nasıl Katkıda Bulunabilirim?

### 🐛 Hata Bildirimi

Hata bildirmeden önce lütfen mevcut issue'ları kontrol edin. Hata raporu oluştururken şunları ekleyin:

- **Açık başlık ve açıklama**
- **Hatayı tekrarlama adımları**
- **Beklenen ve gerçekleşen davranış**
- **Ekran görüntüleri** (varsa)
- **Ortam detayları** (İşletim sistemi, Python sürümü, vb.)

**Hata Raporu Şablonu:**
```markdown
## Hata Açıklaması
[Hatanın net açıklaması]

## Tekrarlama Adımları
1. '...' sayfasına git
2. '...' butonuna tıkla
3. Hatayı gör

## Beklenen Davranış
[Ne olmasını bekliyordunuz]

## Gerçekleşen Davranış
[Ne oldu]

## Ortam Bilgileri
- İşletim Sistemi: [örn. Ubuntu 22.04]
- Python: [örn. 3.10.5]
- Versiyon: [örn. v2.0.0]

## Ekran Görüntüleri
[Varsa ekleyin]
```

### 💡 Özellik Önerme

Özellik önerileri memnuniyetle karşılanır! Lütfen şunları ekleyin:

- **Açık kullanım senaryosu**
- **Detaylı açıklama**
- **Mockup veya örnekler** (varsa)
- **Olası implementasyon yaklaşımı**

**Özellik İsteği Şablonu:**
```markdown
## Özellik Açıklaması
[Özelliğin net açıklaması]

## Kullanım Senaryosu
[Bu özellik ne zaman ve neden kullanılacak?]

## Önerilen Çözüm
[Özelliğin nasıl çalışmasını öneriyorsunuz?]

## Alternatifler
[Düşündüğünüz alternatif çözümler]

## Ek Bilgiler
[Ekran görüntüleri, mockup'lar, vb.]
```

### 📝 Dokümantasyon İyileştirmeleri

Dokümantasyon her zaman iyileştirilebilir:

- Yazım hatalarını düzeltme
- Açıklamaları netleştirme
- Örnekler ekleme
- Türkçe/İngilizce çeviri geliştirmeleri

---

## 💻 Geliştirme Ortamı Kurulumu

### 1. Repository'yi Fork Edin

```bash
# GitHub'da "Fork" butonuna tıklayın
# Sonra klonlayın:
git clone https://github.com/KULLANICI_ADINIZ/cyberguard-ai.git
cd cyberguard-ai
```

### 2. Upstream Remote Ekleyin

```bash
git remote add upstream https://github.com/cyberguard-ai/cyberguard-ai.git
```

### 3. Sanal Ortam Oluşturun

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

### 4. Bağımlılıkları Yükleyin

```bash
# Gerekli paketler
pip install -r requirements.txt

# Geliştirme paketleri
pip install -r requirements-dev.txt
```

### 5. Pre-commit Hook'ları Kurun

```bash
pre-commit install
```

---

## 🔄 Pull Request Süreci

### 1. Branch Oluşturun

```bash
# Feature için
git checkout -b feature/yeni-ozellik-adi

# Bug fix için
git checkout -b bugfix/hata-aciklamasi

# Dokümantasyon için
git checkout -b docs/dokuman-aciklamasi
```

### 2. Değişikliklerinizi Yapın

- Küçük, odaklanmış değişiklikler yapın
- Her commit tek bir konuya odaklanmalı
- Kod standartlarına uyun

### 3. Test Edin

```bash
# Tüm testleri çalıştırın
pytest

# Coverage kontrolü
pytest --cov=src tests/

# Linting
flake8 src/
black --check src/
```

### 4. Commit Edin

```bash
git add .
git commit -m "feat: yeni özellik eklendi"
```

### 5. Push Edin

```bash
git push origin feature/yeni-ozellik-adi
```

### 6. Pull Request Açın

- GitHub'da repository'nize gidin
- "Pull Request" butonuna tıklayın
- Değişikliklerinizi açıklayın
- İlgili issue'ları bağlayın

**PR Şablonu:**
```markdown
## Açıklama
[Değişikliklerinizin kısa açıklaması]

## Değişiklik Tipi
- [ ] 🐛 Bug fix
- [ ] ✨ Yeni özellik
- [ ] 📝 Dokümantasyon
- [ ] 🎨 Stil/formatting
- [ ] ♻️ Refactoring
- [ ] 🔧 Konfigürasyon

## Bağlantılı Issue'lar
Fixes #(issue numarası)

## Test Edilen Senaryolar
- [ ] Test senaryosu 1
- [ ] Test senaryosu 2

## Checklist
- [ ] Kod kodlama standartlarına uygun
- [ ] Testler yazıldı ve geçiyor
- [ ] Dokümantasyon güncellendi
- [ ] CHANGELOG.md güncellendi
```

---

## 📏 Kodlama Standartları

### Python Stil Kılavuzu

**PEP 8 Standartlarına uyun:**

```python
# ✅ İYİ
def calculate_risk_score(vulnerability_data: dict) -> float:
    """
    Zafiyet verilerinden risk skoru hesaplar.
    
    Args:
        vulnerability_data: Zafiyet bilgilerini içeren sözlük
        
    Returns:
        0-10 arası risk skoru
    """
    severity = vulnerability_data.get('severity', 0)
    exploitability = vulnerability_data.get('exploitability', 0)
    return (severity * 0.6) + (exploitability * 0.4)

# ❌ KÖTÜ
def calc(d):
    s=d.get('severity',0)
    e=d.get('exploitability',0)
    return s*0.6+e*0.4
```

### Genel Kurallar

1. **İsimlendirme:**
    - `snake_case` fonksiyonlar ve değişkenler için
    - `PascalCase` sınıflar için
    - `UPPER_CASE` sabitler için

2. **Docstring:**
    - Her fonksiyon ve sınıf için docstring yazın
    - Google style veya NumPy style kullanın

3. **Type Hints:**
    - Mümkün olduğunca type hint kullanın
   ```python
   def process_log(log_file: str) -> List[dict]:
       pass
   ```

4. **Imports:**
   ```python
   # Standart kütüphane
   import os
   import sys
   
   # Üçüncü parti
   import numpy as np
   import pandas as pd
   
   # Yerel
   from src.models import AIModel
   from src.utils import logger
   ```

### Code Formatting

```bash
# Black ile otomatik formatlama
black src/

# isort ile import sıralama
isort src/

# flake8 ile lint kontrolü
flake8 src/
```

---

## 📝 Commit Kuralları

**Conventional Commits** formatını kullanın:

### Commit Mesaj Formatı

```
<tip>(<kapsam>): <kısa açıklama>

[opsiyonel detaylı açıklama]

[opsiyonel footer]
```

### Commit Tipleri

| Tip | Açıklama | Örnek |
|-----|----------|-------|
| `feat` | Yeni özellik | `feat(chatbot): NLP modeli eklendi` |
| `fix` | Hata düzeltme | `fix(scanner): port tarama hatası düzeltildi` |
| `docs` | Dokümantasyon | `docs(readme): kurulum adımları güncellendi` |
| `style` | Kod formatı | `style: black ile formatlama yapıldı` |
| `refactor` | Kod iyileştirme | `refactor(api): endpoint yapısı düzenlendi` |
| `test` | Test ekleme | `test(scanner): unit testler eklendi` |
| `chore` | Genel işler | `chore: dependencies güncellendi` |
| `perf` | Performans | `perf(ml): model inference hızlandırıldı` |

### Örnekler

```bash
# Yeni özellik
git commit -m "feat(chatbot): çoklu dil desteği eklendi"

# Hata düzeltme
git commit -m "fix(database): bağlantı timeout sorunu çözüldü"

# Dokümantasyon
git commit -m "docs(api): endpoint örnekleri eklendi"

# Detaylı commit
git commit -m "feat(scanner): deep scan modu eklendi

- CVE veritabanı entegrasyonu
- Detaylı port analizi
- PDF rapor oluşturma

Closes #123"
```

---

## 🧪 Test Yazma

### Test Yapısı

```
tests/
├── unit/              # Birim testler
│   ├── test_chatbot.py
│   ├── test_scanner.py
│   └── test_models.py
├── integration/       # Entegrasyon testler
│   ├── test_api.py
│   └── test_database.py
└── e2e/              # End-to-end testler
    └── test_workflows.py
```

### Test Yazma Kuralları

**1. Her fonksiyon için test yazın:**

```python
# src/scanner.py
def scan_port(ip: str, port: int) -> bool:
    """Port'un açık olup olmadığını kontrol eder."""
    # implementasyon
    pass

# tests/unit/test_scanner.py
def test_scan_port_open():
    """Açık port doğru tespit edilmeli."""
    result = scan_port("127.0.0.1", 80)
    assert result is True

def test_scan_port_closed():
    """Kapalı port doğru tespit edilmeli."""
    result = scan_port("127.0.0.1", 9999)
    assert result is False

def test_scan_port_invalid_ip():
    """Geçersiz IP ile hata fırlatmalı."""
    with pytest.raises(ValueError):
        scan_port("invalid", 80)
```

**2. Fixture kullanın:**

```python
@pytest.fixture
def sample_vulnerability():
    return {
        'cve_id': 'CVE-2024-1234',
        'severity': 9.8,
        'description': 'Test vulnerability'
    }

def test_process_vulnerability(sample_vulnerability):
    result = process_vulnerability(sample_vulnerability)
    assert result['risk_level'] == 'critical'
```

**3. Mock kullanın:**

```python
from unittest.mock import Mock, patch

@patch('src.scanner.socket.socket')
def test_scan_with_mock(mock_socket):
    mock_socket.return_value.connect_ex.return_value = 0
    result = scan_port("192.168.1.1", 22)
    assert result is True
```

### Test Çalıştırma

```bash
# Tüm testler
pytest

# Belirli bir dosya
pytest tests/unit/test_scanner.py

# Belirli bir test
pytest tests/unit/test_scanner.py::test_scan_port_open

# Coverage ile
pytest --cov=src --cov-report=html

# Verbose mode
pytest -v

# Sadece failed testler
pytest --lf
```

### Coverage Hedefi

- **Minimum %80 coverage** gereklidir
- Kritik modüller için **%90+** hedefleyin
- Coverage raporunu kontrol edin: `htmlcov/index.html`

---

## 🔍 Code Review Süreci

### Review Beklerken

1. ✅ Tüm testlerin geçtiğinden emin olun
2. ✅ CI/CD pipeline'ının başarılı olduğunu kontrol edin
3. ✅ Çakışmaları çözün
4. ✅ Review yorumlarına hızlıca yanıt verin

### Review Yaparken

**Kontrol Edilecekler:**

- [ ] Kod anlaşılır ve bakımı kolay mı?
- [ ] Testler yeterli mi?
- [ ] Dokümantasyon güncel mi?
- [ ] Güvenlik açıkları var mı?
- [ ] Performance etkileri düşünülmüş mü?
- [ ] Error handling yeterli mi?

**Yapıcı Geri Bildirim:**

```markdown
# ❌ Kötü
Bu kod berbat.

# ✅ İyi
Bu fonksiyonda error handling eksik görünüyor. 
`try-except` bloğu ekleyerek daha robust hale getirebiliriz.
Örnek: [link to example]
```

---

## 🏷️ Issue ve PR Etiketleri

### Issue Etiketleri

| Etiket | Açıklama |
|--------|----------|
| `bug` 🐛 | Bir şeyler çalışmıyor |
| `enhancement` ✨ | Yeni özellik veya istek |
| `documentation` 📝 | Dokümantasyon iyileştirmesi |
| `good first issue` 👶 | Yeni katkıcılar için uygun |
| `help wanted` 🆘 | Ekstra dikkat gerekiyor |
| `priority: high` 🔴 | Yüksek öncelikli |
| `priority: low` 🟢 | Düşük öncelikli |
| `wontfix` ⛔ | Üzerinde çalışılmayacak |

### PR Etiketleri

| Etiket | Açıklama |
|--------|----------|
| `WIP` 🚧 | Work in progress |
| `ready for review` 👀 | Review için hazır |
| `needs work` 🔧 | Değişiklik gerekiyor |
| `approved` ✅ | Onaylandı |

---

## 📞 İletişim ve Sorular

### Soru Sormadan Önce

1. 📖 [Dokümantasyonu](docs/) okudunuz mu?
2. 🔍 [Mevcut issue'larda](https://github.com/cyberguard-ai/issues) aradınız mı?
3. 💬 [Discussions](https://github.com/cyberguard-ai/discussions) bölümünü kontrol ettiniz mi?

### İletişim Kanalları

- 💬 **Discord**: [discord.gg/cyberguard](https://discord.gg/cyberguard)
- 📧 **Email**: contribute@cyberguard-ai.com
- 🐦 **Twitter**: [@cyberguard_ai](https://twitter.com/cyberguard_ai)

---

## 🎉 İlk Katkınızı Yapın!

Yeni başlıyorsanız:

1. `good first issue` etiketli issue'lara bakın
2. Küçük bir düzeltme ile başlayın (typo, dokümantasyon)
3. Topluluktan yardım istemekten çekinmeyin!

---

## 🙏 Teşekkürler!

Her katkı, büyük ya da küçük, çok değerlidir. CyberGuard AI'yı daha iyi hale getirmeye yardımcı olduğunuz için teşekkür ederiz! 💙

---

**Not:** Bu kılavuz sürekli geliştirilmektedir. Önerileriniz varsa lütfen issue açın!