# 🔒 Güvenlik Politikası

## 📋 İçindekiler

- [Desteklenen Versiyonlar](#desteklenen-versiyonlar)
- [Güvenlik Açığı Bildirimi](#güvenlik-açığı-bildirimi)
- [Güvenlik Güncellemeleri](#güvenlik-güncellemeleri)
- [Güvenlik En İyi Uygulamaları](#güvenlik-en-iyi-uygulamaları)
- [Güvenlik Denetimi](#güvenlik-denetimi)

---

## 🛡️ Desteklenen Versiyonlar

Aşağıdaki CyberGuard AI versiyonları için güvenlik güncellemeleri sağlanmaktadır:

| Versiyon | Destek Durumu | Destek Bitiş Tarihi |
|----------|---------------|---------------------|
| 2.0.x    | ✅ Tam Destek | 2026-01-15 |
| 1.5.x    | ✅ Güvenlik Yamalar | 2025-06-20 |
| 1.0.x    | ⚠️ Kritik Yamalar | 2025-01-01 |
| < 1.0    | ❌ Desteklenmiyor | - |

### Versiyon Destek Politikası

- **Tam Destek**: Tüm güvenlik ve bug fix'ler
- **Güvenlik Yamalar**: Sadece kritik güvenlik yamaları
- **Kritik Yamalar**: Sadece kritik güvenlik açıkları
- **Desteklenmiyor**: Hiçbir güvenlik güncellemesi yok

**Önemli**: Güvenlik için her zaman en son stabil versiyonu kullanın!

---

## 🚨 Güvenlik Açığı Bildirimi

### Rapor Etme Süreci

Bir güvenlik açığı bulduysanız, lütfen **sorumlu bir şekilde bildirin**.

#### 1. 📧 Özel Bildirim (Tercih Edilen)

Güvenlik açıklarını **ASLA** public issue'larda bildirmeyin!

**Email**: security@cyberguard-ai.com

**Şablon**:
```
Konu: [SECURITY] Kısa Açıklama

# Güvenlik Açığı Raporu

## Özet
[Açığın kısa açıklaması]

## Etkilenen Versiyon(lar)
[Örn: v2.0.0, v1.5.3]

## Zafiyet Türü
[Örn: SQL Injection, XSS, RCE, vb.]

## CVSS Skoru (varsa)
[Örn: 9.8 - Critical]

## Detaylı Açıklama
[Teknik detaylar]

## Tekrarlama Adımları (PoC)
1. [Adım 1]
2. [Adım 2]
3. [Adım 3]

## Etki Analizi
[Bu açığın potansiyel etkileri]

## Önerilen Çözüm
[Varsa çözüm öneriniz]

## Ek Bilgiler
- İletişim: [Email/Twitter/LinkedIn]
- Disclosure Preference: [Koordineli, Public, vb.]
```

#### 2. 🔐 PGP Encrypted Email (Hassas Durumlar)

Çok kritik açıklar için PGP şifreli email kullanın:

```
PGP Public Key Fingerprint:
1234 5678 90AB CDEF 1234 5678 90AB CDEF 1234 5678

PGP Key: https://keybase.io/cyberguard_ai
```

#### 3. 💬 Bug Bounty Platform

Kayıtlı güvenlik araştırmacıları için:
- **HackerOne**: https://hackerone.com/cyberguard-ai
- **Bugcrowd**: https://bugcrowd.com/cyberguard-ai

### Yanıt Süresi

| Aşama | Süre |
|-------|------|
| İlk Yanıt | 24-48 saat |
| İnceleme | 3-5 iş günü |
| Düzeltme Tahmini | 7-30 gün (kritiklik göre) |
| Public Disclosure | 90 gün (koordineli) |

### Güvenlik Açığı Kritiklik Seviyeleri

**Critical (9.0-10.0)** 🔴
- Remote Code Execution (RCE)
- Authentication Bypass
- SQL Injection (kritik)
- **SLA**: 24 saat içinde yama

**High (7.0-8.9)** 🟠
- Privilege Escalation
- Sensitive Data Exposure
- XSS (stored)
- **SLA**: 7 gün içinde yama

**Medium (4.0-6.9)** 🟡
- CSRF
- XSS (reflected)
- Information Disclosure
- **SLA**: 30 gün içinde yama

**Low (0.1-3.9)** 🟢
- Minor information leaks
- Best practice violations
- **SLA**: Bir sonraki release

---
## 📢 Güvenlik Güncellemeleri

### Security Advisory Aboneliği

Güvenlik güncellemelerinden haberdar olmak için:

1. **GitHub Watch**: "Security alerts only" seçeneğini aktif edin
2. **Mailing List**: security-announce@cyberguard-ai.com
3. **RSS Feed**: https://cyberguard-ai.com/security/feed
4. **Twitter**: @cyberguard_security

### Güvenlik Duyuruları

Tüm güvenlik yamaları aşağıdaki kanallarda duyurulur:

- 📧 Email: security-announce@cyberguard-ai.com
- 🐦 Twitter: @cyberguard_security
- 📰 Blog: https://blog.cyberguard-ai.com/security
- 📢 GitHub Security Advisories

### CVE Numaraları

Ciddi güvenlik açıkları için CVE (Common Vulnerabilities and Exposures) numarası alınır ve şu platformlarda yayınlanır:

- NIST National Vulnerability Database
- MITRE CVE List
- GitHub Security Advisories

---

## 🛠️ Güvenlik En İyi Uygulamaları

### Kurulum Güvenliği

**1. Güvenli Konfigürasyon**

```bash
# ❌ ASLA production'da default şifreler kullanmayın!
# ❌ KÖTÜ
DB_PASSWORD=admin123
API_KEY=default_key

# ✅ İYİ
DB_PASSWORD=$(openssl rand -base64 32)
API_KEY=$(uuidgen)
```

**2. Environment Variables**

```bash
# .env dosyasını ASLA commit etmeyin!
# .gitignore'a ekleyin
echo ".env" >> .gitignore

# .env.example kullanın
cp .env.example .env
# Değerleri güncelleyin
```

**3. HTTPS Kullanımı**

```yaml
# config/security.yaml
server:
  ssl:
    enabled: true
    cert: /path/to/cert.pem
    key: /path/to/key.pem
    min_version: TLSv1.3
```

**4. Firewall Kuralları**

```bash
# Sadece gerekli portları açın
ufw allow 443/tcp  # HTTPS
ufw allow 22/tcp   # SSH (IP whitelist ile)
ufw enable
```

### Uygulama Güvenliği

**1. Input Validation**

```python
# ✅ İYİ: Her input'u validate edin
from pydantic import BaseModel, validator

class ScanRequest(BaseModel):
    target: str
    
    @validator('target')
    def validate_target(cls, v):
        if not is_valid_ip(v) and not is_valid_domain(v):
            raise ValueError('Invalid target')
        return v
```

**2. SQL Injection Protection**

```python
# ❌ KÖTÜ: String concatenation
query = f"SELECT * FROM users WHERE id = {user_id}"

# ✅ İYİ: Parameterized queries
query = "SELECT * FROM users WHERE id = %s"
cursor.execute(query, (user_id,))
```

**3. XSS Protection**

```python
# ✅ Output encoding
from markupsafe import escape

user_input = escape(user_input)
```

**4. Authentication**

```python
# ✅ Güçlü şifre politikası
from passlib.hash import argon2

# Argon2 kullanın (bcrypt'ten daha güvenli)
hashed = argon2.hash(password)
```

**5. Rate Limiting**

```python
# ✅ API rate limiting
from flask_limiter import Limiter

limiter = Limiter(
    app,
    default_limits=["100 per hour", "10 per minute"]
)
```

### Database Güvenliği

```sql
-- ✅ Minimum privilege principle
CREATE USER 'cyberguard_app'@'localhost' 
IDENTIFIED BY 'secure_password';

GRANT SELECT, INSERT, UPDATE 
ON cyberguard.* 
TO 'cyberguard_app'@'localhost';

-- Database encryption at rest
ALTER TABLE sensitive_data 
ENCRYPTION='Y';
```

### Logging ve Monitoring

```python
# ✅ Güvenlik olaylarını logla
import logging

logger = logging.getLogger('security')

# Failed login attempts
logger.warning(f"Failed login: {username} from {ip}")

# Successful privilege escalation
logger.critical(f"Privilege escalation: {user} -> admin")

# ASLA hassas bilgileri loglama!
# ❌ KÖTÜ
logger.info(f"Password: {password}")

# ✅ İYİ
logger.info(f"Password changed for user: {user_id}")
```

---

## 🔍 Güvenlik Denetimi

### Otomatik Güvenlik Taramaları

**1. Dependency Scanning**

```bash
# Python dependencies
pip-audit

# GitHub Dependabot
# .github/dependabot.yml
version: 2
updates:
  - package-ecosystem: "pip"
    directory: "/"
    schedule:
      interval: "daily"
```

**2. SAST (Static Application Security Testing)**

```bash
# Bandit - Python security linter
bandit -r src/

# SonarQube
sonar-scanner
```

**3. DAST (Dynamic Application Security Testing)**

```bash
# OWASP ZAP
zap-cli quick-scan https://cyberguard-ai.com

# Burp Suite Professional
```

**4. Container Security**

```bash
# Trivy - Container vulnerability scanner
trivy image cyberguard-ai:latest

# Snyk
snyk container test cyberguard-ai:latest
```

### Manuel Güvenlik Testleri

**Periyodik Denetimler:**

- 📅 **Haftalık**: Dependency updates
- 📅 **Aylık**: Vulnerability scanning
- 📅 **Üç Aylık**: Penetration testing
- 📅 **Yıllık**: Full security audit

### Security Checklist

- [ ] Tüm dependencies güncel mi?
- [ ] Known vulnerabilities var mı?
- [ ] SSL/TLS doğru yapılandırılmış mı?
- [ ] Authentication güçlü mü?
- [ ] Logging ve monitoring aktif mi?
- [ ] Backup stratejisi var mı?
- [ ] Incident response planı hazır mı?
- [ ] Security training yapıldı mı?

---

## 📊 Güvenlik Metrikleri

Güvenlik durumumuzu şu metriklerle takip ediyoruz:

| Metrik | Hedef | Mevcut |
|--------|-------|--------|
| Mean Time to Detect (MTTD) | < 1 saat | 45 dakika |
| Mean Time to Respond (MTTR) | < 4 saat | 3.5 saat |
| Vulnerability Backlog | < 10 | 5 |
| Security Test Coverage | > 80% | 85% |
| False Positive Rate | < 5% | 3% |

---

## 🎓 Güvenlik Eğitimi

Tüm geliştiricilerin tamamlaması gereken:

1. **OWASP Top 10** (yıllık)
2. **Secure Coding Practices** (yıllık)
3. **Security Awareness Training** (6 ayda bir)
4. **Incident Response Training** (yıllık)

---

## 📞 İletişim

### Güvenlik Ekibi

- 📧 **Genel**: security@cyberguard-ai.com
- 🚨 **Acil**: security-urgent@cyberguard-ai.com
- 🔐 **PGP Key**: https://keybase.io/cyberguard_security

### Çalışma Saatleri

- **İş Günleri**: 09:00 - 18:00 (UTC+3)
- **Acil Durumlar**: 7/24 on-call team

---

## 📚 Kaynaklar

### Standartlar ve Frameworks

- [OWASP Top 10](https://owasp.org/www-project-top-ten/)
- [CWE Top 25](https://cwe.mitre.org/top25/)
- [NIST Cybersecurity Framework](https://www.nist.gov/cyberframework)
- [ISO 27001](https://www.iso.org/isoiec-27001-information-security.html)

### Güvenlik Araçları

- [Bandit](https://github.com/PyCQA/bandit) - Python security linter
- [OWASP ZAP](https://www.zaproxy.org/) - Web app security scanner
- [Trivy](https://github.com/aquasecurity/trivy) - Container scanner
- [SonarQube](https://www.sonarqube.org/) - Code quality & security

---

## ⚖️ Yasal Uyarı

CyberGuard AI, sorumlu güvenlik araştırmalarını destekler ve aşağıdaki koşullarda yasal işlem başlatmayacağını taahhüt eder:

- ✅ Açık, sorumlu şekilde bildirildiğinde
- ✅ Test, belirlenen kapsamda yapıldığında
- ✅ Veri çalınmadığında veya tahrip edilmediğinde
- ✅ DoS/DDoS saldırısı yapılmadığında

---

**Son Güncelleme**: 2025-01-15  
**Versiyon**: 2.0  
**Sonraki İnceleme**: 2025-07-15

---

**🔒 Güvenlik, hepimizin sorumluluğudur. Birlikte daha güvenli bir dijital dünya oluşturalım!**