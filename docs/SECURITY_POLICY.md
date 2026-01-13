# 🔒 Security Policy

CyberGuard AI güvenlik politikası ve açık bildirimi

---

## 📋 İçindekiler

- [Desteklenen Sürümler](#desteklenen-sürümler)
- [Güvenlik Açığı Bildirimi](#güvenlik-açığı-bildirimi)
- [Responsible Disclosure](#responsible-disclosure)
- [Güvenlik Önlemleri](#güvenlik-önlemleri)
- [Bug Bounty](#bug-bounty)

---

## ✅ Desteklenen Sürümler

| Sürüm | Destek |
|-------|--------|
| 3.x.x | ✅ Aktif destek |
| 2.x.x | ✅ Güvenlik güncellemeleri |
| 1.x.x | ❌ Destek sona erdi |
| < 1.0 | ❌ Desteklenmiyor |

---

## 🔐 Güvenlik Açığı Bildirimi

### Nasıl Bildirilir?

⚠️ **ÖNEMLİ**: Güvenlik açıklarını **PUBLIC** olarak bildirmeyin!

1. **Email**: <security@cyberguard-ai.com>
2. **GPG Key**: [Public Key](https://cyberguard-ai.com/security.gpg)
3. **HackerOne**: hackerone.com/cyberguard

### Bildirimde Bulunması Gerekenler

```
Konu: [SECURITY] <Kısa açıklama>

1. Açığın Türü: (XSS, SQL Injection, vb.)
2. Etkilenen Bileşen: (API, Frontend, Model, vb.)
3. Etkilenen Sürüm: 
4. Adım Adım Reproduce:
   1. ...
   2. ...
5. Beklenen Davranış:
6. Gerçekleşen Davranış:
7. Proof of Concept: (varsa)
8. Önerilen Düzeltme: (varsa)
```

### Yanıt Süresi

| Aşama | Süre |
|-------|------|
| İlk Yanıt | 24 saat |
| Değerlendirme | 72 saat |
| Fix (Critical) | 7 gün |
| Fix (High) | 30 gün |
| Fix (Medium) | 60 gün |

---

## 📜 Responsible Disclosure

### Kurallar

1. ✅ Sadece kendi test sistemlerinizi kullanın
2. ✅ Verileri modifiye etmeyin veya silmeyin
3. ✅ Hizmet kesintisi yapmayın
4. ✅ Bulduğunuzu bize bildirin, başkalarına değil
5. ✅ Patch yayınlanana kadar bekleyin
6. ❌ Üçüncü taraf verilere erişmeyin
7. ❌ DDoS veya brute force yapmayın

### Safe Harbor

İyi niyetli güvenlik araştırmacılarına karşı **yasal işlem başlatmayız**.

---

## 🛡️ Güvenlik Önlemleri

### Uygulanan

| Önlem | Açıklama |
|-------|----------|
| ✅ TLS 1.3 | Tüm iletişimde |
| ✅ AES-256 | Veri şifreleme |
| ✅ JWT + Refresh | Kimlik doğrulama |
| ✅ Rate Limiting | DoS koruması |
| ✅ Input Validation | Pydantic models |
| ✅ CORS | Origin kontrolü |
| ✅ SQL Parameterization | Injection koruması |
| ✅ XSS Protection | CSP headers |
| ✅ CSRF Tokens | Form güvenliği |
| ✅ Dependency Scanning | Snyk/Dependabot |

### Planlanan

- [ ] Hardware Security Module (HSM)
- [ ] Zero Trust Architecture
- [ ] Quantum-resistant encryption

---

## 💰 Bug Bounty

### Scope

**In Scope:**

- api.cyberguard-ai.com
- app.cyberguard-ai.com
- CyberGuard AI GitHub repo

**Out of Scope:**

- Third-party services
- Physical attacks
- Social engineering

### Ödüller

| Severity | Ödül |
|----------|------|
| Critical (9.0-10.0) | $1,000 - $5,000 |
| High (7.0-8.9) | $500 - $1,000 |
| Medium (4.0-6.9) | $100 - $500 |
| Low (0.1-3.9) | Hall of Fame |

### Hall of Fame

Güvenlik açığı bildiren araştırmacılar (izinleriyle):

- 🏆 [İsim] - Critical XSS (2025)
- 🥈 [İsim] - IDOR (2025)

---

## 📞 İletişim

- **Security Email**: <security@cyberguard-ai.com>
- **GPG Key ID**: 0x1234567890ABCDEF
- **Response Time**: 24 saat içinde

---

## 📅 Son Güncelleme

2026-01-10
