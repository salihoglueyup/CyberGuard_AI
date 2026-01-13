# ⚡ CyberGuard AI - Hızlı Başlangıç (5 Dakika)

> Bu rehber ile 5 dakikada CyberGuard AI'ı çalıştırabilirsiniz.

---

## 📋 Ön Gereksinimler

- ✅ Python 3.9+ kurulu
- ✅ Node.js 18+ kurulu
- ✅ Git kurulu

---

## 🚀 Adım 1: Projeyi İndirin

```bash
git clone https://github.com/salihoglueyup/CyberGuard_AI.git
cd CyberGuard_AI
```

---

## 🐍 Adım 2: Python Bağımlılıkları

```bash
pip install -r requirements.txt
```

---

## 📦 Adım 3: Frontend Bağımlılıkları

```bash
cd frontend
npm install
cd ..
```

---

## 🔑 Adım 4: Ortam Değişkenleri (Opsiyonel)

`.env` dosyası oluşturun:

```bash
# Windows
copy .env.example .env

# Linux/Mac
cp .env.example .env
```

AI Asistan için API anahtarı ekleyin:

```env
GROQ_API_KEY=your_groq_api_key
```

> 💡 **İpucu:** Ücretsiz Groq API anahtarı almak için: <https://console.groq.com>

---

## ▶️ Adım 5: Başlatın

### Kolay Yol (Windows)

```bash
run.bat
```

### Manuel Yol

**Terminal 1 - Backend:**

```bash
cd app
python -m uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

**Terminal 2 - Frontend:**

```bash
cd frontend
npm run dev
```

---

## 🌐 Erişim Adresleri

| Servis | URL |
| ------ | --- |
| 🖥️ Frontend | <http://localhost:5173> |
| 🔌 Backend API | <http://localhost:8000> |
| 📚 API Docs | <http://localhost:8000/api/docs> |
| 📖 ReDoc | <http://localhost:8000/api/redoc> |

---

## ✅ Başarılı Kurulum Kontrolü

1. Tarayıcıda <http://localhost:5173> açın
2. Dashboard yüklenirse ✅
3. Sol menüden "Attack Map" seçin
4. 3D Globe görüntülenirse ✅

---

## 🔧 Sorun Giderme

### Port kullanımda hatası

```bash
# Windows - 8000 portunu kullanan processi bul
netstat -ano | findstr :8000

# Farklı port kullan
uvicorn main:app --port 8001
```

### npm hatası

```bash
# Node modules'ü temizle
rm -rf node_modules
npm cache clean --force
npm install
```

### Backend başlamıyor

```bash
# Eksik paketleri kontrol et
pip install -r requirements.txt --upgrade
```

---

## 📚 Sonraki Adımlar

- 📖 [Kullanım Kılavuzu](KULLANIM_KILAVUZU.md) - Detaylı kullanım
- 🔌 [API Örnekleri](API_EXAMPLES.md) - API kullanımı
- 🌐 [WebSocket Rehberi](WEBSOCKET_GUIDE.md) - Gerçek zamanlı veri

---

**🛡️ Haydi başlayalım!**
