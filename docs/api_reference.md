# 🔌 API Reference

CyberGuard AI API dokümantasyonu

---

## 📋 İçindekiler

- [Genel Bakış](#genel-bakış)
- [Authentication](#authentication)
- [Core Modules](#core-modules)
- [Chatbot API](#chatbot-api)
- [ML Model API](#ml-model-api)
- [Database API](#database-api)
- [Utilities API](#utilities-api)

---

## 🌟 Genel Bakış

CyberGuard AI, modüler bir yapıya sahiptir. Her modül bağımsız olarak kullanılabilir.

### Base Configuration

```python
from src.utils import get_config

config = get_config()
# config.yaml dosyasını yükler
```

---

## 🔐 Authentication

### Gemini API Key

```python
# .env dosyasında
GOOGLE_API_KEY=your_api_key_here

# Kullanım
from src.chatbot import GeminiHandler

chatbot = GeminiHandler()
```

---

## 🧩 Core Modules

### Database Manager

```python
from src.utils.database import DatabaseManager

db = DatabaseManager(db_path='cyberguard.db')

# İstatistik al
stats = db.get_database_stats()

# Saldırı ekle
db.add_attack({
    'attack_type': 'DDoS',
    'source_ip': '192.168.1.100',
    'destination_ip': '192.168.0.10',
    'port': 80,
    'severity': 'high',
    'blocked': True
})

# Saldırıları çek
attacks = db.get_attacks(limit=100)
```

### Logger

```python
from src.utils.logger import Logger

logger = Logger("MyModule")

logger.info("Bilgi mesajı")
logger.warning("Uyarı mesajı")
logger.error("Hata mesajı")
logger.critical("Kritik hata")
```

### Config Manager

```python
from src.utils.config import Config

config = Config()

# Değer al
db_path = config.get('database', 'path')

# Değer set et
config.set('model', 'accuracy', 0.95)
```

---

## 🤖 Chatbot API

### GeminiHandler

```python
from src.chatbot.gemini_handler import GeminiHandler

chatbot = GeminiHandler()

# Basit sohbet
response = chatbot.chat("Merhaba!")

# Context ile sohbet
context = {
    'total_attacks': 5000,
    'by_severity': {'critical': 10, 'high': 20}
}
response = chatbot.chat("Son saldırıları göster", context=context)

# Konuşma geçmişini temizle
chatbot.clear_history()

# Konuşmayı kaydet
filename = chatbot.export_conversation()
```

#### Available Methods

| Method | Parameters | Returns | Description |
|--------|-----------|---------|-------------|
| `chat()` | `message: str, context: dict` | `str` | Mesaj gönder, cevap al |
| `clear_history()` | - | - | Konuşma geçmişini temizle |
| `export_conversation()` | `filename: str` | `str` | Konuşmayı JSON'a aktar |
| `get_attack_context()` | `hours: int` | `dict` | Saldırı context'i oluştur |
| `get_ip_context()` | `ip: str` | `dict` | IP context'i oluştur |
| `get_system_context()` | - | `dict` | Sistem context'i oluştur |

---

## 🧠 RAG System API

### RAG Manager

```python
from src.chatbot.vectorstore.rag_manager import RAGManager

rag = RAGManager()

# Döküman ekle
rag.add_text_document(
    text="DDoS saldırısı nedir...",
    metadata={'title': 'DDoS Rehberi', 'category': 'Security'}
)

# PDF ekle
rag.add_pdf_document(
    pdf_path='security_guide.pdf',
    metadata={'title': 'Security Guide'}
)

# Arama yap
results = rag.search("DDoS saldırısından nasıl korunurum?", k=3)

# Context oluştur
context = rag.get_context_for_query("DDoS nedir?", k=3)

# İstatistikler
stats = rag.get_stats()

# Tümünü sil
rag.delete_all_documents()
```

### Memory Manager

```python
from src.chatbot.vectorstore.memory_manager import MemoryManager

memory = MemoryManager(user_id="user123")

# Konuşma ekle
memory.add_conversation(
    user_message="DDoS nedir?",
    bot_response="DDoS, distributed denial of service...",
    context={'source': 'chatbot'}
)

# Hafızada ara
results = memory.search_memory("saldırı sayısı", k=3)

# Son konuşmaları al
context = memory.get_recent_context(n=5)

# İlgili konuşmaları al
relevant = memory.get_relevant_memory_for_query("DDoS", k=2)

# Temizle
memory.clear_short_term()
memory.clear_all_memory()
```

### Attack Vector Manager

```python
from src.chatbot.vectorstore.attack_vectors import AttackVectorManager

attack_vectors = AttackVectorManager()

# Database'i vektörleştir
attack_vectors.vectorize_attacks(limit=1000)

# Benzer saldırı bul
results = attack_vectors.find_similar_attacks("DDoS saldırısı", k=5)

# Pattern analizi
analysis = attack_vectors.analyze_attack_pattern("DDoS")

# Chatbot için özet
summary = attack_vectors.get_attack_summary_for_chatbot("Port Scan")

# Temizle
attack_vectors.clear_vectors()
```

---

## 🎯 ML Model API

### Model Predictor

```python
from src.models.predictor import AttackPredictor

predictor = AttackPredictor()

# Model yükle
predictor.load_models()

# Tahmin yap
attack_data = {
    'source_ip': '192.168.1.105',
    'destination_ip': '192.168.0.10',
    'port': 80,
    'severity': 'critical',
    'blocked': 1,
    'timestamp': '2024-10-29 14:30:00'
}

result = predictor.predict_single(attack_data)
# Returns: {
#     'predicted_type': 'DDoS',
#     'confidence': 0.98,
#     'probabilities': {...},
#     'risk_level': 'critical'
# }

# Toplu tahmin
results = predictor.predict_batch([attack1, attack2, attack3])

# Model bilgisi
info = predictor.get_model_info()
```

### Model Training

```python
from train_model import ModelTrainer

trainer = ModelTrainer(db_path='cyberguard.db')

# Tam eğitim pipeline
trainer.run_full_training(limit=5000)

# Manuel eğitim
df = trainer.load_data_from_db(limit=1000)
X_train, X_test, y_train, y_test = trainer.prepare_data(df)
trainer.train_model(X_train, y_train)
metrics = trainer.evaluate_model(X_test, y_test)
trainer.save_models()
```

---

## 💾 Database API

### Attack Operations

```python
from src.utils.database import DatabaseManager

db = DatabaseManager()

# Saldırı ekle
attack_id = db.add_attack({
    'attack_type': 'SQL Injection',
    'source_ip': '10.0.0.1',
    'destination_ip': '192.168.1.1',
    'port': 3306,
    'severity': 'high',
    'blocked': True,
    'description': 'SQL injection attempt detected'
})

# Saldırıları çek
attacks = db.get_attacks(limit=100)

# Filtreleme
ddos_attacks = db.get_attacks_by_type('DDoS')
critical_attacks = db.get_attacks_by_severity('critical')

# IP bazlı arama
ip_attacks = db.get_attacks_by_ip('192.168.1.100')

# Zaman aralığı
recent_attacks = db.get_attacks_last_hours(24)

# İstatistikler
stats = db.get_database_stats()
# Returns: {
#     'attacks': 5000,
#     'network_logs': 10000,
#     'scan_results': 500,
#     'db_size_mb': 25.5
# }
```

### Scan Operations

```python
# Tarama ekle
scan_id = db.add_scan_result({
    'filename': 'suspicious.exe',
    'file_hash': 'abc123...',
    'scan_result': 'Threat Detected',
    'threat_type': 'Trojan',
    'risk_score': 95,
    'is_malicious': True
})

# Tarama geçmişi
scans = db.get_scan_history(limit=50)

# Zararlı dosyalar
malicious = db.get_malicious_files()
```

---

## 🛠️ Utilities API

### Mock Data Generator

```python
from src.utils.mock_data_generator import MockDataGenerator

generator = MockDataGenerator(db_path='cyberguard.db')

# Veri üret
generator.generate_all(
    attack_count=5000,
    log_count=10000,
    scan_count=2500,
    clear_first=True
)

# Manuel üretim
attacks = generator.generate_attacks(count=100)
logs = generator.generate_logs(count=200)
scans = generator.generate_network_scans(count=50)

# Database'e ekle
generator.insert_to_database(attacks, logs, scans)

# Temizle
generator.clear_database()
```

### PDF Report Generator

```python
from src.utils.pdf_generator import PDFReportGenerator

pdf_gen = PDFReportGenerator(db_path='cyberguard.db')

# Rapor oluştur
filename = pdf_gen.generate_report(
    output_filename='security_report.pdf',
    days=7,
    include_charts=True
)

# İstatistik al
stats = pdf_gen.get_attack_stats(days=7)

# Grafik oluştur
pie_chart = pdf_gen.create_pie_chart(data, title='Saldırı Dağılımı')
bar_chart = pdf_gen.create_bar_chart(data, title='Severity')
```

### Feature Extractor

```python
from src.utils.feature_extractor import FeatureExtractor

extractor = FeatureExtractor()

# DataFrame'den özellik çıkar
X = extractor.prepare_features(df, fit=True)
y = extractor.prepare_labels(df, fit=True)

# Kaydet/Yükle
extractor.save('models/feature_extractor.pkl')
extractor.load('models/feature_extractor.pkl')

# Sınıf ismi al
attack_name = extractor.get_attack_type_name(encoded_label=5)
```

---

## 🔧 Error Handling

Tüm API fonksiyonları exception fırlatabilir:

```python
try:
    result = predictor.predict_single(attack_data)
except FileNotFoundError:
    print("Model dosyası bulunamadı!")
except ValueError:
    print("Geçersiz veri!")
except Exception as e:
    print(f"Beklenmeyen hata: {e}")
```

---

## 📊 Response Formats

### Standard Response

```json
{
  "success": true,
  "data": {...},
  "message": "İşlem başarılı",
  "timestamp": "2024-10-29T14:30:00"
}
```

### Error Response

```json
{
  "success": false,
  "error": "ValueError",
  "message": "Geçersiz veri formatı",
  "timestamp": "2024-10-29T14:30:00"
}
```

---

## 🔗 Örnek Kullanım Senaryoları

### Senaryo 1: Gerçek Zamanlı Saldırı Tespiti

```python
from src.models.predictor import AttackPredictor
from src.utils.database import DatabaseManager

# Model yükle
predictor = AttackPredictor()
db = DatabaseManager()

# Yeni trafik verisi geldiğinde
new_traffic = {
    'source_ip': '192.168.1.105',
    'destination_ip': '192.168.0.10',
    'port': 80,
    'severity': 'high',
    'blocked': 0,
    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
}

# Tahmin yap
result = predictor.predict_single(new_traffic)

# Eğer saldırıysa database'e kaydet
if result['predicted_type'] != 'Normal':
    new_traffic['attack_type'] = result['predicted_type']
    db.add_attack(new_traffic)
    
    # Alarm gönder
    if result['risk_level'] == 'critical':
        send_alert(f"Kritik saldırı: {result['predicted_type']}")
```

### Senaryo 2: AI Asistan ile Analiz

```python
from src.chatbot.gemini_handler import GeminiHandler
from src.chatbot.vectorstore.attack_vectors import AttackVectorManager

chatbot = GeminiHandler()
attack_vectors = AttackVectorManager()

# Kullanıcı sorusu
user_question = "Son 24 saatte en çok hangi tür saldırı oldu?"

# Context oluştur
context = chatbot.get_attack_context(hours=24)
similar_attacks = attack_vectors.get_attack_summary_for_chatbot(user_question)

# Context'i birleştir
full_context = {**context, 'similar_attacks': similar_attacks}

# Cevap al
response = chatbot.chat(user_question, context=full_context)
print(response)
```

### Senaryo 3: Haftalık Rapor Otomasyonu

```python
from src.utils.pdf_generator import PDFReportGenerator
import schedule
import time

def weekly_report():
    pdf_gen = PDFReportGenerator()
    filename = pdf_gen.generate_report(
        output_filename=f'weekly_report_{datetime.now().strftime("%Y%m%d")}.pdf',
        days=7,
        include_charts=True
    )
    send_email_with_attachment(filename)

# Her pazartesi 09:00'da çalıştır
schedule.every().monday.at("09:00").do(weekly_report)

while True:
    schedule.run_pending()
    time.sleep(60)
```

---

## 📚 İleri Seviye Kullanım

### Custom Model Training

```python
from src.models.random_forest_model import CyberAttackModel
from src.utils.feature_extractor import FeatureExtractor
import pandas as pd

# Kendi verinizi yükleyin
df = pd.read_csv('my_attack_data.csv')

# Feature extraction
extractor = FeatureExtractor()
X = extractor.prepare_features(df, fit=True)
y = extractor.prepare_labels(df, fit=True)

# Model oluştur ve eğit
model = CyberAttackModel(n_estimators=200)
model.train(X_train, y_train)

# Değerlendir
metrics = model.evaluate(X_test, y_test)
print(f"Accuracy: {metrics['accuracy']:.2%}")

# Kaydet
model.save('models/my_custom_model.pkl')
extractor.save('models/my_custom_extractor.pkl')
```

---

## 🆘 Destek

API ile ilgili sorularınız için:

- 📧 Email: api-support@cyberguardai.com
- 📖 Docs: [docs.cyberguardai.com](https://docs.cyberguardai.com)
- 💬 Discord: [discord.gg/cyberguardai](https://discord.gg/cyberguardai)

---

## 📝 Version History

### v1.0.0 (Current)
- ✅ Core API
- ✅ Chatbot API
- ✅ ML Model API
- ✅ RAG System API
- ✅ Database API

### v1.1.0 (Planned)
- REST API endpoints
- WebSocket support
- API rate limiting
- API key authentication

---

[⬆️ Back to Top](#-api-reference)