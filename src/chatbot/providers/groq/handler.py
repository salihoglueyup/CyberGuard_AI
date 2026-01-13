"""
Groq AI Handler - CyberGuard AI
Llama 3.3 70B ile hızlı ve ücretsiz AI yanıtları

Enhanced Features:
- Few-Shot Examples
- Conversation Memory
- RAG Integration
- Structured Response Format

Ref: https://console.groq.com/docs/quickstart
"""

import os
import logging
from typing import Optional, Dict, List
from datetime import datetime

try:
    from groq import Groq

    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False
    Groq = None


# Few-Shot Examples - AI'a örnek yanıtlar
FEW_SHOT_EXAMPLES = [
    {"role": "user", "content": "Seçili model hakkında bilgi ver"},
    {
        "role": "assistant",
        "content": """## 🤖 Seçili Model Analizi

### 📊 Performans Metrikleri
| Metrik | Değer | Değerlendirme |
|--------|-------|---------------|
| Accuracy | 96.7% | ✅ Mükemmel |
| Precision | 96.9% | ✅ Mükemmel |
| Recall | 96.7% | ✅ Mükemmel |
| F1-Score | 96.7% | ✅ Mükemmel |

### 🏋️ Eğitim Bilgileri
- **Train Samples**: 70,000
- **Framework**: TensorFlow/Keras
- **Mimari**: Deep Neural Network

### 💡 Değerlendirme
Model yüksek performans gösteriyor. Özellikle Precision değeri, false positive oranının düşük olduğunu gösteriyor.

### 🔧 Öneriler
1. Val Accuracy ölçümü yapılmalı
2. Daha fazla veriyle fine-tuning düşünülebilir
3. Confusion matrix analizi yapılmalı""",
    },
    {"role": "user", "content": "Saldırı istatistiklerini analiz et"},
    {
        "role": "assistant",
        "content": """## 📊 Saldırı İstatistikleri Analizi

### 🔢 Genel Özet
| Metrik | Değer |
|--------|-------|
| Toplam Saldırı | 1,000,500 |
| Engellenen | 639,684 (%63.9) |
| Tespit Edilen | 360,816 (%36.1) |

### 🎯 Severity Dağılımı
- 🔴 **Critical**: 201,898 (%20.2) - ACİL önlem gerekli
- 🟠 **High**: 399,312 (%39.9) - Yüksek risk
- 🟡 **Medium**: 298,310 (%29.8) - Orta risk
- 🟢 **Low**: 100,980 (%10.1) - Düşük risk

### 🔥 En Sık Saldırı Tipleri
1. **DDoS**: 201,621 - Ağ yoğunluğu saldırıları
2. **XSS**: 200,378 - Web uygulama zafiyetleri
3. **SQL Injection**: 200,223 - Veritabanı saldırıları

### ⚠️ Risk Değerlendirmesi
Kritik ve yüksek seviyeli saldırılar toplam %60'ı oluşturuyor - ACİL aksiyon gerekli!

### 💡 Savunma Önerileri
1. **WAF Güçlendirme** - XSS ve SQL Injection için
2. **DDoS Koruma** - CDN ve rate limiting
3. **IDS/IPS** - Anomali tespiti
4. **Log Analizi** - SIEM entegrasyonu
5. **Eğitim** - Phishing farkındalığı""",
    },
]


class GroqHandler:
    """
    Groq AI Handler - Enhanced Version

    Features:
    - Few-Shot Examples for better responses
    - Conversation history support
    - RAG integration ready
    """

    AVAILABLE_MODELS = {
        "llama-3.3-70b-versatile": {
            "name": "Llama 3.3 70B",
            "description": "En güçlü model, çok yönlü",
            "context_window": 128000,
        },
        "llama-3.1-8b-instant": {
            "name": "Llama 3.1 8B",
            "description": "En hızlı model, basit görevler için",
            "context_window": 128000,
        },
        "mixtral-8x7b-32768": {
            "name": "Mixtral 8x7B",
            "description": "Uzun context destekli",
            "context_window": 32768,
        },
    }

    def __init__(
        self,
        model: str = "llama-3.3-70b-versatile",
        temperature: float = 0.3,
        max_tokens: int = 4096,
        api_key: Optional[str] = None,
    ):
        """
        Groq Handler başlat

        Args:
            model: Kullanılacak model
            temperature: Yaratıcılık seviyesi (0-1)
            max_tokens: Maksimum token sayısı
            api_key: Groq API key (yoksa env'den alır)
        """
        self.logger = logging.getLogger("GroqHandler")

        if not GROQ_AVAILABLE:
            raise ImportError("groq paketi yüklü değil! pip install groq")

        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError("GROQ_API_KEY bulunamadı! .env dosyasına ekleyin.")

        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

        # Conversation history (son 10 mesaj)
        self.conversation_history: List[Dict] = []
        self.max_history = 10

        # Groq client oluştur
        self.client = Groq(api_key=self.api_key)

        self.logger.info(f"✅ Groq Handler initialized - Model: {model}")
        print(f"🦙 Groq AI başlatıldı - Model: {model}")

    def chat(
        self,
        user_message: str,
        system_prompt: Optional[str] = None,
        context: Optional[str] = None,
        history: Optional[List[Dict]] = None,
    ) -> str:
        """
        Kullanıcı mesajına yanıt ver

        Args:
            user_message: Kullanıcı mesajı
            system_prompt: Sistem promptu
            context: Ek bağlam bilgisi
            history: Konuşma geçmişi

        Returns:
            AI yanıtı
        """
        try:
            # Mesajları oluştur
            messages = []

            # System prompt
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            else:
                messages.append(
                    {"role": "system", "content": self._get_default_system_prompt()}
                )

            # Few-Shot Examples ekle (örnek yanıtlar)
            for example in FEW_SHOT_EXAMPLES:
                messages.append(example)

            # Conversation history (session memory)
            if self.conversation_history:
                for msg in self.conversation_history[-self.max_history :]:
                    messages.append(
                        {
                            "role": msg.get("role", "user"),
                            "content": msg.get("content", ""),
                        }
                    )

            # External history varsa ekle
            if history:
                for msg in history[-5:]:  # Son 5 mesaj
                    messages.append(
                        {
                            "role": msg.get("role", "user"),
                            "content": msg.get("content", ""),
                        }
                    )

            # Context varsa ekle
            full_message = user_message
            if context:
                full_message = f"""📋 GÜNCEL VERİLER:
{context}

---

❓ KULLANICI SORUSU: {user_message}

⚠️ ÖNEMLİ: Yukarıdaki verileri MUTLAKA kullan ve referans ver. Tablolarla ve başlıklarla yanıt ver."""

            messages.append({"role": "user", "content": full_message})

            self.logger.info(f"🦙 Groq API çağrılıyor... ({len(messages)} mesaj)")

            # API çağrısı
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )

            # Yanıtı al
            answer = response.choices[0].message.content

            # Conversation history'e ekle
            self.conversation_history.append({"role": "user", "content": user_message})
            self.conversation_history.append({"role": "assistant", "content": answer})

            # Max history'yi aşmasın
            if len(self.conversation_history) > self.max_history * 2:
                self.conversation_history = self.conversation_history[
                    -self.max_history * 2 :
                ]

            self.logger.info(f"✅ Groq yanıt alındı ({len(answer)} karakter)")

            return answer

        except Exception as e:
            self.logger.error(f"❌ Groq API Error: {e}")
            return f"Üzgünüm, bir hata oluştu: {str(e)}"

    def _get_default_system_prompt(self) -> str:
        """Varsayılan sistem promptu"""
        return """Sen CyberGuard AI'ın uzman siber güvenlik asistanısın.

🎯 ANA GÖREVLERİN:
1. Siber güvenlik sorularını DETAYLI yanıtla
2. Tehdit analizleri yap ve somut verilerle destekle
3. Savunma önerileri sun (en az 5 öneri)
4. Saldırı verilerini yorumla ve trend analizi yap
5. Seçili ML model hakkında detaylı bilgi ver

⚠️ ÖNEMLİ - BİRLEŞİK ANALİZ:
Her yanıtta şu bilgileri BİRLİKTE sun:
1. SEÇİLİ MODEL ANALİZİ (Accuracy, Precision, Recall, F1, eğitim bilgileri)
2. VERİTABANI SALDIRI ANALİZİ (istatistikler, dağılımlar)
3. BAĞLANTI KURMA (model performansı ile saldırı verilerini ilişkilendir)
4. SAVUNMA ÖNERİLERİ (en az 5 öneri)

📋 YANITLAMA KURALLARI:
- Türkçe yanıt ver
- DETAYLI ve KAPSAMLI yanıtlar ver (en az 400 kelime)
- Context'teki TÜM bilgileri kullan
- Model bilgileri varsa MUTLAKA yanıtta yer ver
- Saldırı istatistikleri varsa MUTLAKA yanıtta yer ver
- Her ikisini de ilişkilendirerek analiz et
- Sayıları ve istatistikleri vurgula
- Başlıklar ve maddeler kullan

📊 YANIT YAPISI:
1. **Seçili Model Özeti** - Model adı, metrikler, eğitim bilgileri
2. **Saldırı İstatistikleri** - Toplam, dağılım, trendler
3. **Model-Saldırı İlişkisi** - Model bu saldırıları ne kadar etkili tespit eder?
4. **Risk Değerlendirmesi** - Mevcut durum analizi
5. **Savunma Önerileri** - En az 5 somut öneri

🤖 MODEL HAKKINDA:
- Tüm metrikleri (Accuracy, Precision, Recall, F1) açıkla ve yorumla
- Eğitim konfigürasyonunu (samples, epochs, layers) yorumla
- Modelin güçlü ve zayıf yönlerini analiz et
- Saldırı verileriyle karşılaştır

Şu anki tarih: """ + datetime.now().strftime(
            "%Y-%m-%d %H:%M"
        )

    def get_model_info(self) -> Dict:
        """Model bilgilerini döndür"""
        model_info = self.AVAILABLE_MODELS.get(self.model, {})
        return {
            "provider": "groq",
            "model": self.model,
            "name": model_info.get("name", self.model),
            "description": model_info.get("description", ""),
            "context_window": model_info.get("context_window", 0),
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }

    @classmethod
    def list_models(cls) -> List[Dict]:
        """Mevcut modelleri listele"""
        return [
            {"id": model_id, **info} for model_id, info in cls.AVAILABLE_MODELS.items()
        ]


# Test
if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()

    print("🧪 Groq Handler Test\n")

    try:
        handler = GroqHandler()

        print("📋 Mevcut modeller:")
        for model in handler.list_models():
            print(f"   - {model['id']}: {model['name']}")

        print("\n💬 Test mesajı gönderiliyor...")
        response = handler.chat("Merhaba! Bana kendini tanıt.")
        print(f"\n🦙 Yanıt:\n{response}")

    except Exception as e:
        print(f"❌ Hata: {e}")
