"""
Groq AI Handler - CyberGuard AI
Llama 3.3 70B ile hızlı ve ücretsiz AI yanıtları

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


class GroqHandler:
    """
    Groq AI Handler

    Desteklenen modeller:
    - llama-3.3-70b-versatile (en güçlü)
    - llama-3.1-8b-instant (en hızlı)
    - mixtral-8x7b-32768 (uzun context)
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

            # Geçmiş mesajlar
            if history:
                for msg in history[-10:]:  # Son 10 mesaj
                    messages.append(
                        {
                            "role": msg.get("role", "user"),
                            "content": msg.get("content", ""),
                        }
                    )

            # Context varsa ekle
            full_message = user_message
            if context:
                full_message = f"{context}\n\n---\n\nKullanıcı Sorusu: {user_message}"

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

            self.logger.info(f"✅ Groq yanıt alındı ({len(answer)} karakter)")

            return answer

        except Exception as e:
            self.logger.error(f"❌ Groq API Error: {e}")
            return f"Üzgünüm, bir hata oluştu: {str(e)}"

    def _get_default_system_prompt(self) -> str:
        """Varsayılan sistem promptu - model bilgileri dahil"""

        # Model bilgilerini al
        model_info = ""
        try:
            from src.chatbot.model_integration import get_integration

            integration = get_integration()

            if integration.training_results:
                model_info = "\n\n📊 MEVCUT MODEL SONUÇLARI:\n"
                for name, results in list(integration.training_results.items())[:5]:
                    if isinstance(results, dict):
                        acc = results.get("accuracy", 0)
                        if isinstance(acc, float) and acc < 1:
                            acc *= 100
                        model_info += f"  - {name}: %{acc:.2f} accuracy\n"

                model_count = len(integration.get_available_models())
                model_info += f"\n📦 Toplam {model_count} eğitilmiş model var."
        except:
            pass

        return f"""Sen CyberGuard AI'ın gelişmiş siber güvenlik asistanısın.

🎯 TEMEL GÖREVLERİN:
1. Siber güvenlik sorularını uzman düzeyinde yanıtla
2. IDS/IPS modelleri hakkında bilgi ver
3. Saldırı tespiti ve analizi yap
4. Makine öğrenimi modellerini açıkla
5. Savunma stratejileri öner

🤖 MEVCUT SİSTEM DURUMU:
- SSA-LSTMIDS modeli aktif (makale ile birebir)
- CICIDS2017 dataset ile eğitildi
- DDoS ve PortScan modelleri hazır
- Real-time IDS mevcut
{model_info}

📝 CEVAP KURALLARI:
- Her zaman Türkçe yanıt ver
- Teknik ama anlaşılır ol
- Somut ve uygulanabilir öneriler sun
- Emoji kullan ama abartma
- Kod örnekleri göster gerekirse
- Tablo formatını kullan (markdown)
- Karşılaştırmalı bilgi ver

🔧 ÖNEMLİ BİLGİLER:
- Makale: SSA-LSTMIDS (Conv1D + LSTM)
- Parametreler: Conv1D(30), LSTM(120), Dense(512)
- CICIDS2017 accuracy: %99.96
- DDoS model accuracy: %99.62

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
