"""
OpenAI Handler - CyberGuard AI
==============================

GPT-4, GPT-4o, GPT-4-turbo desteği.

Özellikler:
    - Streaming responses
    - Function calling
    - Vision (görsel analiz)
"""

import logging
import os
from collections.abc import Generator
from datetime import datetime

try:
    from openai import OpenAI

    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    OpenAI = None

logger = logging.getLogger("OpenAIHandler")


class OpenAIHandler:
    """
    OpenAI API Handler

    Desteklenen modeller:
    - gpt-4o (en yeni, multimodal)
    - gpt-4-turbo (hızlı)
    - gpt-4 (güçlü)
    - gpt-3.5-turbo (ekonomik)
    """

    AVAILABLE_MODELS = {
        "gpt-4o": {
            "name": "GPT-4o",
            "description": "En yeni, multimodal (görsel + metin)",
            "context_window": 128000,
            "supports_vision": True,
        },
        "gpt-4-turbo": {
            "name": "GPT-4 Turbo",
            "description": "Hızlı ve güçlü",
            "context_window": 128000,
            "supports_vision": True,
        },
        "gpt-4": {
            "name": "GPT-4",
            "description": "En güçlü reasoning",
            "context_window": 8192,
            "supports_vision": False,
        },
        "gpt-3.5-turbo": {
            "name": "GPT-3.5 Turbo",
            "description": "Hızlı ve ekonomik",
            "context_window": 16385,
            "supports_vision": False,
        },
    }

    def __init__(
        self,
        model: str = "gpt-4o",
        temperature: float = 0.3,
        max_tokens: int = 4096,
        api_key: str | None = None,
    ):
        """
        OpenAI Handler başlat

        Args:
            model: Kullanılacak model
            temperature: Yaratıcılık seviyesi (0-1)
            max_tokens: Maksimum token sayısı
            api_key: OpenAI API key (yoksa env'den alır)
        """
        self.logger = logging.getLogger("OpenAIHandler")

        if not OPENAI_AVAILABLE:
            raise ImportError("openai paketi yüklü değil! pip install openai")

        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY bulunamadı! .env dosyasına ekleyin.")

        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

        # OpenAI client
        self.client = OpenAI(api_key=self.api_key)

        self.logger.info(f"✅ OpenAI Handler initialized - Model: {model}")
        print(f"🧠 OpenAI başlatıldı - Model: {model}")

    def chat(
        self,
        user_message: str,
        system_prompt: str | None = None,
        context: str | None = None,
        history: list[dict] | None = None,
        stream: bool = False,
    ) -> str:
        """
        Kullanıcı mesajına yanıt ver

        Args:
            user_message: Kullanıcı mesajı
            system_prompt: Sistem promptu
            context: Ek bağlam bilgisi
            history: Konuşma geçmişi
            stream: Streaming response

        Returns:
            AI yanıtı
        """
        try:
            messages = []

            # System prompt
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            else:
                messages.append(
                    {"role": "system", "content": self._get_default_system_prompt()}
                )

            # History
            if history:
                for msg in history[-10:]:
                    messages.append(
                        {
                            "role": msg.get("role", "user"),
                            "content": msg.get("content", ""),
                        }
                    )

            # Context + user message
            full_message = user_message
            if context:
                full_message = f"{context}\n\n---\n\nKullanıcı Sorusu: {user_message}"

            messages.append({"role": "user", "content": full_message})

            self.logger.info(f"🧠 OpenAI API çağrılıyor... ({len(messages)} mesaj)")

            if stream:
                return self._stream_response(messages)

            # Non-streaming
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )

            answer = response.choices[0].message.content
            self.logger.info(f"✅ OpenAI yanıt alındı ({len(answer)} karakter)")

            return answer

        except Exception as e:
            self.logger.error(f"❌ OpenAI API Error: {e}")
            return f"Üzgünüm, bir hata oluştu: {str(e)}"

    def _stream_response(self, messages: list[dict]) -> Generator[str, None, None]:
        """Streaming response"""
        try:
            stream = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                stream=True,
            )

            for chunk in stream:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content

        except Exception as e:
            yield f"Hata: {str(e)}"

    def chat_with_vision(
        self,
        user_message: str,
        image_url: str,
        system_prompt: str | None = None,
    ) -> str:
        """
        Görsel ile chat (GPT-4o, GPT-4-turbo)

        Args:
            user_message: Kullanıcı mesajı
            image_url: Görsel URL veya base64
            system_prompt: Sistem promptu
        """
        if not self.AVAILABLE_MODELS.get(self.model, {}).get("supports_vision"):
            return "Bu model görsel analizi desteklemiyor. GPT-4o veya GPT-4-turbo kullanın."

        try:
            messages = []

            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})

            messages.append(
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_message},
                        {"type": "image_url", "image_url": {"url": image_url}},
                    ],
                }
            )

            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=self.max_tokens,
            )

            return response.choices[0].message.content

        except Exception as e:
            return f"Görsel analiz hatası: {str(e)}"

    def _get_default_system_prompt(self) -> str:
        """Varsayılan sistem promptu"""
        return f"""Sen CyberGuard AI'ın gelişmiş siber güvenlik asistanısın.

🎯 TEMEL GÖREVLERİN:
1. Siber güvenlik sorularını uzman düzeyinde yanıtla
2. Saldırı tespiti ve analizi yap
3. Savunma stratejileri öner
4. Kod örnekleri ve YARA kuralları üret
5. MITRE ATT&CK mapping yap

📝 CEVAP KURALLARI:
- Her zaman Türkçe yanıt ver
- Teknik ama anlaşılır ol
- Somut ve uygulanabilir öneriler sun
- Kod örnekleri göster gerekirse
- Tablo formatını kullan (markdown)

Şu anki tarih: {datetime.now().strftime('%Y-%m-%d %H:%M')}"""

    def get_model_info(self) -> dict:
        """Model bilgilerini döndür"""
        model_info = self.AVAILABLE_MODELS.get(self.model, {})
        return {
            "provider": "openai",
            "model": self.model,
            "name": model_info.get("name", self.model),
            "description": model_info.get("description", ""),
            "context_window": model_info.get("context_window", 0),
            "supports_vision": model_info.get("supports_vision", False),
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }

    @classmethod
    def list_models(cls) -> list[dict]:
        """Mevcut modelleri listele"""
        return [
            {"id": model_id, **info} for model_id, info in cls.AVAILABLE_MODELS.items()
        ]

    @staticmethod
    def is_available() -> bool:
        """OpenAI kullanılabilir mi?"""
        return OPENAI_AVAILABLE and bool(os.getenv("OPENAI_API_KEY"))


# Test
if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()

    print("🧪 OpenAI Handler Test\n")

    if not OpenAIHandler.is_available():
        print("❌ OpenAI API key bulunamadı!")
    else:
        try:
            handler = OpenAIHandler(model="gpt-4o")

            print("📋 Mevcut modeller:")
            for model in handler.list_models():
                print(f"   - {model['id']}: {model['name']}")

            print("\n💬 Test mesajı gönderiliyor...")
            response = handler.chat("Merhaba! Kendini tanıt.")
            print(f"\n🧠 Yanıt:\n{response}")

        except Exception as e:
            print(f"❌ Hata: {e}")
