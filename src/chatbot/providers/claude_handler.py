"""
Claude Handler - CyberGuard AI
==============================

Anthropic Claude 3.5 Sonnet desteği.

Özellikler:
    - Streaming responses
    - Tool use
    - Kod yazımında güçlü
"""

import logging
import os
from collections.abc import Generator
from datetime import datetime

try:
    from anthropic import Anthropic

    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False
    Anthropic = None

logger = logging.getLogger("ClaudeHandler")


class ClaudeHandler:
    """
    Claude (Anthropic) API Handler

    Desteklenen modeller:
    - claude-3-5-sonnet-20241022 (en yeni)
    - claude-3-opus-20240229 (en güçlü)
    - claude-3-sonnet-20240229
    - claude-3-haiku-20240307 (hızlı)
    """

    AVAILABLE_MODELS = {
        "claude-3-5-sonnet-20241022": {
            "name": "Claude 3.5 Sonnet",
            "description": "En yeni, kod yazımında çok iyi",
            "context_window": 200000,
        },
        "claude-3-opus-20240229": {
            "name": "Claude 3 Opus",
            "description": "En güçlü reasoning",
            "context_window": 200000,
        },
        "claude-3-sonnet-20240229": {
            "name": "Claude 3 Sonnet",
            "description": "Dengeli performans",
            "context_window": 200000,
        },
        "claude-3-haiku-20240307": {
            "name": "Claude 3 Haiku",
            "description": "En hızlı, ekonomik",
            "context_window": 200000,
        },
    }

    def __init__(
        self,
        model: str = "claude-3-5-sonnet-20241022",
        temperature: float = 0.3,
        max_tokens: int = 4096,
        api_key: str | None = None,
    ):
        """
        Claude Handler başlat
        """
        self.logger = logging.getLogger("ClaudeHandler")

        if not ANTHROPIC_AVAILABLE:
            raise ImportError("anthropic paketi yüklü değil! pip install anthropic")

        self.api_key = (
            api_key or os.getenv("ANTHROPIC_API_KEY") or os.getenv("CLAUDE_API_KEY")
        )
        if not self.api_key:
            raise ValueError("ANTHROPIC_API_KEY bulunamadı! .env dosyasına ekleyin.")

        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

        # Anthropic client
        self.client = Anthropic(api_key=self.api_key)

        self.logger.info(f"✅ Claude Handler initialized - Model: {model}")
        print(f"🎭 Claude başlatıldı - Model: {model}")

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
        """
        try:
            messages = []

            # History
            if history:
                for msg in history[-10:]:
                    role = msg.get("role", "user")
                    if role == "system":
                        continue  # Claude system prompt ayrı
                    messages.append(
                        {
                            "role": role,
                            "content": msg.get("content", ""),
                        }
                    )

            # Context + user message
            full_message = user_message
            if context:
                full_message = f"{context}\n\n---\n\nKullanıcı Sorusu: {user_message}"

            messages.append({"role": "user", "content": full_message})

            # System prompt
            sys_prompt = system_prompt or self._get_default_system_prompt()

            self.logger.info(f"🎭 Claude API çağrılıyor... ({len(messages)} mesaj)")

            if stream:
                return self._stream_response(messages, sys_prompt)

            # Non-streaming
            response = self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                system=sys_prompt,
                messages=messages,
            )

            answer = response.content[0].text
            self.logger.info(f"✅ Claude yanıt alındı ({len(answer)} karakter)")

            return answer

        except Exception as e:
            self.logger.error(f"❌ Claude API Error: {e}")
            return f"Üzgünüm, bir hata oluştu: {str(e)}"

    def _stream_response(
        self, messages: list[dict], system_prompt: str
    ) -> Generator[str, None, None]:
        """Streaming response"""
        try:
            with self.client.messages.stream(
                model=self.model,
                max_tokens=self.max_tokens,
                system=system_prompt,
                messages=messages,
            ) as stream:
                for text in stream.text_stream:
                    yield text

        except Exception as e:
            yield f"Hata: {str(e)}"

    def _get_default_system_prompt(self) -> str:
        """Varsayılan sistem promptu"""
        return f"""Sen CyberGuard AI'ın gelişmiş siber güvenlik asistanısın.

🎯 TEMEL GÖREVLERİN:
1. Siber güvenlik sorularını uzman düzeyinde yanıtla
2. Saldırı tespiti ve analizi yap
3. Savunma stratejileri öner
4. Kod örnekleri ve YARA kuralları üret (Claude'un güçlü noktası)
5. MITRE ATT&CK mapping yap

📝 CEVAP KURALLARI:
- Her zaman Türkçe yanıt ver
- Teknik ama anlaşılır ol
- Kod yazarken açıklamalı ol
- Tablo formatını kullan (markdown)

Şu anki tarih: {datetime.now().strftime('%Y-%m-%d %H:%M')}"""

    def get_model_info(self) -> dict:
        """Model bilgilerini döndür"""
        model_info = self.AVAILABLE_MODELS.get(self.model, {})
        return {
            "provider": "anthropic",
            "model": self.model,
            "name": model_info.get("name", self.model),
            "description": model_info.get("description", ""),
            "context_window": model_info.get("context_window", 0),
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
        """Claude kullanılabilir mi?"""
        return ANTHROPIC_AVAILABLE and bool(
            os.getenv("ANTHROPIC_API_KEY") or os.getenv("CLAUDE_API_KEY")
        )


# Test
if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()

    print("🧪 Claude Handler Test\n")

    if not ClaudeHandler.is_available():
        print("❌ Claude API key bulunamadı!")
    else:
        try:
            handler = ClaudeHandler()

            print("📋 Mevcut modeller:")
            for model in handler.list_models():
                print(f"   - {model['id']}: {model['name']}")

            print("\n💬 Test mesajı gönderiliyor...")
            response = handler.chat("Merhaba! Kendini tanıt.")
            print(f"\n🎭 Yanıt:\n{response}")

        except Exception as e:
            print(f"❌ Hata: {e}")
