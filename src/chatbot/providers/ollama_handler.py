"""
Ollama Handler - CyberGuard AI
==============================

Local LLM desteği - İnternet gerektirmez.

Özellikler:
    - Llama 3, Mistral, CodeLlama
    - Tamamen offline çalışır
    - Streaming responses
"""

import logging
from collections.abc import Generator
from datetime import datetime

try:
    import requests

    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

logger = logging.getLogger("OllamaHandler")


class OllamaHandler:
    """
    Ollama Local LLM Handler

    Desteklenen modeller:
    - llama3.2 (en yeni)
    - llama3.1
    - mistral
    - codellama
    - deepseek-coder
    """

    AVAILABLE_MODELS = {
        "llama3.2": {
            "name": "Llama 3.2",
            "description": "Meta'nın en yeni modeli",
            "context_window": 128000,
        },
        "llama3.1": {
            "name": "Llama 3.1",
            "description": "Stabil ve güçlü",
            "context_window": 128000,
        },
        "mistral": {
            "name": "Mistral 7B",
            "description": "Hızlı ve verimli",
            "context_window": 32768,
        },
        "codellama": {
            "name": "CodeLlama",
            "description": "Kod yazımı için optimize",
            "context_window": 16384,
        },
        "deepseek-coder": {
            "name": "DeepSeek Coder",
            "description": "Kod ve güvenlik analizi",
            "context_window": 16384,
        },
    }

    def __init__(
        self,
        model: str = "llama3.2",
        temperature: float = 0.3,
        host: str = "http://localhost:11434",
    ):
        """
        Ollama Handler başlat

        Args:
            model: Kullanılacak model
            temperature: Yaratıcılık seviyesi
            host: Ollama server adresi
        """
        self.logger = logging.getLogger("OllamaHandler")

        self.model = model
        self.temperature = temperature
        self.host = host

        self.logger.info(f"✅ Ollama Handler initialized - Model: {model}")
        print(f"🦙 Ollama başlatıldı - Model: {model} (local)")

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

            # System prompt
            sys_prompt = system_prompt or self._get_default_system_prompt()
            messages.append({"role": "system", "content": sys_prompt})

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

            self.logger.info(f"🦙 Ollama API çağrılıyor... ({len(messages)} mesaj)")

            # API request
            url = f"{self.host}/api/chat"
            payload = {
                "model": self.model,
                "messages": messages,
                "stream": stream,
                "options": {
                    "temperature": self.temperature,
                },
            }

            if stream:
                return self._stream_response(url, payload)

            response = requests.post(url, json=payload)

            if response.status_code != 200:
                return f"Ollama hatası: {response.status_code}"

            result = response.json()
            answer = result.get("message", {}).get("content", "")

            self.logger.info(f"✅ Ollama yanıt alındı ({len(answer)} karakter)")
            return answer

        except requests.exceptions.ConnectionError:
            return "❌ Ollama sunucusu çalışmıyor! `ollama serve` komutu ile başlatın."
        except Exception as e:
            self.logger.error(f"❌ Ollama Error: {e}")
            return f"Üzgünüm, bir hata oluştu: {str(e)}"

    def _stream_response(self, url: str, payload: dict) -> Generator[str, None, None]:
        """Streaming response"""
        try:
            response = requests.post(url, json=payload, stream=True)

            for line in response.iter_lines():
                if line:
                    import json

                    data = json.loads(line)
                    if "message" in data and "content" in data["message"]:
                        yield data["message"]["content"]

        except Exception as e:
            yield f"Hata: {str(e)}"

    def _get_default_system_prompt(self) -> str:
        """Varsayılan sistem promptu"""
        return f"""Sen CyberGuard AI'ın siber güvenlik asistanısın.
Local olarak çalışıyorsun, tamamen offline.

🎯 GÖREVLERİN:
1. Siber güvenlik sorularını yanıtla
2. Saldırı analizi yap
3. Kod örnekleri üret
4. Savunma önerileri sun

📝 KURALLAR:
- Türkçe yanıt ver
- Kısa ve öz ol
- Teknik detaylar ver

Tarih: {datetime.now().strftime('%Y-%m-%d %H:%M')}"""

    def list_local_models(self) -> list[str]:
        """Ollama'da yüklü modelleri listele"""
        try:
            response = requests.get(f"{self.host}/api/tags")
            if response.status_code == 200:
                models = response.json().get("models", [])
                return [m["name"] for m in models]
        except Exception:
            pass
        return []

    def pull_model(self, model_name: str) -> bool:
        """Model indir"""
        try:
            response = requests.post(f"{self.host}/api/pull", json={"name": model_name})
            return response.status_code == 200
        except Exception:
            return False

    def get_model_info(self) -> dict:
        """Model bilgilerini döndür"""
        model_info = self.AVAILABLE_MODELS.get(self.model, {})
        return {
            "provider": "ollama",
            "model": self.model,
            "name": model_info.get("name", self.model),
            "description": model_info.get("description", "Local model"),
            "context_window": model_info.get("context_window", 0),
            "temperature": self.temperature,
            "host": self.host,
            "is_local": True,
        }

    @classmethod
    def list_models(cls) -> list[dict]:
        """Mevcut modelleri listele"""
        return [
            {"id": model_id, **info} for model_id, info in cls.AVAILABLE_MODELS.items()
        ]

    @staticmethod
    def is_available() -> bool:
        """Ollama sunucusu çalışıyor mu?"""
        try:
            response = requests.get("http://localhost:11434/api/tags", timeout=2)
            return response.status_code == 200
        except Exception:
            return False


# Test
if __name__ == "__main__":
    print("🧪 Ollama Handler Test\n")

    if not OllamaHandler.is_available():
        print("❌ Ollama sunucusu çalışmıyor!")
        print("   `ollama serve` komutu ile başlatın.")
    else:
        try:
            handler = OllamaHandler()

            print("📋 Local modeller:")
            for model in handler.list_local_models():
                print(f"   - {model}")

            print("\n💬 Test mesajı gönderiliyor...")
            response = handler.chat("Merhaba! Kendini tanıt.")
            print(f"\n🦙 Yanıt:\n{response}")

        except Exception as e:
            print(f"❌ Hata: {e}")
