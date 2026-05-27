"""
Provider Manager - CyberGuard AI
================================

Tüm LLM provider'larını yöneten unified interface.

Özellikler:
    - Multi-provider desteği
    - Fallback mekanizması
    - Auto-selection
    - Streaming
"""

import logging
import os
from enum import Enum
from typing import Any

logger = logging.getLogger("ProviderManager")


class LLMProvider(Enum):
    GROQ = "groq"
    OPENAI = "openai"
    CLAUDE = "claude"
    GEMINI = "gemini"
    OLLAMA = "ollama"


class ProviderManager:
    """
    LLM Provider Manager

    Unified interface for all LLM providers.
    """

    PROVIDER_PRIORITY = [
        LLMProvider.GROQ,  # Ücretsiz, hızlı
        LLMProvider.GEMINI,  # Güçlü
        LLMProvider.OPENAI,  # Yaygın
        LLMProvider.CLAUDE,  # Kod için iyi
        LLMProvider.OLLAMA,  # Local
    ]

    def __init__(self, preferred_provider: LLMProvider | None = None):
        """
        Provider Manager başlat

        Args:
            preferred_provider: Tercih edilen provider
        """
        self.preferred_provider = preferred_provider
        self.handlers: dict[LLMProvider, Any] = {}
        self.active_provider: LLMProvider | None = None

        self._initialize_handlers()

    def _initialize_handlers(self):
        """Mevcut handler'ları başlat"""

        # Groq
        try:
            from src.chatbot.providers.groq_handler import GroqHandler

            if os.getenv("GROQ_API_KEY"):
                self.handlers[LLMProvider.GROQ] = GroqHandler
                logger.info("✅ Groq provider ready")
        except Exception as e:
            logger.debug(f"Groq not available: {e}")

        # OpenAI
        try:
            from src.chatbot.providers.openai_handler import OpenAIHandler

            if os.getenv("OPENAI_API_KEY"):
                self.handlers[LLMProvider.OPENAI] = OpenAIHandler
                logger.info("✅ OpenAI provider ready")
        except Exception as e:
            logger.debug(f"OpenAI not available: {e}")

        # Claude
        try:
            from src.chatbot.providers.claude_handler import ClaudeHandler

            if os.getenv("ANTHROPIC_API_KEY") or os.getenv("CLAUDE_API_KEY"):
                self.handlers[LLMProvider.CLAUDE] = ClaudeHandler
                logger.info("✅ Claude provider ready")
        except Exception as e:
            logger.debug(f"Claude not available: {e}")

        # Gemini
        try:
            from src.chatbot.providers.gemini_handler import GeminiHandler

            if os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"):
                self.handlers[LLMProvider.GEMINI] = GeminiHandler
                logger.info("✅ Gemini provider ready")
        except Exception as e:
            logger.debug(f"Gemini not available: {e}")

        # Ollama
        try:
            from src.chatbot.providers.ollama_handler import OllamaHandler

            if OllamaHandler.is_available():
                self.handlers[LLMProvider.OLLAMA] = OllamaHandler
                logger.info("✅ Ollama provider ready (local)")
        except Exception as e:
            logger.debug(f"Ollama not available: {e}")

        # Active provider seç
        self._select_active_provider()

    def _select_active_provider(self):
        """Aktif provider seç"""
        if self.preferred_provider and self.preferred_provider in self.handlers:
            self.active_provider = self.preferred_provider
        else:
            # Priority order
            for provider in self.PROVIDER_PRIORITY:
                if provider in self.handlers:
                    self.active_provider = provider
                    break

        if self.active_provider:
            logger.info(f"🎯 Active provider: {self.active_provider.value}")

    def get_handler(self, provider: LLMProvider | None = None):
        """
        Handler instance al

        Args:
            provider: Spesifik provider (None = active)
        """
        target = provider or self.active_provider

        if not target or target not in self.handlers:
            return None

        handler_class = self.handlers[target]
        return handler_class()

    def chat(
        self,
        user_message: str,
        system_prompt: str | None = None,
        context: str | None = None,
        history: list[dict] | None = None,
        provider: LLMProvider | None = None,
        stream: bool = False,
        fallback: bool = True,
    ) -> str:
        """
        Chat with automatic fallback

        Args:
            user_message: Kullanıcı mesajı
            system_prompt: Sistem promptu
            context: Ek bağlam
            history: Konuşma geçmişi
            provider: Spesifik provider
            stream: Streaming response
            fallback: Hata durumunda başka provider dene
        """
        target_provider = provider or self.active_provider

        if not target_provider:
            return "❌ Hiçbir LLM provider mevcut değil!"

        # Önce tercih edilen provider
        providers_to_try = [target_provider]

        if fallback:
            for p in self.PROVIDER_PRIORITY:
                if p != target_provider and p in self.handlers:
                    providers_to_try.append(p)

        last_error = None

        for p in providers_to_try:
            try:
                handler = self.get_handler(p)
                if handler:
                    logger.info(f"💬 Trying {p.value}...")

                    response = handler.chat(
                        user_message=user_message,
                        system_prompt=system_prompt,
                        context=context,
                        history=history,
                        stream=stream,
                    )

                    if response and not response.startswith("Üzgünüm"):
                        return response

            except Exception as e:
                last_error = e
                logger.warning(f"⚠️ {p.value} failed: {e}")
                continue

        return f"❌ Tüm provider'lar başarısız oldu. Son hata: {last_error}"

    def get_available_providers(self) -> list[dict]:
        """Mevcut provider'ları listele"""
        providers = []

        for provider in LLMProvider:
            is_available = provider in self.handlers
            is_active = provider == self.active_provider

            if provider == LLMProvider.GROQ:
                models = ["llama-3.3-70b-versatile", "llama-3.1-8b-instant"]
            elif provider == LLMProvider.OPENAI:
                models = ["gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo"]
            elif provider == LLMProvider.CLAUDE:
                models = ["claude-3-5-sonnet", "claude-3-opus"]
            elif provider == LLMProvider.GEMINI:
                models = ["gemini-pro", "gemini-1.5-pro"]
            elif provider == LLMProvider.OLLAMA:
                models = ["llama3.2", "mistral", "codellama"]
            else:
                models = []

            providers.append(
                {
                    "id": provider.value,
                    "name": provider.value.title(),
                    "available": is_available,
                    "active": is_active,
                    "models": models,
                    "is_local": provider == LLMProvider.OLLAMA,
                    "is_free": provider in [LLMProvider.GROQ, LLMProvider.OLLAMA],
                }
            )

        return providers

    def set_provider(self, provider: LLMProvider) -> bool:
        """Aktif provider değiştir"""
        if provider in self.handlers:
            self.active_provider = provider
            logger.info(f"🔄 Switched to {provider.value}")
            return True
        return False

    def get_active_provider_info(self) -> dict:
        """Aktif provider bilgisi"""
        if not self.active_provider:
            return {"error": "No active provider"}

        handler = self.get_handler()
        if handler and hasattr(handler, "get_model_info"):
            return handler.get_model_info()

        return {"provider": self.active_provider.value, "status": "active"}


# Singleton
_provider_manager: ProviderManager | None = None


def get_provider_manager() -> ProviderManager:
    """Global provider manager"""
    global _provider_manager
    if _provider_manager is None:
        _provider_manager = ProviderManager()
    return _provider_manager


# Test
if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()

    print("🧪 Provider Manager Test\n")

    manager = ProviderManager()

    print("📋 Mevcut Provider'lar:")
    for p in manager.get_available_providers():
        status = (
            "✅ Active" if p["active"] else ("✓ Ready" if p["available"] else "❌ N/A")
        )
        print(f"   {status} {p['name']}: {', '.join(p['models'][:2])}")

    print(
        f"\n🎯 Active: {manager.active_provider.value if manager.active_provider else 'None'}"
    )

    if manager.active_provider:
        print("\n💬 Test mesajı gönderiliyor...")
        response = manager.chat("Merhaba! Kendini tanıt.")
        print(f"\n🤖 Yanıt:\n{response[:500]}...")
