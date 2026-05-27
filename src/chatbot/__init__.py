"""
CyberGuard AI Chatbot
=====================

AI Chatbot modülü - Multi-provider LLM desteği.

Yapı:
    - providers/     : LLM handler'lar (Groq, OpenAI, Claude, Ollama, Gemini)
    - memory/        : Konuşma hafızası ve context
    - integration/   : ML model entegrasyonu
    - vectorstore/   : RAG sistemi
"""

# Providers
# Integration
from .integration import (
    ModelIntegration,
    get_integration,
)

# Memory
from .memory import (
    ContextBuilder,
    ConversationMemory,
    get_context_builder,
    get_memory,
)
from .providers import (
    GroqHandler,
    LLMProvider,
    ProviderManager,
    get_provider_manager,
)

__all__ = [
    # Providers
    "GroqHandler",
    "ProviderManager",
    "LLMProvider",
    "get_provider_manager",
    # Memory
    "ConversationMemory",
    "get_memory",
    "ContextBuilder",
    "get_context_builder",
    # Integration
    "ModelIntegration",
    "get_integration",
]
