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
from .providers import (
    GroqHandler,
    ProviderManager,
    LLMProvider,
    get_provider_manager,
)

# Memory
from .memory import (
    ConversationMemory,
    get_memory,
    ContextBuilder,
    get_context_builder,
)

# Integration
from .integration import (
    ModelIntegration,
    get_integration,
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
