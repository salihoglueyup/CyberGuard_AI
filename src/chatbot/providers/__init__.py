"""
Chatbot Providers - CyberGuard AI
=================================

Tüm LLM provider handler'ları.

Providers:
    - Groq (Llama 3.3) - Ücretsiz
    - OpenAI (GPT-4o)
    - Claude (Claude 3.5)
    - Ollama (Local)
    - Gemini (Google)
"""

from .groq_handler import GroqHandler
from .provider_manager import ProviderManager, LLMProvider, get_provider_manager

# Optional imports
try:
    from .openai_handler import OpenAIHandler
except ImportError:
    OpenAIHandler = None

try:
    from .claude_handler import ClaudeHandler
except ImportError:
    ClaudeHandler = None

try:
    from .ollama_handler import OllamaHandler
except ImportError:
    OllamaHandler = None

try:
    from .gemini_handler import GeminiHandler, EnhancedGeminiHandler
except ImportError:
    GeminiHandler = None
    EnhancedGeminiHandler = None


__all__ = [
    "GroqHandler",
    "OpenAIHandler",
    "ClaudeHandler",
    "OllamaHandler",
    "GeminiHandler",
    "EnhancedGeminiHandler",
    "ProviderManager",
    "LLMProvider",
    "get_provider_manager",
]
