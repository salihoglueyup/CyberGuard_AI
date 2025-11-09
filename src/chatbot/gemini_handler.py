# src/chatbot/gemini_handler.py

"""
Google Gemini AI Handler
Gemini Pro ile chat işlemleri
"""

import os
import sys
import json
from datetime import datetime
from typing import Dict, List, Optional, Any
import time

# Path düzeltmesi (test için)
if __name__ == "__main__":
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

try:
    import google.generativeai as genai

    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    print("⚠️  Warning: google-generativeai not installed")

# Import'ları düzelt
try:
    from src.utils import Logger, get_config, DatabaseManager
except ImportError:
    # Eğer relative import çalışmazsa absolute dene
    from utils import Logger, get_config, DatabaseManager


class GeminiHandler:
    """
    Google Gemini AI Handler

    Özellikleri:
    - Gemini Pro modeli ile chat
    - Context-aware responses
    - Conversation history
    - Intent classification
    - Database integration
    """

    def __init__(self, api_key: Optional[str] = None):
        """
        Gemini handler'ı başlat

        Args:
            api_key (str, optional): Gemini API key
        """

        self.logger = Logger("GeminiHandler")
        self.config = get_config()

        # API Key
        self.api_key = api_key or self.config.gemini_api_key

        if not self.api_key:
            raise ValueError(
                "GOOGLE_API_KEY not found! "
                "Please set it in .env file or pass as parameter."
            )

        if not GEMINI_AVAILABLE:
            raise ImportError(
                "google-generativeai package not installed. "
                "Install it with: pip install google-generativeai"
            )

        # Configure Gemini
        genai.configure(api_key=self.api_key)

        # Chatbot config
        chatbot_config = self.config.get_chatbot_config()

        # Model
        self.model_name = chatbot_config['model_name']
        self.temperature = chatbot_config['temperature']
        self.max_tokens = chatbot_config['max_tokens']

        # Initialize model
        self.model = genai.GenerativeModel(
            model_name=self.model_name,
            generation_config=genai.types.GenerationConfig(
                temperature=self.temperature,
                max_output_tokens=self.max_tokens,
                top_p=chatbot_config['top_p'],
                top_k=chatbot_config['top_k'],
            )
        )

        # Conversation history
        self.conversation_history = []
        self.max_history = chatbot_config['max_history']

        # Database
        self.db = DatabaseManager(self.config.database_path)

        # Session ID
        self.session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        self.logger.info(f"✅ Gemini Handler initialized (model: {self.model_name})")

    # ========================================
    # CHAT FUNCTIONS
    # ========================================

    def chat(self, user_message: str,
             context: Optional[Dict] = None,
             save_to_db: bool = True) -> str:
        """
        Kullanıcı mesajına cevap ver

        Args:
            user_message (str): Kullanıcı mesajı
            context (dict, optional): Ek context bilgisi
            save_to_db (bool): Database'e kaydet

        Returns:
            str: Bot cevabı
        """

        self.logger.info(f"💬 User: {user_message[:50]}...")

        start_time = time.time()

        try:
            # System prompt oluştur
            full_prompt = self._build_prompt(user_message, context)

            # Gemini'ye gönder
            response = self.model.generate_content(full_prompt)

            bot_response = response.text

            # Conversation history'e ekle
            self._add_to_history(user_message, bot_response)

            # Database'e kaydet
            if save_to_db:
                self._save_to_database(
                    user_message,
                    bot_response,
                    context,
                    time.time() - start_time
                )

            self.logger.info(f"🤖 Bot: {bot_response[:50]}...")

            return bot_response

        except Exception as e:
            self.logger.error(f"Gemini error: {str(e)}")
            return self._get_error_response(str(e))

    # src/chatbot/gemini_handler.py içinde _build_prompt metodunu bul ve değiştir

    def _build_prompt(self, user_message: str,
                      context: Optional[Dict] = None) -> str:
        """
        Tam prompt oluştur (system + context + history + user)

        Args:
            user_message (str): Kullanıcı mesajı
            context (dict, optional): Context bilgisi

        Returns:
            str: Tam prompt
        """

        # System prompt - BASITLEŞTIRILMIŞ
        system_prompt = """Sen CyberGuard AI'ın siber güvenlik asistanısın.

    GÖREVLER:
    - Siber güvenlik sorularını yanıtla
    - Sistem verilerini analiz et
    - Saldırıları açıkla
    - Güvenlik önerileri ver

    KURALLAR:
    - Türkçe cevap ver
    - Kısa ve anlaşılır ol
    - Emoji kullan ama az
    - Eğer veri varsa, mutlaka kullan ve göster

    """

        # Context - SADELEŞTIRILMIŞ
        context_str = ""
        if context and len(context) > 0:
            context_str = "\n=== MEVCUT SİSTEM VERİLERİ ===\n"

            # Context'i basit göster
            if 'total_attacks' in context:
                context_str += f"Toplam Saldırı: {context['total_attacks']}\n"

            if 'by_severity' in context:
                context_str += f"Severity Dağılımı: {context['by_severity']}\n"

            if 'by_type' in context:
                context_str += f"Saldırı Türleri: {context['by_type']}\n"

            if 'recent_attacks' in context and context['recent_attacks']:
                context_str += f"\nSon Saldırılar ({len(context['recent_attacks'])} adet):\n"
                for attack in context['recent_attacks'][:3]:  # Sadece ilk 3'ü
                    context_str += f"  - {attack.get('type', 'Unknown')} - {attack.get('source', 'Unknown')} - {attack.get('severity', 'Unknown')}\n"

            if 'status' in context:
                context_str += f"Sistem Durumu: {context['status']}\n"

            if 'total_logs' in context:
                context_str += f"Toplam Log: {context['total_logs']}\n"

            if 'total_scans' in context:
                context_str += f"Tarama Sayısı: {context['total_scans']}\n"

            if 'ip' in context:
                context_str += f"\nIP Analizi: {context['ip']}\n"
                context_str += f"  - Bağlantı: {context.get('total_connections', 0)}\n"
                context_str += f"  - Zararlı: {context.get('malicious_attempts', 0)}\n"
                context_str += f"  - Risk: {context.get('risk_score', 0)}/10\n"

            context_str += "===========================\n"

        # History - BASITLEŞTIRILMIŞ
        history_str = ""
        if self.conversation_history:
            history_str = "\n--- Önceki Konuşma ---\n"
            for msg in self.conversation_history[-2:]:  # Sadece son 2 mesaj
                history_str += f"Kullanıcı: {msg['user']}\n"
                history_str += f"Sen: {msg['bot'][:100]}...\n\n"  # İlk 100 karakter

        # BASIT PROMPT
        full_prompt = f"""{system_prompt}

    {context_str}

    {history_str}

    Kullanıcı Sorusu: {user_message}

    Cevap:"""

        return full_prompt

    def _add_to_history(self, user_message: str, bot_response: str):
        """Conversation history'e ekle"""

        self.conversation_history.append({
            'timestamp': datetime.now().isoformat(),
            'user': user_message,
            'bot': bot_response
        })

        # Limit aşarsa eski mesajları sil
        if len(self.conversation_history) > self.max_history:
            self.conversation_history.pop(0)

    def _save_to_database(self, user_message: str, bot_response: str,
                          context: Optional[Dict], response_time: float):
        """Database'e kaydet"""

        try:
            self.db.add_chat_message({
                'user_message': user_message,
                'bot_response': bot_response,
                'intent': self._detect_intent(user_message),
                'context_used': context or {},
                'response_time': response_time,
                'user_id': 'default_user',
                'session_id': self.session_id
            })
        except Exception as e:
            self.logger.warning(f"Failed to save to database: {e}")

    def _get_error_response(self, error_msg: str) -> str:
        """Hata durumunda cevap"""

        return (
            "❌ Üzgünüm, bir hata oluştu!\n\n"
            "Lütfen:\n"
            "1. Sorunuzu farklı şekilde sormayı deneyin\n"
            "2. Sistem yöneticisiyle iletişime geçin\n\n"
            f"Hata detayı: {error_msg[:100]}"
        )

    # ========================================
    # INTENT DETECTION
    # ========================================

    def _detect_intent(self, message: str) -> str:
        """
        Basit intent detection (keyword based)

        Args:
            message (str): Kullanıcı mesajı

        Returns:
            str: Intent
        """

        message_lower = message.lower()

        # Intent keywords
        intents = {
            'query_attacks': ['saldırı', 'attack', 'kaç', 'son', 'tespit'],
            'analyze_ip': ['ip', 'analiz', 'analyze', 'adres'],
            'scan_file': ['dosya', 'tara', 'scan', 'virus', 'malware'],
            'get_report': ['rapor', 'report', 'özet', 'summary'],
            'system_status': ['durum', 'status', 'sistem', 'system'],
            'ask_info': ['nedir', 'nasıl', 'ne', 'what', 'how'],
        }

        for intent, keywords in intents.items():
            if any(keyword in message_lower for keyword in keywords):
                return intent

        return 'general'

    # ========================================
    # CONTEXT HELPERS
    # ========================================

    def get_attack_context(self, hours: int = 24) -> Dict:
        """Saldırı context'i oluştur"""

        try:
            stats = self.db.get_attack_stats(hours=hours)
            recent_attacks = self.db.get_recent_attacks(limit=5, hours=hours)

            return {
                'total_attacks': stats['total'],
                'by_severity': stats['by_severity'],
                'by_type': stats['by_type'],
                'blocked': stats['blocked'],
                'recent_attacks': [
                    {
                        'time': a['timestamp'],
                        'type': a['attack_type'],
                        'source': a['source_ip'],
                        'severity': a['severity']
                    }
                    for a in recent_attacks
                ]
            }
        except Exception as e:
            self.logger.warning(f"Failed to get attack context: {e}")
            return {}

    def get_ip_context(self, ip_address: str) -> Dict:
        """IP context'i oluştur"""

        try:
            history = self.db.get_ip_history(ip_address, limit=50)

            if not history:
                return {'error': 'IP bulunamadı'}

            # İstatistikler hesapla
            total_connections = len(history)
            malicious_count = sum(1 for h in history if h.get('is_attack'))

            return {
                'ip': ip_address,
                'total_connections': total_connections,
                'malicious_attempts': malicious_count,
                'risk_score': round((malicious_count / total_connections) * 10, 1) if total_connections > 0 else 0,
                'last_seen': history[0]['timestamp'] if history else None
            }
        except Exception as e:
            self.logger.warning(f"Failed to get IP context: {e}")
            return {'error': str(e)}

    def get_system_context(self) -> Dict:
        """Sistem context'i oluştur"""

        try:
            db_stats = self.db.get_database_stats()

            return {
                'status': 'OPERATIONAL',
                'total_attacks': db_stats.get('attacks', 0),
                'total_logs': db_stats.get('network_logs', 0),
                'total_scans': db_stats.get('scan_results', 0),
                'db_size_mb': db_stats.get('db_size_mb', 0)
            }
        except Exception as e:
            self.logger.warning(f"Failed to get system context: {e}")
            return {'status': 'UNKNOWN'}

    # ========================================
    # CONVERSATION MANAGEMENT
    # ========================================

    def clear_history(self):
        """Conversation history'yi temizle"""
        self.conversation_history.clear()
        self.logger.info("🗑️  Conversation history cleared")

    def get_history(self) -> List[Dict]:
        """Conversation history'yi döndür"""
        return self.conversation_history.copy()

    def export_conversation(self, filename: Optional[str] = None) -> str:
        """
        Konuşmayı dosyaya export et

        Args:
            filename (str, optional): Dosya adı

        Returns:
            str: Dosya yolu
        """

        if filename is None:
            filename = f"conversation_{self.session_id}.json"

        with open(filename, 'w', encoding='utf-8') as f:
            json.dump({
                'session_id': self.session_id,
                'model': self.model_name,
                'timestamp': datetime.now().isoformat(),
                'conversation': self.conversation_history
            }, f, indent=2, ensure_ascii=False)

        self.logger.info(f"💾 Conversation exported: {filename}")
        return filename


# ========================================
# CONVENIENCE FUNCTIONS
# ========================================

def create_chatbot(api_key: Optional[str] = None) -> GeminiHandler:
    """
    Chatbot instance oluştur

    Args:
        api_key (str, optional): Gemini API key

    Returns:
        GeminiHandler instance
    """
    return GeminiHandler(api_key=api_key)


# ========================================
# TEST
# ========================================

if __name__ == "__main__":
    print("🧪 Gemini Handler Test\n")
    print("=" * 60)

    try:
        # Chatbot oluştur
        print("\n🤖 Chatbot başlatılıyor...")
        chatbot = GeminiHandler()

        print("✅ Chatbot hazır!\n")
        print("=" * 60)

        # Test soruları
        test_questions = [
            "Merhaba! Kendini tanıt.",
            "DDoS saldırısı nedir?",
            "Sistem durumu nedir?",
        ]

        for i, question in enumerate(test_questions, 1):
            print(f"\n{'=' * 60}")
            print(f"Soru {i}:")
            print(f"👤 {question}")
            print()

            # Context hazırla (bazı sorular için)
            context = None
            if 'durum' in question.lower():
                context = chatbot.get_system_context()

            # Cevap al
            response = chatbot.chat(question, context=context, save_to_db=False)

            print(f"🤖 {response}")
            print()

            # Biraz bekle (rate limiting için)
            time.sleep(2)

        print("=" * 60)

        # History göster
        print("\n📜 Conversation History:")
        history = chatbot.get_history()
        print(f"Total messages: {len(history)}")

        print("\n" + "=" * 60)
        print("✅ Test tamamlandı!")

    except ValueError as e:
        print(f"\n❌ Error: {e}")
        print("\n💡 Tip: .env dosyasına GOOGLE_API_KEY ekleyin:")
        print("   GOOGLE_API_KEY=your_api_key_here")

    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback

        traceback.print_exc()