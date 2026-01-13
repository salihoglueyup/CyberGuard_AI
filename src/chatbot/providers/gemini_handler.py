"""
Enhanced Gemini Handler - CyberGuard AI
RAG + ML Model entegrasyonlu chatbot

Dosya Yolu: src/chatbot/gemini_handler.py
"""

import os
import sys
import json
import time
from datetime import datetime
from typing import Dict, List, Optional, Any

# Path düzeltmesi
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, project_root)

try:
    import google.generativeai as genai

    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    print("⚠️ google-generativeai not installed")

from src.utils import Logger, get_config, DatabaseManager
from src.chatbot.context_manager import ContextManager
from src.chatbot.intent_classifier import IntentClassifier
from src.chatbot.vectorstore.rag_manager import RAGManager
from src.chatbot.vectorstore.memory_manager import MemoryManager
from src.chatbot.model_knowledge import ModelKnowledgeManager
from src.chatbot.vectorstore.attack_vectors import AttackVectorManager


class EnhancedGeminiHandler:
    """
    Enhanced Gemini AI Handler with RAG + ML Integration

    Özellikler:
    - Intent classification
    - Context-aware responses
    - RAG document retrieval
    - ML model integration
    - Memory management
    - Attack vector analysis
    """

    def __init__(self, api_key: Optional[str] = None, user_id: str = "default"):
        """
        Args:
            api_key: Gemini API key
            user_id: Kullanıcı ID
        """

        self.logger = Logger("EnhancedGemini")
        self.config = get_config()
        self.user_id = user_id

        # API Key
        self.api_key = api_key or self.config.gemini_api_key

        if not self.api_key:
            raise ValueError("GOOGLE_API_KEY not found!")

        if not GEMINI_AVAILABLE:
            raise ImportError("google-generativeai not installed")

        # Configure Gemini
        genai.configure(api_key=self.api_key)

        # Chatbot config
        chatbot_config = self.config.get_chatbot_config()

        # Model
        self.model_name = chatbot_config["model_name"]
        self.temperature = chatbot_config["temperature"]
        self.max_tokens = chatbot_config["max_tokens"]

        # Initialize Gemini model
        self.model = genai.GenerativeModel(
            model_name=self.model_name,
            generation_config=genai.types.GenerationConfig(
                temperature=self.temperature,
                max_output_tokens=self.max_tokens,
                top_p=chatbot_config["top_p"],
                top_k=chatbot_config["top_k"],
            ),
        )

        # Components
        self.context_manager = ContextManager()
        self.intent_classifier = IntentClassifier()
        self.rag_manager = RAGManager()
        self.memory_manager = MemoryManager(user_id=user_id)

        try:
            self.model_knowledge = ModelKnowledgeManager()
            self.attack_vectors = AttackVectorManager()
        except Exception as e:
            self.logger.warning(f"Optional components failed: {e}")
            self.model_knowledge = None
            self.attack_vectors = None

        # Database
        self.db = DatabaseManager(self.config.database_path)

        # Session ID
        self.session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        self.logger.info(f"✅ Enhanced Gemini Handler initialized")

    # ========================================
    # MAIN CHAT FUNCTION
    # ========================================

    def chat(
        self, user_message: str, save_to_db: bool = True, model_id: Optional[str] = None
    ) -> str:
        """
        Kullanıcı mesajına akıllı cevap ver

        Args:
            user_message: Kullanıcı mesajı
            save_to_db: Database'e kaydet
            model_id: Kullanılacak ML model ID (None ise en iyi model)

        Returns:
            str: Bot cevabı
        """

        self.logger.info(f"💬 User: {user_message[:50]}...")
        start_time = time.time()

        # Seçili model ID'yi kaydet
        self.active_model_id = model_id

        try:
            # 1. Intent classify
            intent, confidence = self.intent_classifier.classify(user_message)
            self.logger.info(f"🎯 Intent: {intent} (confidence: {confidence:.2%})")

            # 2. Entity extraction
            self.logger.info("📋 Extracting entities...")
            entities = self.intent_classifier.extract_entities(user_message)
            self.logger.info(f"✅ Entities: {entities}")

            # 3. Context oluştur (ML prediction dahil)
            self.logger.info("🔧 Building context...")
            context = self._build_context(user_message, intent, entities)
            self.logger.info(f"✅ Context built")

            # 4. ML Model bilgisini context'e ekle
            if model_id:
                self.logger.info(f"🤖 Getting model info for: {model_id[:30]}...")
                context["active_model_id"] = model_id
                model_info = self._get_model_info(model_id)
                if model_info:
                    context["active_model"] = model_info
                    self.logger.info("✅ Model info added")

                    # Model için saldırı verilerini de ekle
                    self.logger.info("📊 Getting attack data...")
                    attack_data = self._get_attacks_for_model(model_id)
                    if attack_data:
                        context["model_attacks"] = attack_data
                        self.logger.info(
                            f"✅ Attack data: {attack_data.get('total_attacks_all_time', 0)} attacks"
                        )

            # 5. RAG retrieval
            self.logger.info("📚 RAG retrieval...")
            rag_context = self._get_rag_context(user_message, intent)
            self.logger.info(f"✅ RAG done ({len(rag_context)} chars)")

            # 6. Memory retrieval
            self.logger.info("🧠 Memory retrieval...")
            memory_context = self.memory_manager.get_relevant_memory_for_query(
                user_message, k=2
            )
            self.logger.info(f"✅ Memory done")

            # 7. Prompt oluştur
            full_prompt = self._build_enhanced_prompt(
                user_message, intent, context, rag_context, memory_context
            )

            self.logger.info(f"📝 Prompt length: {len(full_prompt)} chars")

            # 8. Gemini'ye gönder
            self.logger.info("🔄 Calling Gemini API...")
            try:
                response = self.model.generate_content(full_prompt)
                bot_response = response.text
                self.logger.info(
                    f"✅ Gemini response received ({len(bot_response)} chars)"
                )
            except Exception as gemini_error:
                self.logger.error(f"❌ Gemini API Error: {gemini_error}")
                return f"❌ Üzgünüm, bir hata oluştu!\n\nLütfen:\n1. Sorunuzu farklı şekilde sormayı deneyin\n2. Sistem yöneticisiyle iletişime geçin\n\nHata: {str(gemini_error)[:100]}"

            # 9. Memory'e kaydet
            self.memory_manager.add_conversation(user_message, bot_response, context)

            # 10. Database'e kaydet
            if save_to_db:
                self._save_to_database(
                    user_message,
                    bot_response,
                    intent,
                    context,
                    time.time() - start_time,
                )

            self.logger.info(f"🤖 Bot: {bot_response[:50]}...")

            return bot_response

        except Exception as e:
            self.logger.error(f"Chat error: {str(e)}")
            import traceback

            traceback.print_exc()
            return self._get_error_response(str(e))

    def _get_model_info(self, model_id: str) -> Optional[Dict]:
        """Model bilgilerini getir"""
        try:
            import json

            registry_path = os.path.join(
                os.path.dirname(
                    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                ),
                "models",
                "model_registry.json",
            )

            if os.path.exists(registry_path):
                with open(registry_path, "r", encoding="utf-8") as f:
                    registry = json.load(f)

                for m in registry.get("models", []):
                    if m.get("id") == model_id or m.get("model_id") == model_id:
                        return {
                            "id": model_id,
                            "name": m.get("name", m.get("model_name")),
                            "type": m.get("type", m.get("model_type", "ML Model")),
                            "accuracy": m.get("metrics", {}).get("accuracy", 0),
                            "f1_score": m.get("metrics", {}).get("f1_score", 0),
                            "train_samples": m.get("training_config", {}).get(
                                "train_samples", 0
                            ),
                            "status": m.get("status"),
                        }
        except Exception as e:
            self.logger.error(f"Model info error: {e}")
        return None

    def _get_attacks_for_model(self, model_id: str) -> Optional[Dict]:
        """Model için veritabanından TÜM saldırı verilerini getir"""
        try:
            from src.utils.database import DatabaseManager

            db = DatabaseManager()

            # TÜM saldırı verilerini al (hours=None ile tüm zamanlar)
            all_stats = db.get_attack_stats(hours=None)  # Tüm veritabanı
            recent_attacks = db.get_recent_attacks(limit=5)  # Son 5 saldırı (hızlı)

            # total zaten stats'ta var
            total_records = all_stats.get("total", 0)

            # Saldırı özeti oluştur
            attack_summary = {
                "total_db_records": total_records,
                "total_attacks_all_time": all_stats.get("total", 0),
                "blocked_all_time": all_stats.get("blocked", 0),
                "block_rate": round(
                    (all_stats.get("blocked", 0) / max(1, all_stats.get("total", 1)))
                    * 100,
                    1,
                ),
                "by_type": all_stats.get("by_type", {}),
                "by_severity": all_stats.get("by_severity", {}),
                "recent_attacks": [],
            }

            # En son 10 kritik saldırıyı ekle
            for attack in recent_attacks[:10]:
                attack_summary["recent_attacks"].append(
                    {
                        "id": attack.get("id"),
                        "type": attack.get("attack_type"),
                        "severity": attack.get("severity"),
                        "source_ip": attack.get("source_ip"),
                        "target_ip": attack.get("destination_ip"),
                        "timestamp": attack.get("timestamp"),
                        "blocked": attack.get("blocked", False),
                    }
                )

            return attack_summary

        except Exception as e:
            self.logger.error(f"Attack data error: {e}")
            return None

    # ========================================
    # CONTEXT BUILDING
    # ========================================

    def _build_context(self, user_message: str, intent: str, entities: Dict) -> Dict:
        """
        Intent ve entity'lere göre context oluştur

        Args:
            user_message: Kullanıcı mesajı
            intent: Tespit edilen intent
            entities: Extract edilen entity'ler

        Returns:
            Dict: Context
        """

        # Intent'e göre gerekli context'leri belirle
        context_req = self.intent_classifier.get_required_context(intent)

        context = {}

        # Attack context
        if context_req.get("attacks"):
            time_period = entities.get("time_periods", [])
            hours = time_period[0]["hours"] if time_period else 24
            context["attacks"] = self.context_manager.get_attack_context(hours=hours)

        # IP context + ML Prediction
        if entities.get("ip_addresses"):
            ip = entities["ip_addresses"][0]
            context["ip_info"] = self.context_manager.get_ip_context(ip)

            # ML Model ile IP analizi yap
            try:
                ml_analysis = self.analyze_ip_with_model(ip)
                if ml_analysis.get("success") or "analysis" in ml_analysis:
                    context["ml_prediction"] = ml_analysis
                    self.logger.info(
                        f"ML Prediction for {ip}: {ml_analysis.get('analysis', {}).get('risk_level')}"
                    )
            except Exception as e:
                self.logger.warning(f"ML prediction failed: {e}")

        # Model context
        if context_req.get("models"):
            context["models"] = self.context_manager.get_model_context(
                query=user_message
            )

        # System context
        if context_req.get("system"):
            context["system"] = self.context_manager.get_system_context()

        return context

    def _get_rag_context(self, user_message: str, intent: str) -> str:
        """
        RAG sisteminden ilgili bilgileri getir

        Args:
            user_message: Kullanıcı mesajı
            intent: Intent

        Returns:
            str: RAG context
        """

        rag_context = ""

        # General documents (her zaman ara)
        doc_results = self.rag_manager.search(user_message, k=2)
        if doc_results:
            rag_context += "İLGİLİ DÖKÜMANLAR:\n\n"
            for i, result in enumerate(doc_results, 1):
                rag_context += f"[Döküman {i}]\n{result['content'][:200]}...\n\n"

        # Model knowledge (model sorularında)
        if (
            intent in ["query_models", "compare_models", "predict_attack"]
            and self.model_knowledge
        ):
            model_ctx = self.model_knowledge.get_model_context_for_chatbot(
                user_message, k=2
            )
            if model_ctx:
                rag_context += model_ctx

        # Attack vectors (saldırı sorularında)
        if intent in ["query_attacks", "predict_attack"] and self.attack_vectors:
            attack_ctx = self.attack_vectors.get_attack_summary_for_chatbot(
                user_message
            )
            if attack_ctx:
                rag_context += attack_ctx

        return rag_context

    def _build_enhanced_prompt(
        self,
        user_message: str,
        intent: str,
        context: Dict,
        rag_context: str,
        memory_context: str,
    ) -> str:
        """
        Gelişmiş prompt oluştur

        Args:
            user_message: Kullanıcı mesajı
            intent: Intent
            context: Database context
            rag_context: RAG context
            memory_context: Memory context

        Returns:
            str: Tam prompt
        """

        # System prompt
        system_prompt = """Sen CyberGuard AI'ın siber güvenlik asistanısın.

GÖREVLER:
✅ Siber güvenlik sorularını yanıtla
✅ Sistem verilerini analiz et ve göster
✅ ML modelleri hakkında bilgi ver
✅ Saldırıları açıkla ve önerilerde bulun
✅ Güvenlik önerileri sun

KURALLAR:
📌 Türkçe cevap ver
📌 Kısa, net ve anlaşılır ol
📌 Emoji kullan ama az (max 3-4)
📌 Eğer veri varsa MUTLAKA kullan ve göster
📌 Rakamları ve istatistikleri öne çıkar
📌 Profesyonel ama arkadaşça ol

"""

        # Intent-specific guidance
        intent_guidance = {
            "query_attacks": "Saldırı istatistiklerini net bir şekilde sun. Kritik olanları vurgula.",
            "analyze_ip": "IP analizini detaylı yap. Risk skorunu ve önerilerini belirt.",
            "query_models": "Model performansını karşılaştırmalı göster. En iyiyi öner.",
            "predict_attack": "Tahmin sonucunu açıkla. Güven skorunu belirt.",
            "system_status": "Sistem durumunu özet geç. Sorun varsa belirt.",
        }

        if intent in intent_guidance:
            system_prompt += f"\n🎯 BU SORU İÇİN: {intent_guidance[intent]}\n"

        # Context
        context_str = ""
        if context:
            context_str = self.context_manager.format_context_for_chatbot(context)

        # RAG context
        rag_str = ""
        if rag_context:
            rag_str = f"\n{rag_context}\n"

        # Memory context
        memory_str = ""
        if memory_context:
            memory_str = f"\n{memory_context}\n"

        model_str = ""
        if context.get("active_model"):
            model = context["active_model"]
            model_str = f"""
🤖 SEÇİLİ ML MODEL:
- İsim: {model.get('name', 'Bilinmiyor')}
- ID: {model.get('id', 'Bilinmiyor')}
- Doğruluk: {model.get('accuracy', 0) * 100:.1f}%
- Tip: {model.get('type', 'ML Model')}
"""

            # Saldırı verileri varsa ekle
            if context.get("model_attacks"):
                attacks = context["model_attacks"]
                model_str += f"""
📊 VERİTABANI SALDIRI VERİLERİ:
- Veritabanı Toplam Kayıt: {attacks.get('total_db_records', 0):,}
- Tüm Zamanlar Saldırı: {attacks.get('total_attacks_all_time', 0):,}
- Tüm Zamanlar Engellenen: {attacks.get('blocked_all_time', 0):,} ({attacks.get('block_rate', 0)}%)
- Son 24 Saat Saldırı: {attacks.get('total_attacks_24h', 0):,}
- Son 24 Saat Engellenen: {attacks.get('blocked_24h', 0):,}

📈 SALDIRI TÜRLERİNE GÖRE DAĞILIM (Tüm Zamanlar):
{attacks.get('by_type', {})}

🔴 CİDDİYET SEVİYESİNE GÖRE:
{attacks.get('by_severity', {})}
"""

                # Son saldırılar
                recent = attacks.get("recent_attacks", [])
                if recent:
                    model_str += "\n⚠️ SON 5 SALDIRI:\n"
                    for att in recent[:5]:
                        model_str += f"  • {att.get('type')} | {att.get('severity')} | {att.get('source_ip')} → {att.get('target_ip')} | {'✅ Engellendi' if att.get('blocked') else '🚨 Aktif!'}\n"

                model_str += """
ÖNEMLI: 
- Bu model veritabanındaki TÜM saldırı verileriyle çalışıyor
- Yukarıdaki GERÇEK saldırı verilerini kullanarak detaylı analiz yap
- Risk değerlendirmesi ve öneriler sun
"""

        # Full prompt
        full_prompt = f"""{system_prompt}

{model_str}

{context_str}

{rag_str}

{memory_str}

Kullanıcı Sorusu: {user_message}

Cevap:"""

        return full_prompt

    # ========================================
    # SPECIAL FUNCTIONS
    # ========================================

    def predict_with_model(
        self, features: Dict, model_id: Optional[str] = None
    ) -> Dict:
        """
        ML modeli ile tahmin yap

        Args:
            features: Özellikler (source_ip, destination_ip, port, vb.)
            model_id: Model ID (None ise en iyi model)

        Returns:
            Dict: Tahmin sonucu
        """

        try:
            # MLPredictionAPI kullan
            from src.api.ml_prediction import MLPredictionAPI

            api = MLPredictionAPI()

            # En iyi model seçimi
            if model_id is None:
                # Model Knowledge'dan en iyi modeli al
                if self.model_knowledge:
                    best_model = self.model_knowledge.get_best_model()
                    if best_model:
                        model_id = best_model["id"]
                        self.logger.info(f"En iyi model seçildi: {best_model['name']}")

            # Tahmin yap
            result = api.predict(features, model_id=model_id)

            if result.get("success"):
                return {
                    "success": True,
                    "model_id": result.get("model_id"),
                    "model_name": result.get("model_name", "Unknown"),
                    "prediction": result.get("predicted_class", "Unknown"),
                    "prediction_label": result.get("prediction_label", "Unknown"),
                    "confidence": result.get("confidence", 0.0),
                    "risk_level": result.get("risk_level", "Unknown"),
                    "explanation": result.get("explanation", ""),
                    "features_used": features,
                }
            else:
                return {"success": False, "error": result.get("error", "Unknown error")}

        except Exception as e:
            self.logger.error(f"Prediction error: {e}")
            return {"success": False, "error": str(e)}

    def analyze_ip_with_model(
        self, ip_address: str, model_id: Optional[str] = None
    ) -> Dict:
        """
        IP adresini ML modeli ile analiz et

        Args:
            ip_address: Analiz edilecek IP
            model_id: Kullanılacak model (None ise en iyi)

        Returns:
            Dict: Analiz sonucu
        """
        # IP için varsayılan özellikler oluştur
        features = {
            "source_ip": ip_address,
            "destination_ip": "10.0.0.1",  # Varsayılan hedef
            "port": 80,
            "severity": "medium",
            "blocked": 0,
        }

        result = self.predict_with_model(features, model_id)

        if result.get("success"):
            return {
                "ip": ip_address,
                "analysis": result,
                "recommendation": self._get_ip_recommendation(result),
            }
        return result

    def _get_ip_recommendation(self, prediction: Dict) -> str:
        """Tahmine göre öneri oluştur"""
        risk = prediction.get("risk_level", "UNKNOWN")
        conf = prediction.get("confidence", 0)

        if risk == "CRITICAL":
            return f"⛔ BU IP HEMEN ENGELLENMELİ! (Güvenilirlik: {conf:.0%})"
        elif risk == "HIGH":
            return f"🔴 Yüksek risk! İzlemeye alınmalı. (Güvenilirlik: {conf:.0%})"
        elif risk == "MEDIUM":
            return f"🟡 Orta risk. Dikkatli olunmalı. (Güvenilirlik: {conf:.0%})"
        else:
            return f"🟢 Düşük risk. Normal trafik olabilir. (Güvenilirlik: {conf:.0%})"

    def get_available_models(self) -> List[Dict]:
        """Kullanılabilir modelleri listele"""
        try:
            from src.api.ml_prediction import MLPredictionAPI

            api = MLPredictionAPI()
            return api.get_available_models()
        except Exception as e:
            self.logger.error(f"Model listesi alınamadı: {e}")
            return []

    # ========================================
    # HELPER FUNCTIONS
    # ========================================

    def _save_to_database(
        self,
        user_message: str,
        bot_response: str,
        intent: str,
        context: Dict,
        response_time: float,
    ):
        """Database'e kaydet"""

        try:
            self.db.add_chat_message(
                {
                    "user_message": user_message,
                    "bot_response": bot_response,
                    "intent": intent,
                    "context_used": context,
                    "response_time": response_time,
                    "user_id": self.user_id,
                    "session_id": self.session_id,
                }
            )
        except Exception as e:
            self.logger.warning(f"Failed to save to database: {e}")

    def _get_error_response(self, error_msg: str) -> str:
        """Hata durumunda cevap"""

        return (
            "❌ Üzgünüm, bir hata oluştu!\n\n"
            "Lütfen:\n"
            "1. Sorunuzu farklı şekilde sormayı deneyin\n"
            "2. Sistem yöneticisiyle iletişime geçin\n\n"
            f"Hata: {error_msg[:100]}"
        )

    # ========================================
    # UTILITY FUNCTIONS
    # ========================================

    def clear_memory(self):
        """Memory'yi temizle"""
        self.memory_manager.clear_short_term()
        self.logger.info("🗑️ Memory cleared")

    def get_stats(self) -> Dict:
        """İstatistikler"""
        return {
            "session_id": self.session_id,
            "user_id": self.user_id,
            "model": self.model_name,
            "memory_stats": self.memory_manager.get_stats(),
            "rag_stats": self.rag_manager.get_stats(),
            "model_knowledge_stats": (
                self.model_knowledge.get_stats() if self.model_knowledge else {}
            ),
        }


# Test
if __name__ == "__main__":
    print("🧪 Enhanced Gemini Handler Test\n")
    print("=" * 60)

    try:
        # Chatbot oluştur
        print("\n🤖 Enhanced Chatbot başlatılıyor...")
        chatbot = EnhancedGeminiHandler()

        print("✅ Chatbot hazır!\n")
        print("=" * 60)

        # Test soruları
        test_questions = [
            "Merhaba! Sistemde kaç saldırı var?",
            "En iyi modelim hangisi?",
            "DDoS saldırısı nedir?",
        ]

        for i, question in enumerate(test_questions, 1):
            print(f"\n{'=' * 60}")
            print(f"Soru {i}: {question}")
            print()

            response = chatbot.chat(question, save_to_db=False)
            print(f"🤖 {response}")

            time.sleep(2)

        print("\n" + "=" * 60)
        print("✅ Test tamamlandı!")

        # Stats
        print("\n📊 İstatistikler:")
        stats = chatbot.get_stats()
        print(json.dumps(stats, indent=2, ensure_ascii=False))

    except ValueError as e:
        print(f"\n❌ Error: {e}")
        print("\n💡 Tip: .env dosyasına GOOGLE_API_KEY ekleyin")

    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback

        traceback.print_exc()
