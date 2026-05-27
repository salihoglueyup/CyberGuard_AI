"""
Chat API Routes - CyberGuard AI
AI Chatbot endpoint - Multi-provider support (Groq, Gemini)

Dosya Yolu: app/api/routes/chat.py
"""

import os

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from app.api.routes.auth import require_auth
from app.paths import PROJECT_ROOT

project_root = str(PROJECT_ROOT)

router = APIRouter()

# Handler cache
_handlers = {}


def get_handler(provider: str = None):
    """Provider'a göre handler döndür"""
    global _handlers

    # Varsayılan provider
    if provider is None:
        provider = os.getenv("DEFAULT_AI_PROVIDER", "groq")

    provider = provider.lower()

    # Cache'den kontrol et
    if provider in _handlers:
        return _handlers[provider], provider

    try:
        if provider == "groq":
            from src.chatbot.groq_handler import GroqHandler

            _handlers[provider] = GroqHandler()
        elif provider == "gemini":
            from src.chatbot.gemini_handler import EnhancedGeminiHandler

            _handlers[provider] = EnhancedGeminiHandler(user_id="api_user")
        else:
            # Bilinmeyen provider, Groq'a düş
            from src.chatbot.groq_handler import GroqHandler

            _handlers["groq"] = GroqHandler()
            return _handlers["groq"], "groq"

        return _handlers[provider], provider
    except Exception as e:
        print(f"[ERROR] Handler init error ({provider}): {e}")
        return None, provider


class ChatRequest(BaseModel):
    message: str = Field(max_length=10000)
    model_id: str | None = None
    provider: str | None = "groq"  # "groq" veya "gemini"


class ChatResponse(BaseModel):
    success: bool
    response: str | None = None
    error: str | None = None
    provider: str | None = None


@router.post("/", response_model=ChatResponse)
async def chat(request: ChatRequest, user: dict = Depends(require_auth)):
    """AI Chatbot ile sohbet - Multi-provider destekli"""
    try:
        handler, provider = get_handler(request.provider)

        if handler is None:
            return ChatResponse(
                success=False,
                error=f"{provider} başlatılamadı. API key kontrol edin.",
                provider=provider,
            )

        # Provider'a göre çağrı yap
        if provider == "groq":
            # Groq için context oluştur
            print(f"[DEBUG] Building context for model_id: {request.model_id}")
            context = _build_groq_context(request.model_id)
            print(f"[DEBUG] Context length: {len(context)} chars")
            print(
                f"[DEBUG] Context preview: {context[:500]}..."
                if context
                else "[WARN] No context"
            )
            response = handler.chat(request.message, context=context)
        else:
            # Gemini handler
            response = handler.chat(
                request.message, save_to_db=True, model_id=request.model_id
            )

        return ChatResponse(success=True, response=response, provider=provider)

    except Exception as e:
        return ChatResponse(success=False, error=str(e), provider=request.provider)


def _build_groq_context(model_id: str = None) -> str:
    """Groq için veritabanı context'i oluştur"""
    try:
        from src.utils.database import DatabaseManager

        db_path = os.path.join(project_root, "src", "database", "cyberguard.db")
        db = DatabaseManager(db_path)

        context_parts = []

        # TÜM saldırı istatistikleri (hours=None = tüm veritabanı)
        stats = db.get_attack_stats(hours=None)
        stats_24h = db.get_attack_stats(hours=24)

        if stats:
            total = stats.get("total", 0)
            blocked = stats.get("blocked", 0)
            block_rate = (blocked / total * 100) if total > 0 else 0

            context_parts.append(
                f"""📊 VERİTABANI SALDIRI İSTATİSTİKLERİ (TÜM VERİLER):

🔢 GENEL ÖZET:
- Toplam Saldırı: {total:,}
- Engellenen: {blocked:,}
- Engelleme Oranı: {block_rate:.1f}%
- Tespit Edilen (Engellenemeyen): {total - blocked:,}

📈 SON 24 SAAT:
- Toplam: {stats_24h.get('total', 0):,}
- Engellenen: {stats_24h.get('blocked', 0):,}

🎯 SEVERİTY DAĞILIMI (Tüm Zamanlar):
{chr(10).join([f'  - {sev}: {cnt:,} saldırı' for sev, cnt in stats.get('by_severity', {}).items()])}

🔥 SALDIRI TİPİ DAĞILIMI (Tüm Zamanlar):
{chr(10).join([f'  - {typ}: {cnt:,} saldırı' for typ, cnt in stats.get('by_type', {}).items()])}"""
            )

        # Seçili model bilgisi - model_registry.json'dan al
        if model_id:
            try:
                import json

                registry_path = os.path.join(
                    project_root, "model_artifacts", "model_registry.json"
                )
                if os.path.exists(registry_path):
                    with open(registry_path, encoding="utf-8") as f:
                        registry = json.load(f)

                    # Model bul
                    for m in registry.get("models", []):
                        mid = m.get("id", m.get("model_id", ""))
                        if mid == model_id:
                            metrics = m.get("metrics", {})
                            training = m.get("training_config", {})

                            # Güvenli format fonksiyonu
                            def safe_format(val, fmt=None):
                                if val is None or val == "N/A":
                                    return "N/A"
                                if fmt == "comma" and isinstance(val, (int, float)):
                                    return f"{int(val):,}"
                                return str(val)

                            # Temel bilgiler
                            context_parts.append(
                                f"""🤖 SEÇİLİ ML MODEL - DETAYLI BİLGİ:

📌 TEMEL BİLGİLER:
- Model ID: {model_id}
- Model Adı: {m.get('name', m.get('model_name', 'N/A'))}
- Description: {m.get('description', 'N/A')}
- Framework: {m.get('framework', 'N/A')}
- Model Type: {m.get('model_type', 'N/A')}
- Status: {m.get('status', 'N/A')}
- Created: {m.get('created_at', 'N/A')}

📊 PERFORMANS METRİKLERİ:
- Accuracy: {metrics.get('accuracy', 0) * 100:.2f}%
- Precision: {metrics.get('precision', 0) * 100:.2f}%
- Recall: {metrics.get('recall', 0) * 100:.2f}%
- F1-Score: {metrics.get('f1_score', 0) * 100:.2f}%
- Loss: {metrics.get('loss', 'N/A')}
- Val Accuracy: {metrics.get('val_accuracy', 0) * 100:.2f}% (varsa)

🏋️ EĞİTİM KONFIGÜRASYONU:
- Train Samples: {safe_format(training.get('train_samples'), 'comma')}
- Test Samples: {safe_format(training.get('test_samples'), 'comma')}
- Epochs: {training.get('epochs', 'N/A')}
- Batch Size: {training.get('batch_size', 'N/A')}
- Learning Rate: {training.get('learning_rate', 'N/A')}
- Hidden Layers: {training.get('hidden_layers', 'N/A')}
- Dropout Rate: {training.get('dropout_rate', 'N/A')}
- Validation Split: {training.get('validation_split', 'N/A')}"""
                            )

                            # Model klasöründen ek bilgi al
                            model_dir = os.path.join(project_root, "model_artifacts", model_id)
                            if os.path.exists(model_dir):
                                # Metadata.json
                                meta_path = os.path.join(model_dir, "metadata.json")
                                if os.path.exists(meta_path):
                                    with open(meta_path, encoding="utf-8") as mf:
                                        meta = json.load(mf)
                                    context_parts.append(
                                        f"""📁 MODEL DOSYA METADATA:
- Type: {meta.get('type', 'N/A')}
- Framework Detail: {meta.get('framework', 'N/A')}
- Created At: {meta.get('created_at', 'N/A')}
- Status: {meta.get('status', 'N/A')}"""
                                    )

                            break
            except Exception as e:
                print(f"Model info error: {e}")

        # Son saldırılar
        recent = db.get_recent_attacks(limit=10)
        if recent:
            context_parts.append(
                """📋 SON 10 SALDIRI:
| Tip | Severity | Kaynak IP | Durum |
|-----|----------|-----------|-------|"""
            )
            for attack in recent[:10]:
                context_parts.append(
                    f"| {attack.get('attack_type', 'N/A')} | {attack.get('severity', 'N/A')} | {attack.get('source_ip', 'N/A')} | {attack.get('status', 'N/A')} |"
                )

        # Top 5 Saldırgan IP
        try:
            top_ips = db.get_top_attacker_ips(limit=5)
            if top_ips:
                context_parts.append(
                    """🎯 TOP 5 SALDIRGAN IP:
| IP Adresi | Saldırı Sayısı |
|-----------|----------------|"""
                )
                for ip_data in top_ips:
                    context_parts.append(
                        f"| {ip_data.get('source_ip', 'N/A')} | {ip_data.get('count', 0):,} |"
                    )
        except Exception as e:
            print(f"Top IPs error: {e}")

        # IDS Model Bilgileri
        try:
            from src.chatbot.model_integration import get_integration

            integration = get_integration()
            available_models = integration.get_available_models()

            if available_models:
                context_parts.append(
                    f"""

🤖 MEVCUT IDS MODELLERİ ({len(available_models)} model):
| Model Adı | Tip | Boyut | Accuracy |
|-----------|-----|-------|----------|"""
                )

                for model in available_models[:5]:  # İlk 5 model
                    params = model.get("params", {})
                    accuracy = params.get("accuracy", 0)
                    if isinstance(accuracy, float) and accuracy < 1:
                        accuracy *= 100

                    context_parts.append(
                        f"| {model['name'][:30]} | {model['type']} | {model['size_mb']} MB | {accuracy:.1f}% |"
                    )

                # Eğitim sonuçları
                if integration.training_results:
                    context_parts.append("\n📊 SON EĞİTİM SONUÇLARI:")
                    for dataset, results in integration.training_results.items():
                        context_parts.append(
                            f"  - {dataset}: Accuracy={results.get('accuracy', 0)*100:.2f}%, "
                            f"F1={results.get('f1_score', 0)*100:.2f}%"
                        )

                context_parts.append(
                    """
🔧 KULLANIM İPUÇLARI:
- "Hangi modelimiz en iyi?" diye sorabilirsin
- "Son 24 saatteki saldırıları analiz et" diyebilirsin
- "Bu IP'yi kontrol et: 192.168.1.1" diyebilirsin
- "Model performanslarını karşılaştır" diyebilirsin"""
                )
        except Exception as e:
            print(f"Model integration error: {e}")

        return "\n".join(context_parts)

    except Exception as e:
        print(f"Context build error: {e}")
        return ""


@router.get("/providers")
async def list_providers(user: dict = Depends(require_auth)):
    """Mevcut AI sağlayıcılarını listele"""
    providers = [
        {
            "id": "groq",
            "name": "🦙 Groq (Llama 3.3)",
            "description": "Hızlı ve ücretsiz - Llama 3.3 70B",
            "model": "llama-3.3-70b-versatile",
            "available": bool(os.getenv("GROQ_API_KEY")),
        },
        {
            "id": "gemini",
            "name": "🔮 Google Gemini",
            "description": "Google'ın AI modeli",
            "model": "gemini-2.0-flash-exp",
            "available": bool(os.getenv("GOOGLE_API_KEY")),
        },
    ]

    default = os.getenv("DEFAULT_AI_PROVIDER", "groq")

    return {
        "providers": providers,
        "default": default,
    }


@router.get("/history")
async def get_chat_history(limit: int = 20, user: dict = Depends(require_auth)):
    """Sohbet geçmişi"""
    limit = max(1, min(limit, 200))
    try:
        from src.utils.database import DatabaseManager

        db_path = os.path.join(project_root, "src", "database", "cyberguard.db")
        db = DatabaseManager(db_path)

        # Chat history tablosu varsa
        history = db.execute_query(
            """
            SELECT * FROM chat_history 
            ORDER BY timestamp DESC 
            LIMIT ?
        """,
            (limit,),
        )

        return {"success": True, "data": history}

    except Exception:
        return {"success": True, "data": [], "note": "Chat history tablosu bulunamadı"}


@router.post("/clear")
async def clear_chat(user: dict = Depends(require_auth)):
    """Sohbet geçmişini temizle"""
    try:
        handler, provider = get_handler()
        if handler and hasattr(handler, 'clear_memory'):
            handler.clear_memory()
        return {"success": True, "message": "Chat memory temizlendi"}
    except Exception as e:
        return {"success": False, "error": str(e)}


@router.get("/stats")
async def get_chat_stats(user: dict = Depends(require_auth)):
    """Chatbot istatistikleri"""
    try:
        handler, provider = get_handler()
        if handler and hasattr(handler, 'get_stats'):
            stats = handler.get_stats()
            return {"success": True, "data": stats}
        return {"success": True, "data": {"provider": provider, "status": "active"}}
    except Exception as e:
        return {"success": False, "error": str(e)}


# ========================================
# STREAMING ENDPOINT
# ========================================
import asyncio
import json

from fastapi.responses import StreamingResponse


class StreamChatRequest(BaseModel):
    message: str
    model_id: str | None = None


async def generate_stream(message: str, model_id: str | None = None, user: dict = Depends(require_auth)):
    """Streaming yanıt oluştur"""
    try:
        handler, provider = get_handler()

        if handler is None:
            yield f"data: {json.dumps({'error': 'Chatbot başlatılamadı'})}\n\n"
            return

        # Gemini ile streaming yanıt al
        try:

            # Intent ve context oluştur (normal chat gibi)
            intent, confidence = handler.intent_classifier.classify(message)
            entities = handler.intent_classifier.extract_entities(message)
            context = handler._build_context(message, intent, entities)

            # Model bilgisini ekle
            if model_id:
                context["active_model_id"] = model_id
                if hasattr(handler, '_get_model_info'):
                    model_info = handler._get_model_info(model_id)
                    if model_info:
                        context["active_model"] = model_info
                        if hasattr(handler, '_get_attacks_for_model'):
                            attack_data = handler._get_attacks_for_model(model_id)
                            if attack_data:
                                context["model_attacks"] = attack_data

            # RAG context
            rag_context = handler._get_rag_context(message, intent) if hasattr(handler, '_get_rag_context') else ""

            # Memory context
            memory_context = ""
            if hasattr(handler, 'memory_manager'):
                memory_context = handler.memory_manager.get_relevant_memory_for_query(
                    message, k=1
                )

            # Prompt oluştur
            full_prompt = handler._build_enhanced_prompt(
                message, intent, context, rag_context, memory_context
            ) if hasattr(handler, '_build_enhanced_prompt') else message

            # Streaming response
            model_obj = getattr(handler, 'model', None)
            if model_obj is None:
                yield f"data: {json.dumps({'error': 'Model not available for streaming'})}\n\n"
                return

            response = model_obj.generate_content(full_prompt, stream=True)

            full_response = ""
            for chunk in response:
                if chunk.text:
                    full_response += chunk.text
                    yield f"data: {json.dumps({'text': chunk.text, 'done': False})}\n\n"
                    await asyncio.sleep(0.01)

            # Memory'e kaydet
            if hasattr(handler, 'memory_manager'):
                handler.memory_manager.add_conversation(message, full_response, context)

            yield f"data: {json.dumps({'text': '', 'done': True})}\n\n"

        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    except Exception as e:
        yield f"data: {json.dumps({'error': str(e)})}\n\n"


@router.post("/stream")
async def chat_stream(request: StreamChatRequest, user: dict = Depends(require_auth)):
    """Streaming AI Chat - Real-time yanıt"""
    return StreamingResponse(
        generate_stream(request.message, request.model_id),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
