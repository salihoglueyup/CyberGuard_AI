# app/pages/ai_assistant.py

"""
AI Assistant - CyberGuard AI
RAG + Memory + Attack Vector + Gemini Pro entegreli akıllı asistan
"""

import streamlit as st
import sys
import os
from datetime import datetime
import time
import re

# Path ayarları
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import importlib.util

# Import managers
try:
    import importlib.util

    # RAG Manager
    rag_spec = importlib.util.spec_from_file_location(
        "rag_manager",
        os.path.join(project_root, 'src', 'chatbot', 'vectorstore', 'rag_manager.py')
    )
    rag_module = importlib.util.module_from_spec(rag_spec)
    rag_spec.loader.exec_module(rag_module)
    RAGManager = rag_module.RAGManager

    # Memory Manager
    mem_spec = importlib.util.spec_from_file_location(
        "memory_manager",
        os.path.join(project_root, 'src', 'chatbot', 'vectorstore', 'memory_manager.py')
    )
    mem_module = importlib.util.module_from_spec(mem_spec)
    mem_spec.loader.exec_module(mem_module)
    MemoryManager = mem_module.MemoryManager

    # Attack Vectors
    attack_spec = importlib.util.spec_from_file_location(
        "attack_vectors",
        os.path.join(project_root, 'src', 'chatbot', 'vectorstore', 'attack_vectors.py')
    )
    attack_module = importlib.util.module_from_spec(attack_spec)
    attack_spec.loader.exec_module(attack_module)
    AttackVectorManager = attack_module.AttackVectorManager

    # Gemini Handler
    gemini_spec = importlib.util.spec_from_file_location(
        "gemini_handler",
        os.path.join(project_root, 'src', 'chatbot', 'gemini_handler.py')
    )
    gemini_module = importlib.util.module_from_spec(gemini_spec)
    gemini_spec.loader.exec_module(gemini_module)
    GeminiChatbot = gemini_module.GeminiHandler  # Sınıf adı GeminiHandler

    RAG_AVAILABLE = True
except Exception as e:
    RAG_AVAILABLE = False
    print(f"⚠️ RAG modülleri yüklenemedi: {e}")

# Custom CSS
st.markdown("""
<style>
    .chat-message {
        padding: 1.5rem;
        border-radius: 15px;
        margin-bottom: 1rem;
        display: flex;
        flex-direction: column;
    }

    .user-message {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        margin-left: 20%;
    }

    .assistant-message {
        background-color: #2d2d2d;
        color: white;
        margin-right: 20%;
        border-left: 4px solid #667eea;
    }

    .message-header {
        font-weight: bold;
        margin-bottom: 0.5rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }

    .message-content {
        line-height: 1.6;
    }

    .quick-question {
        background-color: #3d3d3d;
        padding: 0.8rem 1.2rem;
        border-radius: 20px;
        border: 2px solid #667eea;
        cursor: pointer;
        transition: all 0.3s;
    }

    .stats-box {
        background-color: #2d2d2d;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_managers():
    """Manager'ları yükle ve cache'le"""
    try:
        rag = RAGManager()
        memory = MemoryManager(user_id="streamlit_user")
        attack_vectors = AttackVectorManager()
        gemini = GeminiChatbot()

        return rag, memory, attack_vectors, gemini, True
    except Exception as e:
        print(f"❌ Manager yükleme hatası: {e}")
        return None, None, None, None, False


def show_ai_assistant_page():
    """AI Assistant sayfasını göster"""

    # Initialize
    if RAG_AVAILABLE:
        rag, memory, attack_vectors, gemini, is_initialized = load_managers()
    else:
        is_initialized = False

    # Session state
    if 'messages' not in st.session_state:
        st.session_state.messages = []

    if 'conversation_count' not in st.session_state:
        st.session_state.conversation_count = 0

    if 'rag_enabled' not in st.session_state:
        st.session_state.rag_enabled = True

    if 'memory_enabled' not in st.session_state:
        st.session_state.memory_enabled = True

    if 'attack_vector_enabled' not in st.session_state:
        st.session_state.attack_vector_enabled = True

    # Header
    st.title("🤖 AI Güvenlik Asistanı")
    st.markdown("**RAG + Memory + Attack Vector + Gemini Pro ile Akıllı Siber Güvenlik Danışmanı**")
    st.markdown("---")

    # Sidebar
    with st.sidebar:
        st.markdown("## 📊 İstatistikler")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown(f"""
            <div class="stats-box">
                <h3>{st.session_state.conversation_count}</h3>
                <p>Toplam Soru</p>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown(f"""
            <div class="stats-box">
                <h3>{len(st.session_state.messages) // 2}</h3>
                <p>Konuşma</p>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")

        # RAG Ayarları
        if is_initialized and RAG_AVAILABLE:
            st.markdown("## ⚙️ Akıllı Özellikler")

            st.session_state.rag_enabled = st.checkbox(
                "📚 RAG (Döküman Arama)",
                value=st.session_state.rag_enabled,
                help="Yüklenen dökümanlardan bilgi ara"
            )

            st.session_state.memory_enabled = st.checkbox(
                "🧠 Hafıza (Konuşma Geçmişi)",
                value=st.session_state.memory_enabled,
                help="Önceki konuşmaları hatırla"
            )

            st.session_state.attack_vector_enabled = st.checkbox(
                "🎯 Attack Vectors",
                value=st.session_state.attack_vector_enabled,
                help="Database'deki saldırıları analiz et"
            )

            st.markdown("---")

            # Vektör İstatistikleri
            st.markdown("## 📈 Vektör İstatistikleri")

            rag_stats = rag.get_stats()
            st.metric("📚 Dökümanlar", rag_stats['total_documents'])

            memory_stats = memory.get_stats()
            st.metric("🧠 Konuşmalar", memory_stats['total_conversations'])

            attack_stats = attack_vectors.get_stats()
            st.metric("🎯 Saldırı Vektörleri", attack_stats['total_vectors'])

            st.markdown("---")

        # Yönetim
        st.markdown("## 🛠️ Yönetim")

        if st.button("🗑️ Sohbeti Temizle", use_container_width=True):
            st.session_state.messages = []
            st.session_state.conversation_count = 0
            if is_initialized and gemini:
                gemini.clear_history()
            if is_initialized and memory:
                memory.clear_short_term()
            st.success("✅ Sohbet temizlendi!")
            time.sleep(1)
            st.rerun()

        if is_initialized and gemini:
            if st.button("💾 Sohbeti Kaydet", use_container_width=True):
                try:
                    filename = gemini.export_conversation()
                    st.success(f"✅ Kaydedildi: {filename}")
                except Exception as e:
                    st.error(f"❌ Hata: {e}")

        st.markdown("---")

        st.markdown("## 🎯 Yetenekler")
        st.markdown("""
        - 🔍 Saldırı analizi
        - 📊 İstatistik raporları
        - 🛡️ IP kontrolü
        - 📚 Döküman arama (RAG)
        - 🧠 Akıllı hafıza
        - 🎯 Benzer saldırı tespiti
        - 💡 Güvenlik önerileri
        """)

    # Initialization check
    if not is_initialized:
        st.error("❌ AI Asistan başlatılamadı!")
        st.markdown("""
        **Muhtemel Nedenler:**
        - GOOGLE_API_KEY bulunamadı
        - google-generativeai paketi kurulu değil
        - RAG kütüphaneleri eksik

        **Çözüm:**
        ```bash
        pip install google-generativeai
        pip install langchain-core langchain-huggingface langchain-chroma
        pip install chromadb sentence-transformers
        ```
        """)
        return

    # Tabs
    tab1, tab2, tab3 = st.tabs([
        "💬 Sohbet",
        "📚 Döküman Yönetimi",
        "🎯 Saldırı Vektörleri"
    ])

    # ============================================================
    # TAB 1: Sohbet
    # ============================================================
    with tab1:
        # Hızlı Sorular
        st.markdown("### 💡 Hızlı Sorular")

        quick_questions = [
            "Merhaba! Kendini tanıt",
            "Son 24 saatte kaç saldırı oldu?",
            "DDoS saldırısı nedir?",
            "Sistem durumu nedir?",
            "En tehlikeli 5 IP'yi listele",
            "Siber güvenlik için 5 öneri ver",
        ]

        cols = st.columns(3)
        for i, question in enumerate(quick_questions):
            with cols[i % 3]:
                if st.button(question, key=f"quick_{i}", use_container_width=True):
                    st.session_state.messages.append({
                        "role": "user",
                        "content": question,
                        "timestamp": datetime.now()
                    })
                    st.rerun()

        st.markdown("---")

        # Chat geçmişi
        chat_container = st.container()

        with chat_container:
            for message in st.session_state.messages:
                role = message["role"]
                content = message["content"]
                timestamp = message.get("timestamp", datetime.now())

                if role == "user":
                    st.markdown(f"""
                    <div class="chat-message user-message">
                        <div class="message-header">
                            👤 Siz - {timestamp.strftime('%H:%M:%S')}
                        </div>
                        <div class="message-content">
                            {content}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="chat-message assistant-message">
                        <div class="message-header">
                            🤖 AI Asistan - {timestamp.strftime('%H:%M:%S')}
                        </div>
                        <div class="message-content">
                            {content.replace('\n', '<br>')}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                    # Metadata göster
                    if 'metadata' in message and message['metadata']:
                        with st.expander("🔍 Kaynak Bilgileri"):
                            st.json(message['metadata'])

        # Tips (boş chat için)
        if len(st.session_state.messages) == 0:
            st.info("""
            💡 **İpuçları:**
            - "Son saldırıları göster" diyerek güncel tehditleri görebilirsiniz
            - "192.168.1.100 IP'sini analiz et" diyerek IP kontrolü yapabilirsiniz
            - "DDoS nedir?" gibi siber güvenlik sorularını sorabilirsiniz
            - Döküman yükleyip onlardan soru sorabilirsiniz
            - Yukarıdaki hızlı sorular butonlarını kullanabilirsiniz
            """)

        st.markdown("---")

        # Chat input
        user_input = st.chat_input("Mesajınızı yazın... (Örn: 'Son saldırıları göster')")

        if user_input:
            # Kullanıcı mesajını ekle
            st.session_state.messages.append({
                "role": "user",
                "content": user_input,
                "timestamp": datetime.now()
            })

            st.session_state.conversation_count += 1

            # Asistan cevabı
            with st.spinner("🤔 Düşünüyorum..."):
                try:
                    # Context toplama
                    context_parts = []
                    metadata = {}
                    base_context = None

                    # Gemini context (system context)
                    if any(word in user_input.lower() for word in ['saldırı', 'attack', 'kaç']):
                        base_context = gemini.get_attack_context(hours=24)
                        metadata['gemini_attack_context'] = True
                    elif any(word in user_input.lower() for word in ['ip', 'adres']):
                        ip_match = re.search(r'\b(?:\d{1,3}\.){3}\d{1,3}\b', user_input)
                        if ip_match:
                            base_context = gemini.get_ip_context(ip_match.group(0))
                            metadata['gemini_ip_context'] = ip_match.group(0)
                    elif any(word in user_input.lower() for word in ['durum', 'status', 'sistem']):
                        base_context = gemini.get_system_context()
                        metadata['gemini_system_context'] = True

                    # RAG - Dökümanlardan ara
                    if st.session_state.rag_enabled and RAG_AVAILABLE:
                        rag_context = rag.get_context_for_query(user_input, k=2)
                        if rag_context:
                            context_parts.append("📚 DÖKÜMANLARDAN:\n" + rag_context)
                            metadata['rag_used'] = True

                    # Attack Vectors - Saldırı analizi
                    if st.session_state.attack_vector_enabled and RAG_AVAILABLE:
                        attack_context = attack_vectors.get_attack_summary_for_chatbot(user_input)
                        if attack_context:
                            context_parts.append("🎯 SALDIRI VERİTABANI:\n" + attack_context)
                            metadata['attack_vectors_used'] = True

                    # Memory - Geçmiş konuşmalar
                    if st.session_state.memory_enabled and RAG_AVAILABLE:
                        memory_context = memory.get_relevant_memory_for_query(user_input, k=2)
                        if memory_context:
                            context_parts.append("🧠 GEÇMIŞ KONUŞMALAR:\n" + memory_context)
                            metadata['memory_used'] = True

                    # Context birleştir
                    enhanced_context = "\n\n".join(context_parts) if context_parts else None

                    # Final context (Gemini base + RAG/Memory/Attack)
                    if enhanced_context and base_context:
                        final_context = {**base_context, 'enhanced_info': enhanced_context}
                    elif enhanced_context:
                        final_context = {'enhanced_info': enhanced_context}
                    else:
                        final_context = base_context

                    # Gemini'ye gönder
                    response = gemini.chat(
                        user_input,
                        context=final_context,
                        save_to_db=True
                    )

                    # Cevabı ekle
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": response,
                        "timestamp": datetime.now(),
                        "metadata": metadata
                    })

                    # Memory'ye kaydet
                    if st.session_state.memory_enabled and RAG_AVAILABLE:
                        memory.add_conversation(user_input, response, context=metadata)

                except Exception as e:
                    error_msg = f"❌ Üzgünüm, bir hata oluştu: {str(e)}"
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": error_msg,
                        "timestamp": datetime.now()
                    })

            st.rerun()

    # ============================================================
    # TAB 2: Döküman Yönetimi
    # ============================================================
    with tab2:
        if not RAG_AVAILABLE:
            st.warning("⚠️ RAG özellikleri kullanılamıyor!")
            return

        st.subheader("📚 Döküman Yükleme & Yönetimi")

        col1, col2 = st.columns([2, 1])

        with col1:
            st.markdown("#### 📄 Metin Döküman Ekle")

            doc_title = st.text_input("Döküman Başlığı", placeholder="örn: DDoS Saldırısı Rehberi")
            doc_text = st.text_area(
                "Döküman İçeriği",
                placeholder="Döküman metnini buraya yapıştırın...",
                height=300
            )

            if st.button("➕ Döküman Ekle", type="primary"):
                if doc_text and doc_title:
                    with st.spinner("📝 Döküman işleniyor..."):
                        success = rag.add_text_document(
                            doc_text,
                            metadata={
                                'title': doc_title,
                                'added_by': 'user',
                                'timestamp': datetime.now().isoformat()
                            }
                        )

                        if success:
                            st.success(f"✅ '{doc_title}' eklendi!")
                            st.cache_resource.clear()
                            st.rerun()
                        else:
                            st.error("❌ Döküman eklenemedi!")
                else:
                    st.warning("⚠️ Başlık ve içerik gerekli!")

            st.markdown("---")

            st.markdown("#### 📁 PDF Yükle")

            uploaded_pdf = st.file_uploader("PDF Dosya Seçin", type=['pdf'])

            if uploaded_pdf:
                if st.button("📤 PDF'i İşle ve Ekle"):
                    with st.spinner("📖 PDF okunuyor..."):
                        temp_path = f"temp_{uploaded_pdf.name}"
                        with open(temp_path, "wb") as f:
                            f.write(uploaded_pdf.getbuffer())

                        success = rag.add_pdf_document(
                            temp_path,
                            metadata={
                                'title': uploaded_pdf.name,
                                'added_by': 'user',
                                'timestamp': datetime.now().isoformat()
                            }
                        )

                        os.remove(temp_path)

                        if success:
                            st.success(f"✅ '{uploaded_pdf.name}' eklendi!")
                            st.cache_resource.clear()
                            st.rerun()
                        else:
                            st.error("❌ PDF eklenemedi!")

        with col2:
            st.markdown("#### 📊 Döküman İstatistikleri")

            stats = rag.get_stats()
            st.metric("Toplam Döküman", stats['total_documents'])
            st.metric("VectorStore", "Aktif" if stats['vectorstore_active'] else "Pasif")

            st.markdown("---")

            st.markdown("#### 🧪 Test")

            if st.button("➕ Test Dökümanı", use_container_width=True):
                test_doc = """
                SQL Injection Saldırısı

                SQL Injection, web uygulamalarındaki en yaygın güvenlik açıklarından biridir.
                Saldırganlar, SQL sorgularına kötü amaçlı kod enjekte ederek veritabanına 
                yetkisiz erişim sağlayabilir.

                Korunma Yöntemleri:
                1. Parametreli sorgular kullanın
                2. Input validation yapın
                3. En az yetki prensibi uygulayın
                4. WAF kullanın
                """

                success = rag.add_text_document(
                    test_doc,
                    metadata={'title': 'SQL Injection Rehberi', 'type': 'test'}
                )

                if success:
                    st.success("✅ Test dökümanı eklendi!")
                    st.cache_resource.clear()
                    st.rerun()

    # ============================================================
    # TAB 3: Saldırı Vektörleri
    # ============================================================
    with tab3:
        if not RAG_AVAILABLE:
            st.warning("⚠️ Attack Vector özellikleri kullanılamıyor!")
            return

        st.subheader("🎯 Saldırı Vektörü Yönetimi")

        col1, col2 = st.columns([2, 1])

        with col1:
            st.markdown("#### 🔄 Database Vektörleştirme")

            vector_limit = st.number_input(
                "Vektörleştirilecek Saldırı Sayısı",
                min_value=10,
                max_value=5000,
                value=1000,
                step=100
            )

            if st.button("🚀 Vektörleştir", type="primary"):
                with st.spinner(f"🔄 {vector_limit} saldırı vektörleştiriliyor..."):
                    success = attack_vectors.vectorize_attacks(limit=vector_limit)

                    if success:
                        st.success(f"✅ {vector_limit} saldırı vektörleştirildi!")
                        st.cache_resource.clear()
                        st.rerun()
                    else:
                        st.error("❌ Vektörleştirme başarısız!")

            st.markdown("---")

            st.markdown("#### 🔍 Benzer Saldırı Ara")

            search_query = st.text_input(
                "Saldırı türü veya açıklama",
                placeholder="örn: DDoS, Port Scan"
            )

            if search_query and st.button("🔍 Ara"):
                with st.spinner("🔍 Aranıyor..."):
                    results = attack_vectors.find_similar_attacks(search_query, k=5)

                    if results:
                        st.success(f"✅ {len(results)} benzer saldırı bulundu!")

                        for i, result in enumerate(results, 1):
                            with st.expander(
                                    f"🎯 #{i} - {result['attack_type']} (Benzerlik: {result['similarity_score']:.2f})"):
                                st.write(f"**IP:** {result['source_ip']}")
                                st.write(f"**Severity:** {result['severity']}")
                                st.write(f"**Zaman:** {result['timestamp']}")
                    else:
                        st.warning("⚠️ Sonuç bulunamadı!")

        with col2:
            st.markdown("#### 📊 Vektör İstatistikleri")

            stats = attack_vectors.get_stats()
            st.metric("Vektörleştirilmiş", stats['total_vectors'])
            st.metric("VectorStore", "Aktif" if stats['vectorstore_active'] else "Pasif")

    # Footer
    st.markdown("---")
    st.markdown(f"""
    <p style='text-align: center; color: gray;'>
        🤖 Gemini 2.5 Flash + RAG + Memory + Attack Vectors | 
        Son güncelleme: {datetime.now().strftime('%H:%M:%S')}
    </p>
    """, unsafe_allow_html=True)


# Test
if __name__ == "__main__":
    st.set_page_config(
        page_title="AI Assistant - CyberGuard AI",
        page_icon="🤖",
        layout="wide"
    )

    show_ai_assistant_page()