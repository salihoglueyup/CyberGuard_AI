"""
Memory Manager - CyberGuard AI
Konuşma hafızası ve uzun dönem bellek sistemi
"""

import json
import os
from collections import deque
from datetime import datetime

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings


class MemoryManager:
    """Konuşma hafızası yöneticisi"""

    def __init__(self, user_id: str = "default",
                 persist_directory: str = "src/chatbot/vectorstore/memory_db"):
        """
        Args:
            user_id: Kullanıcı ID
            persist_directory: Hafıza veritabanı yolu
        """
        self.user_id = user_id
        self.persist_directory = os.path.join(persist_directory, user_id)
        os.makedirs(self.persist_directory, exist_ok=True)

        # Embedding modeli
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            model_kwargs={'device': 'cpu'}
        )

        # VectorStore (konuşma geçmişi)
        try:
            self.vectorstore = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embeddings,
                collection_name=f"memory_{user_id}"
            )
            print(f"✅ Memory yüklendi: {self._get_memory_count()} kayıt")
        except Exception:
            self.vectorstore = None
            print("⚠️ Memory oluşturuluyor...")

        # Kısa dönem hafıza (son N mesaj)
        self.short_term_memory = deque(maxlen=10)
        self.max_short_term = 10

    def _get_memory_count(self) -> int:
        """Hafıza kayıt sayısı"""
        try:
            return len(self.vectorstore.get()['ids'])
        except (AttributeError, KeyError, TypeError):
            return 0

    def add_conversation(self, user_message: str, bot_response: str,
                        context: dict | None = None) -> bool:
        """
        Konuşmayı hafızaya ekle

        Args:
            user_message: Kullanıcı mesajı
            bot_response: Bot cevabı
            context: Ek bilgiler

        Returns:
            bool: Başarılı mı?
        """
        try:
            timestamp = datetime.now().isoformat()

            # Metadata
            metadata = {
                'timestamp': timestamp,
                'user_id': self.user_id,
                'type': 'conversation',
                'user_message': user_message,
                'bot_response': bot_response[:500]  # İlk 500 karakter
            }

            if context:
                metadata['context'] = json.dumps(context)

            # Konuşmayı birleştir (arama için)
            conversation_text = f"""
Kullanıcı: {user_message}
Asistan: {bot_response}
Zaman: {timestamp}
"""

            # Document oluştur
            doc = Document(
                page_content=conversation_text,
                metadata=metadata
            )

            # VectorStore'a ekle
            if self.vectorstore is None:
                self.vectorstore = Chroma.from_documents(
                    [doc],
                    self.embeddings,
                    persist_directory=self.persist_directory,
                    collection_name=f"memory_{self.user_id}"
                )
            else:
                self.vectorstore.add_documents([doc])

            # Kısa dönem hafızaya da ekle
            self.short_term_memory.append({
                'user': user_message,
                'bot': bot_response,
                'timestamp': timestamp
            })

            return True

        except Exception as e:
            print(f"❌ Hafızaya eklenemedi: {e}")
            return False

    def search_memory(self, query: str, k: int = 3) -> list[dict]:
        """
        Hafızada ara

        Args:
            query: Arama sorgusu
            k: Kaç sonuç

        Returns:
            List[Dict]: Benzer konuşmalar
        """
        if self.vectorstore is None:
            return []

        try:
            results = self.vectorstore.similarity_search_with_score(query, k=k)

            formatted_results = []
            for doc, score in results:
                formatted_results.append({
                    'content': doc.page_content,
                    'metadata': doc.metadata,
                    'score': float(score),
                    'user_message': doc.metadata.get('user_message', ''),
                    'bot_response': doc.metadata.get('bot_response', ''),
                    'timestamp': doc.metadata.get('timestamp', '')
                })

            return formatted_results

        except Exception as e:
            print(f"❌ Arama hatası: {e}")
            return []

    def get_recent_context(self, n: int = 5) -> str:
        """
        Son N konuşmayı context olarak getir

        Args:
            n: Kaç konuşma

        Returns:
            str: Context metni
        """
        recent = self.short_term_memory[-n:] if len(self.short_term_memory) > 0 else []

        if not recent:
            return ""

        context = "Son Konuşmalar:\n\n"
        for conv in recent:
            context += f"Kullanıcı: {conv['user']}\n"
            context += f"Sen: {conv['bot'][:200]}...\n\n"

        return context

    def get_relevant_memory_for_query(self, query: str, k: int = 2) -> str:
        """
        Sorguyla ilgili geçmiş konuşmaları getir

        Args:
            query: Soru
            k: Kaç konuşma

        Returns:
            str: İlgili konuşmalar
        """
        results = self.search_memory(query, k=k)

        if not results:
            return ""

        context = "İlgili Geçmiş Konuşmalar:\n\n"

        for i, result in enumerate(results, 1):
            context += f"[Konuşma {i}]\n"
            context += f"Kullanıcı: {result['user_message']}\n"
            context += f"Sen: {result['bot_response'][:150]}...\n"
            context += f"Zaman: {result['timestamp']}\n\n"

        return context

    def clear_short_term(self):
        """Kısa dönem hafızayı temizle"""
        self.short_term_memory = []
        print("✅ Kısa dönem hafıza temizlendi")

    def clear_all_memory(self):
        """Tüm hafızayı sil"""
        try:
            if self.vectorstore:
                self.vectorstore.delete_collection()
                self.vectorstore = None

            self.short_term_memory = []
            print("✅ Tüm hafıza silindi")

        except Exception as e:
            print(f"❌ Silme hatası: {e}")

    def get_stats(self) -> dict:
        """Hafıza istatistikleri"""
        try:
            return {
                'user_id': self.user_id,
                'total_conversations': self._get_memory_count(),
                'short_term_count': len(self.short_term_memory),
                'vectorstore_active': self.vectorstore is not None
            }
        except Exception:
            return {
                'user_id': self.user_id,
                'total_conversations': 0,
                'short_term_count': 0,
                'vectorstore_active': False
            }

    def export_memory(self) -> list[dict]:
        """Tüm hafızayı dışa aktar"""
        try:
            if not self.vectorstore:
                return []

            all_data = self.vectorstore.get()

            memories = []
            for i in range(len(all_data['ids'])):
                memories.append({
                    'id': all_data['ids'][i],
                    'content': all_data['documents'][i],
                    'metadata': all_data['metadatas'][i]
                })

            return memories

        except Exception as e:
            print(f"❌ Export hatası: {e}")
            return []


# Test
if __name__ == "__main__":
    print("🧪 Memory Manager Test\n")

    memory = MemoryManager(user_id="test_user")

    # Test konuşmaları ekle
    test_conversations = [
        ("DDoS saldırısı nedir?", "DDoS saldırısı, bir sunucuyu aşırı trafikle hedef alarak çökerten saldırı türüdür."),
        ("Sistemde kaç saldırı var?", "Veritabanında toplam 5000 saldırı kaydı bulunuyor."),
        ("SQL Injection nasıl önlenir?", "SQL Injection'dan korunmak için parameterized queries ve input validation kullanılmalıdır."),
    ]

    for user_msg, bot_msg in test_conversations:
        memory.add_conversation(user_msg, bot_msg)
        print(f"✅ Eklendi: {user_msg[:30]}...")

    # Hafızada ara
    print("\n🔍 Arama: 'saldırı sayısı'")
    results = memory.search_memory("saldırı sayısı", k=2)

    for i, result in enumerate(results, 1):
        print(f"\n[Sonuç {i}] (Skor: {result['score']:.4f})")
        print(f"Kullanıcı: {result['user_message']}")
        print(f"Bot: {result['bot_response'][:100]}...")

    # Son konuşmalar
    print("\n📜 Son Konuşmalar:")
    recent_context = memory.get_recent_context(n=3)
    print(recent_context[:300] + "...")

    # İstatistikler
    print("\n📊 İstatistikler:")
    stats = memory.get_stats()
    print(stats)
