"""
RAG Manager - CyberGuard AI
Retrieval Augmented Generation sistemi
Döküman yönetimi ve akıllı sorgulama
"""

import os
from typing import List, Dict
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
import hashlib


class RAGManager:
    """RAG sistemi yöneticisi"""

    def __init__(self, persist_directory: str = "src/chatbot/vectorstore/documents_db"):
        """
        Args:
            persist_directory: Veritabanı kayıt yolu
        """
        self.persist_directory = persist_directory
        os.makedirs(persist_directory, exist_ok=True)

        # Embedding modeli (Türkçe destekli)
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            model_kwargs={'device': 'cpu'}
        )

        # ChromaDB yükle veya oluştur
        try:
            self.vectorstore = Chroma(
                persist_directory=persist_directory,
                embedding_function=self.embeddings
            )
            print(f"✅ VectorStore yüklendi: {self._get_doc_count()} döküman")
        except:
            self.vectorstore = None
            print("⚠️ VectorStore oluşturuluyor...")

    def _get_doc_count(self) -> int:
        """Döküman sayısı"""
        try:
            return len(self.vectorstore.get()['ids'])
        except:
            return 0

    def add_text_document(self, text: str, metadata: Dict = None) -> bool:
        """
        Metin döküman ekle

        Args:
            text: Döküman metni
            metadata: Metadata (title, source, vb.)

        Returns:
            bool: Başarılı mı?
        """
        try:
            # Text splitter (chunk'lara böl)
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len
            )

            # Chunk'lara böl
            chunks = text_splitter.split_text(text)

            # Metadata ekle
            if metadata is None:
                metadata = {}

            # Döküman ID oluştur
            doc_id = hashlib.md5(text.encode()).hexdigest()
            metadata['doc_id'] = doc_id

            # Document objelerine çevir
            documents = [
                Document(page_content=chunk, metadata=metadata)
                for chunk in chunks
            ]

            # VectorStore'a ekle
            if self.vectorstore is None:
                self.vectorstore = Chroma.from_documents(
                    documents,
                    self.embeddings,
                    persist_directory=self.persist_directory
                )
            else:
                self.vectorstore.add_documents(documents)

            # persist() artık gerekmiyor, otomatik kaydediyor

            print(f"✅ Döküman eklendi: {len(chunks)} chunk")
            return True

        except Exception as e:
            print(f"❌ Döküman eklenemedi: {e}")
            return False

    def add_pdf_document(self, pdf_path: str, metadata: Dict = None) -> bool:
        """
        PDF döküman ekle

        Args:
            pdf_path: PDF dosya yolu
            metadata: Metadata

        Returns:
            bool: Başarılı mı?
        """
        try:
            from PyPDF2 import PdfReader

            reader = PdfReader(pdf_path)
            text = ""

            for page in reader.pages:
                text += page.extract_text() + "\n"

            if metadata is None:
                metadata = {}

            metadata['source'] = os.path.basename(pdf_path)
            metadata['type'] = 'pdf'

            return self.add_text_document(text, metadata)

        except Exception as e:
            print(f"❌ PDF eklenemedi: {e}")
            return False

    def search(self, query: str, k: int = 3) -> List[Dict]:
        """
        Benzer dökümanları ara

        Args:
            query: Arama sorgusu
            k: Kaç sonuç

        Returns:
            List[Dict]: Sonuçlar
        """
        if self.vectorstore is None:
            return []

        try:
            # Similarity search
            results = self.vectorstore.similarity_search_with_score(query, k=k)

            # Format sonuçlar
            formatted_results = []
            for doc, score in results:
                formatted_results.append({
                    'content': doc.page_content,
                    'metadata': doc.metadata,
                    'score': float(score)
                })

            return formatted_results

        except Exception as e:
            print(f"❌ Arama hatası: {e}")
            return []

    def get_context_for_query(self, query: str, k: int = 3) -> str:
        """
        Sorgu için context oluştur

        Args:
            query: Soru
            k: Kaç döküman

        Returns:
            str: Context metni
        """
        results = self.search(query, k=k)

        if not results:
            return ""

        context = "İlgili Dökümanlar:\n\n"

        for i, result in enumerate(results, 1):
            context += f"[Döküman {i}]\n"
            context += f"{result['content']}\n\n"

        return context

    def delete_all_documents(self):
        """Tüm dökümanları sil"""
        try:
            if self.vectorstore:
                # ChromaDB'yi sıfırla
                self.vectorstore.delete_collection()
                self.vectorstore = None
                print("✅ Tüm dökümanlar silindi")
        except Exception as e:
            print(f"❌ Silme hatası: {e}")

    def get_stats(self) -> Dict:
        """İstatistikler"""
        try:
            doc_count = self._get_doc_count()

            return {
                'total_documents': doc_count,
                'vectorstore_active': self.vectorstore is not None,
                'persist_directory': self.persist_directory
            }
        except:
            return {
                'total_documents': 0,
                'vectorstore_active': False,
                'persist_directory': self.persist_directory
            }


# Test
if __name__ == "__main__":
    print("🧪 RAG Manager Test\n")

    rag = RAGManager()

    # Test döküman ekle
    test_doc = """
    DDoS Saldırısı Nedir?
    
    DDoS (Distributed Denial of Service) saldırısı, bir sunucu veya ağı 
    aşırı trafikle hedef alarak hizmet vermesini engelleyen bir siber saldırı türüdür.
    
    Saldırganlar, botnet adı verilen enfekte cihaz ağları kullanarak hedef sisteme 
    aynı anda binlerce istek gönderir. Bu durum, sunucunun meşru kullanıcılara 
    hizmet veremez hale gelmesine neden olur.
    
    Korunma Yöntemleri:
    1. Rate limiting
    2. WAF (Web Application Firewall)
    3. CDN kullanımı
    4. Traffic filtering
    """

    rag.add_text_document(test_doc, metadata={
        'title': 'DDoS Saldırısı Rehberi',
        'category': 'Güvenlik',
        'language': 'tr'
    })

    # Arama yap
    print("\n🔍 Arama: 'DDoS saldırısından nasıl korunurum?'")
    results = rag.search("DDoS saldırısından nasıl korunurum?", k=2)

    for i, result in enumerate(results, 1):
        print(f"\n[Sonuç {i}] (Skor: {result['score']:.4f})")
        print(result['content'][:200] + "...")

    # Context oluştur
    print("\n📄 Context:")
    context = rag.get_context_for_query("DDoS nedir?")
    print(context[:300] + "...")

    # İstatistikler
    print("\n📊 İstatistikler:")
    stats = rag.get_stats()
    print(stats)