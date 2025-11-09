"""
Attack Vectors - CyberGuard AI
Database'deki saldırıları vektörleştir ve benzer saldırı analizi
"""

import os
import sqlite3
import pandas as pd
from typing import List, Dict
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document




class AttackVectorManager:
    """Saldırı vektör yöneticisi"""

    def __init__(self, db_path: str = "cyberguard.db",
                 persist_directory: str = "chatbot/vectorstore/attack_vectors_db"):
        """
        Args:
            db_path: Database yolu
            persist_directory: VectorStore yolu
        """
        self.db_path = db_path
        self.persist_directory = persist_directory
        os.makedirs(persist_directory, exist_ok=True)

        # Embedding modeli
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            model_kwargs={'device': 'cpu'}
        )

        # VectorStore
        try:
            self.vectorstore = Chroma(
                persist_directory=persist_directory,
                embedding_function=self.embeddings,
                collection_name="attack_vectors"
            )
            print(f"✅ Attack VectorStore yüklendi: {self._get_vector_count()} saldırı")
        except:
            self.vectorstore = None
            print("⚠️ Attack VectorStore oluşturuluyor...")

    def _get_vector_count(self) -> int:
        """Vektör sayısı"""
        try:
            return len(self.vectorstore.get()['ids'])
        except:
            return 0

    def vectorize_attacks(self, limit: int = 1000) -> bool:
        """
        Database'deki saldırıları vektörleştir

        Args:
            limit: Maksimum saldırı sayısı

        Returns:
            bool: Başarılı mı?
        """
        try:
            # Database'den saldırıları çek
            conn = sqlite3.connect(self.db_path)

            query = f"""
            SELECT 
                id, timestamp, attack_type, source_ip, destination_ip, 
                port, severity, status, description, blocked
            FROM attacks 
            ORDER BY timestamp DESC 
            LIMIT {limit}
            """

            df = pd.read_sql_query(query, conn)
            conn.close()

            if len(df) == 0:
                print("⚠️ Database'de saldırı yok!")
                return False

            print(f"🔄 {len(df)} saldırı vektörleştiriliyor...")

            # Her saldırı için döküman oluştur
            documents = []

            for _, row in df.iterrows():
                # Saldırı açıklaması oluştur
                attack_text = f"""
Saldırı Türü: {row['attack_type']}
Kaynak IP: {row['source_ip']}
Hedef IP: {row['destination_ip']}
Port: {row['port']}
Severity: {row['severity']}
Durum: {row['status']}
Açıklama: {row['description']}
Engellendi: {'Evet' if row['blocked'] else 'Hayır'}
Zaman: {row['timestamp']}
"""

                # Metadata
                metadata = {
                    'attack_id': int(row['id']),
                    'attack_type': row['attack_type'],
                    'source_ip': row['source_ip'],
                    'severity': row['severity'],
                    'timestamp': row['timestamp'],
                    'blocked': bool(row['blocked'])
                }

                # Document oluştur
                doc = Document(
                    page_content=attack_text,
                    metadata=metadata
                )

                documents.append(doc)

            # VectorStore'a ekle
            if self.vectorstore is None:
                self.vectorstore = Chroma.from_documents(
                    documents,
                    self.embeddings,
                    persist_directory=self.persist_directory,
                    collection_name="attack_vectors"
                )
            else:
                # Önce temizle
                self.vectorstore.delete_collection()
                self.vectorstore = Chroma.from_documents(
                    documents,
                    self.embeddings,
                    persist_directory=self.persist_directory,
                    collection_name="attack_vectors"
                )

            self.vectorstore.persist()

            print(f"✅ {len(documents)} saldırı vektörleştirildi!")
            return True

        except Exception as e:
            print(f"❌ Vektörleştirme hatası: {e}")
            import traceback
            traceback.print_exc()
            return False

    def find_similar_attacks(self, query: str, k: int = 5) -> List[Dict]:
        """
        Benzer saldırıları bul

        Args:
            query: Arama sorgusu (örn: "DDoS saldırısı")
            k: Kaç sonuç

        Returns:
            List[Dict]: Benzer saldırılar
        """
        if self.vectorstore is None:
            return []

        try:
            results = self.vectorstore.similarity_search_with_score(query, k=k)

            similar_attacks = []
            for doc, score in results:
                similar_attacks.append({
                    'content': doc.page_content,
                    'metadata': doc.metadata,
                    'similarity_score': float(score),
                    'attack_type': doc.metadata.get('attack_type', 'Unknown'),
                    'source_ip': doc.metadata.get('source_ip', 'Unknown'),
                    'severity': doc.metadata.get('severity', 'Unknown'),
                    'timestamp': doc.metadata.get('timestamp', 'Unknown')
                })

            return similar_attacks

        except Exception as e:
            print(f"❌ Arama hatası: {e}")
            return []

    def analyze_attack_pattern(self, attack_type: str) -> Dict:
        """
        Belirli bir saldırı türü için pattern analizi

        Args:
            attack_type: Saldırı türü (örn: "DDoS")

        Returns:
            Dict: Analiz sonuçları
        """
        try:
            # Benzer saldırıları bul
            similar = self.find_similar_attacks(attack_type, k=20)

            if not similar:
                return {'error': 'Saldırı bulunamadı'}

            # Analiz
            severities = [s['severity'] for s in similar]
            ips = [s['source_ip'] for s in similar]

            analysis = {
                'attack_type': attack_type,
                'total_found': len(similar),
                'severity_distribution': {
                    'critical': severities.count('critical'),
                    'high': severities.count('high'),
                    'medium': severities.count('medium'),
                    'low': severities.count('low')
                },
                'unique_ips': len(set(ips)),
                'top_ips': list(set(ips))[:5],
                'recent_attacks': similar[:5]
            }

            return analysis

        except Exception as e:
            print(f"❌ Analiz hatası: {e}")
            return {'error': str(e)}

    def get_attack_summary_for_chatbot(self, query: str) -> str:
        """
        Chatbot için saldırı özeti oluştur

        Args:
            query: Kullanıcı sorusu

        Returns:
            str: Özet metin
        """
        similar = self.find_similar_attacks(query, k=3)

        if not similar:
            return ""

        summary = "İlgili Saldırı Kayıtları:\n\n"

        for i, attack in enumerate(similar, 1):
            summary += f"[Saldırı {i}]\n"
            summary += f"Tür: {attack['attack_type']}\n"
            summary += f"Kaynak IP: {attack['source_ip']}\n"
            summary += f"Severity: {attack['severity']}\n"
            summary += f"Zaman: {attack['timestamp']}\n"
            summary += f"Benzerlik: {attack['similarity_score']:.2f}\n\n"

        return summary

    def clear_vectors(self):
        """Tüm vektörleri temizle"""
        try:
            if self.vectorstore:
                self.vectorstore.delete_collection()
                self.vectorstore = None
                print("✅ Tüm vektörler silindi")
        except Exception as e:
            print(f"❌ Silme hatası: {e}")

    def get_stats(self) -> Dict:
        """İstatistikler"""
        try:
            return {
                'total_vectors': self._get_vector_count(),
                'vectorstore_active': self.vectorstore is not None,
                'db_path': self.db_path
            }
        except:
            return {
                'total_vectors': 0,
                'vectorstore_active': False,
                'db_path': self.db_path
            }


# Test
if __name__ == "__main__":
    print("🧪 Attack Vector Manager Test\n")

    attack_vectors = AttackVectorManager()

    # Database'i vektörleştir
    print("🔄 Saldırılar vektörleştiriliyor...")
    attack_vectors.vectorize_attacks(limit=100)

    # Benzer saldırıları bul
    print("\n🔍 Benzer saldırılar: 'DDoS'")
    similar = attack_vectors.find_similar_attacks("DDoS saldırısı", k=3)

    for i, attack in enumerate(similar, 1):
        print(f"\n[Saldırı {i}] (Benzerlik: {attack['similarity_score']:.4f})")
        print(f"Tür: {attack['attack_type']}")
        print(f"IP: {attack['source_ip']}")
        print(f"Severity: {attack['severity']}")

    # Pattern analizi
    print("\n📊 DDoS Pattern Analizi:")
    analysis = attack_vectors.analyze_attack_pattern("DDoS")
    print(f"Toplam: {analysis.get('total_found', 0)}")
    print(f"Severity: {analysis.get('severity_distribution', {})}")

    # İstatistikler
    print("\n📊 İstatistikler:")
    stats = attack_vectors.get_stats()
    print(stats)