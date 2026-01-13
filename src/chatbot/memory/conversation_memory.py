"""
Conversation Memory - CyberGuard AI
===================================

SQLite tabanlı kalıcı konuşma hafızası.

Özellikler:
    - Session yönetimi
    - Mesaj geçmişi
    - User preferences
    - Context caching
"""

import os
import json
import sqlite3
from pathlib import Path
from typing import Optional, Dict, List, Any
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
import logging

PROJECT_ROOT = Path(__file__).parent.parent.parent
DB_PATH = PROJECT_ROOT / "data" / "conversation_memory.db"

logger = logging.getLogger("ConversationMemory")


@dataclass
class Message:
    """Chat mesajı"""

    id: Optional[int]
    session_id: str
    role: str  # user, assistant, system
    content: str
    provider: Optional[str]
    model: Optional[str]
    tokens: Optional[int]
    created_at: str


@dataclass
class Session:
    """Konuşma oturumu"""

    id: str
    user_id: Optional[str]
    title: Optional[str]
    created_at: str
    updated_at: str
    message_count: int
    provider: Optional[str]


class ConversationMemory:
    """
    Kalıcı Konuşma Hafızası
    """

    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path or str(DB_PATH)

        # Ensure directory exists
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)

        self._init_database()

    def _init_database(self):
        """Database tablolarını oluştur"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # Sessions tablosu
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS sessions (
                    id TEXT PRIMARY KEY,
                    user_id TEXT,
                    title TEXT,
                    provider TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
            """
            )

            # Messages tablosu
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    provider TEXT,
                    model TEXT,
                    tokens INTEGER,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (session_id) REFERENCES sessions(id)
                )
            """
            )

            # Preferences tablosu
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS preferences (
                    user_id TEXT PRIMARY KEY,
                    preferred_provider TEXT,
                    preferred_model TEXT,
                    language TEXT DEFAULT 'tr',
                    response_style TEXT DEFAULT 'technical',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
            """
            )

            # Context cache tablosu
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS context_cache (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    expires_at TEXT,
                    created_at TEXT NOT NULL
                )
            """
            )

            # Index'ler
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id)"
            )
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_sessions_user ON sessions(user_id)"
            )

            conn.commit()
            logger.info(f"✅ Database initialized: {self.db_path}")

    # ============= Session Yönetimi =============

    def create_session(
        self,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
        title: Optional[str] = None,
        provider: Optional[str] = None,
    ) -> Session:
        """Yeni oturum oluştur"""
        import uuid

        session_id = session_id or str(uuid.uuid4())[:8]
        now = datetime.now().isoformat()

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO sessions (id, user_id, title, provider, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
            """,
                (session_id, user_id, title, provider, now, now),
            )
            conn.commit()

        return Session(
            id=session_id,
            user_id=user_id,
            title=title,
            created_at=now,
            updated_at=now,
            message_count=0,
            provider=provider,
        )

    def get_session(self, session_id: str) -> Optional[Session]:
        """Oturum getir"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT s.id, s.user_id, s.title, s.created_at, s.updated_at, s.provider,
                       COUNT(m.id) as message_count
                FROM sessions s
                LEFT JOIN messages m ON s.id = m.session_id
                WHERE s.id = ?
                GROUP BY s.id
            """,
                (session_id,),
            )

            row = cursor.fetchone()
            if row:
                return Session(
                    id=row[0],
                    user_id=row[1],
                    title=row[2],
                    created_at=row[3],
                    updated_at=row[4],
                    provider=row[5],
                    message_count=row[6],
                )
        return None

    def list_sessions(
        self,
        user_id: Optional[str] = None,
        limit: int = 20,
    ) -> List[Session]:
        """Oturumları listele"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            if user_id:
                cursor.execute(
                    """
                    SELECT s.id, s.user_id, s.title, s.created_at, s.updated_at, s.provider,
                           COUNT(m.id) as message_count
                    FROM sessions s
                    LEFT JOIN messages m ON s.id = m.session_id
                    WHERE s.user_id = ?
                    GROUP BY s.id
                    ORDER BY s.updated_at DESC
                    LIMIT ?
                """,
                    (user_id, limit),
                )
            else:
                cursor.execute(
                    """
                    SELECT s.id, s.user_id, s.title, s.created_at, s.updated_at, s.provider,
                           COUNT(m.id) as message_count
                    FROM sessions s
                    LEFT JOIN messages m ON s.id = m.session_id
                    GROUP BY s.id
                    ORDER BY s.updated_at DESC
                    LIMIT ?
                """,
                    (limit,),
                )

            sessions = []
            for row in cursor.fetchall():
                sessions.append(
                    Session(
                        id=row[0],
                        user_id=row[1],
                        title=row[2],
                        created_at=row[3],
                        updated_at=row[4],
                        provider=row[5],
                        message_count=row[6],
                    )
                )

            return sessions

    def delete_session(self, session_id: str):
        """Oturum sil"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
            cursor.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
            conn.commit()

    # ============= Message Yönetimi =============

    def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        tokens: Optional[int] = None,
    ) -> Message:
        """Mesaj ekle"""
        now = datetime.now().isoformat()

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # Mesaj ekle
            cursor.execute(
                """
                INSERT INTO messages (session_id, role, content, provider, model, tokens, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
                (session_id, role, content, provider, model, tokens, now),
            )

            message_id = cursor.lastrowid

            # Session update et
            cursor.execute(
                """
                UPDATE sessions SET updated_at = ? WHERE id = ?
            """,
                (now, session_id),
            )

            conn.commit()

        return Message(
            id=message_id,
            session_id=session_id,
            role=role,
            content=content,
            provider=provider,
            model=model,
            tokens=tokens,
            created_at=now,
        )

    def get_messages(
        self,
        session_id: str,
        limit: int = 50,
    ) -> List[Message]:
        """Oturum mesajlarını getir"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT id, session_id, role, content, provider, model, tokens, created_at
                FROM messages
                WHERE session_id = ?
                ORDER BY created_at DESC
                LIMIT ?
            """,
                (session_id, limit),
            )

            messages = []
            for row in cursor.fetchall():
                messages.append(
                    Message(
                        id=row[0],
                        session_id=row[1],
                        role=row[2],
                        content=row[3],
                        provider=row[4],
                        model=row[5],
                        tokens=row[6],
                        created_at=row[7],
                    )
                )

            # Kronolojik sıra
            return list(reversed(messages))

    def get_history_for_prompt(
        self,
        session_id: str,
        limit: int = 10,
    ) -> List[Dict]:
        """LLM için mesaj geçmişi"""
        messages = self.get_messages(session_id, limit)

        return [
            {"role": m.role, "content": m.content}
            for m in messages
            if m.role in ["user", "assistant"]
        ]

    # ============= Preferences =============

    def get_preferences(self, user_id: str) -> Dict:
        """Kullanıcı tercihlerini getir"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT preferred_provider, preferred_model, language, response_style
                FROM preferences WHERE user_id = ?
            """,
                (user_id,),
            )

            row = cursor.fetchone()
            if row:
                return {
                    "preferred_provider": row[0],
                    "preferred_model": row[1],
                    "language": row[2],
                    "response_style": row[3],
                }

        return {
            "preferred_provider": "groq",
            "preferred_model": "llama-3.3-70b-versatile",
            "language": "tr",
            "response_style": "technical",
        }

    def save_preferences(self, user_id: str, preferences: Dict):
        """Kullanıcı tercihlerini kaydet"""
        now = datetime.now().isoformat()

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT OR REPLACE INTO preferences 
                (user_id, preferred_provider, preferred_model, language, response_style, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    user_id,
                    preferences.get("preferred_provider"),
                    preferences.get("preferred_model"),
                    preferences.get("language", "tr"),
                    preferences.get("response_style", "technical"),
                    now,
                    now,
                ),
            )
            conn.commit()

    # ============= Context Cache =============

    def cache_context(self, key: str, value: Any, ttl_hours: int = 24):
        """Context cache'e kaydet"""
        now = datetime.now()
        expires = now + timedelta(hours=ttl_hours)

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT OR REPLACE INTO context_cache (key, value, expires_at, created_at)
                VALUES (?, ?, ?, ?)
            """,
                (key, json.dumps(value), expires.isoformat(), now.isoformat()),
            )
            conn.commit()

    def get_cached_context(self, key: str) -> Optional[Any]:
        """Cache'den context al"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT value, expires_at FROM context_cache WHERE key = ?
            """,
                (key,),
            )

            row = cursor.fetchone()
            if row:
                expires_at = datetime.fromisoformat(row[1])
                if expires_at > datetime.now():
                    return json.loads(row[0])
                else:
                    # Expired, delete
                    cursor.execute("DELETE FROM context_cache WHERE key = ?", (key,))
                    conn.commit()

        return None

    def clear_cache(self):
        """Cache temizle"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM context_cache")
            conn.commit()


# Singleton
_memory_instance: Optional[ConversationMemory] = None


def get_memory() -> ConversationMemory:
    """Global memory instance"""
    global _memory_instance
    if _memory_instance is None:
        _memory_instance = ConversationMemory()
    return _memory_instance


# Test
if __name__ == "__main__":
    print("🧪 Conversation Memory Test\n")

    memory = ConversationMemory()

    # Session oluştur
    session = memory.create_session(title="Test Konuşması")
    print(f"✅ Session created: {session.id}")

    # Mesaj ekle
    memory.add_message(session.id, "user", "Merhaba!")
    memory.add_message(
        session.id, "assistant", "Merhaba! Size nasıl yardımcı olabilirim?"
    )
    memory.add_message(session.id, "user", "DDoS saldırısı nedir?")

    # Mesajları listele
    print("\n📝 Mesajlar:")
    for msg in memory.get_messages(session.id):
        print(f"   [{msg.role}]: {msg.content[:50]}...")

    # History for prompt
    print("\n🔄 Prompt History:")
    history = memory.get_history_for_prompt(session.id)
    print(json.dumps(history, indent=2, ensure_ascii=False))

    # Sessions
    print(f"\n📋 Toplam oturum: {len(memory.list_sessions())}")
