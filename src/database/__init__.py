"""
Database Package - CyberGuard AI
Veritabanı yönetimi ve işlemleri

Dosya Yolu: src/database/__init__.py
"""

try:
    from src.utils.database import DatabaseManager
except ImportError:
    # Eğer import başarısız olursa boş bırak
    DatabaseManager = None

__all__ = ['DatabaseManager']

