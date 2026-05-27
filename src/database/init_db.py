"""
Database Initialization - CyberGuard AI
Veritabanı tablolarını oluştur ve başlangıç verilerini ekle

Dosya Yolu: src/database/init_db.py

NOT: Tablo şeması src/utils/database.py DatabaseManager.create_tables() içinde
     tanımlıdır. Bu dosya sadece CLI wrapper olarak hizmet verir.
"""

import os
import sys

# Proje root
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


def create_tables(db_path: str = None):
    """
    Tüm tabloları oluştur (DatabaseManager'a delegate eder)
    
    Args:
        db_path: Database dosya yolu
    """
    if db_path is None:
        db_path = os.path.join(project_root, 'src', 'database', 'cyberguard.db')

    print(f"[INFO] Database olusturuluyor: {db_path}")

    from src.utils.database import DatabaseManager
    db = DatabaseManager(db_path)

    print(f"[OK] Database basariyla olusturuldu: {db_path}")


def main():
    """Ana fonksiyon"""
    print("\n" + "=" * 50)
    print("CYBERGUARD AI - DATABASE INITIALIZATION")
    print("=" * 50 + "\n")

    create_tables()

    print("\n" + "=" * 50)
    print("[OK] TAMAMLANDI")
    print("=" * 50)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[ERROR] HATA: {e}")
        import traceback
        traceback.print_exc()
