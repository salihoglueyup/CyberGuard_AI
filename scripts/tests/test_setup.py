# test_setup.py

"""
Kurulum testi
"""

import sys
import importlib


def test_imports():
    """Gerekli paketleri test et"""

    required_packages = [
        'tensorflow',
        'keras',
        'sklearn',
        'pandas',
        'numpy',
        'streamlit',
        'google.generativeai',
        'langchain',
        'chromadb',
    ]

    print("📦 Paket kontrolü başlıyor...\n")

    failed = []

    for package in required_packages:
        try:
            importlib.import_module(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package}")
            failed.append(package)

    print("\n" + "=" * 50)

    if failed:
        print(f"\n❌ {len(failed)} paket eksik!")
        print("Yüklemek için: pip install -r requirements.txt")
        return False
    else:
        print("\n✅ Tüm paketler başarıyla yüklendi!")
        return True


if __name__ == "__main__":
    success = test_imports()
    sys.exit(0 if success else 1)