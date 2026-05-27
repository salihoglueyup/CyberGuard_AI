# test_imports.py

"""Kurulum testi"""

import sys


def test_imports():
    packages = {
        'numpy': 'NumPy',
        'pandas': 'Pandas',
        'tensorflow': 'TensorFlow',
        'keras': 'Keras',
        'sklearn': 'Scikit-learn',
        'streamlit': 'Streamlit',
        'google.generativeai': 'Google Gemini',
        'langchain': 'LangChain',
        'chromadb': 'ChromaDB',
        'matplotlib': 'Matplotlib',
        'plotly': 'Plotly',
        'PIL': 'Pillow',
        'cv2': 'OpenCV',
    }

    print("🔍 Paket Kontrolü\n")
    print("=" * 50)

    success_count = 0
    fail_count = 0

    for package, name in packages.items():
        try:
            __import__(package)
            print(f"✅ {name:20s} BAŞARILI")
            success_count += 1
        except ImportError:
            print(f"❌ {name:20s} BAŞARISIZ")
            fail_count += 1

    print("=" * 50)
    print(f"\n📊 Sonuç: {success_count}/{len(packages)} paket başarılı")

    if fail_count == 0:
        print("\n🎉 Tüm paketler başarıyla yüklendi!")
        return True
    else:
        print(f"\n⚠️  {fail_count} paket eksik!")
        return False


if __name__ == "__main__":
    success = test_imports()
    sys.exit(0 if success else 1)