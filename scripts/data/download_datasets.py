"""
Dataset Downloader - CyberGuard AI
NSL-KDD ve CICIDS2017 veri setlerini indirir

Kullanım:
    python scripts/download_datasets.py --dataset all
    python scripts/download_datasets.py --dataset nsl_kdd
    python scripts/download_datasets.py --dataset cicids2017
"""

import os
import sys
import argparse
import zipfile
import urllib.request
from pathlib import Path

# Project root
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "raw"


def download_file(url: str, dest_path: Path, desc: str = ""):
    """URL'den dosya indir"""
    print(f"📥 İndiriliyor: {desc or url}")

    def progress_hook(count, block_size, total_size):
        percent = min(100, count * block_size * 100 // total_size)
        bar = "█" * (percent // 2) + "░" * (50 - percent // 2)
        print(f"\r   [{bar}] {percent}%", end="", flush=True)

    try:
        urllib.request.urlretrieve(url, dest_path, progress_hook)
        print(f"\n✅ İndirildi: {dest_path}")
        return True
    except Exception as e:
        print(f"\n❌ Hata: {e}")
        return False


def extract_zip(zip_path: Path, extract_to: Path):
    """ZIP dosyasını çıkart"""
    print(f"📦 Çıkartılıyor: {zip_path.name}")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extract_to)
    print(f"✅ Çıkartıldı: {extract_to}")


def download_nsl_kdd():
    """NSL-KDD veri setini indir (Kaggle)"""
    print("\n" + "=" * 60)
    print("📊 NSL-KDD Dataset İndiriliyor...")
    print("=" * 60)

    dest_dir = DATA_DIR / "nsl_kdd"
    dest_dir.mkdir(parents=True, exist_ok=True)

    # Kaggle API ile indir
    try:
        import kaggle

        print("🔑 Kaggle API kullanılıyor...")
        kaggle.api.authenticate()
        kaggle.api.dataset_download_files(
            "hassan06/nslkdd", path=str(dest_dir), unzip=True
        )
        print("✅ NSL-KDD başarıyla indirildi!")
        return True
    except Exception as e:
        print(f"⚠️ Kaggle API hatası: {e}")
        print("\n📋 Manuel indirme için:")
        print("   1. https://www.kaggle.com/datasets/hassan06/nslkdd adresine git")
        print("   2. 'Download' butonuna tıkla")
        print(f"   3. ZIP'i şuraya çıkart: {dest_dir}")
        return False


def download_cicids2017():
    """CICIDS2017 veri setini indir (UNB)"""
    print("\n" + "=" * 60)
    print("📊 CICIDS2017 Dataset İndiriliyor...")
    print("=" * 60)

    dest_dir = DATA_DIR / "cicids2017"
    dest_dir.mkdir(parents=True, exist_ok=True)

    print("⚠️ CICIDS2017 büyük bir veri seti (~6GB)")
    print("📋 Manuel indirme önerilir:")
    print("   1. https://www.unb.ca/cic/datasets/ids-2017.html adresine git")
    print("   2. 'MachineLearningCSV.zip' indir")
    print(f"   3. ZIP'i şuraya çıkart: {dest_dir}")
    print("\n🔗 Alternatif Kaggle linki:")
    print("   https://www.kaggle.com/datasets/cicdataset/cicids2017")

    # Kaggle ile dene
    try:
        import kaggle

        print("\n🔑 Kaggle API ile deneniyor...")
        kaggle.api.authenticate()
        kaggle.api.dataset_download_files(
            "cicdataset/cicids2017", path=str(dest_dir), unzip=True
        )
        print("✅ CICIDS2017 başarıyla indirildi!")
        return True
    except Exception as e:
        print(f"⚠️ Kaggle API hatası: {e}")
        return False


def check_kaggle_credentials():
    """Kaggle kimlik bilgilerini kontrol et - .env veya klasik yol"""
    from dotenv import load_dotenv

    # .env'yi yükle
    env_path = PROJECT_ROOT / ".env"
    load_dotenv(env_path)

    kaggle_username = os.getenv("KAGGLE_USERNAME")
    kaggle_key = os.getenv("KAGGLE_KEY")

    # .env'de varsa, Kaggle için gerekli dosyayı oluştur
    if kaggle_username and kaggle_key:
        print("✅ Kaggle kimlik bilgileri .env'den okundu")

        # Kaggle klasörü ve json oluştur
        kaggle_dir = Path.home() / ".kaggle"
        kaggle_dir.mkdir(exist_ok=True)
        kaggle_json = kaggle_dir / "kaggle.json"

        import json

        with open(kaggle_json, "w") as f:
            json.dump({"username": kaggle_username, "key": kaggle_key}, f)

        # Sadece sahibin okuyabilmesi için izinleri ayarla (Windows'ta opsiyonel)
        try:
            os.chmod(kaggle_json, 0o600)
        except:
            pass

        return True

    # Klasik yol kontrolü
    kaggle_dir = Path.home() / ".kaggle"
    kaggle_json = kaggle_dir / "kaggle.json"

    if kaggle_json.exists():
        print("✅ Kaggle kimlik bilgileri bulundu (~/.kaggle/)")
        return True

    print("\n" + "=" * 60)
    print("⚠️ KAGGLE API YAPILANDIRMASI GEREKLİ")
    print("=" * 60)
    print("\n🔧 Yöntem 1: .env dosyasına ekle (ÖNERİLEN)")
    print("   .env dosyasına şunları ekle:")
    print("   KAGGLE_USERNAME=your_username")
    print("   KAGGLE_KEY=your_api_key")
    print("\n🔧 Yöntem 2: Kaggle token indir")
    print("   1. https://www.kaggle.com/settings adresine git")
    print("   2. 'API' bölümünde 'Create New Token' tıkla")
    print(f"   3. kaggle.json'ı şuraya kopyala: {kaggle_dir}")
    return False


def main():
    parser = argparse.ArgumentParser(description="IDS Veri Seti İndirici")
    parser.add_argument(
        "--dataset",
        choices=["all", "nsl_kdd", "cicids2017"],
        default="all",
        help="İndirilecek veri seti",
    )
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("🔐 CyberGuard AI - Dataset Downloader")
    print("=" * 60)

    # Kaggle kontrolü
    has_kaggle = check_kaggle_credentials()

    if args.dataset in ["all", "nsl_kdd"]:
        download_nsl_kdd()

    if args.dataset in ["all", "cicids2017"]:
        download_cicids2017()

    print("\n" + "=" * 60)
    print("📋 İndirme Özeti")
    print("=" * 60)

    # Kontrol
    nsl_dir = DATA_DIR / "nsl_kdd"
    cic_dir = DATA_DIR / "cicids2017"

    print(f"   NSL-KDD: {'✅ Mevcut' if any(nsl_dir.glob('*.csv')) else '❌ Eksik'}")
    print(f"   CICIDS2017: {'✅ Mevcut' if any(cic_dir.glob('*.csv')) else '❌ Eksik'}")

    print("\n💡 Sonraki adım:")
    print("   python scripts/preprocess_datasets.py")


if __name__ == "__main__":
    main()
