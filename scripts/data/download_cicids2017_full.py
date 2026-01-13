"""
CICIDS2017 Full Dataset Downloader
===================================

UNB (University of New Brunswick) kaynaklı CICIDS2017 full dataset indirir.
~700MB compressed, ~2.8M samples

Kullanım:
    python scripts/download_cicids2017_full.py
"""

import os
import sys
import requests
import zipfile
from pathlib import Path
from datetime import datetime
import hashlib

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "raw" / "cicids2017_full_v2"

# CICIDS2017 MachineLearningCSV.zip direct download link
# Source: https://www.unb.ca/cic/datasets/ids-2017.html
DATASET_URL = (
    "https://iscxdownloads.cs.unb.ca/iscxdownloads/CIC-IDS-2017/MachineLearningCSV.zip"
)

# Alternative: Kaggle dataset (requires authentication)
KAGGLE_DATASET = "cicdataset/cicids2017"


def download_with_progress(url, filepath, chunk_size=8192):
    """Download file with progress indicator"""
    print(f"\n📥 Downloading from: {url[:80]}...")

    response = requests.get(url, stream=True, timeout=30)
    response.raise_for_status()

    total_size = int(response.headers.get("content-length", 0))
    downloaded = 0

    with open(filepath, "wb") as f:
        for chunk in response.iter_content(chunk_size=chunk_size):
            if chunk:
                f.write(chunk)
                downloaded += len(chunk)

                if total_size > 0:
                    percent = (downloaded / total_size) * 100
                    mb_downloaded = downloaded / (1024 * 1024)
                    mb_total = total_size / (1024 * 1024)
                    print(
                        f"\r   {mb_downloaded:.1f}MB / {mb_total:.1f}MB ({percent:.1f}%)",
                        end="",
                        flush=True,
                    )

    print(f"\n   ✅ Downloaded: {filepath}")
    return filepath


def extract_zip(zip_path, extract_to):
    """Extract zip file"""
    print(f"\n📦 Extracting to: {extract_to}")

    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        members = zip_ref.namelist()
        print(f"   Files in archive: {len(members)}")

        for i, member in enumerate(members):
            zip_ref.extract(member, extract_to)
            if (i + 1) % 2 == 0 or i == len(members) - 1:
                print(f"\r   Extracted: {i+1}/{len(members)}", end="", flush=True)

    print(f"\n   ✅ Extraction complete!")


def download_from_kaggle():
    """Download from Kaggle (requires kaggle.json)"""
    try:
        import kaggle

        print("\n📥 Downloading from Kaggle...")
        kaggle.api.dataset_download_files(
            KAGGLE_DATASET, path=str(DATA_DIR), unzip=True
        )
        print("   ✅ Kaggle download complete!")
        return True
    except Exception as e:
        print(f"   ⚠️ Kaggle download failed: {e}")
        return False


def count_samples():
    """Count total samples in downloaded CSV files"""
    print("\n📊 Counting samples...")

    csv_files = list(DATA_DIR.glob("**/*.csv"))
    total_rows = 0

    for csv_file in csv_files:
        try:
            with open(csv_file, "r", encoding="utf-8", errors="ignore") as f:
                rows = sum(1 for _ in f) - 1  # Subtract header
                total_rows += rows
                print(f"   {csv_file.name}: {rows:,} rows")
        except Exception as e:
            print(f"   ⚠️ Error counting {csv_file.name}: {e}")

    print(f"\n   📊 Total samples: {total_rows:,}")
    return total_rows


def main():
    print("\n" + "=" * 70)
    print("🎓 CICIDS2017 Full Dataset Downloader")
    print("=" * 70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Create directory
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\n📁 Target directory: {DATA_DIR}")

    # Check if already downloaded
    existing_csvs = list(DATA_DIR.glob("**/*.csv"))
    if existing_csvs:
        print(f"\n⚠️ Found {len(existing_csvs)} existing CSV files")
        print("   Skipping download. Delete files to re-download.")
        count_samples()
        return

    # Try Kaggle first
    print("\n" + "=" * 60)
    print("📥 Attempting download...")
    print("=" * 60)

    # Direct download from UNB
    zip_path = DATA_DIR / "MachineLearningCSV.zip"

    try:
        download_with_progress(DATASET_URL, zip_path)
        extract_zip(zip_path, DATA_DIR)

        # Cleanup
        zip_path.unlink()
        print("\n🗑️ Deleted zip file")

    except Exception as e:
        print(f"\n❌ UNB download failed: {e}")
        print("\n💡 Alternative: Download manually from:")
        print("   https://www.unb.ca/cic/datasets/ids-2017.html")
        print("   Or Kaggle: https://www.kaggle.com/datasets/cicdataset/cicids2017")
        return

    # Count samples
    count_samples()

    print("\n" + "=" * 70)
    print("✅ CICIDS2017 Full Dataset Download Complete!")
    print("=" * 70)
    print(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
