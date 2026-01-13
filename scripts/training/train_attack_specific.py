"""
CICIDS2017 Saldırı Bazlı Eğitim ve Test
========================================

Her saldırı tipini ayrı ayrı eğitir:
- DDoS
- PortScan
- Web Attacks (Brute Force, XSS, SQL Injection)
- Botnet
- Infiltration

Kullanım:
    python scripts/train_attack_specific.py --attack ddos
    python scripts/train_attack_specific.py --attack portscan
    python scripts/train_attack_specific.py --all
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
)

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


# CICIDS2017 dosya-saldırı eşleştirmesi
CICIDS2017_FILES = {
    "ddos": {
        "file": "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv",
        "attacks": ["DDoS"],
        "description": "DDoS saldırıları",
    },
    "portscan": {
        "file": "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv",
        "attacks": ["PortScan"],
        "description": "Port tarama saldırıları",
    },
    "webattacks": {
        "file": "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv",
        "attacks": ["Brute Force", "XSS", "Sql Injection", "Web Attack"],
        "description": "Brute Force, XSS, SQL Injection",
    },
    "botnet": {
        "file": "Friday-WorkingHours-Morning.pcap_ISCX.csv",
        "attacks": ["Bot"],
        "description": "Botnet saldırıları",
    },
    "infiltration": {
        "file": "Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv",
        "attacks": ["Infiltration"],
        "description": "Infiltration saldırıları",
    },
    "dos_heartbleed": {
        "file": "Wednesday-workingHours.pcap_ISCX.csv",
        "attacks": [
            "DoS",
            "Heartbleed",
            "DoS Hulk",
            "DoS GoldenEye",
            "DoS slowloris",
            "DoS Slowhttptest",
        ],
        "description": "DoS ve Heartbleed",
    },
}


def load_attack_data(
    attack_type: str, data_dir: Path, max_samples: int = None
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Belirli saldırı tipini yükle"""

    if attack_type not in CICIDS2017_FILES:
        print(f"❌ Bilinmeyen saldırı tipi: {attack_type}")
        print(f"   Mevcut tipler: {list(CICIDS2017_FILES.keys())}")
        return None, None, None

    config = CICIDS2017_FILES[attack_type]
    file_path = data_dir / config["file"]

    if not file_path.exists():
        print(f"❌ Dosya bulunamadı: {file_path}")
        return None, None, None

    print(f"\n📂 {config['description']} yükleniyor...")
    print(f"   Dosya: {config['file']}")

    # Yükle - encoding fallback
    df = None
    for encoding in ["utf-8", "latin-1", "cp1252", "iso-8859-1"]:
        try:
            df = pd.read_csv(file_path, low_memory=False, encoding=encoding)
            print(f"   ✅ Encoding: {encoding}")
            break
        except UnicodeDecodeError:
            continue

    if df is None:
        print(f"❌ Dosya okunamadı (encoding hatası)")
        return None, None, None

    print(f"   Toplam satır: {len(df):,}")

    # Label sütununu bul
    label_col = None
    for col in [" Label", "Label", "label"]:
        if col in df.columns:
            label_col = col
            break

    if label_col is None:
        print(f"❌ Label sütunu bulunamadı!")
        return None, None, None

    # Sınıf dağılımı
    print(f"\n🎯 Sınıf Dağılımı:")
    class_counts = df[label_col].value_counts()
    for cls, cnt in class_counts.head(10).items():
        print(f"   {cls}: {cnt:,}")

    # Binary classification: BENIGN vs Attack
    df["is_attack"] = df[label_col].apply(
        lambda x: 0 if "BENIGN" in str(x).upper() else 1
    )

    # Features - sadece sayısal sütunları al
    feature_cols = [c for c in df.columns if c not in [label_col, "is_attack"]]

    # Sayısal olmayan sütunları filtrele
    numeric_cols = []
    for col in feature_cols:
        try:
            pd.to_numeric(df[col], errors="raise")
            numeric_cols.append(col)
        except (ValueError, TypeError):
            pass  # Sayısal olmayan sütun, atla

    print(f"\n📊 Kullanılan özellik sayısı: {len(numeric_cols)}")

    # Sayısal verileri al
    X = df[numeric_cols].apply(pd.to_numeric, errors="coerce").values
    y = df["is_attack"].values

    # NaN ve Inf temizle
    X = np.nan_to_num(X, nan=0, posinf=0, neginf=0)
    X = X.astype(np.float32)

    # Sample limit
    if max_samples and len(X) > max_samples:
        # Stratified sampling
        from sklearn.model_selection import train_test_split

        X, _, y, _ = train_test_split(
            X, y, train_size=max_samples, stratify=y, random_state=42
        )
        print(f"\n⚠️ {max_samples:,} sample'a düşürüldü")

    # Attack vs Benign dağılımı
    attack_count = np.sum(y == 1)
    benign_count = np.sum(y == 0)
    print(f"\n📊 Binary Dağılım:")
    print(f"   BENIGN: {benign_count:,} ({benign_count/len(y)*100:.1f}%)")
    print(f"   ATTACK: {attack_count:,} ({attack_count/len(y)*100:.1f}%)")

    return X, y, ["BENIGN", "ATTACK"]


def train_attack_model(
    attack_type: str,
    max_samples: int = 50000,
    epochs: int = 100,
    use_smote: bool = False,
):
    """Saldırı bazlı model eğit"""

    print("\n" + "=" * 70)
    print(f"🎯 {attack_type.upper()} MODEL EĞİTİMİ")
    print("=" * 70)

    data_dir = PROJECT_ROOT / "data" / "raw" / "cicids2017"

    # Veri yükle
    X, y, class_names = load_attack_data(attack_type, data_dir, max_samples)

    if X is None:
        return None

    # Ölçeklendirme
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # SMOTE
    if use_smote:
        try:
            from imblearn.over_sampling import SMOTE

            print("\n⚖️ SMOTE uygulanıyor...")
            smote = SMOTE(random_state=42)
            X, y = smote.fit_resample(X, y)
            print(f"   Yeni dağılım: BENIGN={np.sum(y==0):,}, ATTACK={np.sum(y==1):,}")
        except ImportError:
            print("⚠️ SMOTE için imbalanced-learn gerekli")

    # Train/Test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Sequence oluştur
    seq_len = 10

    def create_sequences(X, y):
        X_seq, y_seq = [], []
        for i in range(len(X) - seq_len + 1):
            X_seq.append(X[i : i + seq_len])
            y_seq.append(y[i + seq_len - 1])
        return np.array(X_seq), np.array(y_seq)

    X_train_seq, y_train_seq = create_sequences(X_train, y_train)
    X_test_seq, y_test_seq = create_sequences(X_test, y_test)

    print(f"\n📊 Train shape: {X_train_seq.shape}")
    print(f"📊 Test shape: {X_test_seq.shape}")

    # Model oluştur
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers

    model = keras.Sequential(
        [
            layers.Input(shape=(seq_len, X.shape[1])),
            layers.Conv1D(30, 5, activation="relu", padding="same"),
            layers.MaxPooling1D(2),
            layers.LSTM(120, dropout=0.2),
            layers.Dense(512, activation="relu"),
            layers.Dropout(0.2),
            layers.Dense(2, activation="softmax"),  # Binary: BENIGN vs ATTACK
        ]
    )

    model.compile(
        optimizer=keras.optimizers.Adam(0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    print("\n📦 Model:")
    model.summary()

    # Eğit
    callbacks = [
        keras.callbacks.EarlyStopping(patience=15, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=7),
    ]

    print(f"\n🏋️ Eğitim başlıyor (epochs={epochs})...")

    history = model.fit(
        X_train_seq,
        y_train_seq,
        batch_size=128,
        epochs=epochs,
        validation_split=0.1,
        callbacks=callbacks,
        verbose=1,
    )

    # Değerlendir
    y_pred = np.argmax(model.predict(X_test_seq, verbose=0), axis=1)

    accuracy = accuracy_score(y_test_seq, y_pred)
    precision = precision_score(y_test_seq, y_pred, average="weighted")
    recall = recall_score(y_test_seq, y_pred, average="weighted")
    f1 = f1_score(y_test_seq, y_pred, average="weighted")

    print("\n" + "=" * 60)
    print(f"📊 {attack_type.upper()} TEST SONUÇLARI")
    print("=" * 60)
    print(f"   Accuracy:  {accuracy*100:.2f}%")
    print(f"   Precision: {precision*100:.2f}%")
    print(f"   Recall:    {recall*100:.2f}%")
    print(f"   F1-Score:  {f1*100:.2f}%")

    print("\n📋 Classification Report:")
    print(classification_report(y_test_seq, y_pred, target_names=class_names))

    # Kaydet
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = PROJECT_ROOT / "models" / f"attack_{attack_type}_{timestamp}.h5"
    model.save(str(model_path))
    print(f"\n💾 Model kaydedildi: {model_path}")

    # Sonuçları kaydet
    results = {
        "attack_type": attack_type,
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "train_samples": len(X_train_seq),
        "test_samples": len(X_test_seq),
        "epochs_trained": len(history.history["loss"]),
        "trained_at": datetime.now().isoformat(),
    }

    results_path = PROJECT_ROOT / "models" / "attack_specific_results.json"

    existing = {}
    if results_path.exists():
        with open(results_path) as f:
            existing = json.load(f)

    existing[attack_type] = results

    with open(results_path, "w") as f:
        json.dump(existing, f, indent=2)

    return results


def train_all_attacks(max_samples: int = 30000):
    """Tüm saldırı tiplerini eğit"""

    print("\n" + "=" * 70)
    print("🎯 TÜM SALDIRI TİPLERİ EĞİTİMİ")
    print("=" * 70)

    all_results = {}

    for attack_type in CICIDS2017_FILES.keys():
        try:
            results = train_attack_model(attack_type, max_samples)
            if results:
                all_results[attack_type] = results
        except Exception as e:
            print(f"❌ {attack_type} hatası: {e}")

    # Özet
    print("\n" + "=" * 70)
    print("📊 SONUÇ ÖZETİ")
    print("=" * 70)
    print(f"{'Saldırı Tipi':<20} {'Accuracy':>10} {'F1-Score':>10}")
    print("-" * 42)

    for attack, res in all_results.items():
        print(f"{attack:<20} {res['accuracy']*100:>9.2f}% {res['f1_score']*100:>9.2f}%")

    return all_results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Saldırı Bazlı Model Eğitimi")
    parser.add_argument(
        "--attack", type=str, choices=list(CICIDS2017_FILES.keys()) + ["all"]
    )
    parser.add_argument("--max_samples", type=int, default=50000)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--smote", action="store_true")
    parser.add_argument("--all", action="store_true")

    args = parser.parse_args()

    if args.all or args.attack == "all":
        train_all_attacks(args.max_samples)
    elif args.attack:
        train_attack_model(args.attack, args.max_samples, args.epochs, args.smote)
    else:
        print("Kullanım: python train_attack_specific.py --attack ddos")
        print("          python train_attack_specific.py --all")
        print(f"\nMevcut saldırı tipleri: {list(CICIDS2017_FILES.keys())}")
