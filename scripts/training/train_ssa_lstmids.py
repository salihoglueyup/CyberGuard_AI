"""
SSA-LSTMIDS Eğitim Script'i
Makaledeki model ile NSL-KDD ve BoT-IoT üzerinde eğitim

Kullanım:
    python scripts/train_ssa_lstmids.py --dataset nsl_kdd --epochs 100
    python scripts/train_ssa_lstmids.py --dataset bot_iot --epochs 50
    python scripts/train_ssa_lstmids.py --dataset all --optimize
"""

import os
import sys
import argparse
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Proje yolu
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from src.network_detection.ssa_lstmids import SSA_LSTMIDS, optimize_with_ssa


def load_nsl_kdd(data_dir: Path, max_samples: int = None):
    """NSL-KDD yükle"""
    print("\n📂 NSL-KDD yükleniyor...")

    train_file = data_dir / "KDDTrain+.txt"
    if not train_file.exists():
        # Alternatif isimler
        for alt in ["KDDTrain+.csv", "Train.csv", "train.csv"]:
            alt_path = data_dir / alt
            if alt_path.exists():
                train_file = alt_path
                break

    if not train_file.exists():
        print(f"❌ NSL-KDD bulunamadı: {data_dir}")
        return None, None, None

    columns = [
        "duration",
        "protocol_type",
        "service",
        "flag",
        "src_bytes",
        "dst_bytes",
        "land",
        "wrong_fragment",
        "urgent",
        "hot",
        "num_failed_logins",
        "logged_in",
        "num_compromised",
        "root_shell",
        "su_attempted",
        "num_root",
        "num_file_creations",
        "num_shells",
        "num_access_files",
        "num_outbound_cmds",
        "is_host_login",
        "is_guest_login",
        "count",
        "srv_count",
        "serror_rate",
        "srv_serror_rate",
        "rerror_rate",
        "srv_rerror_rate",
        "same_srv_rate",
        "diff_srv_rate",
        "srv_diff_host_rate",
        "dst_host_count",
        "dst_host_srv_count",
        "dst_host_same_srv_rate",
        "dst_host_diff_srv_rate",
        "dst_host_same_src_port_rate",
        "dst_host_srv_diff_host_rate",
        "dst_host_serror_rate",
        "dst_host_srv_serror_rate",
        "dst_host_rerror_rate",
        "dst_host_srv_rerror_rate",
        "label",
        "difficulty",
    ]

    df = pd.read_csv(train_file, names=columns, header=None)

    # Attack type mapping
    attack_map = {
        "normal": 0,
        "neptune": 1,
        "smurf": 1,
        "pod": 1,
        "teardrop": 1,
        "land": 1,
        "back": 1,  # DoS
        "ipsweep": 2,
        "portsweep": 2,
        "nmap": 2,
        "satan": 2,  # Probe
        "guess_passwd": 3,
        "ftp_write": 3,
        "imap": 3,
        "phf": 3,  # R2L
        "buffer_overflow": 4,
        "loadmodule": 4,
        "rootkit": 4,
        "perl": 4,  # U2R
    }

    df["attack_class"] = df["label"].str.strip().map(lambda x: attack_map.get(x, 5))

    # One-hot encoding
    df_encoded = pd.get_dummies(df, columns=["protocol_type", "service", "flag"])

    # Features ve labels
    feature_cols = [
        c
        for c in df_encoded.columns
        if c not in ["label", "difficulty", "attack_class"]
    ]
    X = df_encoded[feature_cols].values.astype(np.float32)
    y = df["attack_class"].values

    # NaN temizle
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    if max_samples and len(X) > max_samples:
        idx = np.random.choice(len(X), max_samples, replace=False)
        X, y = X[idx], y[idx]

    print(
        f"   ✅ Yüklendi: {len(X):,} kayıt, {X.shape[1]} feature, {len(np.unique(y))} sınıf"
    )

    class_names = ["Normal", "DoS", "Probe", "R2L", "U2R", "Other"]
    return X, y, class_names


def load_bot_iot(data_dir: Path, max_samples: int = 50000):
    """BoT-IoT yükle"""
    print("\n📂 BoT-IoT yükleniyor...")

    csv_files = list(data_dir.glob("*.csv"))
    if not csv_files:
        print(f"❌ BoT-IoT bulunamadı: {data_dir}")
        return None, None, None

    dfs = []
    for f in csv_files[:5]:  # İlk 5 dosya
        try:
            label = "attack" if "mirai" in f.name or "gafgyt" in f.name else "benign"
            df = pd.read_csv(f, low_memory=False, nrows=max_samples // 5)
            df["label"] = 0 if label == "benign" else 1
            dfs.append(df)
        except:
            continue

    if not dfs:
        return None, None, None

    df = pd.concat(dfs, ignore_index=True)

    # Numeric columns only
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if "label" in numeric_cols:
        numeric_cols.remove("label")

    X = df[numeric_cols].values.astype(np.float32)
    y = df["label"].values

    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    print(
        f"   ✅ Yüklendi: {len(X):,} kayıt, {X.shape[1]} feature, {len(np.unique(y))} sınıf"
    )

    class_names = ["Benign", "Attack"]
    return X, y, class_names


def prepare_data(X, y, test_size=0.2, sequence_length=10):
    """Veriyi hazırla"""
    # Normalize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Sequence oluştur (LSTM için)
    # Her örneği (sequence_length, features) şeklinde yeniden düzenle
    n_samples = len(X_scaled) // sequence_length * sequence_length
    X_seq = X_scaled[:n_samples].reshape(-1, sequence_length, X_scaled.shape[1])
    y_seq = y[:n_samples:sequence_length]  # Her sequence için bir label

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X_seq, y_seq, test_size=test_size, random_state=42, stratify=y_seq
    )

    return X_train, X_test, y_train, y_test, scaler


def main():
    parser = argparse.ArgumentParser(description="SSA-LSTMIDS Training")
    parser.add_argument(
        "--dataset", choices=["nsl_kdd", "bot_iot", "all"], default="nsl_kdd"
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=120)
    parser.add_argument("--max_samples", type=int, default=100000)
    parser.add_argument(
        "--optimize", action="store_true", help="SSA optimizasyonu çalıştır"
    )
    parser.add_argument("--save", action="store_true", help="Modeli kaydet")
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("🧠 SSA-LSTMIDS Eğitim - Scientific Reports 2025")
    print("=" * 60)
    print(f"   Dataset: {args.dataset}")
    print(f"   Epochs: {args.epochs}")
    print(f"   Batch Size: {args.batch_size}")
    print(f"   Max Samples: {args.max_samples}")
    print(f"   Optimize: {'✅' if args.optimize else '❌'}")

    data_dir = PROJECT_ROOT / "data" / "raw"
    results = {}

    # Datasets
    datasets = []
    if args.dataset in ["nsl_kdd", "all"]:
        datasets.append(("nsl_kdd", data_dir / "nsl_kdd", load_nsl_kdd))
    if args.dataset in ["bot_iot", "all"]:
        datasets.append(("bot_iot", data_dir / "bot_iot", load_bot_iot))

    for name, path, loader in datasets:
        print(f"\n{'='*60}")
        print(f"📊 Dataset: {name.upper()}")
        print("=" * 60)

        X, y, class_names = loader(path, args.max_samples)
        if X is None:
            continue

        # Prepare
        X_train, X_test, y_train, y_test, scaler = prepare_data(X, y)
        print(f"\n📊 Veri hazırlandı:")
        print(f"   Train: {len(X_train):,}")
        print(f"   Test: {len(X_test):,}")
        print(f"   Shape: {X_train.shape}")

        num_classes = len(np.unique(y))

        # SSA Optimization
        if args.optimize:
            X_train_opt, X_val_opt, y_train_opt, y_val_opt = train_test_split(
                X_train, y_train, test_size=0.2, random_state=42
            )
            best_params, best_score = optimize_with_ssa(
                X_train_opt, y_train_opt, X_val_opt, y_val_opt, num_classes
            )
            print(f"\n🏆 En iyi parametreler: {best_params}")
            print(f"   Score: {best_score*100:.2f}%")

        # Model
        model = SSA_LSTMIDS(
            input_shape=X_train.shape[1:],
            num_classes=num_classes,
            use_paper_params=not args.optimize,
        )
        model.build()

        # Train
        train_results = model.train(
            X_train,
            y_train,
            X_val=X_test,
            y_val=y_test,
            epochs=args.epochs,
            batch_size=args.batch_size,
        )

        # Evaluate
        eval_results = model.evaluate(X_test, y_test)

        print(f"\n📊 Test Sonuçları ({name}):")
        print(f"   Accuracy:  {eval_results['accuracy']*100:.2f}%")
        print(f"   Precision: {eval_results['precision']*100:.2f}%")
        print(f"   Recall:    {eval_results['recall']*100:.2f}%")
        print(f"   F1-Score:  {eval_results['f1_score']*100:.2f}%")

        results[name] = {**train_results, **eval_results, "class_names": class_names}

        # Save model
        if args.save:
            save_path = (
                PROJECT_ROOT
                / "models"
                / f"ssa_lstmids_{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.h5"
            )
            model.save(str(save_path))

    # Sonuçları kaydet
    results_file = PROJECT_ROOT / "models" / "ssa_lstmids_results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n💾 Sonuçlar kaydedildi: {results_file}")
    print("\n✅ Eğitim tamamlandı!")


if __name__ == "__main__":
    main()
