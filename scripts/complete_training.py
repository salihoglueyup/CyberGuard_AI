"""
Complete Training Pipeline - CyberGuard AI
Makaledeki tüm gereksinimleri karşılayan eğitim pipeline'ı

Özellikler:
    - 10-Fold Cross-Validation
    - SSA Hiperparametre Optimizasyonu
    - Tüm modeller (LSTM, BiLSTM, Transformer, GRU)
    - 3 Dataset (NSL-KDD, CICIDS2017, BoT-IoT)
    - SMOTE veri dengeleme

Kullanım:
    python scripts/complete_training.py --dataset all --epochs 50
    python scripts/complete_training.py --optimize --algorithm ssa
"""

import os
import sys
import argparse
import json
import time
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Callable

# Proje yolu
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# TensorFlow
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

try:
    import tensorflow as tf
    from tensorflow import keras

    print(f"✅ TensorFlow {tf.__version__}")
except ImportError:
    print("❌ TensorFlow gerekli!")
    sys.exit(1)

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)


# ============= Dataset Loaders =============


def load_nsl_kdd(data_dir: Path) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """NSL-KDD veri seti"""
    print("\n📂 NSL-KDD yükleniyor...")

    train_file = data_dir / "KDDTrain+.txt"
    if not train_file.exists():
        train_file = data_dir / "KDDTrain+.csv"

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

    label_map = {
        "normal": "Normal",
        "neptune": "DoS",
        "smurf": "DoS",
        "pod": "DoS",
        "teardrop": "DoS",
        "land": "DoS",
        "back": "DoS",
        "apache2": "DoS",
        "udpstorm": "DoS",
        "ipsweep": "Probe",
        "portsweep": "Probe",
        "nmap": "Probe",
        "satan": "Probe",
        "guess_passwd": "R2L",
        "ftp_write": "R2L",
        "imap": "R2L",
        "phf": "R2L",
        "buffer_overflow": "U2R",
        "loadmodule": "U2R",
        "rootkit": "U2R",
        "perl": "U2R",
    }

    df["attack_type"] = df["label"].str.strip().map(lambda x: label_map.get(x, "Other"))

    df_encoded = pd.get_dummies(df, columns=["protocol_type", "service", "flag"])

    le = LabelEncoder()
    y = le.fit_transform(df["attack_type"])
    class_names = le.classes_.tolist()

    feature_cols = [
        c for c in df_encoded.columns if c not in ["label", "difficulty", "attack_type"]
    ]
    X = df_encoded[feature_cols].values.astype(np.float32)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    print(
        f"   Yüklendi: {len(df):,} kayıt, {X.shape[1]} feature, {len(class_names)} sınıf"
    )
    return X, y, class_names


def load_cicids2017(data_dir: Path) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """CICIDS2017 veri seti"""
    print("\n📂 CICIDS2017 yükleniyor...")

    csv_files = list(data_dir.glob("*.csv"))
    if not csv_files:
        print(f"❌ CICIDS2017 bulunamadı: {data_dir}")
        return None, None, None

    dfs = []
    for f in csv_files[:3]:  # İlk 3 dosya (memory için)
        try:
            df = pd.read_csv(f, low_memory=False)
            dfs.append(df)
        except:
            continue

    if not dfs:
        return None, None, None

    df = pd.concat(dfs, ignore_index=True)

    # Label column
    label_col = None
    for col in df.columns:
        if "label" in col.lower():
            label_col = col
            break

    if label_col is None:
        return None, None, None

    le = LabelEncoder()
    y = le.fit_transform(df[label_col].astype(str))
    class_names = le.classes_.tolist()

    # Features
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if label_col in numeric_cols:
        numeric_cols.remove(label_col)

    X = df[numeric_cols].values.astype(np.float32)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    # Limit rows
    if len(X) > 100000:
        idx = np.random.choice(len(X), 100000, replace=False)
        X, y = X[idx], y[idx]

    print(
        f"   Yüklendi: {len(X):,} kayıt, {X.shape[1]} feature, {len(class_names)} sınıf"
    )
    return X, y, class_names


def load_bot_iot(data_dir: Path) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """BoT-IoT / N-BaIoT veri seti"""
    print("\n📂 BoT-IoT/N-BaIoT yükleniyor...")

    csv_files = list(data_dir.glob("**/*.csv"))
    if not csv_files:
        print(f"❌ BoT-IoT bulunamadı: {data_dir}")
        return None, None, None

    dfs = []
    labels = []

    for f in csv_files[:5]:  # İlk 5 dosya
        try:
            df = pd.read_csv(f, low_memory=False)
            # Dosya adından label çıkar
            label = f.stem.split("_")[0] if "_" in f.stem else "benign"
            df["label"] = label
            dfs.append(df)
        except:
            continue

    if not dfs:
        return None, None, None

    df = pd.concat(dfs, ignore_index=True)

    le = LabelEncoder()
    y = le.fit_transform(df["label"].astype(str))
    class_names = le.classes_.tolist()

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    X = df[numeric_cols].values.astype(np.float32)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    if len(X) > 50000:
        idx = np.random.choice(len(X), 50000, replace=False)
        X, y = X[idx], y[idx]

    print(
        f"   Yüklendi: {len(X):,} kayıt, {X.shape[1]} feature, {len(class_names)} sınıf"
    )
    return X, y, class_names


# ============= Model Builders =============


def create_model(model_type: str, input_shape: Tuple, num_classes: int, **kwargs):
    """Model oluştur"""
    if model_type == "lstm":
        from src.network_detection.model import NetworkAnomalyModel

        model = NetworkAnomalyModel(model_type="detection", use_lstm=True)
        model.build_lstm(input_shape)
        return model.lstm_model

    elif model_type == "bilstm":
        from src.network_detection.advanced_model import AdvancedIDSModel

        m = AdvancedIDSModel(
            input_shape=input_shape,
            num_classes=num_classes,
            lstm_units=kwargs.get("lstm_units", 120),
            use_bidirectional=True,
            use_attention=True,
        )
        m.build()
        return m.model

    elif model_type == "transformer":
        from src.network_detection.transformer_model import TransformerIDSModel

        m = TransformerIDSModel(
            input_shape=input_shape,
            num_classes=num_classes,
            d_model=kwargs.get("d_model", 64),
            num_heads=4,
            num_transformer_blocks=2,
        )
        m.build()
        return m.model

    elif model_type == "gru":
        from src.network_detection.gru_model import GRUIDSModel

        m = GRUIDSModel(
            input_shape=input_shape,
            num_classes=num_classes,
            gru_units=kwargs.get("gru_units", 100),
        )
        m.build()
        return m.model

    else:
        raise ValueError(f"Bilinmeyen model: {model_type}")


# ============= Cross-Validation =============


def cross_validate(
    model_type: str,
    X: np.ndarray,
    y: np.ndarray,
    n_folds: int = 10,
    epochs: int = 30,
    batch_size: int = 64,
    use_smote: bool = False,
) -> Dict:
    """
    K-Fold Cross-Validation

    Returns:
        {accuracy: [], precision: [], recall: [], f1: [], mean_*, std_*}
    """
    print(f"\n🔄 {n_folds}-Fold Cross-Validation: {model_type.upper()}")

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

    results = {"accuracy": [], "precision": [], "recall": [], "f1": []}

    num_classes = len(np.unique(y))

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        print(f"\n   Fold {fold + 1}/{n_folds}")

        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # SMOTE
        if use_smote:
            try:
                from imblearn.over_sampling import SMOTE

                smote = SMOTE(random_state=42)
                X_train_flat = X_train.reshape(X_train.shape[0], -1)
                X_train_flat, y_train = smote.fit_resample(X_train_flat, y_train)
                X_train = X_train_flat.reshape(-1, *X_train.shape[1:])
            except:
                pass

        # Normalize
        scaler = StandardScaler()
        X_train_flat = X_train.reshape(X_train.shape[0], -1)
        X_val_flat = X_val.reshape(X_val.shape[0], -1)

        X_train_scaled = scaler.fit_transform(X_train_flat)
        X_val_scaled = scaler.transform(X_val_flat)

        # Reshape for sequence models
        X_train_seq = X_train_scaled.reshape(-1, 1, X_train_scaled.shape[1])
        X_val_seq = X_val_scaled.reshape(-1, 1, X_val_scaled.shape[1])

        # Model oluştur
        model = create_model(model_type, X_train_seq.shape[1:], num_classes)

        # Eğit
        model.fit(
            X_train_seq,
            y_train,
            validation_data=(X_val_seq, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[
                keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)
            ],
            verbose=0,
        )

        # Değerlendir
        y_pred = np.argmax(model.predict(X_val_seq, verbose=0), axis=1)

        acc = accuracy_score(y_val, y_pred)
        prec = precision_score(y_val, y_pred, average="weighted", zero_division=0)
        rec = recall_score(y_val, y_pred, average="weighted", zero_division=0)
        f1 = f1_score(y_val, y_pred, average="weighted", zero_division=0)

        results["accuracy"].append(acc)
        results["precision"].append(prec)
        results["recall"].append(rec)
        results["f1"].append(f1)

        print(f"      Acc: {acc*100:.2f}%, F1: {f1*100:.2f}%")

        # Memory temizle
        del model
        keras.backend.clear_session()

    # Ortalama ve std
    for key in ["accuracy", "precision", "recall", "f1"]:
        results[f"mean_{key}"] = np.mean(results[key])
        results[f"std_{key}"] = np.std(results[key])

    print(
        f"\n   📊 CV Sonuç: {results['mean_accuracy']*100:.2f}% ± {results['std_accuracy']*100:.2f}%"
    )

    return results


# ============= SSA Optimization =============


def run_ssa_optimization(
    X: np.ndarray,
    y: np.ndarray,
    model_type: str = "bilstm",
    max_iterations: int = 15,
    population_size: int = 10,
) -> Dict:
    """SSA ile hiperparametre optimizasyonu"""
    print(f"\n🦠 SSA Optimizasyonu: {model_type.upper()}")

    from src.network_detection.optimizers.ssa import SSAOptimizer

    num_classes = len(np.unique(y))

    # Objective function
    def objective(params: Dict) -> float:
        lstm_units = params.get("lstm_units", 120)
        dropout = params.get("dropout_rate", 0.3)
        lr = params.get("learning_rate", 0.001)

        # Train/val split
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # Normalize + reshape
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train.reshape(X_train.shape[0], -1))
        X_val_scaled = scaler.transform(X_val.reshape(X_val.shape[0], -1))

        X_train_seq = X_train_scaled.reshape(-1, 1, X_train_scaled.shape[1])
        X_val_seq = X_val_scaled.reshape(-1, 1, X_val_scaled.shape[1])

        # Model
        if model_type == "bilstm":
            from src.network_detection.advanced_model import AdvancedIDSModel

            m = AdvancedIDSModel(
                input_shape=X_train_seq.shape[1:],
                num_classes=num_classes,
                lstm_units=lstm_units,
                dropout_rate=dropout,
            )
            m.build()
            model = m.model
        else:
            model = create_model(model_type, X_train_seq.shape[1:], num_classes)

        # Update learning rate
        keras.backend.set_value(model.optimizer.learning_rate, lr)

        # Train
        model.fit(
            X_train_seq,
            y_train,
            validation_data=(X_val_seq, y_val),
            epochs=10,
            batch_size=64,
            callbacks=[
                keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)
            ],
            verbose=0,
        )

        # Evaluate
        y_pred = np.argmax(model.predict(X_val_seq, verbose=0), axis=1)
        score = f1_score(y_val, y_pred, average="weighted", zero_division=0)

        del model
        keras.backend.clear_session()

        return score

    # Search space
    search_space = {
        "lstm_units": (64, 256, "int"),
        "dropout_rate": (0.1, 0.5, "float"),
        "learning_rate": (0.0001, 0.01, "float"),
    }

    # Optimize
    optimizer = SSAOptimizer(
        objective_function=objective,
        search_space=search_space,
        population_size=population_size,
        max_iterations=max_iterations,
        minimize=False,
        verbose=True,
    )

    best_params, best_score = optimizer.optimize()

    return {
        "best_params": best_params,
        "best_score": best_score,
        "history": optimizer.history,
    }


# ============= Main =============


def main():
    parser = argparse.ArgumentParser(description="Complete Training Pipeline")
    parser.add_argument(
        "--dataset",
        choices=["nsl_kdd", "cicids2017", "bot_iot", "all"],
        default="nsl_kdd",
    )
    parser.add_argument(
        "--model",
        choices=["lstm", "bilstm", "transformer", "gru", "all"],
        default="all",
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--folds", type=int, default=10)
    parser.add_argument("--smote", action="store_true")
    parser.add_argument(
        "--optimize", action="store_true", help="SSA optimizasyonu çalıştır"
    )
    parser.add_argument("--algorithm", choices=["ssa", "pso", "jaya"], default="ssa")
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("🧠 CyberGuard AI - Complete Training Pipeline")
    print("=" * 60)
    print(f"   Dataset: {args.dataset}")
    print(f"   Model: {args.model}")
    print(f"   Epochs: {args.epochs}")
    print(f"   CV Folds: {args.folds}")
    print(f"   SMOTE: {'✅' if args.smote else '❌'}")
    print(f"   Optimize: {'✅' if args.optimize else '❌'}")

    # Dataset yükle
    datasets = {}
    data_dir = PROJECT_ROOT / "data" / "raw"

    if args.dataset in ["nsl_kdd", "all"]:
        X, y, names = load_nsl_kdd(data_dir / "nsl_kdd")
        if X is not None:
            datasets["nsl_kdd"] = (X, y, names)

    if args.dataset in ["cicids2017", "all"]:
        X, y, names = load_cicids2017(data_dir / "cicids2017")
        if X is not None:
            datasets["cicids2017"] = (X, y, names)

    if args.dataset in ["bot_iot", "all"]:
        X, y, names = load_bot_iot(data_dir / "bot_iot")
        if X is not None:
            datasets["bot_iot"] = (X, y, names)

    if not datasets:
        print("❌ Hiçbir dataset yüklenemedi!")
        return

    # Modeller
    models = (
        ["lstm", "bilstm", "transformer", "gru"]
        if args.model == "all"
        else [args.model]
    )

    # Sonuçlar
    all_results = {}

    for ds_name, (X, y, class_names) in datasets.items():
        print(f"\n{'='*60}")
        print(f"📊 Dataset: {ds_name.upper()}")
        print(f"{'='*60}")

        all_results[ds_name] = {}

        # SSA Optimizasyon
        if args.optimize:
            opt_results = run_ssa_optimization(X, y, model_type="bilstm")
            all_results[ds_name]["optimization"] = opt_results
            print(f"\n🏆 En iyi parametreler: {opt_results['best_params']}")

        # Cross-validation
        for model_type in models:
            try:
                cv_results = cross_validate(
                    model_type=model_type,
                    X=X,
                    y=y,
                    n_folds=args.folds,
                    epochs=args.epochs,
                    use_smote=args.smote,
                )
                all_results[ds_name][model_type] = cv_results
            except Exception as e:
                print(f"❌ {model_type} eğitimi başarısız: {e}")

    # Sonuçları kaydet
    results_file = PROJECT_ROOT / "models" / "complete_results.json"
    results_file.parent.mkdir(exist_ok=True)

    # Convert numpy to float
    def convert(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [convert(i) for i in obj]
        return obj

    with open(results_file, "w") as f:
        json.dump(convert(all_results), f, indent=2)

    print(f"\n💾 Sonuçlar kaydedildi: {results_file}")

    # Özet tablo
    print("\n" + "=" * 60)
    print("📊 SONUÇ ÖZETİ")
    print("=" * 60)

    for ds_name, results in all_results.items():
        print(f"\n📁 {ds_name.upper()}:")
        for model_name, metrics in results.items():
            if model_name == "optimization":
                continue
            mean_acc = metrics.get("mean_accuracy", 0) * 100
            std_acc = metrics.get("std_accuracy", 0) * 100
            mean_f1 = metrics.get("mean_f1", 0) * 100
            print(
                f"   {model_name:12} | Acc: {mean_acc:.2f}% ± {std_acc:.2f}% | F1: {mean_f1:.2f}%"
            )

    print("\n✅ Eğitim tamamlandı!")


if __name__ == "__main__":
    main()
