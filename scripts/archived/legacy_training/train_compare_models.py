"""
Model Comparison Training Script
CyberGuard AI - Tüm modelleri NSL-KDD ile karşılaştırmalı eğit

Modeller:
    1. LSTM (Mevcut - Makale mimarisi)
    2. BiLSTM + Attention (Geliştirilmiş)
    3. Transformer (Modern yaklaşım)

Kullanım:
    python scripts/train_compare_models.py
    python scripts/train_compare_models.py --epochs 50 --model all
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
from typing import Dict, List, Tuple

# Proje yolunu ekle
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# TensorFlow
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # TF log seviyesi

try:
    import tensorflow as tf
    from tensorflow import keras

    print(f"✅ TensorFlow {tf.__version__}")
except ImportError:
    print("❌ TensorFlow gerekli!")
    sys.exit(1)

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix


def load_nsl_kdd(data_dir: Path) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    NSL-KDD veri setini yükle

    Returns:
        (X, y, class_names)
    """
    print("\n📂 NSL-KDD veri seti yükleniyor...")

    train_file = data_dir / "KDDTrain+.txt"

    if not train_file.exists():
        # CSV olarak dene
        train_file = data_dir / "KDDTrain+.csv"

    if not train_file.exists():
        raise FileNotFoundError(f"NSL-KDD dosyası bulunamadı: {data_dir}")

    # Kolon isimleri (NSL-KDD)
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

    # Veri yükle
    df = pd.read_csv(train_file, names=columns, header=None)
    print(f"   Yüklenen: {len(df):,} kayıt")

    # Label'ları basitleştir
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
        "processtable": "DoS",
        "mailbomb": "DoS",
        "ipsweep": "Probe",
        "portsweep": "Probe",
        "nmap": "Probe",
        "satan": "Probe",
        "mscan": "Probe",
        "saint": "Probe",
        "guess_passwd": "R2L",
        "ftp_write": "R2L",
        "imap": "R2L",
        "phf": "R2L",
        "multihop": "R2L",
        "warezmaster": "R2L",
        "warezclient": "R2L",
        "spy": "R2L",
        "xlock": "R2L",
        "xsnoop": "R2L",
        "snmpguess": "R2L",
        "snmpgetattack": "R2L",
        "httptunnel": "R2L",
        "sendmail": "R2L",
        "named": "R2L",
        "worm": "R2L",
        "buffer_overflow": "U2R",
        "loadmodule": "U2R",
        "rootkit": "U2R",
        "perl": "U2R",
        "sqlattack": "U2R",
        "xterm": "U2R",
        "ps": "U2R",
    }

    df["attack_type"] = df["label"].str.strip().map(lambda x: label_map.get(x, "Other"))

    # Kategorik değişkenleri encode et
    categorical_cols = ["protocol_type", "service", "flag"]
    df_encoded = pd.get_dummies(df, columns=categorical_cols)

    # Label encoding
    le = LabelEncoder()
    y = le.fit_transform(df["attack_type"])
    class_names = le.classes_.tolist()

    # Feature'ları al
    feature_cols = [
        c for c in df_encoded.columns if c not in ["label", "difficulty", "attack_type"]
    ]
    X = df_encoded[feature_cols].values.astype(np.float32)

    # NaN kontrolü
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    print(f"   Features: {X.shape[1]}")
    print(f"   Classes: {class_names}")
    print(f"   Dağılım: {dict(zip(*np.unique(y, return_counts=True)))}")

    return X, y, class_names


def prepare_sequences(X: np.ndarray, sequence_length: int = 10) -> np.ndarray:
    """
    2D veriyi 3D sequence formatına dönüştür

    LSTM/Transformer modelleri (batch, sequence, features) bekler
    """
    n_samples = X.shape[0]
    n_features = X.shape[1]

    # Sequence = sliding window
    if n_samples < sequence_length:
        # Veri yetersiz, padding
        X_seq = np.zeros((1, sequence_length, n_features))
        X_seq[0, :n_samples, :] = X
        return X_seq

    # Her örneği tek sequence olarak al (feature'ları split et)
    # Ya da basitçe (batch, 1, features) yap
    X_seq = X.reshape(n_samples, 1, n_features)

    # Sequence length > 1 için padding
    if sequence_length > 1:
        X_padded = np.zeros((n_samples, sequence_length, n_features))
        for i in range(n_samples):
            # Son sequence_length örneği al (ya da padding)
            start = max(0, i - sequence_length + 1)
            seq = X[start : i + 1]
            X_padded[i, -len(seq) :, :] = seq
        return X_padded

    return X_seq


def train_model(
    model_type: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    num_classes: int,
    epochs: int = 50,
    batch_size: int = 64,
) -> Tuple[object, Dict]:
    """
    Belirtilen modeli eğit

    Args:
        model_type: "lstm", "bilstm_attention", "transformer"

    Returns:
        (model, results)
    """
    print(f"\n{'='*60}")
    print(f"🏋️ {model_type.upper()} Modeli Eğitiliyor")
    print(f"{'='*60}")

    input_shape = X_train.shape[1:]

    if model_type == "lstm":
        from src.network_detection.model import NetworkAnomalyModel

        model = NetworkAnomalyModel(
            model_type="detection", use_lstm=True, anomaly_threshold=0.8
        )
        model.build_lstm(input_shape)

        # Train
        start = time.time()
        history = model.lstm_model.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[
                keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
                keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=3),
            ],
            verbose=1,
        )
        elapsed = time.time() - start

        # Evaluate
        loss, acc = model.lstm_model.evaluate(X_val, y_val, verbose=0)

        results = {
            "accuracy": float(acc),
            "loss": float(loss),
            "train_time": elapsed,
            "epochs": len(history.history["accuracy"]),
        }

        return model.lstm_model, results

    elif model_type == "bilstm_attention":
        from src.network_detection.advanced_model import AdvancedIDSModel

        model = AdvancedIDSModel(
            input_shape=input_shape,
            num_classes=num_classes,
            lstm_units=120,
            use_bidirectional=True,
            use_attention=True,
        )
        model.build()

        start = time.time()
        results = model.train(
            X_train,
            y_train,
            X_val=X_val,
            y_val=y_val,
            epochs=epochs,
            batch_size=batch_size,
            patience=5,
        )
        elapsed = time.time() - start

        results["train_time"] = elapsed
        results["accuracy"] = results.get(
            "final_val_accuracy", results["final_accuracy"]
        )

        return model.model, results

    elif model_type == "transformer":
        from src.network_detection.transformer_model import TransformerIDSModel

        model = TransformerIDSModel(
            input_shape=input_shape,
            num_classes=num_classes,
            d_model=64,
            num_heads=4,
            num_transformer_blocks=2,
        )
        model.build()

        start = time.time()
        results = model.train(
            X_train,
            y_train,
            X_val=X_val,
            y_val=y_val,
            epochs=epochs,
            batch_size=batch_size,
            patience=5,
        )
        elapsed = time.time() - start

        results["train_time"] = elapsed
        results["accuracy"] = results.get(
            "final_val_accuracy", results["final_accuracy"]
        )

        return model.model, results

    else:
        raise ValueError(f"Bilinmeyen model tipi: {model_type}")


def main():
    parser = argparse.ArgumentParser(description="Model Karşılaştırma Eğitimi")
    parser.add_argument(
        "--model",
        choices=["all", "lstm", "bilstm", "transformer"],
        default="all",
        help="Eğitilecek model",
    )
    parser.add_argument("--epochs", type=int, default=30, help="Epoch sayısı")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch boyutu")
    parser.add_argument(
        "--sequence_length", type=int, default=1, help="Sequence uzunluğu"
    )
    parser.add_argument("--smote", action="store_true", help="SMOTE kullan")
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("🧠 CyberGuard AI - Model Karşılaştırma Eğitimi")
    print("=" * 60)
    print(f"   Epochs: {args.epochs}")
    print(f"   Batch size: {args.batch_size}")
    print(f"   SMOTE: {'✅' if args.smote else '❌'}")

    # Veri yükle
    data_dir = PROJECT_ROOT / "data" / "raw" / "nsl_kdd"
    X, y, class_names = load_nsl_kdd(data_dir)

    # Normalize
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Train/Val/Test split
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )

    print(f"\n📊 Veri Dağılımı:")
    print(f"   Train: {len(X_train):,}")
    print(f"   Val:   {len(X_val):,}")
    print(f"   Test:  {len(X_test):,}")

    # SMOTE (opsiyonel)
    if args.smote:
        from src.network_detection.preprocessing.smote import DataBalancer

        balancer = DataBalancer(method="smote")
        X_train, y_train = balancer.fit_resample(X_train, y_train)

    # Sequence formatına dönüştür
    X_train_seq = prepare_sequences(X_train, args.sequence_length)
    X_val_seq = prepare_sequences(X_val, args.sequence_length)
    X_test_seq = prepare_sequences(X_test, args.sequence_length)

    print(f"\n📐 Sequence Shape: {X_train_seq.shape}")

    # Modelleri seç
    if args.model == "all":
        models_to_train = ["lstm", "bilstm_attention", "transformer"]
    elif args.model == "bilstm":
        models_to_train = ["bilstm_attention"]
    else:
        models_to_train = [args.model]

    # Sonuçlar
    all_results = {}

    for model_type in models_to_train:
        try:
            model, results = train_model(
                model_type=model_type,
                X_train=X_train_seq,
                y_train=y_train,
                X_val=X_val_seq,
                y_val=y_val,
                num_classes=len(class_names),
                epochs=args.epochs,
                batch_size=args.batch_size,
            )

            all_results[model_type] = results

            # Test değerlendirmesi
            y_pred = np.argmax(model.predict(X_test_seq, verbose=0), axis=1)
            report = classification_report(
                y_test, y_pred, target_names=class_names, output_dict=True
            )

            all_results[model_type]["test_accuracy"] = report["accuracy"]
            all_results[model_type]["test_f1"] = report["weighted avg"]["f1-score"]

        except Exception as e:
            print(f"❌ {model_type} eğitimi başarısız: {e}")
            import traceback

            traceback.print_exc()

    # Sonuç özeti
    print("\n" + "=" * 60)
    print("📊 SONUÇ KARŞILAŞTIRMASI")
    print("=" * 60)

    for model_name, results in all_results.items():
        print(f"\n🔹 {model_name.upper()}:")
        print(f"   Val Accuracy:  {results.get('accuracy', 0)*100:.2f}%")
        print(f"   Test Accuracy: {results.get('test_accuracy', 0)*100:.2f}%")
        print(f"   Test F1-Score: {results.get('test_f1', 0)*100:.2f}%")
        print(f"   Train Time:    {results.get('train_time', 0):.1f}s")
        print(f"   Epochs:        {results.get('epochs', 0)}")

    # Sonuçları kaydet
    results_file = PROJECT_ROOT / "models" / "comparison_results.json"
    results_file.parent.mkdir(exist_ok=True)

    with open(results_file, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n💾 Sonuçlar kaydedildi: {results_file}")

    print("\n✅ Eğitim tamamlandı!")


if __name__ == "__main__":
    main()
