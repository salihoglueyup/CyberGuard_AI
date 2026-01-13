"""
Strateji 4: Feature Engineering + Selection
=============================================

Yeni özellikler türeterek ve feature selection ile
daha etkili bir model eğitmeyi hedefliyoruz.

Hedef: %99.88+
Tahmini süre: 45-60 dakika

Kullanım:
    python scripts/strategy4_feature_engineering.py
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
from tensorflow import keras

print(f"✅ TensorFlow {tf.__version__}")

DATA_DIR = PROJECT_ROOT / "data" / "raw" / "cicids2017_full"
MODELS_DIR = PROJECT_ROOT / "models"


def load_and_engineer_features(sample_size=500_000):
    """Feature engineering ile veri hazırla"""
    print("\n📊 Loading and Engineering Features...")

    csv_files = [
        "Monday-WorkingHours.pcap_ISCX.csv",
        "Tuesday-WorkingHours.pcap_ISCX.csv",
        "Wednesday-workingHours.pcap_ISCX.csv",
        "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv",
        "Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv",
        "Friday-WorkingHours-Morning.pcap_ISCX.csv",
        "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv",
        "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv",
    ]

    dfs = []
    for fname in csv_files:
        fpath = DATA_DIR / fname
        if fpath.exists():
            df = pd.read_csv(fpath, low_memory=False)
            dfs.append(df)

    df = pd.concat(dfs, ignore_index=True)
    print(f"   Full: {len(df):,}")

    if len(df) > sample_size:
        df = df.sample(n=sample_size, random_state=42)
    print(f"   Sampled: {len(df):,}")

    label_col = " Label"

    # Feature Engineering
    print("\n   🔧 Engineering Features...")

    # Orijinal feature'lar
    orig_cols = [
        c
        for c in df.columns
        if c
        not in [
            label_col,
            "Timestamp",
            "Flow ID",
            "Source IP",
            "Destination IP",
            "Source Port",
            "Destination Port",
        ]
    ]

    # Tüm sütunları numeric yap
    for col in orig_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.replace([np.inf, -np.inf], np.nan)
    df[orig_cols] = df[orig_cols].fillna(0)

    # Yeni özellikler türet
    print("   Creating derived features...")

    # Packet-based features
    if "Total Fwd Packets" in df.columns and "Total Backward Packets" in df.columns:
        df["packet_ratio"] = df["Total Fwd Packets"] / (
            df["Total Backward Packets"] + 1
        )
        df["total_packets"] = df["Total Fwd Packets"] + df["Total Backward Packets"]

    # Length-based features
    if (
        "Total Length of Fwd Packets" in df.columns
        and "Total Length of Bwd Packets" in df.columns
    ):
        df["length_ratio"] = df["Total Length of Fwd Packets"] / (
            df["Total Length of Bwd Packets"] + 1
        )
        df["total_length"] = (
            df["Total Length of Fwd Packets"] + df["Total Length of Bwd Packets"]
        )

    # Flow duration features
    if "Flow Duration" in df.columns:
        df["flow_duration_log"] = np.log1p(df["Flow Duration"])

    # Bytes per packet
    if "total_packets" in df.columns and "total_length" in df.columns:
        df["bytes_per_packet"] = df["total_length"] / (df["total_packets"] + 1)

    # Flag features
    if "Fwd PSH Flags" in df.columns and "Bwd PSH Flags" in df.columns:
        df["psh_ratio"] = df["Fwd PSH Flags"] / (df["Bwd PSH Flags"] + 1)

    if "FIN Flag Count" in df.columns and "SYN Flag Count" in df.columns:
        df["fin_syn_ratio"] = df["FIN Flag Count"] / (df["SYN Flag Count"] + 1)

    # IAT features
    for prefix in ["Flow IAT", "Fwd IAT", "Bwd IAT"]:
        if f"{prefix} Mean" in df.columns and f"{prefix} Std" in df.columns:
            df[f'{prefix.lower().replace(" ", "_")}_cv'] = df[f"{prefix} Std"] / (
                df[f"{prefix} Mean"] + 1
            )

    # Tüm feature sütunları
    feature_cols = [
        c
        for c in df.columns
        if c
        not in [
            label_col,
            "Timestamp",
            "Flow ID",
            "Source IP",
            "Destination IP",
            "Source Port",
            "Destination Port",
        ]
    ]

    print(f"   Total features: {len(feature_cols)}")

    X = df[feature_cols].values.astype(np.float32)
    X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)

    le = LabelEncoder()
    y = le.fit_transform(df[label_col])
    num_classes = len(le.classes_)

    # Feature Selection
    print("\n   📊 Feature Selection (top 60)...")
    selector = SelectKBest(mutual_info_classif, k=min(60, X.shape[1]))
    X_selected = selector.fit_transform(X, y)

    selected_features = np.array(feature_cols)[selector.get_support()]
    print(f"   Selected: {len(selected_features)} features")
    print(f"   Top 10: {list(selected_features[:10])}")

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X_selected, y, test_size=0.2, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.1, random_state=42, stratify=y_train
    )

    # Scale
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    # Reshape
    timesteps = 10
    if X_train.shape[1] % timesteps != 0:
        pad = timesteps - (X_train.shape[1] % timesteps)
        X_train = np.pad(X_train, ((0, 0), (0, pad)), mode="constant")
        X_val = np.pad(X_val, ((0, 0), (0, pad)), mode="constant")
        X_test = np.pad(X_test, ((0, 0), (0, pad)), mode="constant")

    fstep = X_train.shape[1] // timesteps
    X_train = X_train.reshape(-1, timesteps, fstep)
    X_val = X_val.reshape(-1, timesteps, fstep)
    X_test = X_test.reshape(-1, timesteps, fstep)

    print(f"\n   Train: {X_train.shape}")
    print(f"   Val: {X_val.shape}")
    print(f"   Test: {X_test.shape}")

    return X_train, y_train, X_val, y_val, X_test, y_test, num_classes


def build_model(input_shape, num_classes):
    """Model oluştur"""
    l2 = keras.regularizers.l2(0.0002)

    inputs = keras.layers.Input(shape=input_shape)

    x = keras.layers.Conv1D(
        128, 3, padding="same", activation="relu", kernel_regularizer=l2
    )(inputs)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Conv1D(
        128, 3, padding="same", activation="relu", kernel_regularizer=l2
    )(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.MaxPooling1D(2, padding="same")(x)

    x = keras.layers.Bidirectional(
        keras.layers.LSTM(128, return_sequences=True, dropout=0.15)
    )(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Bidirectional(
        keras.layers.LSTM(64, return_sequences=False, dropout=0.15)
    )(x)

    x = keras.layers.Dense(256, activation="relu", kernel_regularizer=l2)(x)
    x = keras.layers.Dropout(0.3)(x)
    x = keras.layers.Dense(128, activation="relu")(x)
    x = keras.layers.Dropout(0.2)(x)

    outputs = keras.layers.Dense(num_classes, activation="softmax")(x)

    return keras.Model(inputs, outputs, name="FeatureEngineered_LSTMIDS")


def main():
    print("\n" + "=" * 80)
    print("🎯 STRATEJI 4: FEATURE ENGINEERING + SELECTION")
    print("=" * 80)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\nHedef: %99.88+")

    # Load data
    X_train, y_train, X_val, y_val, X_test, y_test, num_classes = (
        load_and_engineer_features(500_000)
    )

    # Build model
    print("\n" + "=" * 60)
    print("🧠 Building Model")
    print("=" * 60)

    model = build_model(X_train.shape[1:], num_classes)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    print(f"   Parameters: {model.count_params():,}")

    # Train
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=10, restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_accuracy", factor=0.5, patience=5, min_lr=1e-7
        ),
    ]

    print("\n🚀 Training...")
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=50,
        batch_size=256,
        callbacks=callbacks,
        verbose=1,
    )

    best_val_acc = max(history.history["val_accuracy"])
    print(f"\n   Best Val Accuracy: {best_val_acc*100:.4f}%")

    # Evaluate
    print("\n" + "=" * 60)
    print("📊 TEST RESULTS")
    print("=" * 60)

    y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    recall = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

    print(f"\n   ✅ Accuracy:  {accuracy*100:.4f}%")
    print(f"   ✅ Precision: {precision*100:.4f}%")
    print(f"   ✅ Recall:    {recall*100:.4f}%")
    print(f"   ✅ F1-Score:  {f1:.6f}")

    print(f"\n   📄 Hedef: 99.88%")
    print(f"   📊 Sonuç: {accuracy*100:.4f}%")

    if accuracy >= 0.9988:
        print("\n   🎉🎉🎉 HEDEF BAŞARILDI! 🎉🎉🎉")
    else:
        print(f"\n   Fark: {(0.9988 - accuracy)*100:+.2f}%")

    # Save
    result = {
        "strategy": "Feature Engineering + Selection",
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "best_val_accuracy": float(best_val_acc),
        "created_at": datetime.now().isoformat(),
    }

    results_path = MODELS_DIR / "strategy4_feature_results.json"
    with open(results_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\n💾 Saved: {results_path}")

    print("\n" + "=" * 80)
    print(f"✅ STRATEJI 4 TAMAMLANDI: {accuracy*100:.4f}%")
    print("=" * 80)

    return accuracy


if __name__ == "__main__":
    main()
