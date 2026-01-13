"""
Strateji 5: Full Dataset + 200 Epoch
=====================================

2.83M sample ile uzun eğitim yaparak maksimum accuracy hedefliyoruz.

Hedef: %99.88+
Tahmini süre: 2-3 saat (CPU)

Kullanım:
    python scripts/strategy5_full_dataset.py
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
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
from tensorflow import keras

print(f"✅ TensorFlow {tf.__version__}")

DATA_DIR = PROJECT_ROOT / "data" / "raw" / "cicids2017_full"
MODELS_DIR = PROJECT_ROOT / "models"

# Sürümü hızlandırmak için 1M sample kullanıyoruz (2.83M yerine)
SAMPLE_SIZE = 1_000_000
EPOCHS = 100  # 200 yerine 100, early stopping zaten durduracak
BATCH_SIZE = 512  # Daha büyük batch = daha hızlı


def load_data():
    """Full dataset yükle (1M sample)"""
    print(f"\n📊 Loading {SAMPLE_SIZE:,} samples from Full Dataset...")

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
            print(f"   ✅ {fname}: {len(df):,}")

    df = pd.concat(dfs, ignore_index=True)
    print(f"\n   Full: {len(df):,}")

    if len(df) > SAMPLE_SIZE:
        df = df.sample(n=SAMPLE_SIZE, random_state=42)
    print(f"   Sampled: {len(df):,}")

    label_col = " Label"
    exclude_cols = [label_col, "Timestamp", "Flow ID", "Source IP", "Destination IP"]
    feature_cols = [c for c in df.columns if c not in exclude_cols]

    for col in feature_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.replace([np.inf, -np.inf], np.nan)
    df[feature_cols] = df[feature_cols].fillna(0)

    X = df[feature_cols].values.astype(np.float32)
    X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)

    le = LabelEncoder()
    y = le.fit_transform(df[label_col])
    num_classes = len(le.classes_)

    print(f"   Features: {len(feature_cols)}")
    print(f"   Classes: {num_classes}")

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
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


def build_deep_model(input_shape, num_classes):
    """Deep model for maximum accuracy"""
    l2 = keras.regularizers.l2(0.0001)

    inputs = keras.layers.Input(shape=input_shape)

    # Conv blocks
    x = keras.layers.Conv1D(
        128, 3, padding="same", activation="relu", kernel_regularizer=l2
    )(inputs)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Conv1D(
        128, 3, padding="same", activation="relu", kernel_regularizer=l2
    )(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.MaxPooling1D(2, padding="same")(x)

    x = keras.layers.Conv1D(
        256, 3, padding="same", activation="relu", kernel_regularizer=l2
    )(x)
    x = keras.layers.BatchNormalization()(x)

    # BiLSTM with attention
    x = keras.layers.Bidirectional(
        keras.layers.LSTM(256, return_sequences=True, dropout=0.1)
    )(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Bidirectional(
        keras.layers.LSTM(128, return_sequences=False, dropout=0.1)
    )(x)
    x = keras.layers.BatchNormalization()(x)

    # Dense
    x = keras.layers.Dense(512, activation="relu", kernel_regularizer=l2)(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Dropout(0.3)(x)
    x = keras.layers.Dense(256, activation="relu", kernel_regularizer=l2)(x)
    x = keras.layers.Dropout(0.2)(x)

    outputs = keras.layers.Dense(num_classes, activation="softmax")(x)

    return keras.Model(inputs, outputs, name="Deep_Full_LSTMIDS")


def main():
    print("\n" + "=" * 80)
    print("🎯 STRATEJI 5: FULL DATASET + LONG TRAINING")
    print("=" * 80)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\nHedef: %99.88+")

    # Load data
    X_train, y_train, X_val, y_val, X_test, y_test, num_classes = load_data()

    # Build model
    print("\n" + "=" * 60)
    print("🧠 Building Deep Model")
    print("=" * 60)

    model = build_deep_model(X_train.shape[1:], num_classes)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    print(f"   Parameters: {model.count_params():,}")
    print(f"   Epochs: {EPOCHS}")
    print(f"   Batch size: {BATCH_SIZE}")

    # Train
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=15, restore_best_weights=True
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
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
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
    model_path = (
        MODELS_DIR / f"strategy5_full_{datetime.now().strftime('%Y%m%d_%H%M%S')}.keras"
    )
    model.save(model_path)
    print(f"\n💾 Saved: {model_path}")

    result = {
        "strategy": "Full Dataset + Long Training",
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "best_val_accuracy": float(best_val_acc),
        "samples": SAMPLE_SIZE,
        "epochs_run": len(history.history["loss"]),
        "created_at": datetime.now().isoformat(),
    }

    results_path = MODELS_DIR / "strategy5_full_results.json"
    with open(results_path, "w") as f:
        json.dump(result, f, indent=2)

    print("\n" + "=" * 80)
    print(f"✅ STRATEJI 5 TAMAMLANDI: {accuracy*100:.4f}%")
    print("=" * 80)

    return accuracy


if __name__ == "__main__":
    main()
