"""
BoT-IoT Dataset Training Script
================================

Dataset: BoT-IoT (Botnet attacks in IoT environments)
Attack Types: Gafgyt (combo, junk, scan, tcp, udp), Mirai (ack, scan, syn, udp, udpplain)
Hedef: %99.99 accuracy (makaledeki)

Kullanım:
    python scripts/train_botiot.py
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from collections import Counter
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
)

# Proje yolu
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
from tensorflow import keras

print(f"✅ TensorFlow {tf.__version__}")

# Paths
DATA_DIR = PROJECT_ROOT / "data" / "raw" / "bot_iot"
MODELS_DIR = PROJECT_ROOT / "models"
REGISTRY_PATH = MODELS_DIR / "model_registry.json"


def load_botiot_data(sample_per_file=5000, max_files=30):
    """BoT-IoT veri yükle"""
    print("\n" + "=" * 60)
    print("📊 Loading BoT-IoT Dataset")
    print("=" * 60)

    # CSV dosyalarını listele
    csv_files = sorted(DATA_DIR.glob("*.csv"))
    csv_files = [
        f
        for f in csv_files
        if f.name
        not in ["data_summary.csv", "device_info.csv", "features.csv", "README.md"]
    ]

    print(f"   Found {len(csv_files)} data files")

    dfs = []

    for i, f in enumerate(csv_files[:max_files]):
        try:
            # Dosya adından label çıkar
            name = f.stem  # e.g., "1.benign", "1.gafgyt.combo"
            parts = name.split(".")

            if len(parts) >= 2:
                if parts[1] == "benign":
                    label = "benign"
                else:
                    # gafgyt.combo -> gafgyt, mirai.ack -> mirai
                    label = parts[1]  # ana kategori
            else:
                label = "unknown"

            # Veri yükle
            df = pd.read_csv(f, low_memory=False, nrows=sample_per_file)
            df["label"] = label
            dfs.append(df)

            if (i + 1) % 10 == 0:
                print(f"   Loaded {i+1}/{min(len(csv_files), max_files)} files...")

        except Exception as e:
            print(f"   ⚠️ Error loading {f.name}: {e}")

    if not dfs:
        print("   ❌ No data loaded!")
        return None, None, None, None, None, None

    df = pd.concat(dfs, ignore_index=True)
    print(f"\n   Total samples: {len(df):,}")

    # Label distribution
    print(f"   Labels: {dict(Counter(df['label']))}")

    # Feature columns - sayısal olanları seç
    label_col = "label"
    feature_cols = [
        c
        for c in df.columns
        if c != label_col and df[c].dtype in ["int64", "float64", "int32", "float32"]
    ]

    print(f"   Numeric features: {len(feature_cols)}")

    if len(feature_cols) < 5:
        print("   ❌ Not enough numeric features!")
        print(f"   Columns: {list(df.columns[:20])}")
        return None, None, None, None, None, None

    X = df[feature_cols].values.astype(np.float32)

    # NaN/Inf temizle
    X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)

    # Label encode
    le = LabelEncoder()
    y = le.fit_transform(df[label_col])

    print(f"   Classes: {list(le.classes_)}")

    # Stratified split
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
    except ValueError:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.1, random_state=42
    )

    print(f"\n   Train: {len(X_train):,}")
    print(f"   Val: {len(X_val):,}")
    print(f"   Test: {len(X_test):,}")

    # Scale
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    # Reshape for LSTM
    timesteps = 10
    n_features = X_train.shape[1]
    if n_features % timesteps != 0:
        pad_size = timesteps - (n_features % timesteps)
        X_train = np.pad(X_train, ((0, 0), (0, pad_size)), mode="constant")
        X_val = np.pad(X_val, ((0, 0), (0, pad_size)), mode="constant")
        X_test = np.pad(X_test, ((0, 0), (0, pad_size)), mode="constant")

    features_per_step = X_train.shape[1] // timesteps
    X_train = X_train.reshape(X_train.shape[0], timesteps, features_per_step)
    X_val = X_val.reshape(X_val.shape[0], timesteps, features_per_step)
    X_test = X_test.reshape(X_test.shape[0], timesteps, features_per_step)

    print(f"   X_train shape: {X_train.shape}")

    return (
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test,
        len(le.classes_),
        list(le.classes_),
    )


def build_botiot_model(input_shape, num_classes):
    """BoT-IoT için optimize edilmiş model"""
    print("\n" + "=" * 60)
    print("🧠 Building BoT-IoT Model")
    print("=" * 60)

    l2_reg = keras.regularizers.l2(0.001)

    inputs = keras.layers.Input(shape=input_shape)

    # Conv1D
    x = keras.layers.Conv1D(
        64, 3, padding="same", activation="relu", kernel_regularizer=l2_reg
    )(inputs)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Conv1D(
        32, 3, padding="same", activation="relu", kernel_regularizer=l2_reg
    )(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.MaxPooling1D(2, padding="same")(x)

    # BiLSTM
    x = keras.layers.Bidirectional(
        keras.layers.LSTM(128, return_sequences=True, dropout=0.2)
    )(x)
    x = keras.layers.Bidirectional(
        keras.layers.LSTM(64, return_sequences=False, dropout=0.2)
    )(x)

    # Dense
    x = keras.layers.Dense(256, activation="relu", kernel_regularizer=l2_reg)(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Dropout(0.4)(x)
    x = keras.layers.Dense(128, activation="relu", kernel_regularizer=l2_reg)(x)
    x = keras.layers.Dropout(0.3)(x)

    outputs = keras.layers.Dense(num_classes, activation="softmax")(x)

    model = keras.Model(inputs, outputs, name="BoT-IoT_SSA_LSTMIDS")

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    print(f"   Total params: {model.count_params():,}")

    return model


def train_and_evaluate(
    model, X_train, y_train, X_val, y_val, X_test, y_test, class_names
):
    """Eğit ve değerlendir"""
    print("\n" + "=" * 60)
    print("🏋️ Training BoT-IoT Model")
    print("=" * 60)

    EPOCHS = 100
    BATCH_SIZE = 256
    PATIENCE = 15

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=PATIENCE,
            restore_best_weights=True,
            mode="max",
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_accuracy", factor=0.5, patience=5, min_lr=1e-6, mode="max"
        ),
    ]

    print("🚀 Starting training...")
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=1,
    )

    # Evaluate
    print("\n" + "=" * 60)
    print("📊 Test Results")
    print("=" * 60)

    y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    recall = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

    print(f"\n   ✅ Accuracy:  {accuracy*100:.2f}%")
    print(f"   ✅ Precision: {precision*100:.2f}%")
    print(f"   ✅ Recall:    {recall*100:.2f}%")
    print(f"   ✅ F1-Score:  {f1:.4f}")

    print(f"\n   📄 Makaledeki hedef: 99.99%")
    print(f"   📊 Bizim sonuç:      {accuracy*100:.2f}%")

    if accuracy >= 0.999:
        print("\n   🎉 HEDEF BAŞARILDI!")
    elif accuracy >= 0.99:
        print("\n   🔥 Çok yakın!")

    print("\n" + "=" * 60)
    print("📋 Classification Report")
    print("=" * 60)
    print(classification_report(y_test, y_pred, target_names=class_names))

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "epochs": len(history.history["loss"]),
        "best_val_accuracy": max(history.history["val_accuracy"]),
    }, model


def save_model(model, results):
    """Modeli kaydet"""
    MODELS_DIR.mkdir(exist_ok=True)

    model_id = f"ssa_lstmids_botiot_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    model_path = MODELS_DIR / f"{model_id}.keras"

    model.save(model_path)
    print(f"\n💾 Model saved: {model_path}")

    # Registry güncelle
    if REGISTRY_PATH.exists():
        with open(REGISTRY_PATH, "r") as f:
            registry = json.load(f)
    else:
        registry = {"models": []}

    entry = {
        "id": model_id,
        "name": "SSA-LSTMIDS_BoT-IoT",
        "model_type": "ssa_lstmids",
        "dataset": "bot_iot",
        "status": "trained",
        "framework": "tensorflow",
        "path": str(model_path),
        "metrics": {
            "accuracy": results["accuracy"],
            "precision": results["precision"],
            "recall": results["recall"],
            "f1_score": results["f1_score"],
        },
        "training_config": {
            "epochs": results["epochs"],
            "best_val_accuracy": results["best_val_accuracy"],
        },
        "created_at": datetime.now().isoformat(),
    }

    registry["models"].append(entry)

    with open(REGISTRY_PATH, "w") as f:
        json.dump(registry, f, indent=2)

    print("📋 Registry updated")


def main():
    print("\n" + "=" * 70)
    print("🎓 BoT-IoT SSA-LSTMIDS Training")
    print("=" * 70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Load data
    result = load_botiot_data(sample_per_file=10000, max_files=30)

    if result[0] is None:
        print("❌ Failed to load data!")
        return

    X_train, y_train, X_val, y_val, X_test, y_test, num_classes, class_names = result

    # Build model
    model = build_botiot_model(X_train.shape[1:], num_classes)
    model.summary()

    # Train
    results, trained_model = train_and_evaluate(
        model, X_train, y_train, X_val, y_val, X_test, y_test, class_names
    )

    # Save
    save_model(trained_model, results)

    print("\n" + "=" * 70)
    print("✅ Training Complete!")
    print("=" * 70)
    print(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
