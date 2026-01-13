"""
CICIDS2017 Fast Training - 500K Stratified Sample
===================================================

Makaleyi geçmek için optimize edilmiş hızlı eğitim:
- 500K stratified sample (her sınıftan orantılı)
- Agresif learning rate
- Büyük batch size (256)
- Early stopping (patience=10)

Hedef: %99.88+
Tahmini süre: 30-40 dakika

Kullanım:
    python scripts/train_cicids_fast.py
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

import tensorflow as tf
from tensorflow import keras

print(f"✅ TensorFlow {tf.__version__}")
print(f"✅ GPU: {tf.config.list_physical_devices('GPU')}")

DATA_DIR = PROJECT_ROOT / "data" / "raw" / "cicids2017_full"
MODELS_DIR = PROJECT_ROOT / "models"
REGISTRY_PATH = MODELS_DIR / "model_registry.json"

# === CONFIGURATION ===
SAMPLE_SIZE = 500_000  # 500K samples
BATCH_SIZE = 256
MAX_EPOCHS = 50
PATIENCE = 10
LEARNING_RATE = 0.002  # Aggressive


def load_stratified_sample():
    """Load 500K stratified sample from full CICIDS2017"""
    print("\n" + "=" * 70)
    print(f"📊 Loading {SAMPLE_SIZE:,} Stratified Samples from CICIDS2017")
    print("=" * 70)

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
    print(f"\n   📊 Full dataset: {len(df):,}")

    # Label column
    label_col = " Label"

    # Stratified sampling - her sınıftan orantılı sample al
    print(f"\n   🔄 Stratified sampling ({SAMPLE_SIZE:,})...")

    # Önce tüm nadir sınıfları al
    label_counts = df[label_col].value_counts()
    rare_threshold = 1000

    sampled_dfs = []
    remaining_budget = SAMPLE_SIZE

    # Nadir sınıfları tamamen al
    for label, count in label_counts.items():
        if count <= rare_threshold:
            sampled_dfs.append(df[df[label_col] == label])
            remaining_budget -= count
            print(f"      {label}: {count} (all)")

    # Kalan sınıflardan orantılı sample
    common_labels = [l for l, c in label_counts.items() if c > rare_threshold]
    common_total = label_counts[common_labels].sum()

    for label in common_labels:
        proportion = label_counts[label] / common_total
        n_samples = int(remaining_budget * proportion)
        sampled = df[df[label_col] == label].sample(
            n=min(n_samples, label_counts[label]), random_state=42
        )
        sampled_dfs.append(sampled)
        print(f"      {label}: {len(sampled):,}")

    df = pd.concat(sampled_dfs, ignore_index=True).sample(
        frac=1, random_state=42
    )  # Shuffle
    print(f"\n   📊 Sampled: {len(df):,}")

    # Features
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

    print(f"\n   Train: {len(X_train):,}")
    print(f"   Val: {len(X_val):,}")
    print(f"   Test: {len(X_test):,}")

    # Scale
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    # Reshape
    timesteps = 10
    n_features = X_train.shape[1]
    if n_features % timesteps != 0:
        pad_size = timesteps - (n_features % timesteps)
        X_train = np.pad(X_train, ((0, 0), (0, pad_size)), mode="constant")
        X_val = np.pad(X_val, ((0, 0), (0, pad_size)), mode="constant")
        X_test = np.pad(X_test, ((0, 0), (0, pad_size)), mode="constant")

    features_per_step = X_train.shape[1] // timesteps
    X_train = X_train.reshape(-1, timesteps, features_per_step)
    X_val = X_val.reshape(-1, timesteps, features_per_step)
    X_test = X_test.reshape(-1, timesteps, features_per_step)

    print(f"   Shape: {X_train.shape}")

    return (
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test,
        num_classes,
        list(le.classes_),
    )


def build_optimized_model(input_shape, num_classes):
    """Optimized SSA-LSTMIDS model"""
    l2_reg = keras.regularizers.l2(0.0003)

    inputs = keras.layers.Input(shape=input_shape)

    # Conv Block
    x = keras.layers.Conv1D(
        64, 3, padding="same", activation="relu", kernel_regularizer=l2_reg
    )(inputs)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Conv1D(
        128, 3, padding="same", activation="relu", kernel_regularizer=l2_reg
    )(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.MaxPooling1D(2, padding="same")(x)

    # BiLSTM
    x = keras.layers.Bidirectional(
        keras.layers.LSTM(128, return_sequences=True, dropout=0.2)
    )(x)
    x = keras.layers.BatchNormalization()(x)

    x = keras.layers.Bidirectional(
        keras.layers.LSTM(64, return_sequences=False, dropout=0.2)
    )(x)
    x = keras.layers.BatchNormalization()(x)

    # Dense
    x = keras.layers.Dense(256, activation="relu", kernel_regularizer=l2_reg)(x)
    x = keras.layers.Dropout(0.3)(x)
    x = keras.layers.Dense(128, activation="relu", kernel_regularizer=l2_reg)(x)
    x = keras.layers.Dropout(0.2)(x)

    outputs = keras.layers.Dense(num_classes, activation="softmax")(x)

    model = keras.Model(inputs, outputs, name="Fast_SSA_LSTMIDS")
    return model


def train():
    """Main training function"""
    print("\n" + "=" * 80)
    print("🚀 FAST CICIDS2017 TRAINING - Target: 99.88%")
    print("=" * 80)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Load data
    data = load_stratified_sample()
    if data is None:
        return

    X_train, y_train, X_val, y_val, X_test, y_test, num_classes, class_names = data

    # Build model
    print("\n" + "=" * 70)
    print("🧠 Building Optimized Model")
    print("=" * 70)

    model = build_optimized_model(X_train.shape[1:], num_classes)

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    print(f"   Parameters: {model.count_params():,}")
    print(f"   Batch size: {BATCH_SIZE}")
    print(f"   Learning rate: {LEARNING_RATE}")

    # Callbacks
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=PATIENCE,
            restore_best_weights=True,
            mode="max",
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_accuracy",
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            mode="max",
            verbose=1,
        ),
    ]

    # Train
    print(f"\n🚀 Training for up to {MAX_EPOCHS} epochs...")

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=MAX_EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=1,
    )

    # Evaluate
    print("\n" + "=" * 70)
    print("📊 TEST RESULTS")
    print("=" * 70)

    y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    recall = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

    print(f"\n   ✅ Accuracy:  {accuracy*100:.4f}%")
    print(f"   ✅ Precision: {precision*100:.4f}%")
    print(f"   ✅ Recall:    {recall*100:.4f}%")
    print(f"   ✅ F1-Score:  {f1:.6f}")

    print(f"\n   📄 Makale hedefi: 99.88%")
    print(f"   📊 Bizim sonuç:   {accuracy*100:.4f}%")

    if accuracy >= 0.9988:
        print("\n   🎉🎉🎉 MAKALEYİ GEÇTİK! 🎉🎉🎉")
    elif accuracy >= 0.9985:
        print("\n   🔥 Çok çok yakın!")
    elif accuracy >= 0.9980:
        print("\n   👍 Çok iyi!")

    # Save
    model_id = f"fast_ssa_lstmids_cicids_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    model_path = MODELS_DIR / f"{model_id}.keras"
    model.save(model_path)
    print(f"\n💾 Saved: {model_path}")

    # Registry
    if REGISTRY_PATH.exists():
        with open(REGISTRY_PATH, "r") as f:
            registry = json.load(f)
    else:
        registry = {"models": []}

    registry["models"].append(
        {
            "id": model_id,
            "name": "Fast_SSA_LSTMIDS_CICIDS",
            "model_type": "ssa_lstmids_fast",
            "dataset": "cicids2017_500k",
            "status": "trained",
            "framework": "tensorflow",
            "path": str(model_path),
            "metrics": {
                "accuracy": float(accuracy),
                "precision": float(precision),
                "recall": float(recall),
                "f1_score": float(f1),
            },
            "training_config": {
                "samples": SAMPLE_SIZE,
                "epochs": len(history.history["loss"]),
                "batch_size": BATCH_SIZE,
                "best_val_accuracy": float(max(history.history["val_accuracy"])),
            },
            "created_at": datetime.now().isoformat(),
        }
    )

    with open(REGISTRY_PATH, "w") as f:
        json.dump(registry, f, indent=2)

    # Final
    print("\n" + "=" * 80)
    print("📊 FINAL SUMMARY")
    print("=" * 80)
    print(f"   Samples: {SAMPLE_SIZE:,}")
    print(f"   Epochs: {len(history.history['loss'])}")
    print(f"   Accuracy: {accuracy*100:.4f}%")
    print(f"   Target: 99.88%")
    print(f"   Status: {'✅ PASSED' if accuracy >= 0.9988 else '❌ Not yet'}")
    print("=" * 80)
    print(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    train()
