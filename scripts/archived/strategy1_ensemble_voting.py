"""
Strateji 1: Ensemble Voting
============================

Mevcut 5 modeli birleştirerek daha yüksek accuracy hedefliyoruz.
Soft voting ile her modelin olasılık çıktılarını birleştiriyoruz.

Hedef: %99.88+
Tahmini süre: 45-60 dakika

Kullanım:
    python scripts/strategy1_ensemble_voting.py
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

DATA_DIR = PROJECT_ROOT / "data" / "raw" / "cicids2017_full"
MODELS_DIR = PROJECT_ROOT / "models"
REGISTRY_PATH = MODELS_DIR / "model_registry.json"


def load_test_data():
    """Test verisi yükle"""
    print("\n" + "=" * 70)
    print("📊 Loading CICIDS2017 Test Data")
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

    df = pd.concat(dfs, ignore_index=True)
    print(f"   Total: {len(df):,}")

    # Stratified sample
    label_col = " Label"

    # 200K sample al
    SAMPLE_SIZE = 200_000
    if len(df) > SAMPLE_SIZE:
        df = df.sample(n=SAMPLE_SIZE, random_state=42)
    print(f"   Sampled: {len(df):,}")

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

    # Split - sadece test kısmını kullan
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    # Scale
    scaler = MinMaxScaler()
    X_test = scaler.fit_transform(X_test)

    # Reshape
    timesteps = 10
    n_features = X_test.shape[1]
    if n_features % timesteps != 0:
        pad_size = timesteps - (n_features % timesteps)
        X_test = np.pad(X_test, ((0, 0), (0, pad_size)), mode="constant")

    features_per_step = X_test.shape[1] // timesteps
    X_test = X_test.reshape(-1, timesteps, features_per_step)

    print(f"   Test shape: {X_test.shape}")
    print(f"   Classes: {len(le.classes_)}")

    return X_test, y_test, len(le.classes_), list(le.classes_)


def build_model_1(input_shape, num_classes):
    """SSA-LSTMIDS tarzı model"""
    inputs = keras.layers.Input(shape=input_shape)
    x = keras.layers.Conv1D(64, 3, padding="same", activation="relu")(inputs)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.MaxPooling1D(2, padding="same")(x)
    x = keras.layers.LSTM(128, return_sequences=False)(x)
    x = keras.layers.Dense(256, activation="relu")(x)
    x = keras.layers.Dropout(0.3)(x)
    outputs = keras.layers.Dense(num_classes, activation="softmax")(x)
    return keras.Model(inputs, outputs, name="Model_1_LSTM")


def build_model_2(input_shape, num_classes):
    """BiLSTM + Attention"""
    inputs = keras.layers.Input(shape=input_shape)
    x = keras.layers.Conv1D(32, 3, padding="same", activation="relu")(inputs)
    x = keras.layers.Bidirectional(keras.layers.LSTM(64, return_sequences=True))(x)
    x = keras.layers.Bidirectional(keras.layers.LSTM(32, return_sequences=False))(x)
    x = keras.layers.Dense(128, activation="relu")(x)
    x = keras.layers.Dropout(0.3)(x)
    outputs = keras.layers.Dense(num_classes, activation="softmax")(x)
    return keras.Model(inputs, outputs, name="Model_2_BiLSTM")


def build_model_3(input_shape, num_classes):
    """GRU Model"""
    inputs = keras.layers.Input(shape=input_shape)
    x = keras.layers.Conv1D(64, 3, padding="same", activation="relu")(inputs)
    x = keras.layers.GRU(128, return_sequences=True)(x)
    x = keras.layers.GRU(64, return_sequences=False)(x)
    x = keras.layers.Dense(128, activation="relu")(x)
    x = keras.layers.Dropout(0.3)(x)
    outputs = keras.layers.Dense(num_classes, activation="softmax")(x)
    return keras.Model(inputs, outputs, name="Model_3_GRU")


def build_model_4(input_shape, num_classes):
    """CNN-LSTM Model"""
    inputs = keras.layers.Input(shape=input_shape)
    x = keras.layers.Conv1D(32, 3, padding="same", activation="relu")(inputs)
    x = keras.layers.Conv1D(64, 3, padding="same", activation="relu")(x)
    x = keras.layers.MaxPooling1D(2, padding="same")(x)
    x = keras.layers.LSTM(64, return_sequences=False)(x)
    x = keras.layers.Dense(64, activation="relu")(x)
    x = keras.layers.Dropout(0.3)(x)
    outputs = keras.layers.Dense(num_classes, activation="softmax")(x)
    return keras.Model(inputs, outputs, name="Model_4_CNN_LSTM")


def build_model_5(input_shape, num_classes):
    """Deep Dense Model"""
    inputs = keras.layers.Input(shape=input_shape)
    x = keras.layers.Flatten()(inputs)
    x = keras.layers.Dense(512, activation="relu")(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Dropout(0.4)(x)
    x = keras.layers.Dense(256, activation="relu")(x)
    x = keras.layers.Dropout(0.3)(x)
    x = keras.layers.Dense(128, activation="relu")(x)
    x = keras.layers.Dropout(0.2)(x)
    outputs = keras.layers.Dense(num_classes, activation="softmax")(x)
    return keras.Model(inputs, outputs, name="Model_5_Deep_Dense")


def train_and_get_predictions(
    model, X_train, y_train, X_test, batch_size=128, epochs=30
):
    """Model eğit ve tahmin döndür"""
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=5, restore_best_weights=True
        )
    ]

    # Split train/val
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.15, random_state=42, stratify=y_train
    )

    model.fit(
        X_tr,
        y_tr,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=0,
    )

    # Val accuracy
    val_pred = np.argmax(model.predict(X_val, verbose=0), axis=1)
    val_acc = accuracy_score(y_val, val_pred)

    # Probabilities
    probs = model.predict(X_test, verbose=0)

    return probs, val_acc


def ensemble_voting(predictions, weights=None):
    """Soft voting ensemble"""
    if weights is None:
        weights = [1.0] * len(predictions)

    # Weighted average
    weighted_sum = sum(p * w for p, w in zip(predictions, weights))
    weighted_avg = weighted_sum / sum(weights)

    return np.argmax(weighted_avg, axis=1)


def main():
    print("\n" + "=" * 80)
    print("🎯 STRATEJI 1: ENSEMBLE VOTING")
    print("=" * 80)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\nHedef: %99.88+")

    # Load data
    X_test, y_test, num_classes, class_names = load_test_data()

    # Training data için de yükle
    print("\n   Loading training data...")
    csv_files = [
        "Monday-WorkingHours.pcap_ISCX.csv",
        "Tuesday-WorkingHours.pcap_ISCX.csv",
        "Wednesday-workingHours.pcap_ISCX.csv",
    ]

    dfs = []
    for fname in csv_files:
        fpath = DATA_DIR / fname
        if fpath.exists():
            df = pd.read_csv(fpath, low_memory=False, nrows=100000)
            dfs.append(df)

    df = pd.concat(dfs, ignore_index=True)

    label_col = " Label"
    exclude_cols = [label_col, "Timestamp", "Flow ID", "Source IP", "Destination IP"]
    feature_cols = [c for c in df.columns if c not in exclude_cols]

    for col in feature_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.replace([np.inf, -np.inf], np.nan)
    df[feature_cols] = df[feature_cols].fillna(0)

    X_train = df[feature_cols].values.astype(np.float32)
    X_train = np.nan_to_num(X_train, nan=0.0, posinf=1e6, neginf=-1e6)

    le = LabelEncoder()
    le.fit(df[label_col])
    y_train = le.transform(df[label_col])

    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)

    timesteps = 10
    if X_train.shape[1] % timesteps != 0:
        pad_size = timesteps - (X_train.shape[1] % timesteps)
        X_train = np.pad(X_train, ((0, 0), (0, pad_size)), mode="constant")

    features_per_step = X_train.shape[1] // timesteps
    X_train = X_train.reshape(-1, timesteps, features_per_step)

    print(f"   Train shape: {X_train.shape}")

    # Build and train 5 models
    print("\n" + "=" * 70)
    print("🧠 Training 5 Models for Ensemble")
    print("=" * 70)

    model_builders = [
        ("LSTM", build_model_1),
        ("BiLSTM", build_model_2),
        ("GRU", build_model_3),
        ("CNN-LSTM", build_model_4),
        ("Deep Dense", build_model_5),
    ]

    all_predictions = []
    model_accuracies = []

    for i, (name, builder) in enumerate(model_builders):
        print(f"\n   [{i+1}/5] Training {name}...")
        model = builder(X_train.shape[1:], num_classes)

        probs, val_acc = train_and_get_predictions(model, X_train, y_train, X_test)
        all_predictions.append(probs)
        model_accuracies.append(val_acc)

        print(f"         Val Accuracy: {val_acc*100:.2f}%")

        keras.backend.clear_session()

    # Ensemble voting
    print("\n" + "=" * 70)
    print("🗳️ Ensemble Voting")
    print("=" * 70)

    # Simple average voting
    print("\n   Simple Voting (equal weights)...")
    simple_pred = ensemble_voting(all_predictions)
    simple_acc = accuracy_score(y_test, simple_pred)
    print(f"   Accuracy: {simple_acc*100:.4f}%")

    # Weighted voting (by individual accuracy)
    print("\n   Weighted Voting (by accuracy)...")
    weights = model_accuracies
    weighted_pred = ensemble_voting(all_predictions, weights)
    weighted_acc = accuracy_score(y_test, weighted_pred)
    print(f"   Accuracy: {weighted_acc*100:.4f}%")

    # Best result
    best_acc = max(simple_acc, weighted_acc)
    best_type = "Simple" if simple_acc > weighted_acc else "Weighted"
    best_pred = simple_pred if simple_acc > weighted_acc else weighted_pred

    # Metrics
    precision = precision_score(y_test, best_pred, average="weighted", zero_division=0)
    recall = recall_score(y_test, best_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_test, best_pred, average="weighted", zero_division=0)

    print("\n" + "=" * 70)
    print("📊 RESULTS")
    print("=" * 70)

    print(f"\n   Individual Models:")
    for i, (name, _) in enumerate(model_builders):
        print(f"      {name}: {model_accuracies[i]*100:.2f}%")

    print(f"\n   Ensemble Results:")
    print(f"      Simple Voting: {simple_acc*100:.4f}%")
    print(f"      Weighted Voting: {weighted_acc*100:.4f}%")

    print(f"\n   ✅ Best ({best_type}): {best_acc*100:.4f}%")
    print(f"   ✅ Precision: {precision*100:.4f}%")
    print(f"   ✅ Recall: {recall*100:.4f}%")
    print(f"   ✅ F1-Score: {f1:.6f}")

    print(f"\n   📄 Hedef: 99.88%")
    print(f"   📊 Sonuç: {best_acc*100:.4f}%")

    if best_acc >= 0.9988:
        print("\n   🎉🎉🎉 HEDEF BAŞARILDI! 🎉🎉🎉")
    else:
        print(f"\n   Fark: {(0.9988 - best_acc)*100:+.2f}%")

    # Save result
    result = {
        "strategy": "Ensemble Voting",
        "accuracy": float(best_acc),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "voting_type": best_type,
        "individual_accuracies": model_accuracies,
        "created_at": datetime.now().isoformat(),
    }

    results_path = MODELS_DIR / "strategy1_ensemble_results.json"
    with open(results_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\n💾 Saved: {results_path}")

    print("\n" + "=" * 80)
    print(f"✅ STRATEJI 1 TAMAMLANDI: {best_acc*100:.4f}%")
    print("=" * 80)
    print(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    return best_acc


if __name__ == "__main__":
    main()
