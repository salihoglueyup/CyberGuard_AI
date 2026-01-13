"""
K-Fold Cross Validation Script
===============================

5-Fold ve 10-Fold cross validation ile model değerlendirmesi.
Daha güvenilir metrikler elde etmek için kullanılır.

Kullanım:
    python scripts/kfold_validation.py
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
from tensorflow import keras

print(f"✅ TensorFlow {tf.__version__}")

DATA_DIR = PROJECT_ROOT / "data" / "raw" / "cicids2017_full"
MODELS_DIR = PROJECT_ROOT / "models"


def load_data():
    """CICIDS2017 verisi yükle"""
    print("\n" + "=" * 60)
    print("📊 Loading Data for K-Fold Validation")
    print("=" * 60)

    train_file = DATA_DIR / "Train_data.csv"
    df = pd.read_csv(train_file, low_memory=False)

    label_col = "class"
    feature_cols = [
        c
        for c in df.columns
        if c not in [label_col, "protocol_type", "service", "flag"]
    ]

    for col in feature_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df[feature_cols] = df[feature_cols].fillna(0)

    X = df[feature_cols].values.astype(np.float32)
    X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)

    le = LabelEncoder()
    y = le.fit_transform(df[label_col])

    print(f"   Samples: {len(X):,}")
    print(f"   Features: {X.shape[1]}")
    print(f"   Classes: {list(le.classes_)}")

    return X, y, len(le.classes_)


def build_model(input_shape, num_classes):
    """SSA-LSTMIDS modeli"""
    inputs = keras.layers.Input(shape=input_shape)
    x = keras.layers.Conv1D(64, 3, padding="same", activation="relu")(inputs)
    x = keras.layers.MaxPooling1D(2, padding="same")(x)
    x = keras.layers.LSTM(128)(x)
    x = keras.layers.Dense(256, activation="relu")(x)
    x = keras.layers.Dropout(0.3)(x)
    outputs = keras.layers.Dense(num_classes, activation="softmax")(x)

    model = keras.Model(inputs, outputs)
    model.compile(
        optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
    )
    return model


def kfold_cross_validation(X, y, num_classes, n_splits=5):
    """K-Fold Cross Validation"""
    print(f"\n" + "=" * 60)
    print(f"🔄 {n_splits}-Fold Cross Validation")
    print("=" * 60)

    # Scale
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    # Reshape for LSTM
    timesteps = 10
    n_features = X_scaled.shape[1]
    if n_features % timesteps != 0:
        pad_size = timesteps - (n_features % timesteps)
        X_scaled = np.pad(X_scaled, ((0, 0), (0, pad_size)), mode="constant")

    features_per_step = X_scaled.shape[1] // timesteps
    X_reshaped = X_scaled.reshape(X_scaled.shape[0], timesteps, features_per_step)

    # K-Fold
    kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    fold_results = []

    for fold, (train_idx, val_idx) in enumerate(kfold.split(X_reshaped, y)):
        print(f"\n   Fold {fold+1}/{n_splits}")

        X_train, X_val = X_reshaped[train_idx], X_reshaped[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # Build fresh model
        model = build_model(X_train.shape[1:], num_classes)

        # Train
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor="val_accuracy", patience=5, restore_best_weights=True
            )
        ]

        model.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val),
            epochs=30,
            batch_size=128,
            callbacks=callbacks,
            verbose=0,
        )

        # Evaluate
        y_pred = np.argmax(model.predict(X_val, verbose=0), axis=1)

        accuracy = accuracy_score(y_val, y_pred)
        precision = precision_score(y_val, y_pred, average="weighted", zero_division=0)
        recall = recall_score(y_val, y_pred, average="weighted", zero_division=0)
        f1 = f1_score(y_val, y_pred, average="weighted", zero_division=0)

        print(f"   Accuracy: {accuracy*100:.2f}% | F1: {f1:.4f}")

        fold_results.append(
            {
                "fold": fold + 1,
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1_score": f1,
            }
        )

        # Clean up
        del model
        keras.backend.clear_session()

    return fold_results


def main():
    print("\n" + "=" * 70)
    print("🎓 K-Fold Cross Validation")
    print("=" * 70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Load data
    X, y, num_classes = load_data()

    # 5-Fold CV
    results_5fold = kfold_cross_validation(X, y, num_classes, n_splits=5)

    # Summary
    print("\n" + "=" * 60)
    print("📊 5-Fold Cross Validation Results")
    print("=" * 60)

    accs = [r["accuracy"] for r in results_5fold]
    f1s = [r["f1_score"] for r in results_5fold]

    print(f"\n   Mean Accuracy: {np.mean(accs)*100:.2f}% ± {np.std(accs)*100:.2f}%")
    print(f"   Mean F1-Score: {np.mean(f1s):.4f} ± {np.std(f1s):.4f}")
    print(f"   Min Accuracy: {np.min(accs)*100:.2f}%")
    print(f"   Max Accuracy: {np.max(accs)*100:.2f}%")

    # Save results
    all_results = {
        "5_fold": {
            "folds": results_5fold,
            "mean_accuracy": float(np.mean(accs)),
            "std_accuracy": float(np.std(accs)),
            "mean_f1": float(np.mean(f1s)),
            "std_f1": float(np.std(f1s)),
        },
        "created_at": datetime.now().isoformat(),
    }

    results_path = MODELS_DIR / "kfold_results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n💾 Results saved: {results_path}")

    print("\n" + "=" * 70)
    print("✅ K-Fold Validation Complete!")
    print("=" * 70)
    print(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
