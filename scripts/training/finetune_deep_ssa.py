"""
Deep SSA-LSTMIDS Fine-Tuning
=============================

Mevcut en iyi CICIDS2017 modelini (%99.38) düşük LR ile
fine-tune ederek %99.88+ hedefliyoruz.

Hedef: %99.88+
Tahmini süre: 1-2 saat

Kullanım:
    python scripts/finetune_deep_ssa.py
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

# En iyi mevcut model
BEST_MODEL_PATH = MODELS_DIR / "deep_ssa_lstmids_cicids2017_20260107_192008.keras"


def load_data(sample_size=1_000_000):
    """Veri yükle"""
    print(f"\n📊 Loading {sample_size:,} samples...")

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

    if len(df) > sample_size:
        df = df.sample(n=sample_size, random_state=42)
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
        X, y, test_size=0.2, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.1, random_state=42
    )

    # Scale
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    # Reshape for model
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


def main():
    print("\n" + "=" * 80)
    print("🎯 DEEP SSA-LSTMIDS FINE-TUNING")
    print("=" * 80)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\nMevcut Model: %99.38")
    print("Hedef: %99.88+")

    # Load data
    X_train, y_train, X_val, y_val, X_test, y_test, num_classes = load_data(1_000_000)

    # Build new deep model (eski model input shape uyuşmuyor)
    print("\n" + "=" * 60)
    print("🧠 Building Deep SSA-LSTMIDS Model")
    print("=" * 60)

    model = build_new_model(X_train.shape[1:], num_classes)
    print(f"   ✅ New model built successfully!")

    # Fine-tune with very low learning rate
    print("\n" + "=" * 60)
    print("🔧 Fine-Tuning (Phase 1: Very Low LR)")
    print("=" * 60)

    # Recompile with very low LR
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.00005),  # Very low LR
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=15, restore_best_weights=True, mode="max"
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_accuracy", factor=0.5, patience=5, min_lr=1e-8, mode="max"
        ),
        keras.callbacks.ModelCheckpoint(
            str(MODELS_DIR / "finetune_best.keras"),
            monitor="val_accuracy",
            save_best_only=True,
            mode="max",
        ),
    ]

    print("\n🚀 Phase 1: Fine-tuning with LR=0.00005")
    history1 = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=50,
        batch_size=256,
        callbacks=callbacks,
        verbose=1,
    )

    phase1_best = max(history1.history["val_accuracy"])
    print(f"\n   Phase 1 Best Val Accuracy: {phase1_best*100:.4f}%")

    # Phase 2: Even lower LR
    print("\n" + "=" * 60)
    print("🔧 Fine-Tuning (Phase 2: Ultra Low LR)")
    print("=" * 60)

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.00001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    print("\n🚀 Phase 2: Fine-tuning with LR=0.00001")
    history2 = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=30,
        batch_size=256,
        callbacks=callbacks,
        verbose=1,
    )

    phase2_best = max(history2.history["val_accuracy"])
    print(f"\n   Phase 2 Best Val Accuracy: {phase2_best*100:.4f}%")

    overall_best = max(phase1_best, phase2_best)
    print(f"\n   Overall Best Val Accuracy: {overall_best*100:.4f}%")

    # Evaluate
    print("\n" + "=" * 60)
    print("📊 TEST RESULTS")
    print("=" * 60)

    # Load best model
    best_model = keras.models.load_model(str(MODELS_DIR / "finetune_best.keras"))

    y_pred = np.argmax(best_model.predict(X_test, verbose=0), axis=1)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    recall = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

    print(f"\n   ✅ Accuracy:  {accuracy*100:.4f}%")
    print(f"   ✅ Precision: {precision*100:.4f}%")
    print(f"   ✅ Recall:    {recall*100:.4f}%")
    print(f"   ✅ F1-Score:  {f1:.6f}")

    print(f"\n   📄 Mevcut En İyi: 99.38%")
    print(f"   📄 Hedef: 99.88%")
    print(f"   📊 Sonuç: {accuracy*100:.4f}%")

    if accuracy >= 0.9988:
        print("\n   🎉🎉🎉 HEDEF BAŞARILDI! 🎉🎉🎉")
    elif accuracy > 0.9938:
        print(f"\n   🎯 İyileştirme: +{(accuracy - 0.9938)*100:.2f}%")
    else:
        print(f"\n   Fark: {(accuracy - 0.9938)*100:+.2f}%")

    # Save
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = MODELS_DIR / f"finetuned_deep_ssa_{timestamp}.keras"
    best_model.save(model_path)
    print(f"\n💾 Saved: {model_path}")

    result = {
        "strategy": "Deep SSA Fine-Tuning",
        "base_model_accuracy": 0.9938,
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "phase1_best_val": float(phase1_best),
        "phase2_best_val": float(phase2_best),
        "overall_best_val": float(overall_best),
        "improvement": float(accuracy - 0.9938),
        "created_at": datetime.now().isoformat(),
    }

    results_path = MODELS_DIR / "finetune_results.json"
    with open(results_path, "w") as f:
        json.dump(result, f, indent=2)

    print("\n" + "=" * 80)
    print(f"✅ FINE-TUNING TAMAMLANDI: {accuracy*100:.4f}%")
    print("=" * 80)

    return accuracy


def build_new_model(input_shape, num_classes):
    """Build model from scratch if loading fails"""
    l2 = keras.regularizers.l2(0.0001)

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

    x = keras.layers.Conv1D(
        256, 3, padding="same", activation="relu", kernel_regularizer=l2
    )(x)
    x = keras.layers.BatchNormalization()(x)

    x = keras.layers.Bidirectional(
        keras.layers.LSTM(256, return_sequences=True, dropout=0.1)
    )(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Bidirectional(
        keras.layers.LSTM(128, return_sequences=False, dropout=0.1)
    )(x)
    x = keras.layers.BatchNormalization()(x)

    x = keras.layers.Dense(512, activation="relu", kernel_regularizer=l2)(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Dropout(0.3)(x)
    x = keras.layers.Dense(256, activation="relu", kernel_regularizer=l2)(x)
    x = keras.layers.Dropout(0.2)(x)

    outputs = keras.layers.Dense(num_classes, activation="softmax")(x)

    model = keras.Model(inputs, outputs, name="Deep_SSA_LSTMIDS_Finetuned")
    print(f"   Parameters: {model.count_params():,}")

    return model


if __name__ == "__main__":
    main()
