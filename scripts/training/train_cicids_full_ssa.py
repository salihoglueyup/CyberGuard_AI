"""
CICIDS2017 Full Dataset Training with SSA Optimizer
=====================================================

Makaledeki sonuçlara ulaşmak için:
- 2.87M sample (full dataset)
- SSA optimizer ile hyperparameter tuning
- 300 epoch eğitim
- Deep BiLSTM+Attention model

Hedef: %99.88 accuracy

Kullanım:
    python scripts/train_cicids_full_ssa.py
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


def load_full_cicids2017():
    """Load all CICIDS2017 CSV files (~2.87M samples)"""
    print("\n" + "=" * 70)
    print("📊 Loading CICIDS2017 FULL Dataset (2.87M samples)")
    print("=" * 70)

    # All daily CSV files (excluding Train/Test which are subsets)
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
            print(f"   Loading {fname}...")
            df = pd.read_csv(fpath, low_memory=False)
            dfs.append(df)
            print(f"      {len(df):,} rows")
        else:
            print(f"   ⚠️ Not found: {fname}")

    if not dfs:
        print("❌ No data loaded!")
        return None

    df = pd.concat(dfs, ignore_index=True)
    print(f"\n   📊 Total loaded: {len(df):,} samples")

    # Find label column
    label_col = None
    for col in [" Label", "Label", "label", "class"]:
        if col in df.columns:
            label_col = col
            break

    if not label_col:
        print(f"   Available columns: {list(df.columns)[:20]}")
        return None

    print(f"   Label column: '{label_col}'")
    print(f"   Label distribution:")
    label_counts = df[label_col].value_counts()
    for label, count in label_counts.items():
        print(f"      {label}: {count:,}")

    # Clean data
    print("\n   🔧 Cleaning data...")

    # Numeric feature columns
    exclude_cols = [label_col, "Timestamp", "Flow ID", "Source IP", "Destination IP"]
    feature_cols = [c for c in df.columns if c not in exclude_cols]

    # Convert to numeric
    for col in feature_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Handle inf/nan
    df = df.replace([np.inf, -np.inf], np.nan)
    df[feature_cols] = df[feature_cols].fillna(0)

    X = df[feature_cols].values.astype(np.float32)
    X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)

    # Labels
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

    # Reshape for LSTM (timesteps=10)
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


def build_deep_model(
    input_shape,
    num_classes,
    conv_filters=64,
    lstm_units=128,
    dense_units=256,
    dropout=0.3,
):
    """Build deep SSA-LSTMIDS model with configurable hyperparameters"""
    l2_reg = keras.regularizers.l2(0.0005)

    inputs = keras.layers.Input(shape=input_shape)

    # Conv Block 1
    x = keras.layers.Conv1D(
        conv_filters, 3, padding="same", activation="relu", kernel_regularizer=l2_reg
    )(inputs)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Conv1D(
        conv_filters, 3, padding="same", activation="relu", kernel_regularizer=l2_reg
    )(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.MaxPooling1D(2, padding="same")(x)

    # Conv Block 2
    x = keras.layers.Conv1D(
        conv_filters * 2,
        3,
        padding="same",
        activation="relu",
        kernel_regularizer=l2_reg,
    )(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.MaxPooling1D(2, padding="same")(x)

    # BiLSTM
    x = keras.layers.Bidirectional(
        keras.layers.LSTM(
            lstm_units, return_sequences=True, dropout=0.2, recurrent_dropout=0.2
        )
    )(x)
    x = keras.layers.BatchNormalization()(x)

    x = keras.layers.Bidirectional(
        keras.layers.LSTM(
            lstm_units // 2, return_sequences=True, dropout=0.2, recurrent_dropout=0.2
        )
    )(x)

    # Attention
    attention = keras.layers.MultiHeadAttention(num_heads=4, key_dim=32)(x, x)
    x = keras.layers.LayerNormalization()(attention + x)

    # Global Pooling
    x = keras.layers.GlobalAveragePooling1D()(x)

    # Dense
    x = keras.layers.Dense(dense_units, activation="relu", kernel_regularizer=l2_reg)(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Dropout(dropout)(x)

    x = keras.layers.Dense(
        dense_units // 2, activation="relu", kernel_regularizer=l2_reg
    )(x)
    x = keras.layers.Dropout(dropout * 0.7)(x)

    outputs = keras.layers.Dense(num_classes, activation="softmax")(x)

    model = keras.Model(inputs, outputs, name="Deep_SSA_LSTMIDS_Full")
    return model


def train_model(
    X_train, y_train, X_val, y_val, X_test, y_test, num_classes, class_names
):
    """Train with optimal hyperparameters"""
    print("\n" + "=" * 70)
    print("🧠 Training Deep SSA-LSTMIDS on Full CICIDS2017")
    print("=" * 70)

    # SSA-optimized hyperparameters (from our SSA optimizer results)
    CONV_FILTERS = 128
    LSTM_UNITS = 256
    DENSE_UNITS = 512
    DROPOUT = 0.34
    BATCH_SIZE = 128
    EPOCHS = 100  # Early stopping will handle

    print(f"\n   Hyperparameters (SSA optimized):")
    print(f"      Conv filters: {CONV_FILTERS}")
    print(f"      LSTM units: {LSTM_UNITS}")
    print(f"      Dense units: {DENSE_UNITS}")
    print(f"      Dropout: {DROPOUT}")
    print(f"      Batch size: {BATCH_SIZE}")

    # Build model
    model = build_deep_model(
        X_train.shape[1:],
        num_classes,
        conv_filters=CONV_FILTERS,
        lstm_units=LSTM_UNITS,
        dense_units=DENSE_UNITS,
        dropout=DROPOUT,
    )

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    print(f"\n   Parameters: {model.count_params():,}")

    # Callbacks
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=15,
            restore_best_weights=True,
            mode="max",
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_accuracy",
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            mode="max",
            verbose=1,
        ),
        keras.callbacks.ModelCheckpoint(
            str(MODELS_DIR / "best_cicids_full.keras"),
            monitor="val_accuracy",
            save_best_only=True,
            mode="max",
        ),
    ]

    print(f"\n🚀 Training for up to {EPOCHS} epochs...")
    print(f"   Target: 99.88%")

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
    print("\n" + "=" * 70)
    print("📊 Test Results")
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

    print(f"\n   📄 Makaledeki hedef: 99.88%")
    print(f"   📊 Bizim sonuç:      {accuracy*100:.4f}%")

    if accuracy >= 0.9988:
        print("\n   🎉🎉🎉 HEDEF BAŞARILDI! 🎉🎉🎉")
    elif accuracy >= 0.9980:
        print("\n   🔥 Çok çok yakın!")
    elif accuracy >= 0.9970:
        print("\n   👍 Çok iyi!")

    # Classification report
    print("\n" + "=" * 60)
    print("📋 Classification Report")
    print("=" * 60)
    # Sadece ilk birkaç sınıf göster (çok fazla sınıf var)
    print(
        classification_report(
            y_test,
            y_pred,
            target_names=class_names[: min(10, len(class_names))],
            zero_division=0,
        )
    )

    # Save model
    model_id = (
        f"deep_ssa_lstmids_cicids_full_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    model_path = MODELS_DIR / f"{model_id}.keras"
    model.save(model_path)
    print(f"\n💾 Model saved: {model_path}")

    # Update registry
    if REGISTRY_PATH.exists():
        with open(REGISTRY_PATH, "r") as f:
            registry = json.load(f)
    else:
        registry = {"models": []}

    registry["models"].append(
        {
            "id": model_id,
            "name": "Deep_SSA_LSTMIDS_CICIDS_Full",
            "model_type": "deep_ssa_lstmids",
            "dataset": "cicids2017_full",
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
                "samples": len(X_train) + len(X_val) + len(X_test),
                "epochs": len(history.history["loss"]),
                "batch_size": BATCH_SIZE,
                "architecture": "deep_bilstm_attention_ssa",
                "best_val_accuracy": float(max(history.history["val_accuracy"])),
            },
            "created_at": datetime.now().isoformat(),
        }
    )

    with open(REGISTRY_PATH, "w") as f:
        json.dump(registry, f, indent=2)

    print("📋 Registry updated")

    return accuracy


def main():
    print("\n" + "=" * 80)
    print("🎓 CICIDS2017 FULL DATASET TRAINING")
    print("=" * 80)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\nHedef: 99.88% (makaledeki sonuç)")

    # Load data
    data = load_full_cicids2017()

    if data is None:
        print("❌ Failed to load data!")
        return

    X_train, y_train, X_val, y_val, X_test, y_test, num_classes, class_names = data

    # Train
    accuracy = train_model(
        X_train, y_train, X_val, y_val, X_test, y_test, num_classes, class_names
    )

    # Final summary
    print("\n" + "=" * 80)
    print("📊 FINAL SUMMARY")
    print("=" * 80)
    print(f"\n   Dataset: CICIDS2017 Full (2.87M samples)")
    print(f"   Model: Deep SSA-LSTMIDS (BiLSTM + Attention)")
    print(f"   Accuracy: {accuracy*100:.4f}%")
    print(f"   Target: 99.88%")
    print(f"   Gap: {(0.9988 - accuracy)*100:+.2f}%")

    print("\n" + "=" * 80)
    print("✅ Training Complete!")
    print("=" * 80)
    print(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
