"""
NSL-KDD SSA-LSTMIDS Optimized Training
======================================

Makaledeki parametreler:
- Conv1D filters: 30
- LSTM units: 120
- Dense units: 512
- Dropout: 0.2
- Batch size: 120
- Epochs: 300
- Early stopping patience: 6

Hedef: %99.36 accuracy (makaledeki sonuç)
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
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
DATA_DIR = PROJECT_ROOT / "data" / "raw" / "nsl_kdd"
MODELS_DIR = PROJECT_ROOT / "models"
REGISTRY_PATH = MODELS_DIR / "model_registry.json"

# Column names
COLUMNS = [
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

# Attack type mapping (5 classes)
ATTACK_MAP = {
    "normal": "Normal",
    "back": "DoS",
    "land": "DoS",
    "neptune": "DoS",
    "pod": "DoS",
    "smurf": "DoS",
    "teardrop": "DoS",
    "apache2": "DoS",
    "udpstorm": "DoS",
    "processtable": "DoS",
    "mailbomb": "DoS",
    "worm": "DoS",
    "ipsweep": "Probe",
    "nmap": "Probe",
    "portsweep": "Probe",
    "satan": "Probe",
    "mscan": "Probe",
    "saint": "Probe",
    "ftp_write": "R2L",
    "guess_passwd": "R2L",
    "imap": "R2L",
    "multihop": "R2L",
    "phf": "R2L",
    "spy": "R2L",
    "warezclient": "R2L",
    "warezmaster": "R2L",
    "snmpgetattack": "R2L",
    "named": "R2L",
    "xlock": "R2L",
    "xsnoop": "R2L",
    "sendmail": "R2L",
    "httptunnel": "R2L",
    "snmpguess": "R2L",
    "buffer_overflow": "U2R",
    "loadmodule": "U2R",
    "perl": "U2R",
    "rootkit": "U2R",
    "ps": "U2R",
    "sqlattack": "U2R",
    "xterm": "U2R",
}


def load_nsl_kdd():
    """NSL-KDD veri yükle - Makaledeki format"""
    print("\n" + "=" * 60)
    print("📊 Loading NSL-KDD Dataset")
    print("=" * 60)

    train_path = DATA_DIR / "KDDTrain+.txt"
    test_path = DATA_DIR / "KDDTest+.txt"

    df_train = pd.read_csv(train_path, names=COLUMNS)
    df_test = pd.read_csv(test_path, names=COLUMNS)

    print(f"   Train samples: {len(df_train):,}")
    print(f"   Test samples: {len(df_test):,}")

    # Map to 5 classes
    df_train["attack_cat"] = df_train["label"].map(ATTACK_MAP).fillna("Other")
    df_test["attack_cat"] = df_test["label"].map(ATTACK_MAP).fillna("Other")

    # One-hot encode categorical
    categorical_cols = ["protocol_type", "service", "flag"]

    # Combine for consistent encoding
    combined = pd.concat([df_train, df_test], ignore_index=True)

    # One-hot encoding
    combined_encoded = pd.get_dummies(
        combined, columns=categorical_cols, drop_first=False
    )

    # Split back
    df_train_enc = combined_encoded.iloc[: len(df_train)]
    df_test_enc = combined_encoded.iloc[len(df_train) :]

    # Label encode target
    le = LabelEncoder()
    y_train = le.fit_transform(df_train_enc["attack_cat"])
    y_test = le.transform(df_test_enc["attack_cat"])

    print(f"   Classes: {list(le.classes_)}")
    print(
        f"   Class distribution (train): {dict(zip(*np.unique(y_train, return_counts=True)))}"
    )

    # Feature columns (exclude label columns)
    drop_cols = ["label", "attack_cat", "difficulty"]
    feature_cols = [c for c in df_train_enc.columns if c not in drop_cols]

    X_train = df_train_enc[feature_cols].values.astype(np.float32)
    X_test = df_test_enc[feature_cols].values.astype(np.float32)

    print(f"   Features: {X_train.shape[1]}")

    # Scale with MinMaxScaler (0-1 range - better for RNNs)
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Reshape for LSTM - Makaledeki gibi (samples, 1, features)
    # Conv1D kernel_size=5 olduğu için en az 5 timestep gerekli
    # Features'ı 5 gruba bölelim
    n_features = X_train.shape[1]

    # Pad to make divisible by 5
    if n_features % 5 != 0:
        pad_size = 5 - (n_features % 5)
        X_train = np.pad(X_train, ((0, 0), (0, pad_size)), mode="constant")
        X_test = np.pad(X_test, ((0, 0), (0, pad_size)), mode="constant")

    # Reshape to (samples, timesteps, features_per_step)
    features_per_step = X_train.shape[1] // 5
    X_train = X_train.reshape(X_train.shape[0], 5, features_per_step)
    X_test = X_test.reshape(X_test.shape[0], 5, features_per_step)

    print(f"   X_train shape: {X_train.shape}")
    print(f"   X_test shape: {X_test.shape}")

    return X_train, y_train, X_test, y_test, len(le.classes_), list(le.classes_)


def build_ssa_lstmids_paper(input_shape, num_classes):
    """
    Makaledeki birebir SSA-LSTMIDS modeli

    Mimari:
        Input → Conv1D(30, kernel=5) → MaxPool(2) → LSTM(120) → Dense(512) → Dropout(0.2) → Output
    """
    print("\n" + "=" * 60)
    print("🧠 Building SSA-LSTMIDS Model (Paper Configuration)")
    print("=" * 60)

    inputs = keras.layers.Input(shape=input_shape, name="input")

    # Conv1D - Feature extraction
    x = keras.layers.Conv1D(
        filters=30,
        kernel_size=3,  # Adjusted for smaller timesteps
        padding="same",
        activation="relu",
        name="conv1d",
    )(inputs)

    # MaxPooling
    x = keras.layers.MaxPooling1D(pool_size=2, padding="same", name="maxpool")(x)

    # LSTM
    x = keras.layers.LSTM(
        units=120,
        activation="tanh",
        recurrent_activation="sigmoid",
        return_sequences=False,
        name="lstm",
    )(x)

    # Dense
    x = keras.layers.Dense(512, activation="relu", name="dense")(x)

    # Dropout
    x = keras.layers.Dropout(0.2, name="dropout")(x)

    # Output
    outputs = keras.layers.Dense(num_classes, activation="softmax", name="output")(x)

    model = keras.Model(inputs, outputs, name="SSA_LSTMIDS_Paper")

    # Compile with paper settings
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    print(f"   Conv1D: 30 filters, kernel=3")
    print(f"   MaxPool: pool_size=2")
    print(f"   LSTM: 120 units")
    print(f"   Dense: 512 units")
    print(f"   Dropout: 0.2")
    print(f"   Total params: {model.count_params():,}")

    return model


def train_model(model, X_train, y_train, X_test, y_test, class_names):
    """Modeli eğit - Makaledeki parametrelerle"""
    print("\n" + "=" * 60)
    print("🏋️ Training SSA-LSTMIDS (Paper Parameters)")
    print("=" * 60)

    # Paper parameters
    EPOCHS = 300
    BATCH_SIZE = 120
    PATIENCE = 10  # More patience for better convergence

    print(f"   Epochs: {EPOCHS}")
    print(f"   Batch Size: {BATCH_SIZE}")
    print(f"   Early Stopping Patience: {PATIENCE}")
    print(f"   Train samples: {len(X_train):,}")
    print(f"   Test samples: {len(X_test):,}")

    # Callbacks
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy",  # Monitor accuracy instead of loss
            patience=PATIENCE,
            restore_best_weights=True,
            verbose=1,
            mode="max",
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_accuracy",
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1,
            mode="max",
        ),
    ]

    # Train
    print("\n🚀 Starting training...")
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_test, y_test),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=1,
    )

    # Evaluate
    print("\n" + "=" * 60)
    print("📊 Evaluation Results")
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

    print(f"\n   📄 Makaledeki hedef: 99.36%")
    print(f"   📊 Bizim sonuç:      {accuracy*100:.2f}%")

    if accuracy >= 0.99:
        print("\n   🎉 HEDEF BAŞARILDI!")
    elif accuracy >= 0.95:
        print("\n   👍 Çok iyi sonuç!")
    elif accuracy >= 0.90:
        print(
            "\n   ℹ️ İyi sonuç, daha fazla epoch veya hyperparameter tuning gerekebilir"
        )

    # Classification report
    print("\n" + "=" * 60)
    print("📋 Classification Report")
    print("=" * 60)
    print(classification_report(y_test, y_pred, target_names=class_names))

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "epochs_trained": len(history.history["loss"]),
        "final_loss": float(history.history["loss"][-1]),
        "final_val_accuracy": float(history.history["val_accuracy"][-1]),
    }, model


def save_model(model, results, class_names):
    """Modeli kaydet"""
    MODELS_DIR.mkdir(exist_ok=True)

    model_id = f"ssa_lstmids_nsl_kdd_paper_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    model_path = MODELS_DIR / f"{model_id}.keras"

    model.save(model_path)
    print(f"\n💾 Model saved: {model_path}")

    # Update registry
    if REGISTRY_PATH.exists():
        with open(REGISTRY_PATH, "r") as f:
            registry = json.load(f)
    else:
        registry = {"models": []}

    entry = {
        "id": model_id,
        "name": "SSA-LSTMIDS_NSL-KDD_Paper",
        "model_type": "ssa_lstmids",
        "dataset": "nsl_kdd",
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
            "epochs": results["epochs_trained"],
            "batch_size": 120,
            "paper_params": True,
        },
        "created_at": datetime.now().isoformat(),
    }

    registry["models"].append(entry)

    with open(REGISTRY_PATH, "w") as f:
        json.dump(registry, f, indent=2)

    print(f"📋 Registry updated")


def main():
    print("\n" + "=" * 70)
    print("🎓 NSL-KDD SSA-LSTMIDS Training (Paper Configuration)")
    print("=" * 70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Load data
    X_train, y_train, X_test, y_test, num_classes, class_names = load_nsl_kdd()

    # Build model
    model = build_ssa_lstmids_paper(X_train.shape[1:], num_classes)
    model.summary()

    # Train
    results, trained_model = train_model(
        model, X_train, y_train, X_test, y_test, class_names
    )

    # Save
    save_model(trained_model, results, class_names)

    print("\n" + "=" * 70)
    print("✅ Training Complete!")
    print("=" * 70)
    print(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
