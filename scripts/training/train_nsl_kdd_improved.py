"""
NSL-KDD SSA-LSTMIDS - İyileştirilmiş Eğitim
=============================================

İyileştirmeler:
1. Stratified Train-Val-Test Split (Train verisinden)
2. SMOTE ile class imbalance düzeltme
3. Dropout ve L2 regularization
4. Learning rate scheduling
5. Batch normalization
6. Binary ve Multi-class karşılaştırma

Hedef: %99+ accuracy
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from collections import Counter
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
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


def load_and_prepare_data(binary_classification=False, use_smote=True):
    """Veri yükle ve hazırla - Stratified split ile"""
    print("\n" + "=" * 60)
    print("📊 Loading NSL-KDD Dataset (Improved)")
    print("=" * 60)

    train_path = DATA_DIR / "KDDTrain+.txt"

    df = pd.read_csv(train_path, names=COLUMNS)
    print(f"   Total samples: {len(df):,}")

    # Map to classes
    if binary_classification:
        df["attack_cat"] = df["label"].apply(
            lambda x: "Attack" if x != "normal" else "Normal"
        )
        print("   Mode: Binary Classification (Normal vs Attack)")
    else:
        df["attack_cat"] = df["label"].map(ATTACK_MAP).fillna("Other")
        print("   Mode: Multi-class Classification (5 classes)")

    # One-hot encode categorical
    categorical_cols = ["protocol_type", "service", "flag"]
    df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=False)

    # Label encode target
    le = LabelEncoder()
    y = le.fit_transform(df_encoded["attack_cat"])

    print(f"   Classes: {list(le.classes_)}")
    print(f"   Class distribution: {dict(Counter(y))}")

    # Feature columns
    drop_cols = ["label", "attack_cat", "difficulty"]
    feature_cols = [c for c in df_encoded.columns if c not in drop_cols]
    X = df_encoded[feature_cols].values.astype(np.float32)

    print(f"   Features: {X.shape[1]}")

    # STRATIFIED SPLIT - Train verisinin kendisinden
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.1, random_state=42, stratify=y_train
    )

    print(f"\n   Train: {len(X_train):,}")
    print(f"   Val: {len(X_val):,}")
    print(f"   Test: {len(X_test):,}")

    # SMOTE oversampling
    if use_smote:
        try:
            from imblearn.over_sampling import SMOTE

            print("\n   🔄 Applying SMOTE oversampling...")
            smote = SMOTE(random_state=42)
            X_train, y_train = smote.fit_resample(X_train, y_train)
            print(f"   Train after SMOTE: {len(X_train):,}")
            print(f"   Class distribution after SMOTE: {dict(Counter(y_train))}")
        except ImportError:
            print("   ⚠️ imblearn not installed, skipping SMOTE")

    # Scale
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    # Reshape for LSTM - (samples, timesteps, features)
    n_features = X_train.shape[1]
    # Keep features grouped properly for Conv1D
    timesteps = 10
    if n_features % timesteps != 0:
        pad_size = timesteps - (n_features % timesteps)
        X_train = np.pad(X_train, ((0, 0), (0, pad_size)), mode="constant")
        X_val = np.pad(X_val, ((0, 0), (0, pad_size)), mode="constant")
        X_test = np.pad(X_test, ((0, 0), (0, pad_size)), mode="constant")

    features_per_step = X_train.shape[1] // timesteps
    X_train = X_train.reshape(X_train.shape[0], timesteps, features_per_step)
    X_val = X_val.reshape(X_val.shape[0], timesteps, features_per_step)
    X_test = X_test.reshape(X_test.shape[0], timesteps, features_per_step)

    print(f"\n   X_train shape: {X_train.shape}")

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


def build_improved_model(input_shape, num_classes):
    """
    İyileştirilmiş SSA-LSTMIDS modeli

    İyileştirmeler:
    - Batch Normalization
    - Daha yüksek Dropout
    - L2 Regularization
    - Bidirectional LSTM
    """
    print("\n" + "=" * 60)
    print("🧠 Building Improved SSA-LSTMIDS Model")
    print("=" * 60)

    l2_reg = keras.regularizers.l2(0.001)

    inputs = keras.layers.Input(shape=input_shape, name="input")

    # Conv1D with BatchNorm
    x = keras.layers.Conv1D(
        filters=64,
        kernel_size=3,
        padding="same",
        activation="relu",
        kernel_regularizer=l2_reg,
        name="conv1d_1",
    )(inputs)
    x = keras.layers.BatchNormalization()(x)

    x = keras.layers.Conv1D(
        filters=32,
        kernel_size=3,
        padding="same",
        activation="relu",
        kernel_regularizer=l2_reg,
        name="conv1d_2",
    )(x)
    x = keras.layers.BatchNormalization()(x)

    # MaxPooling
    x = keras.layers.MaxPooling1D(pool_size=2, padding="same", name="maxpool")(x)

    # Bidirectional LSTM
    x = keras.layers.Bidirectional(
        keras.layers.LSTM(
            units=128,
            activation="tanh",
            recurrent_activation="sigmoid",
            return_sequences=True,
            dropout=0.2,
            recurrent_dropout=0.2,
            kernel_regularizer=l2_reg,
        ),
        name="bilstm_1",
    )(x)

    x = keras.layers.Bidirectional(
        keras.layers.LSTM(
            units=64,
            activation="tanh",
            recurrent_activation="sigmoid",
            return_sequences=False,
            dropout=0.2,
            recurrent_dropout=0.2,
            kernel_regularizer=l2_reg,
        ),
        name="bilstm_2",
    )(x)

    # Dense layers with heavy dropout
    x = keras.layers.Dense(
        256, activation="relu", kernel_regularizer=l2_reg, name="dense_1"
    )(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Dropout(0.4, name="dropout_1")(x)

    x = keras.layers.Dense(
        128, activation="relu", kernel_regularizer=l2_reg, name="dense_2"
    )(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Dropout(0.3, name="dropout_2")(x)

    # Output
    outputs = keras.layers.Dense(num_classes, activation="softmax", name="output")(x)

    model = keras.Model(inputs, outputs, name="Improved_SSA_LSTMIDS")

    # Compile with standard learning rate (ReduceLROnPlateau will handle decay)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    print(f"   Conv1D: 64→32 filters")
    print(f"   BiLSTM: 128→64 units")
    print(f"   Dense: 256→128 units")
    print(f"   Dropout: 0.4, 0.3")
    print(f"   L2 Regularization: 0.001")
    print(f"   Total params: {model.count_params():,}")

    return model


def train_and_evaluate(
    model, X_train, y_train, X_val, y_val, X_test, y_test, class_names
):
    """Eğit ve değerlendir"""
    print("\n" + "=" * 60)
    print("🏋️ Training Improved Model")
    print("=" * 60)

    EPOCHS = 100
    BATCH_SIZE = 256
    PATIENCE = 15

    print(f"   Epochs: {EPOCHS}")
    print(f"   Batch Size: {BATCH_SIZE}")
    print(f"   Early Stopping Patience: {PATIENCE}")

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
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

    print("\n🚀 Starting training...")
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=1,
    )

    # Evaluate on TEST set
    print("\n" + "=" * 60)
    print("📊 Test Set Evaluation")
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
    elif accuracy >= 0.98:
        print("\n   🔥 Çok yakın!")
    elif accuracy >= 0.95:
        print("\n   👍 Çok iyi sonuç!")

    # Classification Report
    print("\n" + "=" * 60)
    print("📋 Classification Report")
    print("=" * 60)
    print(classification_report(y_test, y_pred, target_names=class_names))

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    print("\nConfusion Matrix:")
    print(cm)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "epochs_trained": len(history.history["loss"]),
        "best_val_accuracy": max(history.history["val_accuracy"]),
    }, model


def save_model(model, results, mode):
    """Modeli kaydet"""
    MODELS_DIR.mkdir(exist_ok=True)

    model_id = f"ssa_lstmids_nsl_kdd_improved_{mode}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
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
        "name": f"SSA-LSTMIDS_NSL-KDD_Improved_{mode}",
        "model_type": "ssa_lstmids_improved",
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
            "mode": mode,
            "improvements": [
                "stratified_split",
                "smote",
                "bilstm",
                "batch_norm",
                "l2_reg",
            ],
        },
        "created_at": datetime.now().isoformat(),
    }

    registry["models"].append(entry)

    with open(REGISTRY_PATH, "w") as f:
        json.dump(registry, f, indent=2)

    print(f"📋 Registry updated")


def main():
    print("\n" + "=" * 70)
    print("🎓 NSL-KDD SSA-LSTMIDS - İYİLEŞTİRİLMİŞ EĞİTİM")
    print("=" * 70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # BINARY CLASSIFICATION (Normal vs Attack)
    print("\n\n" + "🔷" * 30)
    print("BINARY CLASSIFICATION (Normal vs Attack)")
    print("🔷" * 30)

    X_train, y_train, X_val, y_val, X_test, y_test, num_classes, class_names = (
        load_and_prepare_data(binary_classification=True, use_smote=True)
    )

    model_binary = build_improved_model(X_train.shape[1:], num_classes)
    results_binary, trained_binary = train_and_evaluate(
        model_binary, X_train, y_train, X_val, y_val, X_test, y_test, class_names
    )
    save_model(trained_binary, results_binary, "binary")

    # MULTI-CLASS CLASSIFICATION (5 classes)
    print("\n\n" + "🔶" * 30)
    print("MULTI-CLASS CLASSIFICATION (5 classes)")
    print("🔶" * 30)

    X_train, y_train, X_val, y_val, X_test, y_test, num_classes, class_names = (
        load_and_prepare_data(binary_classification=False, use_smote=True)
    )

    model_multi = build_improved_model(X_train.shape[1:], num_classes)
    results_multi, trained_multi = train_and_evaluate(
        model_multi, X_train, y_train, X_val, y_val, X_test, y_test, class_names
    )
    save_model(trained_multi, results_multi, "multiclass")

    # Summary
    print("\n" + "=" * 70)
    print("📊 FINAL SUMMARY")
    print("=" * 70)
    print(f"\n   Binary (Normal vs Attack): {results_binary['accuracy']*100:.2f}%")
    print(f"   Multi-class (5 classes):   {results_multi['accuracy']*100:.2f}%")
    print(f"\n   Makaledeki hedef:          99.36%")

    print("\n" + "=" * 70)
    print("✅ Training Complete!")
    print("=" * 70)
    print(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
