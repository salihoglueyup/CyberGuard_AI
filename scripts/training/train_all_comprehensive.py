"""
Complete Model Training Script - CyberGuard AI
================================================

12 Model Eğitim Pipeline:
- Sprint 1: SSA-LSTMIDS × 3 Dataset (Makaledeki birebir)
- Sprint 2: Alternatif modeller (BiLSTM, GRU, Transformer, CNN-Transformer)
- Sprint 3: Ensemble modeller
- Sprint 4: AI Decision Layer components

Kullanım:
    python scripts/train_all_comprehensive.py --sprint 1
    python scripts/train_all_comprehensive.py --all
"""

import os
import sys
import json
import time
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional

# Proje yolu
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# TensorFlow import
try:
    import tensorflow as tf
    from tensorflow import keras

    print(f"✅ TensorFlow {tf.__version__}")
except ImportError:
    print("❌ TensorFlow gerekli!")
    sys.exit(1)

# Sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
)

# ============= DATA PATHS =============

DATA_DIR = PROJECT_ROOT / "data" / "raw"
MODELS_DIR = PROJECT_ROOT / "models"
REGISTRY_PATH = MODELS_DIR / "model_registry.json"

NSL_KDD_PATH = DATA_DIR / "nsl_kdd"
CICIDS_PATH = DATA_DIR / "cicids2017_full"
BOTIOT_PATH = DATA_DIR / "bot_iot"

# ============= DATA LOADERS =============


def load_nsl_kdd(
    sample_size: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """NSL-KDD veri yükle"""
    print("\n📊 Loading NSL-KDD dataset...")

    # Column names
    columns = [
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

    # Load train
    train_path = NSL_KDD_PATH / "KDDTrain+.txt"
    test_path = NSL_KDD_PATH / "KDDTest+.txt"

    df_train = pd.read_csv(train_path, names=columns)
    df_test = pd.read_csv(test_path, names=columns)

    print(f"   Train: {len(df_train):,} samples")
    print(f"   Test: {len(df_test):,} samples")

    # Sample if needed
    if sample_size and len(df_train) > sample_size:
        df_train = df_train.sample(n=sample_size, random_state=42)
        print(f"   Sampled to: {len(df_train):,}")

    # Label mapping (5 classes like paper)
    attack_map = {
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

    df_train["attack_cat"] = df_train["label"].map(attack_map).fillna("Other")
    df_test["attack_cat"] = df_test["label"].map(attack_map).fillna("Other")

    # Encode categorical
    for col in ["protocol_type", "service", "flag"]:
        le = LabelEncoder()
        df_train[col] = le.fit_transform(df_train[col])
        df_test[col] = le.transform(
            df_test[col].apply(lambda x: x if x in le.classes_ else le.classes_[0])
        )

    # Label encode
    le_label = LabelEncoder()
    y_train = le_label.fit_transform(df_train["attack_cat"])
    y_test = le_label.transform(df_test["attack_cat"])

    # Features (drop label columns)
    feature_cols = [
        c for c in df_train.columns if c not in ["label", "attack_cat", "difficulty"]
    ]
    X_train = df_train[feature_cols].values.astype(np.float32)
    X_test = df_test[feature_cols].values.astype(np.float32)

    # Scale
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Reshape for LSTM (samples, timesteps, features)
    # Timesteps=10 for proper MaxPooling operation
    n_features = X_train.shape[1]
    timesteps = 10
    # Pad features if needed
    if n_features % timesteps != 0:
        pad_size = timesteps - (n_features % timesteps)
        X_train = np.pad(X_train, ((0, 0), (0, pad_size)), mode="constant")
        X_test = np.pad(X_test, ((0, 0), (0, pad_size)), mode="constant")

    n_features_per_step = X_train.shape[1] // timesteps
    X_train = X_train.reshape(X_train.shape[0], timesteps, n_features_per_step)
    X_test = X_test.reshape(X_test.shape[0], timesteps, n_features_per_step)

    class_names = list(le_label.classes_)
    print(f"   Classes: {class_names}")
    print(f"   X_train shape: {X_train.shape}")

    return X_train, y_train, X_test, y_test, class_names


def load_cicids2017(
    sample_size: Optional[int] = 100000,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """CICIDS2017 veri yükle"""
    print("\n📊 Loading CICIDS2017 dataset...")

    # First check for Train_data.csv and Test_data.csv (preferred)
    train_file = CICIDS_PATH / "Train_data.csv"
    test_file = CICIDS_PATH / "Test_data.csv"

    if train_file.exists() and test_file.exists():
        print("   Using Train_data.csv / Test_data.csv")
        try:
            df_train = pd.read_csv(train_file, low_memory=False)
            df_test = pd.read_csv(test_file, low_memory=False)

            # Find label column
            label_col = None
            for col in [" Label", "Label", "label", "attack_cat", "class"]:
                if col in df_train.columns:
                    label_col = col
                    break

            if label_col is None:
                print("   ❌ Label column not found in Train_data.csv")
            else:
                # Clean
                for df in [df_train, df_test]:
                    df.replace([np.inf, -np.inf], np.nan, inplace=True)
                    df.dropna(inplace=True)

                # Features
                feature_cols = [
                    c
                    for c in df_train.columns
                    if c != label_col and df_train[c].dtype in ["int64", "float64"]
                ]
                X_train = df_train[feature_cols].values.astype(np.float32)
                X_test = df_test[feature_cols].values.astype(np.float32)

                # Replace inf/nan
                X_train = np.nan_to_num(X_train, nan=0.0, posinf=1e6, neginf=-1e6)
                X_test = np.nan_to_num(X_test, nan=0.0, posinf=1e6, neginf=-1e6)

                # Label encode
                le = LabelEncoder()
                y_train = le.fit_transform(df_train[label_col])
                y_test = le.transform(
                    df_test[label_col].apply(
                        lambda x: x if x in le.classes_ else le.classes_[0]
                    )
                )

                # Scale
                scaler = StandardScaler()
                X_train = scaler.fit_transform(X_train)
                X_test = scaler.transform(X_test)

                # Reshape
                timesteps = 10
                n_features = X_train.shape[1]
                if n_features % timesteps != 0:
                    pad_size = timesteps - (n_features % timesteps)
                    X_train = np.pad(X_train, ((0, 0), (0, pad_size)), mode="constant")
                    X_test = np.pad(X_test, ((0, 0), (0, pad_size)), mode="constant")
                n_features_per_step = X_train.shape[1] // timesteps
                X_train = X_train.reshape(
                    X_train.shape[0], timesteps, n_features_per_step
                )
                X_test = X_test.reshape(X_test.shape[0], timesteps, n_features_per_step)

                class_names = list(le.classes_)
                print(f"   Train: {len(X_train):,}, Test: {len(X_test):,}")
                print(f"   Classes: {class_names}")
                print(f"   X_train shape: {X_train.shape}")

                return X_train, y_train, X_test, y_test, class_names

        except Exception as e:
            print(f"   ⚠️ Error loading Train/Test files: {e}")

    # Fallback to individual CSV files
    csv_files = list(CICIDS_PATH.glob("*.csv"))
    csv_files = [
        f for f in csv_files if f.name not in ["Train_data.csv", "Test_data.csv"]
    ]

    if not csv_files:
        print("   ❌ No CSV files found in CICIDS path")
        return None, None, None, None, None

    print(f"   Found {len(csv_files)} CSV files")

    # Load and combine (sample from each)
    dfs = []
    samples_per_file = sample_size // len(csv_files) if sample_size else None

    for f in csv_files[:5]:  # Limit to 5 files for speed
        try:
            df = pd.read_csv(f, low_memory=False)
            if samples_per_file and len(df) > samples_per_file:
                df = df.sample(n=samples_per_file, random_state=42)
            dfs.append(df)
            print(f"   Loaded {f.name}: {len(df):,} samples")
        except Exception as e:
            print(f"   ⚠️ Error loading {f.name}: {e}")

    if not dfs:
        return None, None, None, None, None

    df = pd.concat(dfs, ignore_index=True)
    print(f"   Total: {len(df):,} samples")

    # Find label column
    label_col = None
    for col in [" Label", "Label", "label", "attack_cat"]:
        if col in df.columns:
            label_col = col
            break

    if label_col is None:
        print("   ❌ Label column not found")
        return None, None, None, None, None

    # Clean more aggressively
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna()

    # Features and labels
    feature_cols = [
        c for c in df.columns if c != label_col and df[c].dtype in ["int64", "float64"]
    ]
    X = df[feature_cols].values.astype(np.float32)

    # Replace any remaining inf/nan
    X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)

    le = LabelEncoder()
    y = le.fit_transform(df[label_col])

    # Scale with error handling
    scaler = StandardScaler()
    try:
        X = scaler.fit_transform(X)
    except ValueError as e:
        print(f"   ⚠️ Scaler error: {e}")
        # Clip extreme values
        X = np.clip(X, -1e9, 1e9)
        X = scaler.fit_transform(X)

    # Split - try stratified first, fallback to random
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
    except ValueError:
        print("   ⚠️ Stratified split failed, using random split")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

    # Reshape for LSTM (timesteps=10)
    timesteps = 10
    n_features = X_train.shape[1]
    if n_features % timesteps != 0:
        pad_size = timesteps - (n_features % timesteps)
        X_train = np.pad(X_train, ((0, 0), (0, pad_size)), mode="constant")
        X_test = np.pad(X_test, ((0, 0), (0, pad_size)), mode="constant")
    n_features_per_step = X_train.shape[1] // timesteps
    X_train = X_train.reshape(X_train.shape[0], timesteps, n_features_per_step)
    X_test = X_test.reshape(X_test.shape[0], timesteps, n_features_per_step)

    class_names = list(le.classes_)
    print(
        f"   Classes: {class_names[:5]}..."
        if len(class_names) > 5
        else f"   Classes: {class_names}"
    )
    print(f"   X_train shape: {X_train.shape}")

    return X_train, y_train, X_test, y_test, class_names


def load_botiot(
    sample_size: Optional[int] = 100000,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """BoT-IoT veri yükle"""
    print("\n📊 Loading BoT-IoT dataset...")

    csv_files = list(BOTIOT_PATH.glob("*.csv"))
    if not csv_files:
        print("   ❌ No CSV files found")
        return None, None, None, None, None

    print(f"   Found {len(csv_files)} CSV files")

    # Load first file(s)
    dfs = []
    for f in csv_files[:3]:
        try:
            df = pd.read_csv(
                f, low_memory=False, nrows=sample_size // 3 if sample_size else None
            )
            dfs.append(df)
            print(f"   Loaded {f.name}: {len(df):,} samples")
        except Exception as e:
            print(f"   ⚠️ Error: {e}")

    if not dfs:
        return None, None, None, None, None

    df = pd.concat(dfs, ignore_index=True)

    # Find label column
    label_col = None
    for col in ["category", "attack", "label", "Label"]:
        if col in df.columns:
            label_col = col
            break

    if label_col is None:
        print("   ❌ Label column not found")
        return None, None, None, None, None

    # Clean
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna()

    # Features
    feature_cols = [
        c for c in df.columns if c != label_col and df[c].dtype in ["int64", "float64"]
    ]
    if len(feature_cols) < 5:
        print(f"   ❌ Not enough numeric features: {len(feature_cols)}")
        return None, None, None, None, None

    X = df[feature_cols].values.astype(np.float32)

    # Replace any remaining inf/nan
    X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)

    le = LabelEncoder()
    y = le.fit_transform(df[label_col])

    # Scale with error handling
    scaler = StandardScaler()
    try:
        X = scaler.fit_transform(X)
    except ValueError as e:
        print(f"   ⚠️ Scaler error: {e}")
        X = np.clip(X, -1e9, 1e9)
        X = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Reshape for LSTM (timesteps=10)
    timesteps = 10
    n_features = X_train.shape[1]
    if n_features % timesteps != 0:
        pad_size = timesteps - (n_features % timesteps)
        X_train = np.pad(X_train, ((0, 0), (0, pad_size)), mode="constant")
        X_test = np.pad(X_test, ((0, 0), (0, pad_size)), mode="constant")
    n_features_per_step = X_train.shape[1] // timesteps
    X_train = X_train.reshape(X_train.shape[0], timesteps, n_features_per_step)
    X_test = X_test.reshape(X_test.shape[0], timesteps, n_features_per_step)

    class_names = list(le.classes_)
    print(f"   Classes: {class_names}")
    print(f"   X_train shape: {X_train.shape}")

    return X_train, y_train, X_test, y_test, class_names


# ============= MODEL BUILDERS =============


def build_ssa_lstmids(input_shape: Tuple, num_classes: int):
    """
    Makaledeki SSA-LSTMIDS modeli - Inline tanım

    Mimari (Makaledeki):
        Input → Conv1D(30) → MaxPool → LSTM(120) → Dense(512) → Dropout(0.2) → Output

    SSA ile optimize edilmiş parametreler:
        - Conv1D filters: 30
        - LSTM units: 120
        - Dense units: 512
        - Dropout: 0.2
    """
    # Paper parameters
    conv_filters = 30
    kernel_size = 5
    lstm_units = 120
    dense_units = 512
    dropout_rate = 0.2

    inputs = keras.layers.Input(shape=input_shape, name="input")

    # Conv1D Layer - Feature extraction
    x = keras.layers.Conv1D(
        filters=conv_filters,
        kernel_size=kernel_size,
        padding="same",
        activation="relu",
        name="conv1d",
    )(inputs)

    # MaxPooling
    x = keras.layers.MaxPooling1D(pool_size=2, name="maxpool")(x)

    # LSTM Layer - Temporal pattern learning
    x = keras.layers.LSTM(
        units=lstm_units,
        activation="tanh",
        recurrent_activation="sigmoid",
        return_sequences=False,
        name="lstm",
    )(x)

    # Dense Layer
    x = keras.layers.Dense(units=dense_units, activation="relu", name="dense")(x)

    # Dropout
    x = keras.layers.Dropout(rate=dropout_rate, name="dropout")(x)

    # Output Layer
    outputs = keras.layers.Dense(
        units=num_classes, activation="softmax", name="output"
    )(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="SSA_LSTMIDS")

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    print("🧠 SSA-LSTMIDS Model (Paper params):")
    print(f"   Conv1D: {conv_filters} filters, kernel={kernel_size}")
    print(f"   LSTM: {lstm_units} units")
    print(f"   Dense: {dense_units} units")
    print(f"   Dropout: {dropout_rate}")
    print(f"   Total params: {model.count_params():,}")

    return model


def build_bilstm_attention(input_shape: Tuple, num_classes: int) -> keras.Model:
    """BiLSTM + Attention modeli"""
    inputs = keras.layers.Input(shape=input_shape)

    x = keras.layers.Bidirectional(keras.layers.LSTM(64, return_sequences=True))(inputs)
    x = keras.layers.Bidirectional(keras.layers.LSTM(32, return_sequences=True))(x)

    # Simple attention
    attention = keras.layers.Dense(1, activation="tanh")(x)
    attention = keras.layers.Flatten()(attention)
    attention = keras.layers.Activation("softmax")(attention)
    attention = keras.layers.RepeatVector(64)(attention)
    attention = keras.layers.Permute([2, 1])(attention)

    x = keras.layers.GlobalAveragePooling1D()(x)
    x = keras.layers.Dense(128, activation="relu")(x)
    x = keras.layers.Dropout(0.3)(x)
    outputs = keras.layers.Dense(num_classes, activation="softmax")(x)

    model = keras.Model(inputs, outputs, name="BiLSTM_Attention")
    model.compile(
        optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
    )
    return model


def build_gru_ids(input_shape: Tuple, num_classes: int) -> keras.Model:
    """GRU-IDS modeli (IoT için hafif)"""
    inputs = keras.layers.Input(shape=input_shape)

    x = keras.layers.GRU(64, return_sequences=True)(inputs)
    x = keras.layers.GRU(32)(x)
    x = keras.layers.Dense(64, activation="relu")(x)
    x = keras.layers.Dropout(0.2)(x)
    outputs = keras.layers.Dense(num_classes, activation="softmax")(x)

    model = keras.Model(inputs, outputs, name="GRU_IDS")
    model.compile(
        optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
    )
    return model


def build_transformer_ids(input_shape: Tuple, num_classes: int) -> keras.Model:
    """Transformer-IDS modeli"""
    inputs = keras.layers.Input(shape=input_shape)

    # Positional encoding (simplified)
    x = keras.layers.Dense(64)(inputs)

    # Multi-head attention
    attention_output = keras.layers.MultiHeadAttention(num_heads=4, key_dim=16)(x, x)
    x = keras.layers.Add()([x, attention_output])
    x = keras.layers.LayerNormalization()(x)

    # Feed-forward
    ff = keras.layers.Dense(128, activation="relu")(x)
    ff = keras.layers.Dense(64)(ff)
    x = keras.layers.Add()([x, ff])
    x = keras.layers.LayerNormalization()(x)

    x = keras.layers.GlobalAveragePooling1D()(x)
    x = keras.layers.Dense(64, activation="relu")(x)
    x = keras.layers.Dropout(0.2)(x)
    outputs = keras.layers.Dense(num_classes, activation="softmax")(x)

    model = keras.Model(inputs, outputs, name="Transformer_IDS")
    model.compile(
        optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
    )
    return model


def build_cnn_transformer(input_shape: Tuple, num_classes: int) -> keras.Model:
    """CNN-Transformer Hybrid modeli"""
    inputs = keras.layers.Input(shape=input_shape)

    # CNN feature extraction
    x = keras.layers.Conv1D(32, 3, padding="same", activation="relu")(inputs)
    x = keras.layers.Conv1D(64, 3, padding="same", activation="relu")(x)

    # Transformer encoder
    attention_output = keras.layers.MultiHeadAttention(num_heads=4, key_dim=16)(x, x)
    x = keras.layers.Add()([x, attention_output])
    x = keras.layers.LayerNormalization()(x)

    x = keras.layers.GlobalAveragePooling1D()(x)
    x = keras.layers.Dense(128, activation="relu")(x)
    x = keras.layers.Dropout(0.3)(x)
    outputs = keras.layers.Dense(num_classes, activation="softmax")(x)

    model = keras.Model(inputs, outputs, name="CNN_Transformer")
    model.compile(
        optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
    )
    return model


# ============= TRAINING FUNCTIONS =============


def train_model(
    model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    epochs: int = 50,
    batch_size: int = 128,
    model_name: str = "model",
) -> Dict:
    """Model eğit ve değerlendir"""

    print(f"\n🏋️ Training {model_name}...")
    print(f"   Epochs: {epochs}, Batch: {batch_size}")

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=5, restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3),
    ]

    start_time = time.time()

    # SSA_LSTMIDS has its own training method
    if hasattr(model, "train") and hasattr(model, "model"):
        history = model.train(
            X_train, y_train, X_test, y_test, epochs=epochs, batch_size=batch_size
        )
        keras_model = model.model
    else:
        history = model.fit(
            X_train,
            y_train,
            validation_data=(X_test, y_test),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1,
        )
        keras_model = model

    train_time = time.time() - start_time

    # Evaluate
    y_pred = np.argmax(keras_model.predict(X_test, verbose=0), axis=1)

    results = {
        "model_name": model_name,
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(
            precision_score(y_test, y_pred, average="weighted", zero_division=0)
        ),
        "recall": float(
            recall_score(y_test, y_pred, average="weighted", zero_division=0)
        ),
        "f1_score": float(
            f1_score(y_test, y_pred, average="weighted", zero_division=0)
        ),
        "train_time_seconds": train_time,
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "epochs_trained": (
            len(history.history["loss"]) if hasattr(history, "history") else epochs
        ),
        "timestamp": datetime.now().isoformat(),
    }

    print(f"\n✅ {model_name} Results:")
    print(f"   Accuracy: {results['accuracy']*100:.2f}%")
    print(f"   F1-Score: {results['f1_score']:.4f}")
    print(f"   Train time: {train_time:.1f}s")

    return results, keras_model


def save_to_registry(model: keras.Model, results: Dict, dataset: str, model_type: str):
    """Model registry'e kaydet"""

    MODELS_DIR.mkdir(exist_ok=True)

    # Model kaydet
    model_id = f"{model_type}_{dataset}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    model_path = MODELS_DIR / f"{model_id}.keras"
    model.save(model_path)

    # Registry yükle/oluştur
    if REGISTRY_PATH.exists():
        with open(REGISTRY_PATH, "r") as f:
            registry = json.load(f)
    else:
        registry = {"models": []}

    # Model entry
    entry = {
        "id": model_id,
        "name": results["model_name"],
        "model_type": model_type,
        "dataset": dataset,
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
            "train_samples": results["train_samples"],
            "test_samples": results["test_samples"],
            "epochs": results["epochs_trained"],
            "train_time_seconds": results["train_time_seconds"],
        },
        "created_at": results["timestamp"],
    }

    registry["models"].append(entry)

    with open(REGISTRY_PATH, "w") as f:
        json.dump(registry, f, indent=2)

    print(f"💾 Saved to registry: {model_id}")


# ============= SPRINT FUNCTIONS =============


def run_sprint_1():
    """Sprint 1: SSA-LSTMIDS × 3 Dataset"""
    print("\n" + "=" * 60)
    print("🔴 SPRINT 1: SSA-LSTMIDS (Makaledeki Birebir)")
    print("=" * 60)

    results_all = []

    # 1. NSL-KDD
    print("\n" + "-" * 40)
    print("📊 Dataset 1/3: NSL-KDD")
    print("-" * 40)

    X_train, y_train, X_test, y_test, classes = load_nsl_kdd()

    if X_train is not None:
        model = build_ssa_lstmids(X_train.shape[1:], len(classes))
        results, keras_model = train_model(
            model,
            X_train,
            y_train,
            X_test,
            y_test,
            epochs=100,
            batch_size=120,
            model_name="SSA-LSTMIDS_NSL-KDD",
        )
        save_to_registry(keras_model, results, "nsl_kdd", "ssa_lstmids")
        results_all.append(results)

    # 2. CICIDS2017
    print("\n" + "-" * 40)
    print("📊 Dataset 2/3: CICIDS2017")
    print("-" * 40)

    X_train, y_train, X_test, y_test, classes = load_cicids2017(sample_size=100000)

    if X_train is not None:
        model = build_ssa_lstmids(X_train.shape[1:], len(classes))
        results, keras_model = train_model(
            model,
            X_train,
            y_train,
            X_test,
            y_test,
            epochs=100,
            batch_size=120,
            model_name="SSA-LSTMIDS_CICIDS2017",
        )
        save_to_registry(keras_model, results, "cicids2017", "ssa_lstmids")
        results_all.append(results)

    # 3. BoT-IoT
    print("\n" + "-" * 40)
    print("📊 Dataset 3/3: BoT-IoT")
    print("-" * 40)

    X_train, y_train, X_test, y_test, classes = load_botiot(sample_size=100000)

    if X_train is not None:
        model = build_ssa_lstmids(X_train.shape[1:], len(classes))
        results, keras_model = train_model(
            model,
            X_train,
            y_train,
            X_test,
            y_test,
            epochs=100,
            batch_size=120,
            model_name="SSA-LSTMIDS_BoT-IoT",
        )
        save_to_registry(keras_model, results, "bot_iot", "ssa_lstmids")
        results_all.append(results)

    return results_all


def run_sprint_2():
    """Sprint 2: Alternatif Modeller"""
    print("\n" + "=" * 60)
    print("🟡 SPRINT 2: Alternatif Modeller")
    print("=" * 60)

    results_all = []

    # CICIDS2017 veri yükle (tüm modeller için)
    X_train, y_train, X_test, y_test, classes = load_cicids2017(sample_size=80000)

    if X_train is None:
        print("❌ CICIDS2017 yüklenemedi!")
        return results_all

    # 1. BiLSTM+Attention
    print("\n" + "-" * 40)
    print("🧠 Model 1/4: BiLSTM+Attention")
    print("-" * 40)

    model = build_bilstm_attention(X_train.shape[1:], len(classes))
    results, keras_model = train_model(
        model,
        X_train,
        y_train,
        X_test,
        y_test,
        epochs=50,
        batch_size=128,
        model_name="BiLSTM_Attention",
    )
    save_to_registry(keras_model, results, "cicids2017", "bilstm_attention")
    results_all.append(results)

    # 2. GRU-IDS
    print("\n" + "-" * 40)
    print("🧠 Model 2/4: GRU-IDS")
    print("-" * 40)

    model = build_gru_ids(X_train.shape[1:], len(classes))
    results, keras_model = train_model(
        model,
        X_train,
        y_train,
        X_test,
        y_test,
        epochs=50,
        batch_size=128,
        model_name="GRU_IDS",
    )
    save_to_registry(keras_model, results, "cicids2017", "gru_ids")
    results_all.append(results)

    # 3. Transformer-IDS
    print("\n" + "-" * 40)
    print("🧠 Model 3/4: Transformer-IDS")
    print("-" * 40)

    model = build_transformer_ids(X_train.shape[1:], len(classes))
    results, keras_model = train_model(
        model,
        X_train,
        y_train,
        X_test,
        y_test,
        epochs=50,
        batch_size=128,
        model_name="Transformer_IDS",
    )
    save_to_registry(keras_model, results, "cicids2017", "transformer_ids")
    results_all.append(results)

    # 4. CNN-Transformer
    print("\n" + "-" * 40)
    print("🧠 Model 4/4: CNN-Transformer")
    print("-" * 40)

    model = build_cnn_transformer(X_train.shape[1:], len(classes))
    results, keras_model = train_model(
        model,
        X_train,
        y_train,
        X_test,
        y_test,
        epochs=50,
        batch_size=128,
        model_name="CNN_Transformer",
    )
    save_to_registry(keras_model, results, "cicids2017", "cnn_transformer")
    results_all.append(results)

    return results_all


def run_sprint_3():
    """Sprint 3: Ensemble Modeller"""
    print("\n" + "=" * 60)
    print("🟢 SPRINT 3: Ensemble Modeller")
    print("=" * 60)

    # Bu sprint için önce base modellerin eğitilmiş olması gerekir
    # Şimdilik skip, sonra implement edilecek
    print("⚠️ Ensemble modeller için önce base modeller eğitilmeli!")
    print("   Sprint 1 ve 2 tamamlandıktan sonra çalıştırın.")

    return []


def run_sprint_4():
    """Sprint 4: AI Decision Layer Components"""
    print("\n" + "=" * 60)
    print("🔵 SPRINT 4: AI Decision Layer")
    print("=" * 60)

    # CICIDS2017 normal verileri yükle
    X_train, y_train, X_test, y_test, classes = load_cicids2017(sample_size=50000)

    if X_train is None:
        print("❌ Veri yüklenemedi!")
        return []

    results_all = []

    # 1. VAE Zero-Day
    print("\n" + "-" * 40)
    print("🧠 Component 1/3: VAE Zero-Day")
    print("-" * 40)

    try:
        from src.ai_decision.zero_day_detector import ZeroDayDetector

        # Normal verileri filtrele
        normal_mask = y_train == 0  # Assuming 0 is normal class
        X_normal = X_train[normal_mask].reshape(-1, X_train.shape[-1])

        if len(X_normal) > 100:
            detector = ZeroDayDetector(input_dim=X_normal.shape[1])
            detector.build()
            vae_result = detector.fit(X_normal, epochs=30, verbose=1)
            print(f"   ✅ VAE trained! Threshold: {detector.threshold:.4f}")
            results_all.append({"component": "vae", "trained": True, **vae_result})
    except Exception as e:
        print(f"   ❌ VAE error: {e}")

    # 2. RL Threshold Agent
    print("\n" + "-" * 40)
    print("🧠 Component 2/3: RL Threshold Agent")
    print("-" * 40)

    try:
        from src.ai_decision.rl_threshold import RLThresholdAgent

        agent = RLThresholdAgent()
        rl_result = agent.train(episodes=100, steps_per_episode=50)
        print(
            f"   ✅ RL Agent trained! Episodes: {rl_result.get('total_episodes', 100)}"
        )
        results_all.append({"component": "rl", "trained": True, **rl_result})
    except Exception as e:
        print(f"   ❌ RL error: {e}")

    # 3. Meta-Selector (placeholder - needs trained models)
    print("\n" + "-" * 40)
    print("🧠 Component 3/3: Meta-Selector")
    print("-" * 40)
    print("   ⚠️ Meta-selector için eğitilmiş modeller gerekli!")

    return results_all


# ============= MAIN =============


def main():
    parser = argparse.ArgumentParser(description="CyberGuard AI Model Training")
    parser.add_argument(
        "--sprint", type=int, choices=[1, 2, 3, 4], help="Sprint number to run"
    )
    parser.add_argument("--all", action="store_true", help="Run all sprints")
    args = parser.parse_args()

    print("\n" + "=" * 70)
    print("🎓 CyberGuard AI - Comprehensive Model Training")
    print("=" * 70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    all_results = []

    if args.all:
        all_results.extend(run_sprint_1())
        all_results.extend(run_sprint_2())
        all_results.extend(run_sprint_3())
        all_results.extend(run_sprint_4())
    elif args.sprint == 1:
        all_results = run_sprint_1()
    elif args.sprint == 2:
        all_results = run_sprint_2()
    elif args.sprint == 3:
        all_results = run_sprint_3()
    elif args.sprint == 4:
        all_results = run_sprint_4()
    else:
        print("\n⚠️ Kullanım: python train_all_comprehensive.py --sprint 1")
        print("        veya: python train_all_comprehensive.py --all")
        return

    # Summary
    print("\n" + "=" * 70)
    print("📊 TRAINING SUMMARY")
    print("=" * 70)

    for r in all_results:
        if "model_name" in r:
            print(
                f"   {r['model_name']}: Acc={r['accuracy']*100:.2f}%, F1={r['f1_score']:.4f}"
            )
        elif "component" in r:
            print(f"   {r['component'].upper()}: Trained={r.get('trained', False)}")

    print(f"\nCompleted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)


if __name__ == "__main__":
    main()
