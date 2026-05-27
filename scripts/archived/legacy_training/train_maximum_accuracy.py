"""
Maximum Accuracy Training Script
=================================

CICIDS2017 ve BoT-IoT için makaledeki sonuçlara ulaşmak için:
1. Full dataset kullanımı
2. Deeper model mimarisi
3. 300 epoch eğitim
4. SSA-optimized parametreler
5. Advanced preprocessing

Hedef:
- CICIDS2017: %99.88
- BoT-IoT: %99.99

Kullanım:
    python scripts/train_maximum_accuracy.py
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

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
from tensorflow import keras

print(f"✅ TensorFlow {tf.__version__}")
print(f"✅ GPU: {tf.config.list_physical_devices('GPU')}")

MODELS_DIR = PROJECT_ROOT / "models"
REGISTRY_PATH = MODELS_DIR / "model_registry.json"


# ==================== DEEP MODEL ====================


def build_deep_ssa_lstmids(input_shape, num_classes):
    """
    Derin SSA-LSTMIDS modeli - makalenin ötesinde

    Improvements:
    - Multiple Conv1D layers
    - Bidirectional LSTM with attention
    - Batch Normalization everywhere
    - Residual connection
    - L2 regularization
    """
    l2_reg = keras.regularizers.l2(0.0005)

    inputs = keras.layers.Input(shape=input_shape, name="input")

    # ======= CONV BLOCK 1 =======
    x = keras.layers.Conv1D(
        64,
        3,
        padding="same",
        activation="relu",
        kernel_regularizer=l2_reg,
        name="conv1",
    )(inputs)
    x = keras.layers.BatchNormalization(name="bn1")(x)
    x = keras.layers.Conv1D(
        64,
        3,
        padding="same",
        activation="relu",
        kernel_regularizer=l2_reg,
        name="conv2",
    )(x)
    x = keras.layers.BatchNormalization(name="bn2")(x)
    x = keras.layers.MaxPooling1D(2, padding="same", name="pool1")(x)

    # ======= CONV BLOCK 2 =======
    x = keras.layers.Conv1D(
        128,
        3,
        padding="same",
        activation="relu",
        kernel_regularizer=l2_reg,
        name="conv3",
    )(x)
    x = keras.layers.BatchNormalization(name="bn3")(x)
    x = keras.layers.MaxPooling1D(2, padding="same", name="pool2")(x)

    # ======= BiLSTM with Attention =======
    lstm_out = keras.layers.Bidirectional(
        keras.layers.LSTM(
            128,
            return_sequences=True,
            dropout=0.2,
            recurrent_dropout=0.2,
            kernel_regularizer=l2_reg,
        ),
        name="bilstm1",
    )(x)
    lstm_out = keras.layers.BatchNormalization(name="bn4")(lstm_out)

    lstm_out2 = keras.layers.Bidirectional(
        keras.layers.LSTM(
            64,
            return_sequences=True,
            dropout=0.2,
            recurrent_dropout=0.2,
            kernel_regularizer=l2_reg,
        ),
        name="bilstm2",
    )(lstm_out)

    # Self-Attention
    attention = keras.layers.MultiHeadAttention(
        num_heads=4, key_dim=32, name="attention"
    )(lstm_out2, lstm_out2)
    attention = keras.layers.LayerNormalization(name="ln1")(attention + lstm_out2)

    # Global pooling
    x = keras.layers.GlobalAveragePooling1D(name="gap")(attention)

    # ======= DENSE BLOCK =======
    x = keras.layers.Dense(
        512, activation="relu", kernel_regularizer=l2_reg, name="dense1"
    )(x)
    x = keras.layers.BatchNormalization(name="bn5")(x)
    x = keras.layers.Dropout(0.4, name="drop1")(x)

    x = keras.layers.Dense(
        256, activation="relu", kernel_regularizer=l2_reg, name="dense2"
    )(x)
    x = keras.layers.BatchNormalization(name="bn6")(x)
    x = keras.layers.Dropout(0.3, name="drop2")(x)

    x = keras.layers.Dense(
        128, activation="relu", kernel_regularizer=l2_reg, name="dense3"
    )(x)
    x = keras.layers.Dropout(0.2, name="drop3")(x)

    # Output
    outputs = keras.layers.Dense(num_classes, activation="softmax", name="output")(x)

    model = keras.Model(inputs, outputs, name="Deep_SSA_LSTMIDS")

    return model


# ==================== CICIDS2017 FULL TRAINING ====================


def load_cicids2017_full():
    """CICIDS2017 tam veri yükle"""
    print("\n" + "=" * 70)
    print("📊 Loading CICIDS2017 FULL Dataset")
    print("=" * 70)

    data_dir = PROJECT_ROOT / "data" / "raw" / "cicids2017_full"

    # Train_data.csv kullan (en temiz format)
    train_file = data_dir / "Train_data.csv"

    if not train_file.exists():
        print("❌ Train_data.csv not found!")
        return None

    df = pd.read_csv(train_file, low_memory=False)
    print(f"   Total samples: {len(df):,}")

    # Label
    label_col = "class"
    print(f"   Labels: {df[label_col].value_counts().to_dict()}")

    # Features
    feature_cols = [
        c
        for c in df.columns
        if c not in [label_col, "protocol_type", "service", "flag"]
    ]

    for col in feature_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df[feature_cols] = df[feature_cols].fillna(0)
    df = df.replace([np.inf, -np.inf], 0)

    X = df[feature_cols].values.astype(np.float32)
    X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)

    le = LabelEncoder()
    y = le.fit_transform(df[label_col])

    print(f"   Features: {len(feature_cols)}")
    print(f"   Classes: {list(le.classes_)}")

    # SMOTE for imbalanced data
    try:
        from imblearn.over_sampling import SMOTE

        print("   🔄 Applying SMOTE...")
        smote = SMOTE(random_state=42)
        X, y = smote.fit_resample(X, y)
        print(f"   After SMOTE: {len(X):,}")
    except ImportError:
        print("   ⚠️ imblearn not installed, skipping SMOTE")

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.1, random_state=42, stratify=y_train
    )

    print(f"   Train: {len(X_train):,}")
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
        len(le.classes_),
        list(le.classes_),
    )


# ==================== BOT-IOT FULL TRAINING ====================


def load_botiot_full(max_samples_per_file=50000):
    """BoT-IoT tam veri yükle - tüm dosyalardan"""
    print("\n" + "=" * 70)
    print("📊 Loading BoT-IoT FULL Dataset")
    print("=" * 70)

    data_dir = PROJECT_ROOT / "data" / "raw" / "bot_iot"

    csv_files = sorted(data_dir.glob("*.csv"))
    csv_files = [
        f
        for f in csv_files
        if f.name
        not in ["data_summary.csv", "device_info.csv", "features.csv", "README.md"]
    ]

    print(f"   Found {len(csv_files)} data files")

    dfs = []

    for i, f in enumerate(csv_files):
        try:
            name = f.stem
            parts = name.split(".")

            if len(parts) >= 2:
                if parts[1] == "benign":
                    label = "benign"
                else:
                    label = parts[1]  # gafgyt, mirai
            else:
                continue

            df = pd.read_csv(f, low_memory=False, nrows=max_samples_per_file)
            df["label"] = label
            dfs.append(df)

            if (i + 1) % 20 == 0:
                print(f"   Loaded {i+1}/{len(csv_files)} files...")

        except Exception as e:
            print(f"   ⚠️ Error loading {f.name}: {e}")

    if not dfs:
        print("   ❌ No data loaded!")
        return None

    df = pd.concat(dfs, ignore_index=True)
    print(f"\n   Total samples: {len(df):,}")
    print(f"   Labels: {dict(Counter(df['label']))}")

    # Features
    label_col = "label"
    feature_cols = [
        c
        for c in df.columns
        if c != label_col and df[c].dtype in ["int64", "float64", "int32", "float32"]
    ]

    print(f"   Numeric features: {len(feature_cols)}")

    X = df[feature_cols].values.astype(np.float32)
    X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)

    le = LabelEncoder()
    y = le.fit_transform(df[label_col])

    print(f"   Classes: {list(le.classes_)}")

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.1, random_state=42, stratify=y_train
    )

    print(f"   Train: {len(X_train):,}")
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
        len(le.classes_),
        list(le.classes_),
    )


# ==================== TRAINING FUNCTION ====================


def train_deep_model(
    X_train,
    y_train,
    X_val,
    y_val,
    X_test,
    y_test,
    num_classes,
    class_names,
    dataset_name,
):
    """Deep model ile 300 epoch eğitim"""
    print("\n" + "=" * 70)
    print(f"🧠 Training Deep SSA-LSTMIDS for {dataset_name}")
    print("=" * 70)

    # Build model
    model = build_deep_ssa_lstmids(X_train.shape[1:], num_classes)

    # Standard learning rate - ReduceLROnPlateau will handle decay
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    print(f"   Parameters: {model.count_params():,}")
    model.summary()

    # Callbacks
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=30,  # Makaledeki gibi daha sabırlı
            restore_best_weights=True,
            mode="max",
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_accuracy",
            factor=0.5,
            patience=10,
            min_lr=1e-7,
            mode="max",
            verbose=1,
        ),
        keras.callbacks.ModelCheckpoint(
            str(MODELS_DIR / f"best_{dataset_name.lower()}.keras"),
            monitor="val_accuracy",
            save_best_only=True,
            mode="max",
        ),
    ]

    # Training config - makaledeki gibi
    EPOCHS = 300
    BATCH_SIZE = 128

    print(f"\n🚀 Training for {EPOCHS} epochs...")
    print(f"   Batch size: {BATCH_SIZE}")
    print(f"   Target: {'99.88%' if 'cicids' in dataset_name.lower() else '99.99%'}")

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
    print(f"📊 Test Results - {dataset_name}")
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

    target = 0.9988 if "cicids" in dataset_name.lower() else 0.9999
    print(f"\n   📄 Makaledeki hedef: {target*100:.2f}%")
    print(f"   📊 Bizim sonuç:      {accuracy*100:.4f}%")

    if accuracy >= target:
        print("\n   🎉🎉🎉 HEDEF BAŞARILDI! 🎉🎉🎉")
    elif accuracy >= target - 0.001:
        print("\n   🔥 Çok çok yakın!")
    elif accuracy >= target - 0.005:
        print("\n   👍 Çok iyi!")

    print("\n" + "=" * 60)
    print("📋 Classification Report")
    print("=" * 60)
    print(classification_report(y_test, y_pred, target_names=class_names))

    # Save model
    model_id = f"deep_ssa_lstmids_{dataset_name.lower()}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
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
            "name": f"Deep_SSA_LSTMIDS_{dataset_name}",
            "model_type": "deep_ssa_lstmids",
            "dataset": dataset_name.lower(),
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
                "epochs": len(history.history["loss"]),
                "max_epochs": EPOCHS,
                "batch_size": BATCH_SIZE,
                "architecture": "deep_bilstm_attention",
                "best_val_accuracy": float(max(history.history["val_accuracy"])),
            },
            "created_at": datetime.now().isoformat(),
        }
    )

    with open(REGISTRY_PATH, "w") as f:
        json.dump(registry, f, indent=2)

    print("📋 Registry updated")

    return accuracy, model


def main():
    print("\n" + "=" * 80)
    print("🎓 MAXIMUM ACCURACY TRAINING")
    print("=" * 80)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\nHedefler:")
    print("   - CICIDS2017: 99.88%")
    print("   - BoT-IoT: 99.99%")

    results = {}

    # ==================== CICIDS2017 ====================
    print("\n" + "=" * 80)
    print("🔵 PHASE 1: CICIDS2017 Training")
    print("=" * 80)

    cicids_data = load_cicids2017_full()

    if cicids_data:
        X_train, y_train, X_val, y_val, X_test, y_test, num_classes, class_names = (
            cicids_data
        )

        cicids_acc, cicids_model = train_deep_model(
            X_train,
            y_train,
            X_val,
            y_val,
            X_test,
            y_test,
            num_classes,
            class_names,
            "CICIDS2017",
        )
        results["CICIDS2017"] = cicids_acc

        # Clean up
        del cicids_model
        keras.backend.clear_session()

    # ==================== BoT-IoT ====================
    print("\n" + "=" * 80)
    print("🟢 PHASE 2: BoT-IoT Training")
    print("=" * 80)

    botiot_data = load_botiot_full(max_samples_per_file=100000)  # Daha fazla sample

    if botiot_data:
        X_train, y_train, X_val, y_val, X_test, y_test, num_classes, class_names = (
            botiot_data
        )

        botiot_acc, botiot_model = train_deep_model(
            X_train,
            y_train,
            X_val,
            y_val,
            X_test,
            y_test,
            num_classes,
            class_names,
            "BoT-IoT",
        )
        results["BoT-IoT"] = botiot_acc

    # ==================== FINAL SUMMARY ====================
    print("\n" + "=" * 80)
    print("📊 FINAL SUMMARY")
    print("=" * 80)

    print("\n   | Dataset     | Hedef   | Sonuç   | Durum |")
    print("   |-------------|---------|---------|-------|")

    if "CICIDS2017" in results:
        status = "✅ BAŞARILI" if results["CICIDS2017"] >= 0.9988 else "⚡ Yakın"
        print(
            f"   | CICIDS2017  | 99.88%  | {results['CICIDS2017']*100:.2f}%  | {status} |"
        )

    if "BoT-IoT" in results:
        status = "✅ BAŞARILI" if results["BoT-IoT"] >= 0.9999 else "⚡ Yakın"
        print(
            f"   | BoT-IoT     | 99.99%  | {results['BoT-IoT']*100:.2f}%  | {status} |"
        )

    print("\n" + "=" * 80)
    print("✅ Training Complete!")
    print("=" * 80)
    print(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
