"""
Ensemble Models Training Script
================================

Ensemble Türleri:
1. Voting Ensemble (Soft voting)
2. Stacking Ensemble (Meta-learner)
3. Weighted Ensemble (Learned weights)

Base Models: SSA-LSTMIDS, BiLSTM+Attention, GRU-IDS, Transformer-IDS, CNN-Transformer

Kullanım:
    python scripts/train_ensemble.py
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
DATA_DIR = PROJECT_ROOT / "data" / "raw" / "cicids2017_full"
MODELS_DIR = PROJECT_ROOT / "models"
REGISTRY_PATH = MODELS_DIR / "model_registry.json"


def load_cicids_data():
    """CICIDS2017 veri yükle - Train verisinden split"""
    print("\n" + "=" * 60)
    print("📊 Loading CICIDS2017 Data for Ensemble Training")
    print("=" * 60)

    train_file = DATA_DIR / "Train_data.csv"

    if not train_file.exists():
        print("❌ Train_data.csv not found!")
        return None

    df = pd.read_csv(train_file, low_memory=False)
    print(f"   Total samples: {len(df):,}")

    # Label column
    label_col = "class"
    print(f"   Label column: {label_col}")
    print(f"   Labels: {df[label_col].unique()}")

    # Sayısal feature sütunları
    feature_cols = [
        c
        for c in df.columns
        if c not in [label_col, "protocol_type", "service", "flag"]
    ]
    print(f"   Features: {len(feature_cols)}")

    # NaN/Inf temizle
    for col in feature_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df[feature_cols] = df[feature_cols].fillna(0)

    # Features ve Labels
    X = df[feature_cols].values.astype(np.float32)
    X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)

    le = LabelEncoder()
    y = le.fit_transform(df[label_col])

    print(f"   Classes: {list(le.classes_)}")

    # Stratified split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"   Train: {len(X_train):,}")
    print(f"   Test: {len(X_test):,}")

    # Scale
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Reshape
    timesteps = 10
    n_features = X_train.shape[1]
    if n_features % timesteps != 0:
        pad_size = timesteps - (n_features % timesteps)
        X_train = np.pad(X_train, ((0, 0), (0, pad_size)), mode="constant")
        X_test = np.pad(X_test, ((0, 0), (0, pad_size)), mode="constant")

    features_per_step = X_train.shape[1] // timesteps
    X_train = X_train.reshape(X_train.shape[0], timesteps, features_per_step)
    X_test = X_test.reshape(X_test.shape[0], timesteps, features_per_step)

    print(f"   X_train shape: {X_train.shape}")
    print(f"   X_test shape: {X_test.shape}")

    return X_train, y_train, X_test, y_test, len(le.classes_), list(le.classes_)


# ============= BASE MODELS =============


def build_ssa_lstmids(input_shape, num_classes):
    """SSA-LSTMIDS base model"""
    inputs = keras.layers.Input(shape=input_shape)
    x = keras.layers.Conv1D(30, 3, padding="same", activation="relu")(inputs)
    x = keras.layers.MaxPooling1D(2, padding="same")(x)
    x = keras.layers.LSTM(120, return_sequences=False)(x)
    x = keras.layers.Dense(512, activation="relu")(x)
    x = keras.layers.Dropout(0.2)(x)
    outputs = keras.layers.Dense(num_classes, activation="softmax")(x)
    return keras.Model(inputs, outputs, name="SSA_LSTMIDS")


def build_bilstm(input_shape, num_classes):
    """BiLSTM base model"""
    inputs = keras.layers.Input(shape=input_shape)
    x = keras.layers.Bidirectional(keras.layers.LSTM(64, return_sequences=True))(inputs)
    x = keras.layers.Bidirectional(keras.layers.LSTM(32))(x)
    x = keras.layers.Dense(128, activation="relu")(x)
    x = keras.layers.Dropout(0.3)(x)
    outputs = keras.layers.Dense(num_classes, activation="softmax")(x)
    return keras.Model(inputs, outputs, name="BiLSTM")


def build_gru(input_shape, num_classes):
    """GRU base model"""
    inputs = keras.layers.Input(shape=input_shape)
    x = keras.layers.GRU(64, return_sequences=True)(inputs)
    x = keras.layers.GRU(32)(x)
    x = keras.layers.Dense(64, activation="relu")(x)
    x = keras.layers.Dropout(0.2)(x)
    outputs = keras.layers.Dense(num_classes, activation="softmax")(x)
    return keras.Model(inputs, outputs, name="GRU")


def build_transformer(input_shape, num_classes):
    """Transformer base model"""
    inputs = keras.layers.Input(shape=input_shape)
    x = keras.layers.Dense(64)(inputs)
    x = keras.layers.MultiHeadAttention(num_heads=4, key_dim=16)(x, x)
    x = keras.layers.LayerNormalization()(x)
    x = keras.layers.GlobalAveragePooling1D()(x)
    x = keras.layers.Dense(64, activation="relu")(x)
    x = keras.layers.Dropout(0.2)(x)
    outputs = keras.layers.Dense(num_classes, activation="softmax")(x)
    return keras.Model(inputs, outputs, name="Transformer")


def build_cnn_lstm(input_shape, num_classes):
    """CNN-LSTM base model"""
    inputs = keras.layers.Input(shape=input_shape)
    x = keras.layers.Conv1D(32, 3, padding="same", activation="relu")(inputs)
    x = keras.layers.Conv1D(64, 3, padding="same", activation="relu")(x)
    x = keras.layers.MaxPooling1D(2, padding="same")(x)
    x = keras.layers.LSTM(64)(x)
    x = keras.layers.Dense(128, activation="relu")(x)
    x = keras.layers.Dropout(0.3)(x)
    outputs = keras.layers.Dense(num_classes, activation="softmax")(x)
    return keras.Model(inputs, outputs, name="CNN_LSTM")


# ============= ENSEMBLE CLASSES =============


class VotingEnsemble:
    """Soft Voting Ensemble"""

    def __init__(self, models):
        self.models = models
        self.name = "Voting_Ensemble"

    def predict(self, X):
        """Soft voting - olasılık ortalaması"""
        predictions = []
        for model in self.models:
            pred = model.predict(X, verbose=0)
            predictions.append(pred)

        # Ortalama olasılık
        avg_pred = np.mean(predictions, axis=0)
        return np.argmax(avg_pred, axis=1)

    def predict_proba(self, X):
        predictions = []
        for model in self.models:
            pred = model.predict(X, verbose=0)
            predictions.append(pred)
        return np.mean(predictions, axis=0)


class WeightedEnsemble:
    """Weighted Voting Ensemble"""

    def __init__(self, models, weights=None):
        self.models = models
        self.weights = weights or [1.0 / len(models)] * len(models)
        self.name = "Weighted_Ensemble"

    def predict(self, X):
        """Ağırlıklı olasılık ortalaması"""
        predictions = []
        for model, weight in zip(self.models, self.weights):
            pred = model.predict(X, verbose=0) * weight
            predictions.append(pred)

        weighted_pred = np.sum(predictions, axis=0)
        return np.argmax(weighted_pred, axis=1)


class StackingEnsemble:
    """Stacking Ensemble with Meta-Learner"""

    def __init__(self, base_models, meta_model=None):
        self.base_models = base_models
        self.meta_model = meta_model
        self.name = "Stacking_Ensemble"

    def fit_meta(self, X, y, num_classes):
        """Meta-learner'ı eğit"""
        # Base model predictions
        base_predictions = []
        for model in self.base_models:
            pred = model.predict(X, verbose=0)
            base_predictions.append(pred)

        # Stack predictions
        stacked_X = np.concatenate(base_predictions, axis=1)

        # Meta-learner model
        self.meta_model = keras.Sequential(
            [
                keras.layers.Dense(
                    64, activation="relu", input_shape=(stacked_X.shape[1],)
                ),
                keras.layers.Dropout(0.3),
                keras.layers.Dense(32, activation="relu"),
                keras.layers.Dense(num_classes, activation="softmax"),
            ]
        )

        self.meta_model.compile(
            optimizer="adam",
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )

        self.meta_model.fit(
            stacked_X, y, epochs=20, batch_size=64, validation_split=0.1, verbose=1
        )

    def predict(self, X):
        base_predictions = []
        for model in self.base_models:
            pred = model.predict(X, verbose=0)
            base_predictions.append(pred)

        stacked_X = np.concatenate(base_predictions, axis=1)
        meta_pred = self.meta_model.predict(stacked_X, verbose=0)
        return np.argmax(meta_pred, axis=1)


# ============= TRAINING =============


def train_base_models(X_train, y_train, X_test, y_test, num_classes):
    """Base modelleri eğit"""
    print("\n" + "=" * 60)
    print("🏋️ Training Base Models")
    print("=" * 60)

    builders = [
        ("SSA-LSTMIDS", build_ssa_lstmids),
        ("BiLSTM", build_bilstm),
        ("GRU", build_gru),
        ("Transformer", build_transformer),
        ("CNN-LSTM", build_cnn_lstm),
    ]

    trained_models = []

    for name, builder in builders:
        print(f"\n   Training {name}...")

        model = builder(X_train.shape[1:], num_classes)
        model.compile(
            optimizer="adam",
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )

        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor="val_accuracy", patience=5, restore_best_weights=True
            )
        ]

        model.fit(
            X_train,
            y_train,
            validation_data=(X_test, y_test),
            epochs=30,
            batch_size=128,
            callbacks=callbacks,
            verbose=0,
        )

        # Evaluate
        y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
        acc = accuracy_score(y_test, y_pred)
        print(f"   {name}: {acc*100:.2f}%")

        trained_models.append(model)

    return trained_models


def evaluate_ensemble(ensemble, X_test, y_test, class_names):
    """Ensemble değerlendir"""
    y_pred = ensemble.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    recall = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
    }


def save_ensemble_results(results, ensemble_name):
    """Sonuçları registry'e kaydet"""
    if REGISTRY_PATH.exists():
        with open(REGISTRY_PATH, "r") as f:
            registry = json.load(f)
    else:
        registry = {"models": []}

    entry = {
        "id": f"ensemble_{ensemble_name.lower()}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        "name": ensemble_name,
        "model_type": "ensemble",
        "dataset": "cicids2017",
        "status": "trained",
        "framework": "tensorflow",
        "metrics": {
            "accuracy": results["accuracy"],
            "precision": results["precision"],
            "recall": results["recall"],
            "f1_score": results["f1_score"],
        },
        "created_at": datetime.now().isoformat(),
    }

    registry["models"].append(entry)

    with open(REGISTRY_PATH, "w") as f:
        json.dump(registry, f, indent=2)


def main():
    print("\n" + "=" * 70)
    print("🎓 Ensemble Models Training")
    print("=" * 70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Load data
    result = load_cicids_data()
    if result is None:
        return

    X_train, y_train, X_test, y_test, num_classes, class_names = result

    # Split train for meta-learner
    X_train_base, X_train_meta, y_train_base, y_train_meta = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42
    )

    # Train base models
    base_models = train_base_models(
        X_train_base, y_train_base, X_test, y_test, num_classes
    )

    # ============= VOTING ENSEMBLE =============
    print("\n" + "=" * 60)
    print("🗳️ Voting Ensemble")
    print("=" * 60)

    voting_ensemble = VotingEnsemble(base_models)
    voting_results = evaluate_ensemble(voting_ensemble, X_test, y_test, class_names)

    print(f"   ✅ Accuracy: {voting_results['accuracy']*100:.2f}%")
    print(f"   ✅ F1-Score: {voting_results['f1_score']:.4f}")

    save_ensemble_results(voting_results, "Voting_Ensemble")

    # ============= WEIGHTED ENSEMBLE =============
    print("\n" + "=" * 60)
    print("⚖️ Weighted Ensemble")
    print("=" * 60)

    # Accuracy'ye göre ağırlık
    accuracies = []
    for model in base_models:
        y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
        acc = accuracy_score(y_test, y_pred)
        accuracies.append(acc)

    # Normalize weights
    total = sum(accuracies)
    weights = [a / total for a in accuracies]
    print(f"   Weights: {[f'{w:.3f}' for w in weights]}")

    weighted_ensemble = WeightedEnsemble(base_models, weights)
    weighted_results = evaluate_ensemble(weighted_ensemble, X_test, y_test, class_names)

    print(f"   ✅ Accuracy: {weighted_results['accuracy']*100:.2f}%")
    print(f"   ✅ F1-Score: {weighted_results['f1_score']:.4f}")

    save_ensemble_results(weighted_results, "Weighted_Ensemble")

    # ============= STACKING ENSEMBLE =============
    print("\n" + "=" * 60)
    print("📚 Stacking Ensemble")
    print("=" * 60)

    stacking_ensemble = StackingEnsemble(base_models)
    print("   Training meta-learner...")
    stacking_ensemble.fit_meta(X_train_meta, y_train_meta, num_classes)

    stacking_results = evaluate_ensemble(stacking_ensemble, X_test, y_test, class_names)

    print(f"   ✅ Accuracy: {stacking_results['accuracy']*100:.2f}%")
    print(f"   ✅ F1-Score: {stacking_results['f1_score']:.4f}")

    save_ensemble_results(stacking_results, "Stacking_Ensemble")

    # ============= SUMMARY =============
    print("\n" + "=" * 70)
    print("📊 ENSEMBLE SUMMARY")
    print("=" * 70)
    print(f"\n   Voting Ensemble:   {voting_results['accuracy']*100:.2f}%")
    print(f"   Weighted Ensemble: {weighted_results['accuracy']*100:.2f}%")
    print(f"   Stacking Ensemble: {stacking_results['accuracy']*100:.2f}%")

    best = max(
        [
            ("Voting", voting_results["accuracy"]),
            ("Weighted", weighted_results["accuracy"]),
            ("Stacking", stacking_results["accuracy"]),
        ],
        key=lambda x: x[1],
    )

    print(f"\n   🏆 Best: {best[0]} Ensemble ({best[1]*100:.2f}%)")

    print("\n" + "=" * 70)
    print("✅ Training Complete!")
    print("=" * 70)
    print(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
