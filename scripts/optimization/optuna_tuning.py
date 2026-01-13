"""
Optuna Hyperparameter Tuning Script
=====================================

Bayesian optimization ile otomatik hyperparameter tuning.
Grid search'den çok daha verimli.

Kullanım:
    pip install optuna
    python scripts/optuna_tuning.py
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
from sklearn.metrics import accuracy_score

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
from tensorflow import keras

print(f"✅ TensorFlow {tf.__version__}")

try:
    import optuna

    print(f"✅ Optuna {optuna.__version__}")
except ImportError:
    print("❌ Optuna not installed. Run: pip install optuna")
    sys.exit(1)

DATA_DIR = PROJECT_ROOT / "data" / "raw" / "cicids2017_full"
MODELS_DIR = PROJECT_ROOT / "models"


def load_data():
    """Veri yükle ve preprocess"""
    print("\n" + "=" * 60)
    print("📊 Loading Data for Optuna Tuning")
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

    # Scale
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)

    # Reshape
    timesteps = 10
    if X.shape[1] % timesteps != 0:
        pad_size = timesteps - (X.shape[1] % timesteps)
        X = np.pad(X, ((0, 0), (0, pad_size)), mode="constant")

    features_per_step = X.shape[1] // timesteps
    X = X.reshape(X.shape[0], timesteps, features_per_step)

    # Split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"   Train: {len(X_train):,}")
    print(f"   Val: {len(X_val):,}")
    print(f"   Shape: {X_train.shape}")

    return X_train, y_train, X_val, y_val, len(le.classes_)


def create_objective(X_train, y_train, X_val, y_val, num_classes):
    """Optuna objective function factory"""

    def objective(trial):
        # Hyperparameters to optimize
        conv_filters = trial.suggest_int("conv_filters", 16, 128, step=16)
        lstm_units = trial.suggest_int("lstm_units", 32, 256, step=32)
        dense_units = trial.suggest_int("dense_units", 64, 512, step=64)
        dropout = trial.suggest_float("dropout", 0.1, 0.5, step=0.1)
        learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
        batch_size = trial.suggest_categorical("batch_size", [32, 64, 128, 256])

        # Build model
        inputs = keras.layers.Input(shape=X_train.shape[1:])
        x = keras.layers.Conv1D(conv_filters, 3, padding="same", activation="relu")(
            inputs
        )
        x = keras.layers.MaxPooling1D(2, padding="same")(x)
        x = keras.layers.LSTM(lstm_units)(x)
        x = keras.layers.Dense(dense_units, activation="relu")(x)
        x = keras.layers.Dropout(dropout)(x)
        outputs = keras.layers.Dense(num_classes, activation="softmax")(x)

        model = keras.Model(inputs, outputs)
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )

        # Train
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor="val_accuracy", patience=3, restore_best_weights=True
            )
        ]

        model.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val),
            epochs=20,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=0,
        )

        # Evaluate
        y_pred = np.argmax(model.predict(X_val, verbose=0), axis=1)
        accuracy = accuracy_score(y_val, y_pred)

        # Clean up
        del model
        keras.backend.clear_session()

        return accuracy

    return objective


def main():
    print("\n" + "=" * 70)
    print("🎓 Optuna Hyperparameter Tuning")
    print("=" * 70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Load data
    X_train, y_train, X_val, y_val, num_classes = load_data()

    # Create study
    print("\n" + "=" * 60)
    print("🔧 Starting Optuna Optimization")
    print("=" * 60)

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(),
    )

    objective = create_objective(X_train, y_train, X_val, y_val, num_classes)

    # Run optimization
    N_TRIALS = 20
    print(f"   Trials: {N_TRIALS}")

    study.optimize(
        objective, n_trials=N_TRIALS, show_progress_bar=True, gc_after_trial=True
    )

    # Results
    print("\n" + "=" * 60)
    print("📊 Optuna Optimization Results")
    print("=" * 60)

    print(f"\n   Best trial: {study.best_trial.number}")
    print(f"   Best accuracy: {study.best_value*100:.2f}%")
    print(f"\n   Best hyperparameters:")
    for key, value in study.best_params.items():
        print(f"      {key}: {value}")

    # Save results
    results = {
        "best_accuracy": study.best_value,
        "best_params": study.best_params,
        "n_trials": N_TRIALS,
        "trials": [
            {"number": t.number, "value": t.value, "params": t.params}
            for t in study.trials
            if t.value is not None
        ],
        "created_at": datetime.now().isoformat(),
    }

    results_path = MODELS_DIR / "optuna_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n💾 Results saved: {results_path}")

    # Retrain best model
    print("\n" + "=" * 60)
    print("🏆 Training Best Model")
    print("=" * 60)

    best = study.best_params

    inputs = keras.layers.Input(shape=X_train.shape[1:])
    x = keras.layers.Conv1D(best["conv_filters"], 3, padding="same", activation="relu")(
        inputs
    )
    x = keras.layers.MaxPooling1D(2, padding="same")(x)
    x = keras.layers.LSTM(best["lstm_units"])(x)
    x = keras.layers.Dense(best["dense_units"], activation="relu")(x)
    x = keras.layers.Dropout(best["dropout"])(x)
    outputs = keras.layers.Dense(num_classes, activation="softmax")(x)

    best_model = keras.Model(inputs, outputs, name="Optuna_Best_Model")
    best_model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=best["learning_rate"]),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=10, restore_best_weights=True
        )
    ]

    best_model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=50,
        batch_size=best["batch_size"],
        callbacks=callbacks,
        verbose=1,
    )

    # Final evaluation
    y_pred = np.argmax(best_model.predict(X_val, verbose=0), axis=1)
    final_accuracy = accuracy_score(y_val, y_pred)

    print(f"\n   ✅ Final accuracy: {final_accuracy*100:.2f}%")

    # Save best model
    model_path = (
        MODELS_DIR / f"optuna_best_{datetime.now().strftime('%Y%m%d_%H%M%S')}.keras"
    )
    best_model.save(model_path)
    print(f"   💾 Model saved: {model_path}")

    print("\n" + "=" * 70)
    print("✅ Optuna Tuning Complete!")
    print("=" * 70)
    print(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
