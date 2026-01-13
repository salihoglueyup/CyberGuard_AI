"""
Strateji 6: SSA Real Hyperparameter Tuning
============================================

SSA (Sparrow Search Algorithm) ile gerçek hyperparameter
optimizasyonu yaparak maksimum accuracy hedefliyoruz.

Hedef: %99.88+
Tahmini süre: 2-4 saat

Kullanım:
    python scripts/strategy6_ssa_tuning.py
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

# SSA Configuration
N_SPARROWS = 10  # Population size
MAX_ITERATIONS = 15  # SSA iterations
EPOCHS_PER_TRIAL = 20  # Quick evaluation


class SSAOptimizer:
    """Sparrow Search Algorithm for hyperparameter optimization"""

    def __init__(self, bounds, n_sparrows=10, max_iter=15):
        self.bounds = bounds  # Dict of {param: (min, max)}
        self.n_sparrows = n_sparrows
        self.max_iter = max_iter
        self.param_names = list(bounds.keys())
        self.dim = len(self.param_names)

        # Initialize population
        self.population = self._init_population()
        self.fitness = np.zeros(n_sparrows)
        self.best_position = None
        self.best_fitness = 0
        self.history = []

    def _init_population(self):
        """Initialize random population"""
        pop = np.zeros((self.n_sparrows, self.dim))
        for i, param in enumerate(self.param_names):
            low, high = self.bounds[param]
            pop[:, i] = np.random.uniform(low, high, self.n_sparrows)
        return pop

    def _decode(self, position):
        """Decode position to hyperparameters"""
        params = {}
        for i, name in enumerate(self.param_names):
            val = position[i]
            # Round integer parameters
            if name in ["conv_filters", "lstm_units", "dense_units", "batch_size"]:
                val = int(round(val))
            params[name] = val
        return params

    def update(self, iteration):
        """SSA update step"""
        n = self.n_sparrows

        # Sort by fitness (descending)
        sorted_idx = np.argsort(-self.fitness)

        # Producers (top 20%)
        n_producers = max(1, int(0.2 * n))

        # Update producers
        for i in range(n_producers):
            idx = sorted_idx[i]
            R2 = np.random.random()

            if R2 < 0.8:
                # Safe exploration
                self.population[idx] += np.random.randn(self.dim) * 0.1
            else:
                # Danger - move to safe area
                self.population[idx] = (
                    self.best_position + np.random.randn(self.dim) * 0.05
                )

        # Update scroungers (follow producers)
        for i in range(n_producers, n):
            idx = sorted_idx[i]
            producer_idx = sorted_idx[np.random.randint(0, n_producers)]

            if self.fitness[idx] > self.fitness[producer_idx] / 2:
                # Follow producer
                self.population[idx] += (
                    np.random.randn(self.dim)
                    * (self.population[producer_idx] - self.population[idx])
                    * 0.5
                )
            else:
                # Random exploration
                self.population[idx] += np.random.randn(self.dim) * 0.2

        # Clip to bounds
        for i, param in enumerate(self.param_names):
            low, high = self.bounds[param]
            self.population[:, i] = np.clip(self.population[:, i], low, high)

    def optimize(self, fitness_func):
        """Run optimization"""
        print(
            f"\n🦅 SSA Optimization: {self.n_sparrows} sparrows, {self.max_iter} iterations"
        )

        # Evaluate initial population
        print("\n📊 Evaluating initial population...")
        for i in range(self.n_sparrows):
            params = self._decode(self.population[i])
            self.fitness[i] = fitness_func(params)

            if self.fitness[i] > self.best_fitness:
                self.best_fitness = self.fitness[i]
                self.best_position = self.population[i].copy()

            print(f"   Sparrow {i+1}/{self.n_sparrows}: {self.fitness[i]*100:.2f}%")

        print(f"\n   🏆 Initial best: {self.best_fitness*100:.4f}%")
        self.history.append(self.best_fitness)

        # Optimization loop
        for it in range(self.max_iter):
            print(f"\n🔄 Iteration {it+1}/{self.max_iter}")

            self.update(it)

            # Evaluate new positions (top 5 only for speed)
            sorted_idx = np.argsort(-self.fitness)
            for rank, i in enumerate(sorted_idx[:5]):
                params = self._decode(self.population[i])
                new_fitness = fitness_func(params)

                if new_fitness > self.fitness[i]:
                    self.fitness[i] = new_fitness

                if new_fitness > self.best_fitness:
                    self.best_fitness = new_fitness
                    self.best_position = self.population[i].copy()
                    print(f"   🎯 New best: {self.best_fitness*100:.4f}%")

            self.history.append(self.best_fitness)
            print(f"   Best so far: {self.best_fitness*100:.4f}%")

        return self._decode(self.best_position), self.best_fitness


def load_data(sample_size=300_000):
    """Veri yükle"""
    print(f"\n📊 Loading {sample_size:,} samples...")

    csv_files = [
        "Monday-WorkingHours.pcap_ISCX.csv",
        "Tuesday-WorkingHours.pcap_ISCX.csv",
        "Wednesday-workingHours.pcap_ISCX.csv",
        "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv",
        "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv",
    ]

    dfs = []
    for fname in csv_files:
        fpath = DATA_DIR / fname
        if fpath.exists():
            df = pd.read_csv(fpath, low_memory=False)
            dfs.append(df)

    df = pd.concat(dfs, ignore_index=True)

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

    # Split (stratify kullanma - bazı sınıflarda çok az sample var)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.15, random_state=42
    )

    # Scale
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    # Reshape
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

    print(f"   Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

    return X_train, y_train, X_val, y_val, X_test, y_test, num_classes


def build_model(input_shape, num_classes, params):
    """Build model with given hyperparameters"""
    l2 = keras.regularizers.l2(0.0001)

    conv_filters = params["conv_filters"]
    lstm_units = params["lstm_units"]
    dense_units = params["dense_units"]
    dropout = params["dropout"]

    inputs = keras.layers.Input(shape=input_shape)

    x = keras.layers.Conv1D(
        conv_filters, 3, padding="same", activation="relu", kernel_regularizer=l2
    )(inputs)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Conv1D(
        conv_filters, 3, padding="same", activation="relu", kernel_regularizer=l2
    )(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.MaxPooling1D(2, padding="same")(x)

    x = keras.layers.Bidirectional(
        keras.layers.LSTM(lstm_units, return_sequences=True, dropout=dropout * 0.5)
    )(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Bidirectional(
        keras.layers.LSTM(
            lstm_units // 2, return_sequences=False, dropout=dropout * 0.5
        )
    )(x)

    x = keras.layers.Dense(dense_units, activation="relu", kernel_regularizer=l2)(x)
    x = keras.layers.Dropout(dropout)(x)
    x = keras.layers.Dense(dense_units // 2, activation="relu")(x)
    x = keras.layers.Dropout(dropout * 0.7)(x)

    outputs = keras.layers.Dense(num_classes, activation="softmax")(x)

    return keras.Model(inputs, outputs)


def main():
    print("\n" + "=" * 80)
    print("🎯 STRATEJI 6: SSA REAL HYPERPARAMETER TUNING")
    print("=" * 80)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\nHedef: %99.88+")

    # Load data
    X_train, y_train, X_val, y_val, X_test, y_test, num_classes = load_data(300_000)

    # Define fitness function
    def fitness_func(params):
        """Evaluate model with given params"""
        try:
            model = build_model(X_train.shape[1:], num_classes, params)
            model.compile(
                optimizer=keras.optimizers.Adam(learning_rate=params["learning_rate"]),
                loss="sparse_categorical_crossentropy",
                metrics=["accuracy"],
            )

            callbacks = [
                keras.callbacks.EarlyStopping(
                    monitor="val_accuracy", patience=5, restore_best_weights=True
                )
            ]

            batch_size = params["batch_size"]

            history = model.fit(
                X_train,
                y_train,
                validation_data=(X_val, y_val),
                epochs=EPOCHS_PER_TRIAL,
                batch_size=batch_size,
                callbacks=callbacks,
                verbose=0,
            )

            val_acc = max(history.history["val_accuracy"])

            keras.backend.clear_session()

            return val_acc

        except Exception as e:
            print(f"   Error: {e}")
            return 0.0

    # SSA bounds
    bounds = {
        "conv_filters": (32, 256),
        "lstm_units": (64, 256),
        "dense_units": (128, 512),
        "dropout": (0.1, 0.4),
        "learning_rate": (0.0001, 0.005),
        "batch_size": (64, 512),
    }

    # Run SSA
    print("\n" + "=" * 60)
    print("🦅 Starting SSA Optimization")
    print("=" * 60)

    ssa = SSAOptimizer(bounds, n_sparrows=N_SPARROWS, max_iter=MAX_ITERATIONS)
    best_params, best_fitness = ssa.optimize(fitness_func)

    print("\n" + "=" * 60)
    print("🏆 SSA OPTIMIZATION COMPLETE")
    print("=" * 60)
    print(f"\n   Best Val Accuracy: {best_fitness*100:.4f}%")
    print(f"\n   Best Hyperparameters:")
    for k, v in best_params.items():
        print(f"      {k}: {v}")

    # Final training with best params
    print("\n" + "=" * 60)
    print("🚀 Final Training with Best Hyperparameters")
    print("=" * 60)

    # Load more data for final training
    X_train, y_train, X_val, y_val, X_test, y_test, num_classes = load_data(700_000)

    model = build_model(X_train.shape[1:], num_classes, best_params)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=best_params["learning_rate"]),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    print(f"   Parameters: {model.count_params():,}")

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=15, restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_accuracy", factor=0.5, patience=5, min_lr=1e-7
        ),
    ]

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=80,
        batch_size=best_params["batch_size"],
        callbacks=callbacks,
        verbose=1,
    )

    final_val_acc = max(history.history["val_accuracy"])
    print(f"\n   Final Best Val Accuracy: {final_val_acc*100:.4f}%")

    # Evaluate
    print("\n" + "=" * 60)
    print("📊 TEST RESULTS")
    print("=" * 60)

    y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    recall = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

    print(f"\n   ✅ Accuracy:  {accuracy*100:.4f}%")
    print(f"   ✅ Precision: {precision*100:.4f}%")
    print(f"   ✅ Recall:    {recall*100:.4f}%")
    print(f"   ✅ F1-Score:  {f1:.6f}")

    print(f"\n   📄 Hedef: 99.88%")
    print(f"   📊 Sonuç: {accuracy*100:.4f}%")

    if accuracy >= 0.9988:
        print("\n   🎉🎉🎉 HEDEF BAŞARILDI! 🎉🎉🎉")
    else:
        print(f"\n   Fark: {(0.9988 - accuracy)*100:+.2f}%")

    # Save
    model_path = (
        MODELS_DIR / f"strategy6_ssa_{datetime.now().strftime('%Y%m%d_%H%M%S')}.keras"
    )
    model.save(model_path)
    print(f"\n💾 Saved: {model_path}")

    result = {
        "strategy": "SSA Real Hyperparameter Tuning",
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "best_val_accuracy": float(final_val_acc),
        "best_hyperparameters": {
            k: float(v) if isinstance(v, (int, float)) else v
            for k, v in best_params.items()
        },
        "ssa_history": [float(x) for x in ssa.history],
        "created_at": datetime.now().isoformat(),
    }

    results_path = MODELS_DIR / "strategy6_ssa_results.json"
    with open(results_path, "w") as f:
        json.dump(result, f, indent=2)

    print("\n" + "=" * 80)
    print(f"✅ STRATEJI 6 TAMAMLANDI: {accuracy*100:.4f}%")
    print("=" * 80)

    return accuracy


if __name__ == "__main__":
    main()
