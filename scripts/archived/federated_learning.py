"""
Federated Learning Framework
=============================

Distributed training without centralizing data.
Privacy-preserving machine learning approach.

Simulates multiple clients training locally and aggregating.

Kullanım:
    python scripts/federated_learning.py
"""

import os
import sys
import json
import numpy as np
from pathlib import Path
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
from tensorflow import keras

print(f"✅ TensorFlow {tf.__version__}")

DATA_DIR = PROJECT_ROOT / "data" / "raw" / "cicids2017_full"
MODELS_DIR = PROJECT_ROOT / "models"


class FederatedClient:
    """Simulated federated learning client"""

    def __init__(self, client_id, X, y, input_shape, num_classes):
        self.client_id = client_id
        self.X = X
        self.y = y
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = self._build_model()

    def _build_model(self):
        """Build local model"""
        inputs = keras.layers.Input(shape=self.input_shape)
        x = keras.layers.Conv1D(32, 3, padding="same", activation="relu")(inputs)
        x = keras.layers.MaxPooling1D(2, padding="same")(x)
        x = keras.layers.LSTM(64)(x)
        x = keras.layers.Dense(64, activation="relu")(x)
        x = keras.layers.Dropout(0.3)(x)
        outputs = keras.layers.Dense(self.num_classes, activation="softmax")(x)

        model = keras.Model(inputs, outputs)
        model.compile(
            optimizer="adam",
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        return model

    def set_weights(self, weights):
        """Set model weights from server"""
        self.model.set_weights(weights)

    def get_weights(self):
        """Get model weights"""
        return self.model.get_weights()

    def train(self, epochs=5, batch_size=32):
        """Train locally"""
        self.model.fit(self.X, self.y, epochs=epochs, batch_size=batch_size, verbose=0)
        return len(self.X)  # Return number of samples


class FederatedServer:
    """Federated learning server (aggregator)"""

    def __init__(self, clients, input_shape, num_classes):
        self.clients = clients
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.global_model = self._build_global_model()
        self.history = []

    def _build_global_model(self):
        """Build global model"""
        inputs = keras.layers.Input(shape=self.input_shape)
        x = keras.layers.Conv1D(32, 3, padding="same", activation="relu")(inputs)
        x = keras.layers.MaxPooling1D(2, padding="same")(x)
        x = keras.layers.LSTM(64)(x)
        x = keras.layers.Dense(64, activation="relu")(x)
        x = keras.layers.Dropout(0.3)(x)
        outputs = keras.layers.Dense(self.num_classes, activation="softmax")(x)

        model = keras.Model(inputs, outputs, name="FederatedGlobalModel")
        model.compile(
            optimizer="adam",
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        return model

    def broadcast_weights(self):
        """Send global weights to all clients"""
        global_weights = self.global_model.get_weights()
        for client in self.clients:
            client.set_weights(global_weights)

    def aggregate_weights(self, client_sample_counts):
        """FedAvg: Weighted average of client weights"""
        total_samples = sum(client_sample_counts)

        # Initialize aggregated weights
        aggregated_weights = [np.zeros_like(w) for w in self.global_model.get_weights()]

        # Weighted sum
        for client, sample_count in zip(self.clients, client_sample_counts):
            client_weights = client.get_weights()
            weight_factor = sample_count / total_samples

            for i, w in enumerate(client_weights):
                aggregated_weights[i] += w * weight_factor

        # Update global model
        self.global_model.set_weights(aggregated_weights)

    def train_round(self, local_epochs=5, local_batch_size=32):
        """One round of federated training"""
        # Broadcast global weights
        self.broadcast_weights()

        # Local training
        sample_counts = []
        for client in self.clients:
            n_samples = client.train(epochs=local_epochs, batch_size=local_batch_size)
            sample_counts.append(n_samples)

        # Aggregate
        self.aggregate_weights(sample_counts)

        return sample_counts

    def evaluate(self, X_test, y_test):
        """Evaluate global model"""
        y_pred = np.argmax(self.global_model.predict(X_test, verbose=0), axis=1)
        accuracy = np.mean(y_pred == y_test)
        return accuracy


def load_and_partition_data(n_clients=5):
    """Load data and partition for clients"""
    print("\n" + "=" * 60)
    print(f"📊 Loading and Partitioning Data for {n_clients} Clients")
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
    num_classes = len(le.classes_)

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

    # Split train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Partition training data for clients (non-IID simulation)
    client_data = []
    indices = np.random.permutation(len(X_train))
    chunk_size = len(indices) // n_clients

    for i in range(n_clients):
        start = i * chunk_size
        end = start + chunk_size if i < n_clients - 1 else len(indices)
        client_indices = indices[start:end]
        client_data.append((X_train[client_indices], y_train[client_indices]))
        print(f"   Client {i+1}: {len(client_indices)} samples")

    print(f"   Test set: {len(X_test)} samples")

    return client_data, X_test, y_test, X_train.shape[1:], num_classes


def main():
    print("\n" + "=" * 70)
    print("🎓 Federated Learning Training")
    print("=" * 70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Hyperparameters
    N_CLIENTS = 5
    N_ROUNDS = 10
    LOCAL_EPOCHS = 5
    LOCAL_BATCH_SIZE = 32

    print(f"\n   Clients: {N_CLIENTS}")
    print(f"   Rounds: {N_ROUNDS}")
    print(f"   Local epochs: {LOCAL_EPOCHS}")

    # Load and partition data
    client_data, X_test, y_test, input_shape, num_classes = load_and_partition_data(
        N_CLIENTS
    )

    # Create clients
    clients = []
    for i, (X, y) in enumerate(client_data):
        client = FederatedClient(i + 1, X, y, input_shape, num_classes)
        clients.append(client)

    # Create server
    server = FederatedServer(clients, input_shape, num_classes)

    # Training rounds
    print("\n" + "=" * 60)
    print("🔄 Federated Training Rounds")
    print("=" * 60)

    round_results = []

    for round_num in range(N_ROUNDS):
        # Train
        sample_counts = server.train_round(
            local_epochs=LOCAL_EPOCHS, local_batch_size=LOCAL_BATCH_SIZE
        )

        # Evaluate
        accuracy = server.evaluate(X_test, y_test)
        round_results.append(
            {
                "round": round_num + 1,
                "accuracy": accuracy,
                "total_samples": sum(sample_counts),
            }
        )

        print(f"   Round {round_num+1}/{N_ROUNDS} - Accuracy: {accuracy*100:.2f}%")

    # Final results
    print("\n" + "=" * 60)
    print("📊 Federated Learning Results")
    print("=" * 60)

    final_accuracy = round_results[-1]["accuracy"]
    print(f"\n   Final accuracy: {final_accuracy*100:.2f}%")
    print(
        f"   Improvement: {(round_results[-1]['accuracy'] - round_results[0]['accuracy'])*100:.2f}%"
    )

    # Save model
    model_path = (
        MODELS_DIR
        / f"federated_global_{datetime.now().strftime('%Y%m%d_%H%M%S')}.keras"
    )
    server.global_model.save(model_path)
    print(f"\n💾 Global model saved: {model_path}")

    # Save results
    results = {
        "n_clients": N_CLIENTS,
        "n_rounds": N_ROUNDS,
        "local_epochs": LOCAL_EPOCHS,
        "final_accuracy": final_accuracy,
        "round_history": round_results,
        "created_at": datetime.now().isoformat(),
    }

    results_path = MODELS_DIR / "federated_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"💾 Results saved: {results_path}")

    print("\n" + "=" * 70)
    print("✅ Federated Learning Complete!")
    print("=" * 70)
    print(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
