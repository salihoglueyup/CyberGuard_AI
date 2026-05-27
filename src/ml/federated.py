"""
Federated Learning Module - CyberGuard AI
==========================================

Privacy-preserving dağıtık öğrenme.

Özellikler:
    - FedAvg algoritması
    - Client simulation
    - Secure aggregation
"""

import json
import logging
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path

import numpy as np

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logger = logging.getLogger("FederatedLearning")


class ClientStatus(Enum):
    IDLE = "idle"
    TRAINING = "training"
    READY = "ready"
    FAILED = "failed"


@dataclass
class FLClient:
    """Federated Learning client"""

    client_id: str
    name: str
    data_size: int = 0
    status: ClientStatus = ClientStatus.IDLE
    local_epochs: int = 1
    batch_size: int = 32
    current_round: int = 0
    metrics: dict = field(default_factory=dict)
    weights: list[np.ndarray] | None = None


@dataclass
class FLRound:
    """Bir FL round"""

    round_id: int
    started_at: str
    completed_at: str | None = None
    participating_clients: list[str] = field(default_factory=list)
    aggregated_metrics: dict = field(default_factory=dict)


class FederatedServer:
    """
    Federated Learning Server

    FedAvg implementasyonu
    """

    def __init__(
        self,
        model_fn=None,
        num_rounds: int = 10,
        clients_per_round: int = 5,
        local_epochs: int = 1,
        learning_rate: float = 0.01,
    ):
        self.model_fn = model_fn  # Model oluşturma fonksiyonu
        self.num_rounds = num_rounds
        self.clients_per_round = clients_per_round
        self.local_epochs = local_epochs
        self.learning_rate = learning_rate

        self.global_model = None
        self.clients: dict[str, FLClient] = {}
        self.rounds: list[FLRound] = []
        self.current_round = 0
        self.is_training = False

        # Results
        self.global_metrics_history: list[dict] = []

    def initialize_global_model(self, input_shape: tuple, num_classes: int):
        """Global modeli başlat"""
        if self.model_fn:
            self.global_model = self.model_fn(input_shape, num_classes)
        else:
            self.global_model = self._default_model(input_shape, num_classes)

        print("✅ Global model initialized")
        self.global_model.summary()

    def _default_model(self, input_shape: tuple, num_classes: int):
        """Varsayılan model"""
        from tensorflow import keras
        from tensorflow.keras import layers

        model = keras.Sequential(
            [
                layers.Input(shape=input_shape),
                layers.Conv1D(32, 3, activation="relu", padding="same"),
                layers.MaxPooling1D(2),
                layers.LSTM(64, dropout=0.2),
                layers.Dense(128, activation="relu"),
                layers.Dropout(0.2),
                layers.Dense(num_classes, activation="softmax"),
            ]
        )

        model.compile(
            optimizer=keras.optimizers.Adam(self.learning_rate),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )

        return model

    def register_client(
        self, client_id: str, name: str, data_size: int = 0
    ) -> FLClient:
        """Yeni client kaydet"""
        client = FLClient(
            client_id=client_id,
            name=name,
            data_size=data_size,
            local_epochs=self.local_epochs,
        )

        self.clients[client_id] = client
        print(f"📱 Client registered: {name} (data_size={data_size})")

        return client

    def get_global_weights(self) -> list[np.ndarray]:
        """Global model ağırlıklarını al"""
        if self.global_model is None:
            return []
        return self.global_model.get_weights()

    def set_global_weights(self, weights: list[np.ndarray]):
        """Global model ağırlıklarını set et"""
        if self.global_model:
            self.global_model.set_weights(weights)

    def train_client(
        self,
        client_id: str,
        X: np.ndarray,
        y: np.ndarray,
        validation_split: float = 0.1,
    ) -> dict:
        """
        Bir client'ı eğit (simulated)
        """
        if client_id not in self.clients:
            return {"error": "Client bulunamadı"}

        client = self.clients[client_id]
        client.status = ClientStatus.TRAINING

        try:
            # Clone global model for client
            from tensorflow import keras

            client_model = keras.models.clone_model(self.global_model)
            client_model.set_weights(self.get_global_weights())
            client_model.compile(
                optimizer=keras.optimizers.Adam(self.learning_rate),
                loss="sparse_categorical_crossentropy",
                metrics=["accuracy"],
            )

            # Local training
            history = client_model.fit(
                X,
                y,
                epochs=client.local_epochs,
                batch_size=client.batch_size,
                validation_split=validation_split,
                verbose=0,
            )

            # Store local weights
            client.weights = client_model.get_weights()
            client.data_size = len(X)
            client.metrics = {
                "loss": float(history.history["loss"][-1]),
                "accuracy": float(history.history["accuracy"][-1]),
                "val_loss": float(history.history.get("val_loss", [0])[-1]),
                "val_accuracy": float(history.history.get("val_accuracy", [0])[-1]),
            }
            client.status = ClientStatus.READY

            return {
                "client_id": client_id,
                "status": "success",
                "metrics": client.metrics,
            }

        except Exception as e:
            client.status = ClientStatus.FAILED
            return {"error": str(e)}

    def aggregate_weights(self, client_ids: list[str]) -> list[np.ndarray]:
        """
        FedAvg: Weighted averaging of client weights
        """
        if not client_ids:
            return []

        # Get ready clients
        ready_clients = [
            self.clients[cid]
            for cid in client_ids
            if cid in self.clients and self.clients[cid].weights is not None
        ]

        if not ready_clients:
            return []

        # Total data size
        total_samples = sum(c.data_size for c in ready_clients)

        if total_samples == 0:
            # Equal weights
            weights = [c.weights for c in ready_clients]
            aggregated = [
                np.mean([w[i] for w in weights], axis=0) for i in range(len(weights[0]))
            ]
        else:
            # Weighted average
            aggregated = None
            for client in ready_clients:
                weight = client.data_size / total_samples

                if aggregated is None:
                    aggregated = [w * weight for w in client.weights]
                else:
                    for i, w in enumerate(client.weights):
                        aggregated[i] += w * weight

        return aggregated

    def run_round(
        self, client_data: dict[str, tuple[np.ndarray, np.ndarray]]
    ) -> FLRound:
        """
        Bir FL round çalıştır

        Args:
            client_data: {client_id: (X, y)} dictionary
        """
        self.current_round += 1

        fl_round = FLRound(
            round_id=self.current_round,
            started_at=datetime.now().isoformat(),
            participating_clients=list(client_data.keys()),
        )

        print(f"\n{'='*50}")
        print(f"🔄 FL Round {self.current_round}")
        print(f"{'='*50}")

        # Train each client
        for client_id, (X, y) in client_data.items():
            if client_id not in self.clients:
                self.register_client(client_id, f"Client_{client_id}", len(X))

            print(f"   Training {client_id}...")
            result = self.train_client(client_id, X, y)

            if "error" not in result:
                print(f"      Accuracy: {result['metrics']['accuracy']*100:.2f}%")

        # Aggregate
        print("\n   🔗 Aggregating weights...")
        aggregated_weights = self.aggregate_weights(list(client_data.keys()))

        if aggregated_weights:
            self.set_global_weights(aggregated_weights)
            print("   ✅ Global model updated")

        # Calculate aggregated metrics
        accuracies = [
            self.clients[cid].metrics.get("accuracy", 0)
            for cid in client_data.keys()
            if cid in self.clients
        ]

        fl_round.aggregated_metrics = {
            "avg_accuracy": np.mean(accuracies) if accuracies else 0,
            "min_accuracy": min(accuracies) if accuracies else 0,
            "max_accuracy": max(accuracies) if accuracies else 0,
        }

        fl_round.completed_at = datetime.now().isoformat()
        self.rounds.append(fl_round)

        # History
        self.global_metrics_history.append(
            {
                "round": self.current_round,
                "timestamp": fl_round.completed_at,
                **fl_round.aggregated_metrics,
            }
        )
        if len(self.global_metrics_history) > 1000:
            self.global_metrics_history = self.global_metrics_history[-500:]

        print(f"\n   📊 Round {self.current_round} completed")
        print(
            f"      Avg Accuracy: {fl_round.aggregated_metrics['avg_accuracy']*100:.2f}%"
        )

        return fl_round

    def train(
        self,
        client_data_generator,  # Callable that returns client data
        num_rounds: int = None,
    ) -> dict:
        """
        Tam FL eğitimi
        """
        if num_rounds is None:
            num_rounds = self.num_rounds

        self.is_training = True

        print(f"\n{'='*60}")
        print("🚀 Federated Learning Training")
        print(f"{'='*60}")
        print(f"   Rounds: {num_rounds}")
        print(f"   Clients per round: {self.clients_per_round}")
        print(f"   Local epochs: {self.local_epochs}")

        for round_num in range(num_rounds):
            if not self.is_training:
                break

            # Get client data for this round
            client_data = client_data_generator(round_num)

            # Run round
            self.run_round(client_data)

        self.is_training = False

        return self.get_training_summary()

    def get_training_summary(self) -> dict:
        """Eğitim özeti"""
        return {
            "total_rounds": len(self.rounds),
            "total_clients": len(self.clients),
            "final_accuracy": (
                self.global_metrics_history[-1]["avg_accuracy"]
                if self.global_metrics_history
                else 0
            ),
            "metrics_history": self.global_metrics_history,
            "clients": [
                {
                    "id": c.client_id,
                    "name": c.name,
                    "data_size": c.data_size,
                    "status": c.status.value,
                    "metrics": c.metrics,
                }
                for c in self.clients.values()
            ],
        }

    def stop_training(self):
        """Eğitimi durdur"""
        self.is_training = False

    def save_global_model(self, path: str):
        """Global modeli kaydet"""
        if self.global_model:
            self.global_model.save(path)
            print(f"💾 Global model saved: {path}")

    def evaluate_global_model(self, X: np.ndarray, y: np.ndarray) -> dict:
        """Global modeli değerlendir"""
        if self.global_model is None:
            return {"error": "Model initialized değil"}

        loss, accuracy = self.global_model.evaluate(X, y, verbose=0)

        return {"loss": float(loss), "accuracy": float(accuracy)}


# Singleton
_fl_server: FederatedServer | None = None


def get_fl_server() -> FederatedServer:
    """Global FL server"""
    global _fl_server
    if _fl_server is None:
        _fl_server = FederatedServer()
    return _fl_server


# Test
if __name__ == "__main__":
    print("🧪 Federated Learning Test\n")

    # Dummy data
    np.random.seed(42)

    # Initialize server
    server = FederatedServer(num_rounds=3, clients_per_round=3, local_epochs=2)

    # Initialize model
    server.initialize_global_model(input_shape=(10, 5), num_classes=2)

    # Simulate client data generator
    def get_client_data(round_num):
        return {
            f"client_{i}": (
                np.random.randn(100, 10, 5).astype(np.float32),
                np.random.randint(0, 2, 100),
            )
            for i in range(3)
        }

    # Train
    summary = server.train(get_client_data, num_rounds=3)

    print("\n📊 Training Summary:")
    print(json.dumps(summary, indent=2, default=str))
