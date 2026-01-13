"""
Sparrow Search Algorithm (SSA) Optimizer
==========================================

Makaledeki SSA optimizasyon algoritması implementasyonu.

SSA, doğadaki serçelerin yiyecek arama davranışından esinlenen
metaheuristik bir optimizasyon algoritmasıdır.

Roller:
- Producers (Liderler): En iyi konumlara sahip serçeler
- Scroungers (Takipçiler): Liderleri takip eder
- Scouters (Gözcüler): Tehlike durumunda kaçış

Kullanım:
    python scripts/ssa_optimize.py
"""

import os
import sys
import numpy as np
from pathlib import Path
from datetime import datetime
import json

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
from tensorflow import keras


class SparrowSearchAlgorithm:
    """
    Sparrow Search Algorithm (SSA)

    Reference:
    Xue, J., & Shen, B. (2020). A novel swarm intelligence optimization
    approach: sparrow search algorithm. Systems Science & Control Engineering.
    """

    def __init__(
        self,
        objective_func,
        bounds,
        n_sparrows=30,
        max_iter=100,
        producer_ratio=0.2,
        safety_threshold=0.8,
        scouter_ratio=0.1,
    ):
        """
        Args:
            objective_func: Objective function to minimize (lower is better)
            bounds: List of (min, max) tuples for each dimension
            n_sparrows: Population size
            max_iter: Maximum iterations
            producer_ratio: Ratio of producers in population
            safety_threshold: Safety threshold for anti-predator behavior
            scouter_ratio: Ratio of scouters
        """
        self.objective_func = objective_func
        self.bounds = np.array(bounds)
        self.n_dim = len(bounds)
        self.n_sparrows = n_sparrows
        self.max_iter = max_iter
        self.n_producers = max(1, int(n_sparrows * producer_ratio))
        self.n_scouters = max(1, int(n_sparrows * scouter_ratio))
        self.safety_threshold = safety_threshold

        # Initialize population
        self.population = self._initialize_population()
        self.fitness = np.array([self.objective_func(ind) for ind in self.population])

        # Best solution
        self.best_idx = np.argmin(self.fitness)
        self.best_position = self.population[self.best_idx].copy()
        self.best_fitness = self.fitness[self.best_idx]

        # History
        self.history = []

    def _initialize_population(self):
        """Initialize random population within bounds"""
        population = np.zeros((self.n_sparrows, self.n_dim))
        for i in range(self.n_dim):
            population[:, i] = np.random.uniform(
                self.bounds[i, 0], self.bounds[i, 1], self.n_sparrows
            )
        return population

    def _clip_to_bounds(self, position):
        """Clip position to bounds"""
        for i in range(self.n_dim):
            position[i] = np.clip(position[i], self.bounds[i, 0], self.bounds[i, 1])
        return position

    def _update_producers(self, iteration):
        """Update producer positions"""
        # Sort by fitness
        sorted_idx = np.argsort(self.fitness)
        producers_idx = sorted_idx[: self.n_producers]

        alpha = np.random.random()  # Warning factor
        R2 = np.random.random()  # Alarm value

        for idx in producers_idx:
            if R2 < self.safety_threshold:
                # Safe, normal foraging
                Q = np.random.randn()
                self.population[idx] = self.population[idx] * np.exp(
                    -idx / (alpha * self.max_iter + 1e-10)
                )
            else:
                # Danger, move randomly
                Q = np.random.randn()
                L = np.ones(self.n_dim)
                self.population[idx] = self.best_position + Q * L

            self.population[idx] = self._clip_to_bounds(self.population[idx])

    def _update_scroungers(self, iteration):
        """Update scrounger positions"""
        sorted_idx = np.argsort(self.fitness)
        producers_idx = sorted_idx[: self.n_producers]
        scroungers_idx = sorted_idx[self.n_producers :]

        for idx in scroungers_idx:
            A = np.random.choice([-1, 1], size=self.n_dim)
            A_plus = A.T / (A.T @ A + 1e-10)

            if idx > self.n_sparrows / 2:
                # Hungry scroungers - search more actively
                Q = np.random.randn()
                self.population[idx] = Q * np.exp(
                    (self.population[sorted_idx[-1]] - self.population[idx])
                    / (idx**2 + 1e-10)
                )
            else:
                # Less hungry - follow producers
                best_producer = self.population[np.random.choice(producers_idx)]
                self.population[idx] = (
                    best_producer
                    + np.abs(self.population[idx] - best_producer) * A_plus
                )

            self.population[idx] = self._clip_to_bounds(self.population[idx])

    def _update_scouters(self, iteration):
        """Update scouter positions (anti-predator behavior)"""
        sorted_idx = np.argsort(self.fitness)

        # Select random scouters
        scouter_idx = np.random.choice(
            range(self.n_sparrows), size=self.n_scouters, replace=False
        )

        for idx in scouter_idx:
            if self.fitness[idx] > self.best_fitness:
                # Worse than best - escape
                beta = np.random.randn()
                K = np.random.choice([-1, 1])
                self.population[idx] = self.best_position + beta * np.abs(
                    self.population[idx] - self.best_position
                )
            else:
                # Close to best - explore nearby
                K = np.random.choice([-1, 1])
                self.population[idx] = self.population[idx] + K * (
                    np.abs(self.population[idx] - self.population[sorted_idx[-1]])
                ) / (np.abs(self.fitness[idx] - self.fitness[sorted_idx[-1]]) + 1e-10)

            self.population[idx] = self._clip_to_bounds(self.population[idx])

    def optimize(self, verbose=True):
        """Run SSA optimization"""
        if verbose:
            print("\n" + "=" * 60)
            print("🐦 Sparrow Search Algorithm (SSA) Optimization")
            print("=" * 60)
            print(f"   Population: {self.n_sparrows}")
            print(f"   Producers: {self.n_producers}")
            print(f"   Scouters: {self.n_scouters}")
            print(f"   Max iterations: {self.max_iter}")
            print(f"   Dimensions: {self.n_dim}")

        for iteration in range(self.max_iter):
            # Update positions
            self._update_producers(iteration)
            self._update_scroungers(iteration)
            self._update_scouters(iteration)

            # Evaluate fitness
            self.fitness = np.array(
                [self.objective_func(ind) for ind in self.population]
            )

            # Update best
            current_best_idx = np.argmin(self.fitness)
            if self.fitness[current_best_idx] < self.best_fitness:
                self.best_fitness = self.fitness[current_best_idx]
                self.best_position = self.population[current_best_idx].copy()

            self.history.append(self.best_fitness)

            if verbose and (iteration + 1) % 10 == 0:
                print(
                    f"   Iteration {iteration+1}/{self.max_iter} - Best fitness: {self.best_fitness:.6f}"
                )

        if verbose:
            print(f"\n   ✅ Optimization complete!")
            print(f"   📊 Best fitness: {self.best_fitness:.6f}")
            print(f"   📊 Best position: {self.best_position}")

        return self.best_position, self.best_fitness


def create_model_from_params(params, input_shape, num_classes):
    """Create model from SSA parameters"""
    conv_filters = int(params[0])
    lstm_units = int(params[1])
    dense_units = int(params[2])
    dropout = params[3]

    inputs = keras.layers.Input(shape=input_shape)
    x = keras.layers.Conv1D(conv_filters, 3, padding="same", activation="relu")(inputs)
    x = keras.layers.MaxPooling1D(2, padding="same")(x)
    x = keras.layers.LSTM(lstm_units)(x)
    x = keras.layers.Dense(dense_units, activation="relu")(x)
    x = keras.layers.Dropout(dropout)(x)
    outputs = keras.layers.Dense(num_classes, activation="softmax")(x)

    model = keras.Model(inputs, outputs)
    model.compile(
        optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
    )
    return model


def main():
    print("\n" + "=" * 70)
    print("🎓 SSA Optimizer for Neural Network Hyperparameters")
    print("=" * 70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Demo: Optimize Rastrigin function
    def rastrigin(x):
        """Rastrigin benchmark function"""
        A = 10
        n = len(x)
        return A * n + sum([(xi**2 - A * np.cos(2 * np.pi * xi)) for xi in x])

    # Bounds for 5D Rastrigin
    bounds = [(-5.12, 5.12)] * 5

    # Create optimizer
    ssa = SparrowSearchAlgorithm(
        objective_func=rastrigin,
        bounds=bounds,
        n_sparrows=30,
        max_iter=100,
        producer_ratio=0.2,
        safety_threshold=0.8,
        scouter_ratio=0.1,
    )

    # Run optimization
    best_position, best_fitness = ssa.optimize(verbose=True)

    print("\n" + "=" * 60)
    print("🧠 Neural Network Hyperparameter Optimization Demo")
    print("=" * 60)

    # Define bounds for neural network hyperparameters
    # [conv_filters, lstm_units, dense_units, dropout]
    nn_bounds = [
        (16, 128),  # conv_filters
        (32, 256),  # lstm_units
        (64, 512),  # dense_units
        (0.1, 0.5),  # dropout
    ]

    # Demo: Simple objective function (would use actual training in practice)
    def nn_objective(params):
        """Simulated accuracy (would train model in real scenario)"""
        conv_filters = int(params[0])
        lstm_units = int(params[1])
        dense_units = int(params[2])
        dropout = params[3]

        # Simulate accuracy based on params (in practice, train and evaluate)
        # Higher values generally better, but with diminishing returns
        accuracy = (
            0.95
            + 0.03 * np.tanh((conv_filters - 50) / 50)
            + 0.01 * np.tanh((lstm_units - 100) / 100)
            + 0.005 * np.tanh((dense_units - 200) / 200)
            - 0.02 * (dropout - 0.3) ** 2
        )

        # We minimize, so return negative accuracy
        return -accuracy

    print("\n   Optimizing neural network hyperparameters...")

    ssa_nn = SparrowSearchAlgorithm(
        objective_func=nn_objective,
        bounds=nn_bounds,
        n_sparrows=20,
        max_iter=50,
        producer_ratio=0.2,
        safety_threshold=0.8,
    )

    best_params, best_obj = ssa_nn.optimize(verbose=True)

    print("\n" + "=" * 60)
    print("📊 Optimal Hyperparameters")
    print("=" * 60)
    print(f"   Conv filters: {int(best_params[0])}")
    print(f"   LSTM units: {int(best_params[1])}")
    print(f"   Dense units: {int(best_params[2])}")
    print(f"   Dropout: {best_params[3]:.4f}")
    print(f"   Estimated accuracy: {-best_obj*100:.2f}%")

    # Save results
    results = {
        "algorithm": "SSA",
        "best_params": {
            "conv_filters": int(best_params[0]),
            "lstm_units": int(best_params[1]),
            "dense_units": int(best_params[2]),
            "dropout": float(best_params[3]),
        },
        "best_objective": float(best_obj),
        "convergence_history": [float(x) for x in ssa_nn.history],
        "created_at": datetime.now().isoformat(),
    }

    results_path = PROJECT_ROOT / "models" / "ssa_optimization_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n💾 Results saved: {results_path}")

    print("\n" + "=" * 70)
    print("✅ SSA Optimization Complete!")
    print("=" * 70)
    print(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
