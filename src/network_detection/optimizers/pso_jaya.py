"""
PSO Optimizer - Particle Swarm Optimization
CyberGuard AI için hiperparametre optimizasyonu

PSO:
    - Kennedy & Eberhart (1995)
    - Parçacık sürüsü davranışı
    - Global arama yeteneği
    - Hızlı yakınsama

Ref: Makaledeki PSO-LSTMIDS yaklaşımı
"""

import random
import time
from collections.abc import Callable

import numpy as np


class PSOOptimizer:
    """
    Particle Swarm Optimization (PSO)

    Makaledeki PSO-LSTMIDS için kullanılır.
    """

    DEFAULT_SEARCH_SPACE = {
        "lstm_units": (32, 256, "int"),
        "conv_filters": (16, 64, "int"),
        "dropout_rate": (0.1, 0.5, "float"),
        "learning_rate": (0.0001, 0.01, "float"),
        "batch_size": (32, 256, "int"),
    }

    def __init__(
        self,
        objective_function: Callable,
        search_space: dict | None = None,
        n_particles: int = 20,
        max_iterations: int = 30,
        w: float = 0.7,  # Inertia weight
        c1: float = 1.5,  # Cognitive component
        c2: float = 1.5,  # Social component
        minimize: bool = False,
        verbose: bool = True,
    ):
        self.objective_function = objective_function
        self.search_space = search_space or self.DEFAULT_SEARCH_SPACE
        self.n_particles = n_particles
        self.max_iterations = max_iterations
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.minimize = minimize
        self.verbose = verbose

        self.param_names = list(self.search_space.keys())
        self.n_dims = len(self.param_names)

        self.lower_bounds = np.array(
            [self.search_space[p][0] for p in self.param_names]
        )
        self.upper_bounds = np.array(
            [self.search_space[p][1] for p in self.param_names]
        )

        self.best_params: dict = {}
        self.best_score: float = float("-inf") if not minimize else float("inf")
        self.history: list[dict] = []

        if verbose:
            print("🐦 PSO Optimizer başlatıldı")
            print(f"   Particles: {n_particles}, Iterations: {max_iterations}")

    def _decode(self, position: np.ndarray) -> dict:
        params = {}
        for i, name in enumerate(self.param_names):
            _, _, dtype = self.search_space[name]
            if dtype == "int":
                params[name] = int(round(position[i]))
            else:
                params[name] = float(position[i])
        return params

    def optimize(self) -> tuple[dict, float]:
        print("\n🚀 PSO Optimizasyonu başlıyor...")
        start_time = time.time()

        # Initialize particles
        positions = np.random.uniform(
            self.lower_bounds, self.upper_bounds, (self.n_particles, self.n_dims)
        )
        velocities = np.zeros((self.n_particles, self.n_dims))

        # Personal best
        pbest_positions = positions.copy()
        pbest_scores = np.full(
            self.n_particles, float("-inf") if not self.minimize else float("inf")
        )

        # Global best
        gbest_position = None
        gbest_score = float("-inf") if not self.minimize else float("inf")

        # Evaluate initial
        for i in range(self.n_particles):
            params = self._decode(positions[i])
            try:
                score = self.objective_function(params)
            except Exception:
                score = float("-inf") if not self.minimize else float("inf")

            if self.minimize:
                if score < pbest_scores[i]:
                    pbest_scores[i] = score
                    pbest_positions[i] = positions[i].copy()
                if score < gbest_score:
                    gbest_score = score
                    gbest_position = positions[i].copy()
            else:
                if score > pbest_scores[i]:
                    pbest_scores[i] = score
                    pbest_positions[i] = positions[i].copy()
                if score > gbest_score:
                    gbest_score = score
                    gbest_position = positions[i].copy()

        # Main loop
        for iteration in range(self.max_iterations):
            for i in range(self.n_particles):
                r1, r2 = random.random(), random.random()

                # Update velocity
                velocities[i] = (
                    self.w * velocities[i]
                    + self.c1 * r1 * (pbest_positions[i] - positions[i])
                    + self.c2 * r2 * (gbest_position - positions[i])
                )

                # Update position
                positions[i] = positions[i] + velocities[i]
                positions[i] = np.clip(
                    positions[i], self.lower_bounds, self.upper_bounds
                )

                # Evaluate
                params = self._decode(positions[i])
                try:
                    score = self.objective_function(params)
                except Exception:
                    continue

                # Update personal best
                if (self.minimize and score < pbest_scores[i]) or (
                    not self.minimize and score > pbest_scores[i]
                ):
                    pbest_scores[i] = score
                    pbest_positions[i] = positions[i].copy()

                    # Update global best
                    if (self.minimize and score < gbest_score) or (
                        not self.minimize and score > gbest_score
                    ):
                        gbest_score = score
                        gbest_position = positions[i].copy()

            self.history.append({"iteration": iteration + 1, "best_score": gbest_score})

            if self.verbose and (iteration + 1) % 5 == 0:
                print(f"   İterasyon {iteration+1}: Best = {gbest_score:.4f}")

        self.best_score = gbest_score
        self.best_params = self._decode(gbest_position)

        elapsed = time.time() - start_time
        print(f"\n✅ PSO tamamlandı! Süre: {elapsed:.1f}s")
        print(f"   🏆 Best score: {self.best_score:.4f}")

        return self.best_params, self.best_score


class JAYAOptimizer:
    """
    JAYA Algorithm - Parameter-free optimization

    Avantaj: Hiçbir algoritma parametresi yok!
    Sadece populasyon boyutu ve iterasyon sayısı.

    Ref: Rao (2016) - "JAYA: A simple and new optimization algorithm"
    """

    DEFAULT_SEARCH_SPACE = {
        "lstm_units": (32, 256, "int"),
        "conv_filters": (16, 64, "int"),
        "dropout_rate": (0.1, 0.5, "float"),
        "learning_rate": (0.0001, 0.01, "float"),
    }

    def __init__(
        self,
        objective_function: Callable,
        search_space: dict | None = None,
        population_size: int = 20,
        max_iterations: int = 30,
        minimize: bool = False,
        verbose: bool = True,
    ):
        self.objective_function = objective_function
        self.search_space = search_space or self.DEFAULT_SEARCH_SPACE
        self.population_size = population_size
        self.max_iterations = max_iterations
        self.minimize = minimize
        self.verbose = verbose

        self.param_names = list(self.search_space.keys())
        self.n_dims = len(self.param_names)

        self.lower_bounds = np.array(
            [self.search_space[p][0] for p in self.param_names]
        )
        self.upper_bounds = np.array(
            [self.search_space[p][1] for p in self.param_names]
        )

        self.best_params: dict = {}
        self.best_score: float = float("-inf") if not minimize else float("inf")

        if verbose:
            print("🎯 JAYA Optimizer başlatıldı (parametresiz!)")

    def _decode(self, solution: np.ndarray) -> dict:
        params = {}
        for i, name in enumerate(self.param_names):
            _, _, dtype = self.search_space[name]
            if dtype == "int":
                params[name] = int(round(solution[i]))
            else:
                params[name] = float(solution[i])
        return params

    def optimize(self) -> tuple[dict, float]:
        print("\n🚀 JAYA Optimizasyonu başlıyor...")
        start_time = time.time()

        # Initialize population
        population = np.random.uniform(
            self.lower_bounds, self.upper_bounds, (self.population_size, self.n_dims)
        )
        fitness = np.zeros(self.population_size)

        # Evaluate initial population
        for i in range(self.population_size):
            params = self._decode(population[i])
            try:
                fitness[i] = self.objective_function(params)
            except Exception:
                fitness[i] = float("-inf") if not self.minimize else float("inf")

        # Main loop
        for iteration in range(self.max_iterations):
            # Find best and worst
            if self.minimize:
                best_idx = np.argmin(fitness)
                worst_idx = np.argmax(fitness)
            else:
                best_idx = np.argmax(fitness)
                worst_idx = np.argmin(fitness)

            best_solution = population[best_idx].copy()
            worst_solution = population[worst_idx].copy()

            # Update each solution
            for i in range(self.population_size):
                r1, r2 = random.random(), random.random()

                # JAYA update rule (parametresiz!)
                new_solution = (
                    population[i]
                    + r1 * (best_solution - np.abs(population[i]))
                    - r2 * (worst_solution - np.abs(population[i]))
                )

                new_solution = np.clip(
                    new_solution, self.lower_bounds, self.upper_bounds
                )

                # Evaluate new solution
                params = self._decode(new_solution)
                try:
                    new_fitness = self.objective_function(params)
                except Exception:
                    continue

                # Greedy selection
                if (self.minimize and new_fitness < fitness[i]) or (
                    not self.minimize and new_fitness > fitness[i]
                ):
                    population[i] = new_solution
                    fitness[i] = new_fitness

            # Update global best
            if self.minimize:
                current_best = np.min(fitness)
            else:
                current_best = np.max(fitness)

            if (self.minimize and current_best < self.best_score) or (
                not self.minimize and current_best > self.best_score
            ):
                self.best_score = current_best
                best_idx = (
                    np.argmax(fitness) if not self.minimize else np.argmin(fitness)
                )
                self.best_params = self._decode(population[best_idx])

            if self.verbose and (iteration + 1) % 5 == 0:
                print(f"   İterasyon {iteration+1}: Best = {self.best_score:.4f}")

        elapsed = time.time() - start_time
        print(f"\n✅ JAYA tamamlandı! Süre: {elapsed:.1f}s")
        print(f"   🏆 Best score: {self.best_score:.4f}")

        return self.best_params, self.best_score


# Test
if __name__ == "__main__":
    print("🧪 Optimizer Test\n")

    def test_func(params):
        x = params.get("lstm_units", 100) - 100
        y = params.get("conv_filters", 40) - 40
        return -(x**2 + y**2)

    space = {
        "lstm_units": (50, 150, "int"),
        "conv_filters": (20, 60, "int"),
    }

    # PSO Test
    pso = PSOOptimizer(test_func, space, n_particles=10, max_iterations=10)
    pso_params, pso_score = pso.optimize()

    # JAYA Test
    jaya = JAYAOptimizer(test_func, space, population_size=10, max_iterations=10)
    jaya_params, jaya_score = jaya.optimize()

    print("\n📊 Sonuçlar:")
    print(f"   PSO:  {pso_params} → {pso_score}")
    print(f"   JAYA: {jaya_params} → {jaya_score}")
