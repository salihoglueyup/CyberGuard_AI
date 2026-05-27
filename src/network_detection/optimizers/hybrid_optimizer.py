"""
Hybrid Optimizer: SSA + Bayesian Optimization
Makale için geliştirilmiş hiperparametre optimizasyonu

SSA (Salp Swarm Algorithm) + Bayesian Optimization kombinasyonu:
- SSA: Global arama, explorationiyi
- Bayesian: Local arama, exploitation iyi
- Hibrit: İkisinin güçlü yanlarını birleştir

Kullanım:
    optimizer = HybridOptimizer(objective, search_space)
    best_params, best_score = optimizer.optimize()
"""

import warnings
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

warnings.filterwarnings("ignore")


@dataclass
class OptimizationResult:
    """Optimizasyon sonucu"""

    best_params: dict
    best_score: float
    history: list[dict]
    total_evaluations: int
    convergence_epoch: int


class SSABayesianOptimizer:
    """
    SSA + Bayesian Hibrit Optimizer

    Aşama 1: SSA ile global arama (exploration)
    Aşama 2: Bayesian ile local refinement (exploitation)

    Args:
        objective_function: Maximize edilecek fonksiyon
        search_space: Parametre arama alanı
        ssa_iterations: SSA iterasyon sayısı
        bayesian_iterations: Bayesian iterasyon sayısı
        population_size: SSA popülasyon boyutu
        minimize: True ise minimize, False ise maximize
    """

    def __init__(
        self,
        objective_function: Callable[[dict], float],
        search_space: dict[str, tuple],
        ssa_iterations: int = 20,
        bayesian_iterations: int = 15,
        population_size: int = 15,
        minimize: bool = False,
        verbose: bool = True,
    ):
        self.objective = objective_function
        self.search_space = search_space
        self.ssa_iterations = ssa_iterations
        self.bayesian_iterations = bayesian_iterations
        self.population_size = population_size
        self.minimize = minimize
        self.verbose = verbose

        self.param_names = list(search_space.keys())
        self.bounds = self._parse_bounds()
        self.history: list[dict] = []
        self.best_params: dict | None = None
        self.best_score: float = float("inf") if minimize else float("-inf")

    def _parse_bounds(self) -> list[tuple[float, float, str]]:
        """Sınırları parse et"""
        bounds = []
        for name, spec in self.search_space.items():
            lower, upper, dtype = spec
            bounds.append((lower, upper, dtype))
        return bounds

    def _decode_params(self, position: np.ndarray) -> dict:
        """Pozisyonu parametrelere dönüştür"""
        params = {}
        for i, (name, (lower, upper, dtype)) in enumerate(
            zip(self.param_names, self.bounds)
        ):
            val = position[i]
            val = np.clip(val, lower, upper)
            if dtype == "int":
                params[name] = int(round(val))
            else:
                params[name] = float(val)
        return params

    def _encode_params(self, params: dict) -> np.ndarray:
        """Parametreleri pozisyona dönüştür"""
        position = np.zeros(len(self.param_names))
        for i, name in enumerate(self.param_names):
            position[i] = params.get(name, 0)
        return position

    def _evaluate(self, params: dict) -> float:
        """Objective function'ı değerlendir"""
        try:
            score = self.objective(params)
            self.history.append({"params": params, "score": score})

            # Best güncelle
            if self.minimize:
                if score < self.best_score:
                    self.best_score = score
                    self.best_params = params.copy()
            else:
                if score > self.best_score:
                    self.best_score = score
                    self.best_params = params.copy()

            return score
        except Exception as e:
            if self.verbose:
                print(f"   ⚠️ Evaluation error: {e}")
            return float("inf") if self.minimize else float("-inf")

    def _ssa_phase(self) -> tuple[np.ndarray, float]:
        """
        SSA (Salp Swarm Algorithm) fazı - Global exploration
        """
        if self.verbose:
            print("\n🦠 Aşama 1: SSA Global Arama")
            print("=" * 50)

        dim = len(self.param_names)

        # Popülasyon başlat
        population = np.zeros((self.population_size, dim))
        for i in range(self.population_size):
            for j, (lower, upper, _) in enumerate(self.bounds):
                population[i, j] = np.random.uniform(lower, upper)

        # Fitness hesapla
        fitness = np.zeros(self.population_size)
        for i in range(self.population_size):
            params = self._decode_params(population[i])
            fitness[i] = self._evaluate(params)

        # En iyi çözümü bul
        if self.minimize:
            best_idx = np.argmin(fitness)
        else:
            best_idx = np.argmax(fitness)

        food_pos = population[best_idx].copy()
        food_score = fitness[best_idx]

        # SSA iterasyonları
        for t in range(self.ssa_iterations):
            c1 = 2 * np.exp(-((4 * t / self.ssa_iterations) ** 2))

            for i in range(self.population_size):
                if i < self.population_size // 2:
                    # Leader salps
                    for j in range(dim):
                        c2 = np.random.random()
                        c3 = np.random.random()
                        lower, upper, _ = self.bounds[j]

                        if c3 < 0.5:
                            population[i, j] = food_pos[j] + c1 * (
                                (upper - lower) * c2 + lower
                            )
                        else:
                            population[i, j] = food_pos[j] - c1 * (
                                (upper - lower) * c2 + lower
                            )
                else:
                    # Follower salps
                    population[i] = (population[i] + population[i - 1]) / 2

                # Sınırları kontrol et
                for j, (lower, upper, _) in enumerate(self.bounds):
                    population[i, j] = np.clip(population[i, j], lower, upper)

            # Fitness güncelle
            for i in range(self.population_size):
                params = self._decode_params(population[i])
                current_fitness = self._evaluate(params)
                fitness[i] = current_fitness

                # Food güncelle
                if self.minimize:
                    if current_fitness < food_score:
                        food_score = current_fitness
                        food_pos = population[i].copy()
                else:
                    if current_fitness > food_score:
                        food_score = current_fitness
                        food_pos = population[i].copy()

            if self.verbose and (t + 1) % 5 == 0:
                print(
                    f"   SSA Iteration {t+1}/{self.ssa_iterations}: Best = {food_score:.4f}"
                )

        return food_pos, food_score

    def _bayesian_phase(self, initial_point: np.ndarray) -> tuple[dict, float]:
        """
        Bayesian Optimization fazı - Local refinement
        Gaussian Process tabanlı
        """
        if self.verbose:
            print("\n🎯 Aşama 2: Bayesian Local Refinement")
            print("=" * 50)

        try:
            from sklearn.gaussian_process import GaussianProcessRegressor
            from sklearn.gaussian_process.kernels import Matern

            USE_GP = True
        except ImportError:
            USE_GP = False
            if self.verbose:
                print("   ⚠️ sklearn GP bulunamadı, basit local search kullanılacak")

        # Başlangıç noktaları (SSA'dan gelen + random)
        X_observed = [initial_point]
        y_observed = [self.best_score if not self.minimize else -self.best_score]

        # SSA history'den ek noktalar al
        for item in self.history[-self.population_size :]:
            X_observed.append(self._encode_params(item["params"]))
            y_observed.append(item["score"] if not self.minimize else -item["score"])

        X_observed = np.array(X_observed)
        y_observed = np.array(y_observed)

        if USE_GP:
            # Gaussian Process ile Bayesian Optimization
            kernel = Matern(nu=2.5)
            gp = GaussianProcessRegressor(
                kernel=kernel, n_restarts_optimizer=5, random_state=42
            )

            for i in range(self.bayesian_iterations):
                # GP'yi fit et
                gp.fit(X_observed, y_observed)

                # Acquisition function ile sonraki noktayı seç (UCB)
                best_next = None
                best_acq = float("-inf")

                for _ in range(100):
                    candidate = np.zeros(len(self.param_names))
                    for j, (lower, upper, _) in enumerate(self.bounds):
                        # Başlangıç noktası etrafında arama
                        center = initial_point[j]
                        spread = (upper - lower) * 0.2  # %20 aralık
                        candidate[j] = np.clip(
                            np.random.normal(center, spread), lower, upper
                        )

                    mu, sigma = gp.predict(candidate.reshape(1, -1), return_std=True)
                    kappa = 2.0
                    acq = mu[0] + kappa * sigma[0]

                    if acq > best_acq:
                        best_acq = acq
                        best_next = candidate

                # Değerlendir
                params = self._decode_params(best_next)
                score = self._evaluate(params)

                X_observed = np.vstack([X_observed, best_next])
                y_observed = np.append(
                    y_observed, score if not self.minimize else -score
                )

                if self.verbose and (i + 1) % 5 == 0:
                    print(
                        f"   Bayesian Iteration {i+1}/{self.bayesian_iterations}: Score = {score:.4f}"
                    )
        else:
            # Basit local search
            for i in range(self.bayesian_iterations):
                candidate = initial_point.copy()
                for j, (lower, upper, _) in enumerate(self.bounds):
                    perturbation = np.random.normal(0, (upper - lower) * 0.1)
                    candidate[j] = np.clip(candidate[j] + perturbation, lower, upper)

                params = self._decode_params(candidate)
                score = self._evaluate(params)

                if (self.minimize and score < self.best_score) or (
                    not self.minimize and score > self.best_score
                ):
                    initial_point = candidate.copy()

        return self.best_params, self.best_score

    def optimize(self) -> OptimizationResult:
        """
        Hibrit optimizasyon çalıştır

        Returns:
            OptimizationResult
        """
        if self.verbose:
            print("\n" + "=" * 60)
            print("🚀 SSA + Bayesian Hibrit Optimizasyon")
            print("=" * 60)
            print(f"   SSA iterations: {self.ssa_iterations}")
            print(f"   Bayesian iterations: {self.bayesian_iterations}")
            print(f"   Population: {self.population_size}")
            print(f"   Parameters: {self.param_names}")

        # Aşama 1: SSA
        ssa_best_pos, ssa_best_score = self._ssa_phase()

        if self.verbose:
            print(f"\n   SSA En İyi: {self._decode_params(ssa_best_pos)}")
            print(f"   SSA Score: {ssa_best_score:.4f}")

        # Aşama 2: Bayesian
        final_params, final_score = self._bayesian_phase(ssa_best_pos)

        if self.verbose:
            print("\n🏆 Final En İyi Parametreler:")
            for k, v in final_params.items():
                print(f"   {k}: {v}")
            print(f"   Score: {final_score:.4f}")

        # Convergence epoch bul
        convergence = len(self.history)
        for i, item in enumerate(self.history):
            if abs(item["score"] - final_score) < 1e-4:
                convergence = i
                break

        return OptimizationResult(
            best_params=final_params,
            best_score=final_score,
            history=self.history,
            total_evaluations=len(self.history),
            convergence_epoch=convergence,
        )


# Alias
HybridOptimizer = SSABayesianOptimizer


# Test
if __name__ == "__main__":
    print("🧪 SSA + Bayesian Optimizer Test\n")

    # Test objective: Sphere function
    def sphere(params: dict) -> float:
        x = params.get("x", 0)
        y = params.get("y", 0)
        return -(x**2 + y**2)  # Maximize için negatif

    search_space = {
        "x": (-5, 5, "float"),
        "y": (-5, 5, "float"),
    }

    optimizer = HybridOptimizer(
        objective_function=sphere,
        search_space=search_space,
        ssa_iterations=10,
        bayesian_iterations=10,
        population_size=10,
        minimize=False,
        verbose=True,
    )

    result = optimizer.optimize()

    print("\n📊 Sonuç:")
    print(f"   Best: {result.best_params}")
    print(f"   Score: {result.best_score}")
    print(f"   Evaluations: {result.total_evaluations}")
