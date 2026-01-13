"""
Feature Selection - CyberGuard AI
=================================

Meta-heuristic ve klasik feature selection teknikleri.

Özellikler:
    - Mutual Information
    - Recursive Feature Elimination (RFE)
    - PSO-based Feature Selection
    - SSA-based Feature Selection
    - Hybrid Selection
"""

import numpy as np
from typing import Tuple, Optional, List, Dict, Callable
from dataclasses import dataclass
import logging

logger = logging.getLogger("FeatureSelection")


@dataclass
class FeatureSelectionResult:
    """Feature selection sonucu"""

    selected_indices: List[int]
    selected_names: List[str]
    scores: Dict[str, float]
    n_original: int
    n_selected: int
    reduction_ratio: float


class MutualInformationSelector:
    """
    Mutual Information based Feature Selection

    Her feature'ın target ile bilgi paylaşımını ölçer.
    """

    def __init__(self, n_features: int = 20, random_state: int = 42):
        self.n_features = n_features
        self.random_state = random_state
        self.scores_ = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "MutualInformationSelector":
        """Feature importance hesapla"""
        from sklearn.feature_selection import mutual_info_classif

        # 3D → 2D
        if len(X.shape) == 3:
            X = X.reshape(X.shape[0], -1)

        self.scores_ = mutual_info_classif(X, y, random_state=self.random_state)
        return self

    def get_selected_indices(self) -> List[int]:
        """En iyi feature indekslerini döndür"""
        if self.scores_ is None:
            raise ValueError("fit() önce çağrılmalı")

        return np.argsort(self.scores_)[-self.n_features :].tolist()

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Feature'ları filtrele"""
        indices = self.get_selected_indices()

        if len(X.shape) == 3:
            # Sequence data için sadece son dimension'dan seç
            return X[:, :, indices]
        return X[:, indices]


class RFESelector:
    """
    Recursive Feature Elimination

    Model ile iteratif olarak en önemsiz feature'ı çıkar.
    """

    def __init__(
        self,
        n_features: int = 20,
        step: int = 1,
        random_state: int = 42,
    ):
        self.n_features = n_features
        self.step = step
        self.random_state = random_state
        self.selected_indices_ = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RFESelector":
        """RFE uygula"""
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.feature_selection import RFE

        # 3D → 2D
        original_shape = X.shape
        if len(original_shape) == 3:
            X_flat = X.reshape(X.shape[0], -1)
        else:
            X_flat = X

        # Base model
        estimator = RandomForestClassifier(
            n_estimators=50,
            random_state=self.random_state,
            n_jobs=-1,
        )

        # RFE
        rfe = RFE(
            estimator=estimator,
            n_features_to_select=self.n_features,
            step=self.step,
        )
        rfe.fit(X_flat, y)

        self.selected_indices_ = np.where(rfe.support_)[0].tolist()
        self.ranking_ = rfe.ranking_

        return self

    def get_selected_indices(self) -> List[int]:
        """Seçilen indeksler"""
        if self.selected_indices_ is None:
            raise ValueError("fit() önce çağrılmalı")
        return self.selected_indices_

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Feature'ları filtrele"""
        indices = self.get_selected_indices()

        if len(X.shape) == 3:
            return X[:, :, indices]
        return X[:, indices]


class PSOFeatureSelector:
    """
    Particle Swarm Optimization based Feature Selection

    PSO ile optimal feature subset bulma.
    """

    def __init__(
        self,
        n_particles: int = 20,
        n_iterations: int = 50,
        w: float = 0.7,  # Inertia weight
        c1: float = 1.5,  # Cognitive parameter
        c2: float = 1.5,  # Social parameter
        min_features: int = 5,
        random_state: int = 42,
    ):
        self.n_particles = n_particles
        self.n_iterations = n_iterations
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.min_features = min_features
        self.random_state = random_state

        self.best_position_ = None
        self.best_fitness_ = 0
        self.history_ = []

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        fitness_func: Callable = None,
    ) -> "PSOFeatureSelector":
        """
        PSO ile feature selection

        Args:
            X: Features
            y: Labels
            fitness_func: Custom fitness function (X, y, mask) -> score
        """
        np.random.seed(self.random_state)

        # Flatten if 3D
        if len(X.shape) == 3:
            n_features = X.shape[2]
        else:
            n_features = X.shape[1]

        # Initialize particles
        positions = np.random.rand(self.n_particles, n_features)
        velocities = np.random.rand(self.n_particles, n_features) * 0.1

        # Personal best
        p_best = positions.copy()
        p_best_fitness = np.zeros(self.n_particles)

        # Global best
        g_best = positions[0].copy()
        g_best_fitness = 0

        # Fitness function
        if fitness_func is None:
            fitness_func = self._default_fitness

        print(f"🔄 PSO Feature Selection başlıyor...")
        print(f"   Particles: {self.n_particles}, Iterations: {self.n_iterations}")

        for iteration in range(self.n_iterations):
            for i in range(self.n_particles):
                # Binary mask
                mask = positions[i] > 0.5

                # Minimum feature constraint
                if mask.sum() < self.min_features:
                    top_indices = np.argsort(positions[i])[-self.min_features :]
                    mask = np.zeros(n_features, dtype=bool)
                    mask[top_indices] = True

                # Evaluate fitness
                fitness = fitness_func(X, y, mask)

                # Update personal best
                if fitness > p_best_fitness[i]:
                    p_best_fitness[i] = fitness
                    p_best[i] = positions[i].copy()

                # Update global best
                if fitness > g_best_fitness:
                    g_best_fitness = fitness
                    g_best = positions[i].copy()

            # Update velocities and positions
            r1 = np.random.rand(self.n_particles, n_features)
            r2 = np.random.rand(self.n_particles, n_features)

            velocities = (
                self.w * velocities
                + self.c1 * r1 * (p_best - positions)
                + self.c2 * r2 * (g_best - positions)
            )

            # Velocity clamping
            velocities = np.clip(velocities, -0.5, 0.5)

            positions += velocities
            positions = np.clip(positions, 0, 1)

            self.history_.append(g_best_fitness)

            if (iteration + 1) % 10 == 0:
                n_selected = (g_best > 0.5).sum()
                print(
                    f"   Iteration {iteration+1}: Fitness={g_best_fitness:.4f}, Features={n_selected}"
                )

        # Store best
        self.best_position_ = g_best
        self.best_fitness_ = g_best_fitness

        n_selected = (g_best > 0.5).sum()
        logger.info(f"✅ PSO completed: {n_features} → {n_selected} features")

        return self

    def _default_fitness(
        self,
        X: np.ndarray,
        y: np.ndarray,
        mask: np.ndarray,
    ) -> float:
        """Default fitness: accuracy with penalty for many features"""
        from sklearn.model_selection import cross_val_score
        from sklearn.ensemble import RandomForestClassifier

        if mask.sum() == 0:
            return 0

        # Flatten and select features
        if len(X.shape) == 3:
            X_flat = X[:, :, mask].reshape(X.shape[0], -1)
        else:
            X_flat = X[:, mask]

        # Quick evaluation
        clf = RandomForestClassifier(n_estimators=10, random_state=42, n_jobs=-1)
        scores = cross_val_score(clf, X_flat, y, cv=3)

        # Fitness = accuracy - penalty for many features
        accuracy = scores.mean()
        n_features = mask.sum()
        penalty = 0.001 * n_features  # Small penalty

        return accuracy - penalty

    def get_selected_indices(self) -> List[int]:
        """Seçilen feature indeksleri"""
        if self.best_position_ is None:
            raise ValueError("fit() önce çağrılmalı")

        mask = self.best_position_ > 0.5
        return np.where(mask)[0].tolist()

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Feature'ları filtrele"""
        indices = self.get_selected_indices()

        if len(X.shape) == 3:
            return X[:, :, indices]
        return X[:, indices]


class SSAFeatureSelector:
    """
    Salp Swarm Algorithm based Feature Selection

    Makaleyle uyumlu meta-heuristic.
    """

    def __init__(
        self,
        n_salps: int = 20,
        n_iterations: int = 50,
        min_features: int = 5,
        random_state: int = 42,
    ):
        self.n_salps = n_salps
        self.n_iterations = n_iterations
        self.min_features = min_features
        self.random_state = random_state

        self.best_position_ = None
        self.best_fitness_ = 0
        self.history_ = []

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        fitness_func: Callable = None,
    ) -> "SSAFeatureSelector":
        """SSA ile feature selection"""
        np.random.seed(self.random_state)

        if len(X.shape) == 3:
            n_features = X.shape[2]
        else:
            n_features = X.shape[1]

        # Initialize salps
        positions = np.random.rand(self.n_salps, n_features)

        # Food position (best solution)
        food_position = positions[0].copy()
        food_fitness = 0

        if fitness_func is None:
            fitness_func = self._default_fitness

        print(f"🔄 SSA Feature Selection başlıyor...")

        lb, ub = 0, 1  # Lower and upper bounds

        for iteration in range(self.n_iterations):
            # c1 decreases linearly
            c1 = 2 * np.exp(-((4 * iteration / self.n_iterations) ** 2))

            for i in range(self.n_salps):
                # Evaluate fitness
                mask = positions[i] > 0.5

                if mask.sum() < self.min_features:
                    top_indices = np.argsort(positions[i])[-self.min_features :]
                    mask = np.zeros(n_features, dtype=bool)
                    mask[top_indices] = True

                fitness = fitness_func(X, y, mask)

                # Update food
                if fitness > food_fitness:
                    food_fitness = fitness
                    food_position = positions[i].copy()

            # Update positions
            for i in range(self.n_salps):
                if i == 0:  # Leader salp
                    for j in range(n_features):
                        c2 = np.random.rand()
                        c3 = np.random.rand()

                        if c3 < 0.5:
                            positions[i, j] = food_position[j] + c1 * (
                                (ub - lb) * c2 + lb
                            )
                        else:
                            positions[i, j] = food_position[j] - c1 * (
                                (ub - lb) * c2 + lb
                            )
                else:  # Follower salps
                    positions[i] = 0.5 * (positions[i] + positions[i - 1])

            # Bound positions
            positions = np.clip(positions, lb, ub)

            self.history_.append(food_fitness)

            if (iteration + 1) % 10 == 0:
                n_selected = (food_position > 0.5).sum()
                print(
                    f"   Iteration {iteration+1}: Fitness={food_fitness:.4f}, Features={n_selected}"
                )

        self.best_position_ = food_position
        self.best_fitness_ = food_fitness

        return self

    def _default_fitness(self, X, y, mask):
        """Default fitness function"""
        from sklearn.model_selection import cross_val_score
        from sklearn.ensemble import RandomForestClassifier

        if mask.sum() == 0:
            return 0

        if len(X.shape) == 3:
            X_flat = X[:, :, mask].reshape(X.shape[0], -1)
        else:
            X_flat = X[:, mask]

        clf = RandomForestClassifier(n_estimators=10, random_state=42, n_jobs=-1)
        scores = cross_val_score(clf, X_flat, y, cv=3)

        return scores.mean() - 0.001 * mask.sum()

    def get_selected_indices(self) -> List[int]:
        if self.best_position_ is None:
            raise ValueError("fit() önce çağrılmalı")
        return np.where(self.best_position_ > 0.5)[0].tolist()

    def transform(self, X: np.ndarray) -> np.ndarray:
        indices = self.get_selected_indices()
        if len(X.shape) == 3:
            return X[:, :, indices]
        return X[:, indices]


# ============= Utility Functions =============


def select_features(
    X: np.ndarray,
    y: np.ndarray,
    method: str = "mutual_info",
    n_features: int = 20,
    feature_names: List[str] = None,
    **kwargs,
) -> FeatureSelectionResult:
    """
    Feature selection wrapper

    Args:
        X: Features
        y: Labels
        method: "mutual_info", "rfe", "pso", "ssa"
        n_features: Kaç feature seçilecek
        feature_names: Feature isimleri
    """
    selectors = {
        "mutual_info": MutualInformationSelector,
        "rfe": RFESelector,
        "pso": PSOFeatureSelector,
        "ssa": SSAFeatureSelector,
    }

    if method not in selectors:
        raise ValueError(f"Unknown method: {method}")

    if method in ["pso", "ssa"]:
        selector = selectors[method](**kwargs)
    else:
        selector = selectors[method](n_features=n_features, **kwargs)

    selector.fit(X, y)
    indices = selector.get_selected_indices()

    # Feature names
    if feature_names:
        names = [feature_names[i] for i in indices if i < len(feature_names)]
    else:
        names = [f"feature_{i}" for i in indices]

    n_original = X.shape[-1]
    n_selected = len(indices)

    return FeatureSelectionResult(
        selected_indices=indices,
        selected_names=names,
        scores={"n_selected": n_selected},
        n_original=n_original,
        n_selected=n_selected,
        reduction_ratio=(n_original - n_selected) / n_original,
    )


# Test
if __name__ == "__main__":
    print("🧪 Feature Selection Test\n")

    # Create test data
    np.random.seed(42)
    n_samples = 500
    n_features = 41

    X = np.random.randn(n_samples, 10, n_features).astype(np.float32)
    y = np.random.randint(0, 5, n_samples)

    print(f"📊 Original features: {n_features}")

    # Test Mutual Information
    print("\n🔍 Mutual Information:")
    result = select_features(X, y, method="mutual_info", n_features=20)
    print(f"   Selected: {result.n_selected}")
    print(f"   Reduction: {result.reduction_ratio*100:.1f}%")

    # Test PSO (quick)
    print("\n🔍 PSO Feature Selection:")
    pso = PSOFeatureSelector(n_particles=10, n_iterations=10)
    pso.fit(X[:100], y[:100])  # Quick test
    print(f"   Selected: {len(pso.get_selected_indices())}")
