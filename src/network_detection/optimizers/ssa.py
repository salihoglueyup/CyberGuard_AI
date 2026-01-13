"""
SSA Optimizer - Salp Swarm Algorithm
CyberGuard AI için hiperparametre optimizasyonu

SSA (Salp Swarm Algorithm):
    - Deniz salp'larının davranışına dayalı meta-heuristic algoritma
    - Makaledeki SSA-LSTMIDS için kullanılan yöntem
    - Hiperparametreleri otomatik optimize eder

Referans:
    Mirjalili, S., et al. "Salp Swarm Algorithm" (2017)
    Scientific Reports 2025 - SSA-LSTMIDS
"""

import numpy as np
from typing import Dict, List, Callable, Tuple, Optional
import random
import time


class SSAOptimizer:
    """
    Salp Swarm Algorithm (SSA) Optimizer

    Makaledeki SSA-LSTMIDS implementasyonu için kullanılır.
    LSTM/BiLSTM modellerinin hiperparametrelerini optimize eder.
    """

    # Optimize edilecek hiperparametreler ve aralıkları
    DEFAULT_SEARCH_SPACE = {
        "lstm_units": (32, 256, "int"),  # LSTM hücre sayısı
        "conv_filters": (16, 64, "int"),  # Conv1D filtre sayısı
        "dropout_rate": (0.1, 0.5, "float"),  # Dropout oranı
        "learning_rate": (0.0001, 0.01, "float"),  # Öğrenme oranı
        "batch_size": (32, 256, "int"),  # Batch boyutu
        "attention_units": (32, 128, "int"),  # Attention boyutu
    }

    def __init__(
        self,
        objective_function: Callable,
        search_space: Optional[Dict] = None,
        population_size: int = 20,
        max_iterations: int = 30,
        minimize: bool = False,  # False = maximize (accuracy)
        verbose: bool = True,
    ):
        """
        SSA Optimizer başlat

        Args:
            objective_function: Değerlendirilecek fonksiyon (params -> score)
            search_space: Hiperparametre arama alanı
            population_size: Popülasyon boyutu (salp sayısı)
            max_iterations: Maksimum iterasyon
            minimize: True=minimize, False=maximize
            verbose: Detaylı çıktı
        """
        self.objective_function = objective_function
        self.search_space = search_space or self.DEFAULT_SEARCH_SPACE
        self.population_size = population_size
        self.max_iterations = max_iterations
        self.minimize = minimize
        self.verbose = verbose

        self.param_names = list(self.search_space.keys())
        self.n_params = len(self.param_names)

        # Arama alanı sınırları
        self.lower_bounds = np.array(
            [self.search_space[p][0] for p in self.param_names]
        )
        self.upper_bounds = np.array(
            [self.search_space[p][1] for p in self.param_names]
        )

        # Sonuçlar
        self.best_params: Dict = {}
        self.best_score: float = float("-inf") if not minimize else float("inf")
        self.history: List[Dict] = []

        if verbose:
            print("🦠 SSA Optimizer başlatıldı")
            print(f"   Popülasyon: {population_size}")
            print(f"   İterasyonlar: {max_iterations}")
            print(f"   Parametreler: {self.param_names}")

    def _initialize_population(self) -> np.ndarray:
        """Rastgele popülasyon oluştur"""
        population = np.zeros((self.population_size, self.n_params))

        for i in range(self.population_size):
            for j, param in enumerate(self.param_names):
                lb, ub, dtype = self.search_space[param]
                if dtype == "int":
                    population[i, j] = random.randint(int(lb), int(ub))
                else:
                    population[i, j] = random.uniform(lb, ub)

        return population

    def _decode_solution(self, solution: np.ndarray) -> Dict:
        """Numpy array'i parametre dict'ine dönüştür"""
        params = {}
        for i, param in enumerate(self.param_names):
            _, _, dtype = self.search_space[param]
            if dtype == "int":
                params[param] = int(round(solution[i]))
            else:
                params[param] = float(solution[i])
        return params

    def _bound_solution(self, solution: np.ndarray) -> np.ndarray:
        """Çözümü sınırlar içinde tut"""
        return np.clip(solution, self.lower_bounds, self.upper_bounds)

    def optimize(self) -> Tuple[Dict, float]:
        """
        SSA optimizasyonu çalıştır

        Returns:
            (best_params, best_score)
        """
        print("\n🚀 SSA Optimizasyonu başlıyor...")
        start_time = time.time()

        # Popülasyonu başlat
        population = self._initialize_population()
        fitness = np.zeros(self.population_size)

        # İlk değerlendirme
        print("📊 İlk popülasyon değerlendiriliyor...")
        for i in range(self.population_size):
            params = self._decode_solution(population[i])
            try:
                fitness[i] = self.objective_function(params)
            except Exception as e:
                print(f"⚠️ Değerlendirme hatası: {e}")
                fitness[i] = float("-inf") if not self.minimize else float("inf")

        # En iyi çözümü bul
        if self.minimize:
            best_idx = np.argmin(fitness)
        else:
            best_idx = np.argmax(fitness)

        food_position = population[best_idx].copy()
        food_fitness = fitness[best_idx]

        self.best_score = food_fitness
        self.best_params = self._decode_solution(food_position)

        print(f"   İlk en iyi skor: {food_fitness:.4f}")

        # Ana iterasyon döngüsü
        for iteration in range(self.max_iterations):
            # c1 parametresi (azalan)
            c1 = 2 * np.exp(-((4 * iteration / self.max_iterations) ** 2))

            for i in range(self.population_size):
                if i < self.population_size // 2:
                    # Leader salps (food'a doğru hareket)
                    for j in range(self.n_params):
                        c2, c3 = random.random(), random.random()

                        if c3 < 0.5:
                            population[i, j] = food_position[j] + c1 * (
                                (self.upper_bounds[j] - self.lower_bounds[j]) * c2
                                + self.lower_bounds[j]
                            )
                        else:
                            population[i, j] = food_position[j] - c1 * (
                                (self.upper_bounds[j] - self.lower_bounds[j]) * c2
                                + self.lower_bounds[j]
                            )
                else:
                    # Follower salps
                    population[i] = (population[i] + population[i - 1]) / 2

                # Sınırları kontrol et
                population[i] = self._bound_solution(population[i])

            # Fitness değerlendirme
            for i in range(self.population_size):
                params = self._decode_solution(population[i])
                try:
                    fitness[i] = self.objective_function(params)
                except Exception as e:
                    fitness[i] = float("-inf") if not self.minimize else float("inf")

                # En iyi güncelle
                if self.minimize:
                    is_better = fitness[i] < self.best_score
                else:
                    is_better = fitness[i] > self.best_score

                if is_better:
                    self.best_score = fitness[i]
                    self.best_params = self._decode_solution(population[i])
                    food_position = population[i].copy()
                    food_fitness = fitness[i]

            # İterasyon sonucu kaydet
            self.history.append(
                {
                    "iteration": iteration + 1,
                    "best_score": self.best_score,
                    "best_params": self.best_params.copy(),
                    "mean_fitness": np.mean(fitness),
                }
            )

            if self.verbose and (iteration + 1) % 5 == 0:
                print(
                    f"   İterasyon {iteration + 1}/{self.max_iterations}: "
                    f"Best = {self.best_score:.4f}"
                )

        elapsed = time.time() - start_time

        print(f"\n✅ SSA Optimizasyonu tamamlandı!")
        print(f"   ⏱️ Süre: {elapsed:.1f}s")
        print(f"   🏆 En iyi skor: {self.best_score:.4f}")
        print(f"   📋 En iyi parametreler:")
        for k, v in self.best_params.items():
            print(f"      {k}: {v}")

        return self.best_params, self.best_score

    def get_optimization_history(self) -> List[Dict]:
        """Optimizasyon geçmişini döndür"""
        return self.history


class HyperparameterTuner:
    """
    IDS Modeli için Hiperparametre Tuner

    SSA optimizer'ı kullanarak model hiperparametrelerini optimize eder.
    """

    def __init__(
        self,
        model_class,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        search_space: Optional[Dict] = None,
        epochs_per_trial: int = 20,
        verbose: bool = True,
    ):
        """
        Tuner başlat

        Args:
            model_class: Model sınıfı (AdvancedIDSModel gibi)
            X_train, y_train: Eğitim verisi
            X_val, y_val: Validation verisi
            search_space: Arama alanı
            epochs_per_trial: Her deneme için epoch sayısı
        """
        self.model_class = model_class
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.search_space = search_space
        self.epochs_per_trial = epochs_per_trial
        self.verbose = verbose

        self.trial_count = 0

    def objective(self, params: Dict) -> float:
        """
        Objective function - model eğit ve accuracy döndür
        """
        self.trial_count += 1

        if self.verbose:
            print(f"\n🧪 Trial {self.trial_count}: {params}")

        try:
            # Model oluştur
            model = self.model_class(
                input_shape=self.X_train.shape[1:],
                num_classes=len(np.unique(self.y_train)),
                lstm_units=params.get("lstm_units", 120),
                conv_filters=params.get("conv_filters", 30),
                dropout_rate=params.get("dropout_rate", 0.3),
                attention_units=params.get("attention_units", 64),
            )
            model.build()

            # Eğit
            results = model.train(
                self.X_train,
                self.y_train,
                X_val=self.X_val,
                y_val=self.y_val,
                epochs=self.epochs_per_trial,
                batch_size=params.get("batch_size", 64),
                patience=5,
            )

            score = results.get("final_val_accuracy", results.get("final_accuracy", 0))

            if self.verbose:
                print(f"   ✅ Score: {score:.4f}")

            # Memory temizle
            import gc

            del model
            gc.collect()

            return score

        except Exception as e:
            print(f"   ❌ Trial failed: {e}")
            return 0.0

    def tune(
        self, population_size: int = 10, max_iterations: int = 15
    ) -> Tuple[Dict, float]:
        """
        SSA ile hiperparametre tune et

        Returns:
            (best_params, best_score)
        """
        print("\n🔧 Hiperparametre Tuning başlıyor...")

        optimizer = SSAOptimizer(
            objective_function=self.objective,
            search_space=self.search_space,
            population_size=population_size,
            max_iterations=max_iterations,
            minimize=False,  # Accuracy maximize
            verbose=self.verbose,
        )

        best_params, best_score = optimizer.optimize()

        return best_params, best_score


# Test
if __name__ == "__main__":
    print("🧪 SSA Optimizer Test\n")

    # Basit test fonksiyonu (sphere function)
    def sphere_function(params: Dict) -> float:
        x = params.get("lstm_units", 100) - 100
        y = params.get("conv_filters", 30) - 30
        return -(x**2 + y**2)  # Negative because we maximize

    # Test search space
    test_space = {
        "lstm_units": (50, 150, "int"),
        "conv_filters": (20, 50, "int"),
    }

    optimizer = SSAOptimizer(
        objective_function=sphere_function,
        search_space=test_space,
        population_size=10,
        max_iterations=10,
        minimize=False,
        verbose=True,
    )

    best_params, best_score = optimizer.optimize()

    print(f"\n🏆 Test Sonucu:")
    print(f"   Best params: {best_params}")
    print(f"   Best score: {best_score}")
    print(f"   (Beklenen: lstm_units≈100, conv_filters≈30)")
