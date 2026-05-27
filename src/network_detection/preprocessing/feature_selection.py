"""
Feature Selection Module
CyberGuard AI için optimum feature seçimi

Yöntemler:
    - Variance Threshold
    - SelectKBest (chi2, mutual_info)
    - Recursive Feature Elimination (RFE)
    - Feature Importance (tree-based)
    - Correlation filtering

Ref: Makaledeki "Feature Selection" önerisi
"""


import numpy as np

try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.feature_selection import (
        RFE,
        SelectKBest,
        VarianceThreshold,
        chi2,
        f_classif,
        mutual_info_classif,
    )

    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("⚠️ scikit-learn gerekli!")


class FeatureSelector:
    """
    Feature Selection Pipeline

    Farklı yöntemlerle en iyi feature'ları seçer.
    """

    METHODS = ["variance", "kbest", "rfe", "importance", "correlation"]

    def __init__(
        self,
        method: str = "kbest",
        n_features: int = 50,
        threshold: float = 0.01,
        verbose: bool = True,
    ):
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn gerekli!")

        self.method = method
        self.n_features = n_features
        self.threshold = threshold
        self.verbose = verbose

        self.selector = None
        self.selected_indices: np.ndarray | None = None
        self.feature_scores: np.ndarray | None = None
        self.feature_names: list[str] | None = None

        if verbose:
            print("🔍 Feature Selector başlatıldı")
            print(f"   Method: {method}, N features: {n_features}")

    def fit(
        self, X: np.ndarray, y: np.ndarray, feature_names: list[str] = None
    ) -> "FeatureSelector":
        """Feature selection uygula"""

        self.feature_names = feature_names or [f"f_{i}" for i in range(X.shape[1])]

        if self.verbose:
            print(f"\n📊 Feature selection ({self.method})...")
            print(f"   Input: {X.shape[1]} features")

        if self.method == "variance":
            self._fit_variance(X)
        elif self.method == "kbest":
            self._fit_kbest(X, y)
        elif self.method == "rfe":
            self._fit_rfe(X, y)
        elif self.method == "importance":
            self._fit_importance(X, y)
        elif self.method == "correlation":
            self._fit_correlation(X)
        else:
            raise ValueError(f"Bilinmeyen method: {self.method}")

        if self.verbose:
            print(f"   Selected: {len(self.selected_indices)} features")

        return self

    def _fit_variance(self, X: np.ndarray):
        """Variance threshold based selection"""
        self.selector = VarianceThreshold(threshold=self.threshold)
        self.selector.fit(X)

        self.selected_indices = np.where(self.selector.get_support())[0]
        self.feature_scores = self.selector.variances_

    def _fit_kbest(self, X: np.ndarray, y: np.ndarray):
        """SelectKBest with mutual information"""
        k = min(self.n_features, X.shape[1])
        self.selector = SelectKBest(score_func=mutual_info_classif, k=k)
        self.selector.fit(X, y)

        self.selected_indices = np.where(self.selector.get_support())[0]
        self.feature_scores = self.selector.scores_

    def _fit_rfe(self, X: np.ndarray, y: np.ndarray):
        """Recursive Feature Elimination"""
        estimator = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
        n_features = min(self.n_features, X.shape[1])

        self.selector = RFE(estimator, n_features_to_select=n_features, step=0.1)
        self.selector.fit(X, y)

        self.selected_indices = np.where(self.selector.get_support())[0]
        self.feature_scores = self.selector.ranking_

    def _fit_importance(self, X: np.ndarray, y: np.ndarray):
        """Tree-based feature importance"""
        rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        rf.fit(X, y)

        self.feature_scores = rf.feature_importances_

        # Top N features
        top_indices = np.argsort(self.feature_scores)[::-1][: self.n_features]
        self.selected_indices = np.sort(top_indices)

    def _fit_correlation(self, X: np.ndarray):
        """Correlation-based filtering (remove highly correlated)"""
        corr_matrix = np.corrcoef(X, rowvar=False)

        # Üst üçgen matris
        upper = np.triu(np.abs(corr_matrix), k=1)

        # Yüksek korelasyonlu feature'ları bul
        high_corr = np.where(upper > self.threshold)
        to_remove = set(high_corr[1])

        all_indices = set(range(X.shape[1]))
        self.selected_indices = np.array(sorted(all_indices - to_remove))
        self.feature_scores = np.std(X, axis=0)

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Seçili feature'ları uygula"""
        if self.selected_indices is None:
            raise ValueError("Önce fit() çağrılmalı!")

        return X[:, self.selected_indices]

    def fit_transform(
        self, X: np.ndarray, y: np.ndarray, feature_names: list[str] = None
    ) -> np.ndarray:
        """Fit ve transform birlikte"""
        self.fit(X, y, feature_names)
        return self.transform(X)

    def get_selected_features(self) -> list[str]:
        """Seçili feature isimlerini döndür"""
        if self.selected_indices is None or self.feature_names is None:
            return []
        return [self.feature_names[i] for i in self.selected_indices]

    def get_feature_ranking(self) -> list[tuple[str, float]]:
        """Feature ranking döndür"""
        if self.feature_scores is None or self.feature_names is None:
            return []

        ranking = list(zip(self.feature_names, self.feature_scores))
        return sorted(ranking, key=lambda x: x[1], reverse=True)

    def plot_importance(self, top_n: int = 20):
        """Feature importance plot"""
        try:
            import matplotlib.pyplot as plt

            ranking = self.get_feature_ranking()[:top_n]
            names, scores = zip(*ranking)

            plt.figure(figsize=(10, 6))
            plt.barh(range(len(names)), scores)
            plt.yticks(range(len(names)), names)
            plt.xlabel("Importance Score")
            plt.title(f"Top {top_n} Features ({self.method})")
            plt.tight_layout()
            plt.savefig("feature_importance.png")
            plt.close()

            print("📊 Feature importance plot kaydedildi: feature_importance.png")
        except ImportError:
            print("⚠️ matplotlib gerekli!")


# Wrapper for pipeline
def select_features(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray = None,
    method: str = "kbest",
    n_features: int = 50,
) -> tuple[np.ndarray, np.ndarray, FeatureSelector]:
    """
    Feature selection wrapper

    Returns:
        (X_train_selected, X_test_selected, selector)
    """
    selector = FeatureSelector(method=method, n_features=n_features)
    X_train_selected = selector.fit_transform(X_train, y_train)

    X_test_selected = None
    if X_test is not None:
        X_test_selected = selector.transform(X_test)

    return X_train_selected, X_test_selected, selector


# Test
if __name__ == "__main__":
    print("🧪 Feature Selection Test\n")

    # Test data
    X = np.random.rand(1000, 100)
    y = np.random.randint(0, 5, 1000)

    # Test each method
    for method in ["variance", "kbest", "importance"]:
        print(f"\n--- {method.upper()} ---")
        selector = FeatureSelector(method=method, n_features=20)
        X_selected = selector.fit_transform(X, y)
        print(f"Selected shape: {X_selected.shape}")

        top_features = selector.get_feature_ranking()[:5]
        print(f"Top 5: {top_features}")

    print("\n✅ Test tamamlandı!")
