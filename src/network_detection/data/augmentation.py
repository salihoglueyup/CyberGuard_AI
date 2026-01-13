"""
Data Augmentation - CyberGuard AI
=================================

Dengesiz veri setleri için augmentation teknikleri.

Özellikler:
    - SMOTE (Synthetic Minority Over-sampling)
    - ADASYN (Adaptive Synthetic Sampling)
    - Random Oversampling
    - Class balancing
"""

import numpy as np
from typing import Tuple, Optional, Dict, List
import logging

logger = logging.getLogger("DataAugmentation")


class SMOTEAugmenter:
    """
    SMOTE (Synthetic Minority Over-sampling Technique)

    Az sayıda olan sınıflar için sentetik örnekler oluşturur.
    """

    def __init__(
        self,
        k_neighbors: int = 5,
        sampling_strategy: str = "auto",  # "auto", "minority", "not majority"
        random_state: int = 42,
    ):
        self.k_neighbors = k_neighbors
        self.sampling_strategy = sampling_strategy
        self.random_state = random_state
        np.random.seed(random_state)

    def fit_resample(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        SMOTE uygula

        Args:
            X: Features (n_samples, n_features) veya (n_samples, seq_len, n_features)
            y: Labels

        Returns:
            X_resampled, y_resampled
        """
        try:
            from imblearn.over_sampling import SMOTE

            # Flatten if 3D
            original_shape = X.shape
            if len(original_shape) == 3:
                X_flat = X.reshape(X.shape[0], -1)
            else:
                X_flat = X

            smote = SMOTE(
                k_neighbors=min(self.k_neighbors, min(np.bincount(y)) - 1),
                sampling_strategy=self.sampling_strategy,
                random_state=self.random_state,
            )

            X_resampled, y_resampled = smote.fit_resample(X_flat, y)

            # Reshape back if needed
            if len(original_shape) == 3:
                X_resampled = X_resampled.reshape(
                    -1, original_shape[1], original_shape[2]
                )

            logger.info(f"✅ SMOTE: {len(y)} → {len(y_resampled)} samples")
            return X_resampled, y_resampled

        except ImportError:
            logger.warning("imbalanced-learn not installed. Using fallback.")
            return self._smote_fallback(X, y)

    def _smote_fallback(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Manual SMOTE implementation"""
        classes, counts = np.unique(y, return_counts=True)
        max_count = counts.max()

        X_new = [X]
        y_new = [y]

        for cls, count in zip(classes, counts):
            if count < max_count:
                # Get samples of this class
                cls_mask = y == cls
                X_cls = X[cls_mask]

                # Number of synthetic samples needed
                n_synthetic = max_count - count

                # Generate synthetic samples
                for _ in range(n_synthetic):
                    # Random sample from class
                    idx = np.random.randint(0, len(X_cls))
                    sample = X_cls[idx]

                    # Add small noise
                    noise = np.random.randn(*sample.shape) * 0.01
                    synthetic = sample + noise

                    X_new.append(synthetic[np.newaxis])
                    y_new.append(np.array([cls]))

        X_resampled = np.concatenate(X_new, axis=0)
        y_resampled = np.concatenate(y_new, axis=0)

        return X_resampled, y_resampled


class ADASYNAugmenter:
    """
    ADASYN (Adaptive Synthetic Sampling)

    Öğrenmesi zor bölgelere daha fazla örnek üretir.
    """

    def __init__(
        self,
        n_neighbors: int = 5,
        sampling_strategy: str = "auto",
        random_state: int = 42,
    ):
        self.n_neighbors = n_neighbors
        self.sampling_strategy = sampling_strategy
        self.random_state = random_state

    def fit_resample(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """ADASYN uygula"""
        try:
            from imblearn.over_sampling import ADASYN

            # Flatten if 3D
            original_shape = X.shape
            if len(original_shape) == 3:
                X_flat = X.reshape(X.shape[0], -1)
            else:
                X_flat = X

            adasyn = ADASYN(
                n_neighbors=min(self.n_neighbors, min(np.bincount(y)) - 1),
                sampling_strategy=self.sampling_strategy,
                random_state=self.random_state,
            )

            X_resampled, y_resampled = adasyn.fit_resample(X_flat, y)

            if len(original_shape) == 3:
                X_resampled = X_resampled.reshape(
                    -1, original_shape[1], original_shape[2]
                )

            logger.info(f"✅ ADASYN: {len(y)} → {len(y_resampled)} samples")
            return X_resampled, y_resampled

        except ImportError:
            logger.warning("imbalanced-learn not installed. Using SMOTE fallback.")
            return SMOTEAugmenter().fit_resample(X, y)


class RandomOversamplerWrapper:
    """
    Random Oversampling

    Minority class'tan random tekrar.
    """

    def __init__(
        self,
        sampling_strategy: str = "auto",
        random_state: int = 42,
    ):
        self.sampling_strategy = sampling_strategy
        self.random_state = random_state

    def fit_resample(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Random oversampling"""
        try:
            from imblearn.over_sampling import RandomOverSampler

            original_shape = X.shape
            if len(original_shape) == 3:
                X_flat = X.reshape(X.shape[0], -1)
            else:
                X_flat = X

            ros = RandomOverSampler(
                sampling_strategy=self.sampling_strategy,
                random_state=self.random_state,
            )

            X_resampled, y_resampled = ros.fit_resample(X_flat, y)

            if len(original_shape) == 3:
                X_resampled = X_resampled.reshape(
                    -1, original_shape[1], original_shape[2]
                )

            logger.info(f"✅ RandomOverSampler: {len(y)} → {len(y_resampled)} samples")
            return X_resampled, y_resampled

        except ImportError:
            # Manual implementation
            classes, counts = np.unique(y, return_counts=True)
            max_count = counts.max()

            X_new = [X]
            y_new = [y]

            for cls, count in zip(classes, counts):
                if count < max_count:
                    cls_mask = y == cls
                    X_cls = X[cls_mask]

                    n_oversample = max_count - count
                    indices = np.random.choice(len(X_cls), n_oversample, replace=True)

                    X_new.append(X_cls[indices])
                    y_new.append(np.full(n_oversample, cls))

            return np.concatenate(X_new), np.concatenate(y_new)


# ============= Utility Functions =============


def balance_dataset(
    X: np.ndarray,
    y: np.ndarray,
    method: str = "smote",  # "smote", "adasyn", "random", "undersample"
    target_ratio: float = 1.0,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Dataset dengeleme

    Args:
        X: Features
        y: Labels
        method: Dengeleme metodu
        target_ratio: Hedef oran (1.0 = eşit)
        random_state: Random seed

    Returns:
        X_balanced, y_balanced
    """
    if method == "smote":
        augmenter = SMOTEAugmenter(random_state=random_state)
    elif method == "adasyn":
        augmenter = ADASYNAugmenter(random_state=random_state)
    elif method == "random":
        augmenter = RandomOversamplerWrapper(random_state=random_state)
    elif method == "undersample":
        return undersample_majority(X, y, random_state=random_state)
    else:
        raise ValueError(f"Unknown method: {method}")

    return augmenter.fit_resample(X, y)


def undersample_majority(
    X: np.ndarray,
    y: np.ndarray,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Majority class'ı undersample et
    """
    np.random.seed(random_state)

    classes, counts = np.unique(y, return_counts=True)
    min_count = counts.min()

    X_new = []
    y_new = []

    for cls in classes:
        cls_mask = y == cls
        X_cls = X[cls_mask]

        if len(X_cls) > min_count:
            indices = np.random.choice(len(X_cls), min_count, replace=False)
            X_new.append(X_cls[indices])
        else:
            X_new.append(X_cls)

        y_new.append(np.full(min(len(X_cls), min_count), cls))

    return np.concatenate(X_new), np.concatenate(y_new)


def get_class_distribution(y: np.ndarray) -> Dict[int, int]:
    """Sınıf dağılımını döndür"""
    classes, counts = np.unique(y, return_counts=True)
    return dict(zip(classes.tolist(), counts.tolist()))


def calculate_class_weights(y: np.ndarray) -> Dict[int, float]:
    """
    Dengesiz sınıflar için ağırlık hesapla

    Keras model.fit() için class_weight parametresi
    """
    from sklearn.utils import class_weight

    classes = np.unique(y)
    weights = class_weight.compute_class_weight(
        class_weight="balanced", classes=classes, y=y
    )

    return dict(zip(classes.tolist(), weights.tolist()))


def create_balanced_batches(
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int = 32,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Her batch'te dengeli sınıf dağılımı
    """
    classes = np.unique(y)
    n_classes = len(classes)
    samples_per_class = batch_size // n_classes

    X_batch = []
    y_batch = []

    for cls in classes:
        cls_mask = y == cls
        X_cls = X[cls_mask]

        if len(X_cls) >= samples_per_class:
            indices = np.random.choice(len(X_cls), samples_per_class, replace=False)
        else:
            indices = np.random.choice(len(X_cls), samples_per_class, replace=True)

        X_batch.append(X_cls[indices])
        y_batch.append(np.full(samples_per_class, cls))

    X_batch = np.concatenate(X_batch)
    y_batch = np.concatenate(y_batch)

    # Shuffle
    perm = np.random.permutation(len(X_batch))
    return X_batch[perm], y_batch[perm]


# Test
if __name__ == "__main__":
    print("🧪 Data Augmentation Test\n")

    # Create imbalanced dataset
    np.random.seed(42)

    # Class 0: 1000 samples, Class 1: 50 samples
    X_0 = np.random.randn(1000, 10, 41)
    y_0 = np.zeros(1000)

    X_1 = np.random.randn(50, 10, 41)
    y_1 = np.ones(50)

    X = np.concatenate([X_0, X_1]).astype(np.float32)
    y = np.concatenate([y_0, y_1]).astype(np.int32)

    print(f"📊 Original distribution: {get_class_distribution(y)}")

    # Test SMOTE
    print("\n🔄 SMOTE:")
    X_smote, y_smote = balance_dataset(X, y, method="smote")
    print(f"   Result: {get_class_distribution(y_smote)}")

    # Test Random Oversampling
    print("\n🔄 Random Oversampling:")
    X_ros, y_ros = balance_dataset(X, y, method="random")
    print(f"   Result: {get_class_distribution(y_ros)}")

    # Test Undersampling
    print("\n🔄 Undersampling:")
    X_under, y_under = balance_dataset(X, y, method="undersample")
    print(f"   Result: {get_class_distribution(y_under)}")

    # Class weights
    print("\n⚖️ Class Weights:")
    weights = calculate_class_weights(y)
    print(f"   {weights}")
