"""
Data Balancing - SMOTE/ADASYN
CyberGuard AI için dengesiz veri çözümü

Neden Gerekli?
    - Gerçek ağ trafiğinde normal trafik >> saldırı trafiği
    - Nadir saldırılar (U2R, R2L) öğrenilemiyor
    - SMOTE ile azınlık sınıfları artırılır

Yöntemler:
    - SMOTE: Synthetic Minority Oversampling
    - ADASYN: Adaptive Synthetic Sampling
    - Class Weights: Ağırlıklı loss function
"""

from collections import Counter

import numpy as np

try:
    from imblearn.combine import SMOTEENN, SMOTETomek
    from imblearn.over_sampling import ADASYN, SMOTE, BorderlineSMOTE

    IMBLEARN_AVAILABLE = True
except ImportError:
    IMBLEARN_AVAILABLE = False
    print("⚠️ imbalanced-learn yüklü değil! pip install imbalanced-learn")


class DataBalancer:
    """
    Veri Dengeleme Sınıfı

    Dengesiz IDS veri setlerini dengeler.
    SMOTE, ADASYN ve hybrid yöntemler desteklenir.
    """

    SUPPORTED_METHODS = ["smote", "adasyn", "borderline", "smote_enn", "smote_tomek"]

    def __init__(
        self,
        method: str = "smote",
        sampling_strategy: str | dict | float = "auto",
        random_state: int = 42,
        k_neighbors: int = 5,
        verbose: bool = True,
    ):
        """
        DataBalancer başlat

        Args:
            method: Dengeleme yöntemi (smote, adasyn, borderline, smote_enn, smote_tomek)
            sampling_strategy: Örnekleme stratejisi
                - "auto": Tüm sınıfları çoğunluk sınıfına eşitle
                - "minority": Sadece azınlık sınıfını artır
                - Dict: {class: count} şeklinde hedef sayılar
                - float: Azınlık/çoğunluk oranı
            random_state: Random seed
            k_neighbors: SMOTE için komşu sayısı
            verbose: Detaylı çıktı
        """
        if not IMBLEARN_AVAILABLE:
            raise ImportError("imbalanced-learn gerekli! pip install imbalanced-learn")

        self.method = method.lower()
        self.sampling_strategy = sampling_strategy
        self.random_state = random_state
        self.k_neighbors = k_neighbors
        self.verbose = verbose

        self.sampler = None
        self._create_sampler()

        if verbose:
            print("⚖️ DataBalancer başlatıldı")
            print(f"   Yöntem: {self.method.upper()}")
            print(f"   Strateji: {sampling_strategy}")

    def _create_sampler(self):
        """Sampler objesi oluştur"""
        if self.method == "smote":
            self.sampler = SMOTE(
                sampling_strategy=self.sampling_strategy,
                random_state=self.random_state,
                k_neighbors=self.k_neighbors,
            )
        elif self.method == "adasyn":
            self.sampler = ADASYN(
                sampling_strategy=self.sampling_strategy,
                random_state=self.random_state,
                n_neighbors=self.k_neighbors,
            )
        elif self.method == "borderline":
            self.sampler = BorderlineSMOTE(
                sampling_strategy=self.sampling_strategy,
                random_state=self.random_state,
                k_neighbors=self.k_neighbors,
            )
        elif self.method == "smote_enn":
            self.sampler = SMOTEENN(
                sampling_strategy=self.sampling_strategy, random_state=self.random_state
            )
        elif self.method == "smote_tomek":
            self.sampler = SMOTETomek(
                sampling_strategy=self.sampling_strategy, random_state=self.random_state
            )
        else:
            raise ValueError(
                f"Bilinmeyen yöntem: {self.method}. "
                f"Desteklenen: {self.SUPPORTED_METHODS}"
            )

    def fit_resample(
        self, X: np.ndarray, y: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Veriyi dengele

        Args:
            X: Özellik matrisi (n_samples, n_features)
            y: Etiketler

        Returns:
            (X_resampled, y_resampled)
        """
        if self.verbose:
            print(f"\n📊 Veri dengeleniyor ({self.method.upper()})...")
            print(f"   Girdi: {X.shape[0]:,} örnek")
            self._print_distribution("Önce", y)

        # 3D veriyi 2D'ye dönüştür (SMOTE 2D bekler)
        original_shape = X.shape
        if len(X.shape) == 3:
            X_2d = X.reshape(X.shape[0], -1)
        else:
            X_2d = X

        # Resample
        try:
            X_resampled, y_resampled = self.sampler.fit_resample(X_2d, y)
        except Exception as e:
            print(f"⚠️ SMOTE hatası: {e}")
            print("   Orijinal veri döndürülüyor...")
            return X, y

        # 3D'ye geri dönüştür
        if len(original_shape) == 3:
            X_resampled = X_resampled.reshape(-1, original_shape[1], original_shape[2])

        if self.verbose:
            print(f"   Çıktı: {X_resampled.shape[0]:,} örnek")
            self._print_distribution("Sonra", y_resampled)

        return X_resampled, y_resampled

    def _print_distribution(self, label: str, y: np.ndarray):
        """Sınıf dağılımını yazdır"""
        counter = Counter(y)
        total = len(y)

        print(f"   {label}:")
        for cls, count in sorted(counter.items()):
            pct = count / total * 100
            bar = "█" * int(pct / 5) + "░" * (20 - int(pct / 5))
            print(f"      Sınıf {cls}: {count:,} ({pct:.1f}%) {bar}")


def compute_class_weights(y: np.ndarray) -> dict[int, float]:
    """
    Class weights hesapla

    Dengesiz veri için loss function ağırlıkları.
    SMOTE'a alternatif veya tamamlayıcı olarak kullanılabilir.

    Args:
        y: Etiketler

    Returns:
        {class_id: weight} dictionary
    """
    from sklearn.utils.class_weight import compute_class_weight

    classes = np.unique(y)
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=y)

    class_weights = {int(cls): float(weight) for cls, weight in zip(classes, weights)}

    print("⚖️ Class Weights:")
    for cls, weight in class_weights.items():
        print(f"   Sınıf {cls}: {weight:.3f}")

    return class_weights


def analyze_imbalance(y: np.ndarray) -> dict:
    """
    Veri dengesizliğini analiz et

    Args:
        y: Etiketler

    Returns:
        Analiz sonuçları
    """
    counter = Counter(y)
    total = len(y)

    majority_class = max(counter, key=counter.get)
    minority_class = min(counter, key=counter.get)

    imbalance_ratio = counter[majority_class] / counter[minority_class]

    analysis = {
        "total_samples": total,
        "n_classes": len(counter),
        "class_distribution": dict(counter),
        "majority_class": majority_class,
        "majority_count": counter[majority_class],
        "minority_class": minority_class,
        "minority_count": counter[minority_class],
        "imbalance_ratio": imbalance_ratio,
        "is_imbalanced": imbalance_ratio > 3,  # 3:1'den fazla dengesiz
    }

    print("\n📊 Dengesizlik Analizi:")
    print(f"   Toplam örnek: {total:,}")
    print(f"   Sınıf sayısı: {len(counter)}")
    print(f"   Çoğunluk sınıfı: {majority_class} ({counter[majority_class]:,})")
    print(f"   Azınlık sınıfı: {minority_class} ({counter[minority_class]:,})")
    print(f"   Dengesizlik oranı: {imbalance_ratio:.1f}:1")
    print(f"   Dengesiz mi?: {'⚠️ EVET' if analysis['is_imbalanced'] else '✅ Hayır'}")

    return analysis


# Test
if __name__ == "__main__":
    print("🧪 Data Balancer Test\n")

    if not IMBLEARN_AVAILABLE:
        print("❌ imbalanced-learn yüklü değil!")
        exit(1)

    # Dengesiz test verisi oluştur
    np.random.seed(42)
    X = np.random.rand(1000, 10, 78).astype(np.float32)
    y = np.array([0] * 800 + [1] * 150 + [2] * 50)  # Dengesiz

    # Analiz
    analyze_imbalance(y)

    # SMOTE
    balancer = DataBalancer(method="smote", sampling_strategy="auto")
    X_balanced, y_balanced = balancer.fit_resample(X, y)

    # Class weights
    weights = compute_class_weights(y)

    print("\n✅ Test tamamlandı!")
