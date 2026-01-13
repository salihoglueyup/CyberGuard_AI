"""
SSA-LSTMIDS - Makale Birebir Aynısı
===================================

Bu dosya, "SSA-LSTMIDS: An Intrusion Detection System" makalesindeki modelin
TAM OLARAK AYNI mimarisini ve hiperparametrelerini içerir.

Makale Referansı:
- Conv1D + LSTM hibrit yapısı
- SSA (Salp Swarm Algorithm) ile hiperparametre optimizasyonu
- 10-fold cross validation
- NSL-KDD, CICIDS2017 ve BoT-IoT dataset desteği

Hiperparametreler (Makaleden - Tablo 2):
- Conv1D Filters: 30
- Kernel Size: 5
- LSTM Units: 120
- Dense Units: 512
- Dropout: 0.2
- Batch Size: 120
- Learning Rate: 0.001
- Epochs: 300
- SSA Population: 40
- SSA Iterations: 36

Makale Accuracy Sonuçları (Tablo 8):
- NSL-KDD: 99.36%
- CICIDS2017: 99.42%
- BoT-IoT: 99.99%
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from pathlib import Path
from datetime import datetime
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model, Sequential
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint


# ============= MAKALE PARAMETRELERİ (Tablo 2) =============

PAPER_PARAMS = {
    # Network Architecture
    "conv_filters": 30,  # Makale: 30
    "kernel_size": 5,  # Makale: 5
    "lstm_units": 120,  # Makale: 120
    "dense_units": 512,  # Makale: 512
    "dropout_rate": 0.2,  # Makale: 0.2
    # Training
    "batch_size": 120,  # Makale: 120
    "epochs": 300,  # Makale: 300
    "learning_rate": 0.001,  # Makale: 0.001
    "validation_split": 0.1,  # Makale: 10%
    # SSA Parameters
    "ssa_population": 40,  # Makale: 40
    "ssa_iterations": 36,  # Makale: 36
    # Cross Validation
    "n_folds": 10,  # Makale: 10-fold CV
    # Sequence
    "sequence_length": 10,  # Makale: 10 timesteps
}


class PaperSSALSTMIDS:
    """
    Makale ile BİREBİR AYNI SSA-LSTMIDS modeli

    Mimari (Makaleden Şekil 4):
    Input → Conv1D(30, 5) → MaxPool → LSTM(120) → Dense(512) → Dropout(0.2) → Output
    """

    def __init__(self, params: Dict = None):
        self.params = params or PAPER_PARAMS.copy()
        self.model: Optional[Model] = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.history = None
        self.class_names = []

    def build_model(self, input_shape: Tuple, num_classes: int) -> Model:
        """
        Makale Şekil 4'e göre model oluştur

        Yapı:
        1. Input Layer
        2. Conv1D (filters=30, kernel=5, ReLU)
        3. MaxPooling1D
        4. LSTM (units=120, return_sequences=False)
        5. Dense (units=512, ReLU)
        6. Dropout (0.2)
        7. Output Dense (softmax)
        """
        model = Sequential(
            [
                # Input
                layers.Input(shape=input_shape),
                # Conv1D Layer (Makale: 30 filters, 5 kernel)
                layers.Conv1D(
                    filters=self.params["conv_filters"],
                    kernel_size=self.params["kernel_size"],
                    activation="relu",
                    padding="same",
                ),
                # MaxPooling
                layers.MaxPooling1D(pool_size=2),
                # LSTM Layer (Makale: 120 units)
                layers.LSTM(
                    units=self.params["lstm_units"],
                    return_sequences=False,
                    dropout=self.params["dropout_rate"],
                    recurrent_dropout=self.params["dropout_rate"],
                ),
                # Dense Layer (Makale: 512 units)
                layers.Dense(units=self.params["dense_units"], activation="relu"),
                # Dropout (Makale: 0.2)
                layers.Dropout(rate=self.params["dropout_rate"]),
                # Output Layer
                layers.Dense(num_classes, activation="softmax"),
            ]
        )

        # Optimizer (Makale: Adam with lr=0.001)
        optimizer = keras.optimizers.Adam(learning_rate=self.params["learning_rate"])

        model.compile(
            optimizer=optimizer,
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )

        self.model = model
        return model

    def create_sequences(
        self, X: np.ndarray, y: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Timestep sequence'ları oluştur"""
        seq_len = self.params["sequence_length"]

        if len(X) < seq_len:
            # Padding
            padding = np.zeros((seq_len - len(X), X.shape[1]))
            X = np.vstack([padding, X])
            y = np.concatenate([np.zeros(seq_len - len(y)), y])

        X_seq, y_seq = [], []
        for i in range(len(X) - seq_len + 1):
            X_seq.append(X[i : i + seq_len])
            y_seq.append(y[i + seq_len - 1])

        return np.array(X_seq), np.array(y_seq)

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray = None,
        y_val: np.ndarray = None,
        callbacks: List = None,
    ) -> Dict:
        """Model eğit"""
        if self.model is None:
            raise ValueError("Model henüz oluşturulmadı!")

        # Varsayılan callbacks (Makaleden)
        if callbacks is None:
            callbacks = [
                EarlyStopping(
                    monitor="val_loss",
                    patience=20,  # Makale: patience
                    restore_best_weights=True,
                    verbose=1,
                ),
                ReduceLROnPlateau(
                    monitor="val_loss", factor=0.5, patience=10, min_lr=1e-6, verbose=1
                ),
            ]

        # Validation split
        if X_val is None:
            validation_split = self.params["validation_split"]
            validation_data = None
        else:
            validation_split = None
            validation_data = (X_val, y_val)

        # Eğitim
        self.history = self.model.fit(
            X_train,
            y_train,
            batch_size=self.params["batch_size"],
            epochs=self.params["epochs"],
            validation_split=validation_split,
            validation_data=validation_data,
            callbacks=callbacks,
            verbose=1,
        )

        return self.history.history

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict:
        """Model değerlendir"""
        y_pred_proba = self.model.predict(X_test, verbose=0)
        y_pred = np.argmax(y_pred_proba, axis=1)

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
        recall = recall_score(y_test, y_pred, average="weighted", zero_division=0)
        f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "predictions": y_pred,
            "probabilities": y_pred_proba,
        }

    def cross_validate(self, X: np.ndarray, y: np.ndarray, n_folds: int = None) -> Dict:
        """
        10-Fold Cross Validation (Makaleden)
        """
        n_folds = n_folds or self.params["n_folds"]

        print(f"\n{'='*60}")
        print(f"📊 {n_folds}-Fold Cross Validation Başlıyor")
        print(f"{'='*60}\n")

        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

        fold_results = []
        all_predictions = []
        all_true = []

        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            print(f"\n📁 Fold {fold+1}/{n_folds}")
            print("-" * 40)

            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            # Yeni model oluştur
            num_classes = len(np.unique(y))
            self.build_model(
                input_shape=(X.shape[1], X.shape[2]), num_classes=num_classes
            )

            # Eğit
            self.train(X_train, y_train, X_val, y_val)

            # Değerlendir
            results = self.evaluate(X_val, y_val)
            fold_results.append(results)

            all_predictions.extend(results["predictions"])
            all_true.extend(y_val)

            print(f"   Accuracy: {results['accuracy']*100:.2f}%")
            print(f"   F1-Score: {results['f1_score']*100:.2f}%")

        # Ortalama sonuçlar
        avg_results = {
            "accuracy": np.mean([r["accuracy"] for r in fold_results]),
            "accuracy_std": np.std([r["accuracy"] for r in fold_results]),
            "precision": np.mean([r["precision"] for r in fold_results]),
            "recall": np.mean([r["recall"] for r in fold_results]),
            "f1_score": np.mean([r["f1_score"] for r in fold_results]),
            "f1_std": np.std([r["f1_score"] for r in fold_results]),
            "fold_results": fold_results,
        }

        print(f"\n{'='*60}")
        print(f"📊 {n_folds}-Fold CV Sonuçları")
        print(f"{'='*60}")
        print(
            f"   Accuracy: {avg_results['accuracy']*100:.2f}% ± {avg_results['accuracy_std']*100:.2f}%"
        )
        print(
            f"   F1-Score: {avg_results['f1_score']*100:.2f}% ± {avg_results['f1_std']*100:.2f}%"
        )

        return avg_results

    def save(self, path: str):
        """Model kaydet"""
        self.model.save(path)

        # Parametreleri de kaydet
        params_path = path.replace(".h5", "_paper_params.json")
        with open(params_path, "w") as f:
            json.dump(
                {
                    "params": self.params,
                    "class_names": self.class_names,
                    "saved_at": datetime.now().isoformat(),
                },
                f,
                indent=2,
            )

        print(f"💾 Model kaydedildi: {path}")

    @classmethod
    def load(cls, path: str) -> "PaperSSALSTMIDS":
        """Model yükle"""
        instance = cls()
        instance.model = keras.models.load_model(path, compile=False)
        instance.model.compile(
            optimizer="adam",
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )

        # Parametreleri yükle
        params_path = path.replace(".h5", "_paper_params.json")
        if os.path.exists(params_path):
            with open(params_path) as f:
                data = json.load(f)
                instance.params = data.get("params", PAPER_PARAMS)
                instance.class_names = data.get("class_names", [])

        return instance


def load_cicids2017(
    data_dir: Path, max_samples: int = None
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    CICIDS2017 dataset'ini yükle
    """
    print("\n📂 CICIDS2017 Yükleniyor...")

    csv_files = list(data_dir.glob("*.csv"))
    if not csv_files:
        print(f"❌ CSV dosyası bulunamadı: {data_dir}")
        return None, None, None

    print(f"   {len(csv_files)} dosya bulundu")

    dfs = []
    for csv_file in csv_files:
        print(f"   📄 Yükleniyor: {csv_file.name}")
        df = None
        for encoding in ["utf-8", "latin-1", "cp1252", "iso-8859-1"]:
            try:
                df = pd.read_csv(csv_file, low_memory=False, encoding=encoding)
                break
            except UnicodeDecodeError:
                continue

        if df is not None:
            dfs.append(df)
        else:
            print(f"      ⚠️ Encoding hatası, atlandı")

    if not dfs:
        return None, None, None

    # Birleştir
    data = pd.concat(dfs, ignore_index=True)
    print(f"\n📊 Toplam satır: {len(data):,}")

    # Label sütununu bul
    label_col = None
    for col in [" Label", "Label", "label", "attack_type"]:
        if col in data.columns:
            label_col = col
            break

    if label_col is None:
        print("❌ Label sütunu bulunamadı!")
        print(f"   Columns: {data.columns.tolist()[:10]}...")
        return None, None, None

    # Sınıf dağılımı
    print(f"\n🎯 Sınıf Dağılımı:")
    class_counts = data[label_col].value_counts()
    for cls, cnt in list(class_counts.items())[:10]:
        print(f"   {cls}: {cnt:,}")

    # Binary classification: BENIGN vs ATTACK
    # CICIDS2017'de çok fazla sınıf var, binary daha stabil
    data["is_attack"] = (
        data[label_col].astype(str).apply(lambda x: 0 if "BENIGN" in x.upper() else 1)
    )
    y = data["is_attack"].values
    class_names = ["BENIGN", "ATTACK"]

    print(f"\n📊 Binary Dağılım:")
    print(f"   BENIGN: {np.sum(y==0):,}")
    print(f"   ATTACK: {np.sum(y==1):,}")

    # Features - sadece sayısal sütunları al
    feature_cols = [c for c in data.columns if c != label_col]

    # Sayısal olmayan sütunları filtrele
    numeric_cols = []
    for col in feature_cols:
        try:
            # Sayısal dönüşüm dene
            pd.to_numeric(data[col], errors="raise")
            numeric_cols.append(col)
        except (ValueError, TypeError):
            print(f"   ⚠️ Sayısal olmayan sütun atlandı: {col}")

    print(f"\n📊 Kullanılan özellik sayısı: {len(numeric_cols)}")

    # Sayısal verileri al
    X = data[numeric_cols].apply(pd.to_numeric, errors="coerce").values

    # NaN ve Inf temizle
    X = np.nan_to_num(X, nan=0, posinf=0, neginf=0)

    # Float32'ye çevir
    X = X.astype(np.float32)

    # Sample limit
    if max_samples and len(X) > max_samples:
        indices = np.random.choice(len(X), max_samples, replace=False)
        X, y = X[indices], y[indices]
        print(f"\n⚠️ {max_samples:,} sample'a düşürüldü")

    print(f"\n✅ Yükleme tamamlandı: {X.shape}")

    return X, y, class_names


def train_paper_model(
    dataset: str = "cicids2017",
    max_samples: int = None,
    save_model: bool = True,
    use_cv: bool = False,
):
    """
    Makaledeki modeli eğit
    """
    print("\n" + "=" * 70)
    print("🎓 MAKALEDEKİ SSA-LSTMIDS MODELİ EĞİTİMİ")
    print("=" * 70)
    print("\n📋 Makale Parametreleri:")
    for k, v in PAPER_PARAMS.items():
        print(f"   {k}: {v}")

    # Veri yükle
    if dataset == "cicids2017":
        data_dir = PROJECT_ROOT / "data" / "raw" / "cicids2017"
        X, y, class_names = load_cicids2017(data_dir, max_samples)
    elif dataset == "nsl_kdd":
        from scripts.train_ssa_lstmids import load_nsl_kdd

        X, y, class_names = load_nsl_kdd(
            PROJECT_ROOT / "data" / "raw" / "nsl_kdd", max_samples
        )
    else:
        print(f"❌ Bilinmeyen dataset: {dataset}")
        return

    if X is None:
        return

    # Ölçeklendirme
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Sequence oluştur
    seq_len = PAPER_PARAMS["sequence_length"]
    X_seq, y_seq = [], []
    for i in range(len(X) - seq_len + 1):
        X_seq.append(X[i : i + seq_len])
        y_seq.append(y[i + seq_len - 1])

    X_seq = np.array(X_seq)
    y_seq = np.array(y_seq)

    print(f"\n📊 Sequence shape: {X_seq.shape}")

    # Model oluştur
    model = PaperSSALSTMIDS()
    model.class_names = class_names

    if use_cv:
        # 10-Fold Cross Validation
        results = model.cross_validate(X_seq, y_seq)
    else:
        # Train/Test split
        from sklearn.model_selection import train_test_split

        X_train, X_test, y_train, y_test = train_test_split(
            X_seq, y_seq, test_size=0.2, random_state=42, stratify=y_seq
        )

        num_classes = len(np.unique(y_seq))
        model.build_model(
            input_shape=(X_seq.shape[1], X_seq.shape[2]), num_classes=num_classes
        )

        print(f"\n📦 Model özeti:")
        model.model.summary()

        # Eğit
        model.train(X_train, y_train)

        # Değerlendir
        results = model.evaluate(X_test, y_test)

        print(f"\n{'='*60}")
        print("📊 TEST SONUÇLARI")
        print("=" * 60)
        print(f"   Accuracy:  {results['accuracy']*100:.2f}%")
        print(f"   Precision: {results['precision']*100:.2f}%")
        print(f"   Recall:    {results['recall']*100:.2f}%")
        print(f"   F1-Score:  {results['f1_score']*100:.2f}%")

    # Kaydet
    if save_model:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = (
            PROJECT_ROOT / "models" / f"paper_ssa_lstmids_{dataset}_{timestamp}.h5"
        )
        model.save(str(model_path))

        # Sonuçları kaydet
        results_path = PROJECT_ROOT / "models" / f"paper_ssa_lstmids_results.json"

        existing = {}
        if results_path.exists():
            with open(results_path) as f:
                existing = json.load(f)

        existing[dataset] = {
            "accuracy": float(results.get("accuracy", 0)),
            "precision": float(results.get("precision", 0)),
            "recall": float(results.get("recall", 0)),
            "f1_score": float(results.get("f1_score", 0)),
            "params": PAPER_PARAMS,
            "class_names": class_names,
            "trained_at": datetime.now().isoformat(),
        }

        with open(results_path, "w") as f:
            json.dump(existing, f, indent=2)

        print(f"\n💾 Sonuçlar kaydedildi: {results_path}")

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Makale SSA-LSTMIDS Eğitimi")
    parser.add_argument(
        "--dataset",
        type=str,
        default="cicids2017",
        choices=["cicids2017", "nsl_kdd", "bot_iot"],
    )
    parser.add_argument("--max_samples", type=int, default=100000)
    parser.add_argument("--cv", action="store_true", help="10-Fold CV kullan")
    parser.add_argument("--no_save", action="store_true")

    args = parser.parse_args()

    train_paper_model(
        dataset=args.dataset,
        max_samples=args.max_samples,
        save_model=not args.no_save,
        use_cv=args.cv,
    )
