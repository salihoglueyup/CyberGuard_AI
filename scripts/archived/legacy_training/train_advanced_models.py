"""
Train Advanced Models - CyberGuard AI
======================================

LSTM, BiLSTM+Attention, GRU modellerini CICIDS2017 ile eğit.
"""

import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime

# Proje root
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler


def load_cicids2017(max_samples: int = 50000):
    """CICIDS2017 veri setini yükle"""
    data_dir = os.path.join(project_root, "data", "raw", "cicids2017")

    if not os.path.exists(data_dir):
        print(f"❌ Veri seti bulunamadı: {data_dir}")
        return None, None, None

    csv_files = [f for f in os.listdir(data_dir) if f.endswith(".csv")]

    if not csv_files:
        print("❌ CSV dosyası bulunamadı")
        return None, None, None

    print(f"📂 {len(csv_files)} dosya bulundu")

    dfs = []
    for csv_file in csv_files[:3]:  # İlk 3 dosya
        filepath = os.path.join(data_dir, csv_file)
        print(f"   📄 Yükleniyor: {csv_file}")

        try:
            df = pd.read_csv(filepath, low_memory=False)
            df.columns = df.columns.str.strip()

            # Sample
            if len(df) > max_samples // 3:
                df = df.sample(n=max_samples // 3, random_state=42)

            dfs.append(df)
        except Exception as e:
            print(f"   ⚠️ Hata: {e}")

    if not dfs:
        return None, None, None

    data = pd.concat(dfs, ignore_index=True)
    print(f"✅ Toplam: {len(data)} satır")

    # Label column
    label_col = None
    for col in ["Label", "label", " Label"]:
        if col in data.columns:
            label_col = col
            break

    if not label_col:
        print("❌ Label kolonu bulunamadı")
        return None, None, None

    # Preprocess
    y = data[label_col].values
    X = data.drop(columns=[label_col])

    # Sadece numeric
    X = X.select_dtypes(include=[np.number])

    # NaN ve Inf temizle
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(0)

    # Label encode
    le = LabelEncoder()
    y = le.fit_transform(y)

    print(f"   Features: {X.shape[1]}")
    print(f"   Classes: {len(np.unique(y))}")

    return X.values, y, le.classes_


def prepare_data(X, y, seq_len: int = 10):
    """Veriyi model için hazırla"""
    # Scale
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Eğer yeterli veri yoksa padding
    n_samples = len(X_scaled)
    n_features = X_scaled.shape[1]

    # Reshape to (samples, seq_len, features)
    n_complete = (n_samples // seq_len) * seq_len
    X_scaled = X_scaled[:n_complete]
    y = y[:n_complete]

    X_seq = X_scaled.reshape(-1, seq_len, n_features)
    y_seq = y[::seq_len]  # Her sequence için 1 label

    print(f"   Sequence shape: {X_seq.shape}")

    return X_seq, y_seq, n_features


def train_lstm_model(
    X_train, y_train, X_val, y_val, num_classes, n_features, epochs=30
):
    """LSTM modeli eğit"""
    from src.network_detection.models.advanced_model import build_lstm_model

    print("\n🔄 LSTM Modeli Eğitiliyor...")

    model = build_lstm_model(
        input_shape=(X_train.shape[1], n_features),
        num_classes=num_classes,
        lstm_units=128,
    )

    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    callbacks = [
        EarlyStopping(patience=5, restore_best_weights=True),
        ModelCheckpoint(
            os.path.join(project_root, "models", "lstm_cicids2017.h5"),
            save_best_only=True,
        ),
    ]

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=64,
        callbacks=callbacks,
        verbose=1,
    )

    # Evaluate
    loss, acc = model.evaluate(X_val, y_val, verbose=0)
    print(f"✅ LSTM Accuracy: {acc*100:.2f}%")

    return model, acc


def train_bilstm_attention_model(
    X_train, y_train, X_val, y_val, num_classes, n_features, epochs=30
):
    """BiLSTM + Attention modeli eğit"""
    from src.network_detection.models.attention import build_bilstm_attention

    print("\n🔄 BiLSTM + Attention Modeli Eğitiliyor...")

    model = build_bilstm_attention(
        input_shape=(X_train.shape[1], n_features),
        num_classes=num_classes,
        lstm_units=128,
    )

    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    callbacks = [
        EarlyStopping(patience=5, restore_best_weights=True),
        ModelCheckpoint(
            os.path.join(project_root, "models", "bilstm_attention_cicids2017.h5"),
            save_best_only=True,
        ),
    ]

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=64,
        callbacks=callbacks,
        verbose=1,
    )

    loss, acc = model.evaluate(X_val, y_val, verbose=0)
    print(f"✅ BiLSTM+Attention Accuracy: {acc*100:.2f}%")

    return model, acc


def train_gru_model(X_train, y_train, X_val, y_val, num_classes, n_features, epochs=30):
    """GRU modeli eğit"""
    from src.network_detection.models.gru_model import GRUIDSModel

    print("\n🔄 GRU Modeli Eğitiliyor...")

    gru = GRUIDSModel(
        input_shape=(X_train.shape[1], n_features),
        num_classes=num_classes,
        gru_units=128,
    )

    model = gru.build()

    callbacks = [
        EarlyStopping(patience=5, restore_best_weights=True),
        ModelCheckpoint(
            os.path.join(project_root, "models", "gru_cicids2017.h5"),
            save_best_only=True,
        ),
    ]

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=64,
        callbacks=callbacks,
        verbose=1,
    )

    loss, acc = model.evaluate(X_val, y_val, verbose=0)
    print(f"✅ GRU Accuracy: {acc*100:.2f}%")

    return model, acc


def main():
    print("=" * 60)
    print("🧠 CyberGuard AI - Advanced Model Training")
    print("=" * 60)
    print(f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # Veri yükle
    print("📂 CICIDS2017 Veri Seti Yükleniyor...")
    X, y, classes = load_cicids2017(max_samples=30000)

    if X is None:
        print("❌ Veri yüklenemedi!")
        return

    # Hazırla
    X_seq, y_seq, n_features = prepare_data(X, y)
    num_classes = len(np.unique(y_seq))

    # Split
    X_train, X_val, y_train, y_val = train_test_split(
        X_seq, y_seq, test_size=0.2, random_state=42, stratify=y_seq
    )

    print(f"\n📊 Train: {len(X_train)}, Val: {len(X_val)}")
    print(f"   Classes: {num_classes}")

    results = {}

    # 1. LSTM
    try:
        _, acc = train_lstm_model(
            X_train, y_train, X_val, y_val, num_classes, n_features
        )
        results["LSTM"] = acc
    except Exception as e:
        print(f"❌ LSTM hatası: {e}")
        results["LSTM"] = 0

    # 2. BiLSTM + Attention
    try:
        _, acc = train_bilstm_attention_model(
            X_train, y_train, X_val, y_val, num_classes, n_features
        )
        results["BiLSTM+Attention"] = acc
    except Exception as e:
        print(f"❌ BiLSTM+Attention hatası: {e}")
        results["BiLSTM+Attention"] = 0

    # 3. GRU
    try:
        _, acc = train_gru_model(
            X_train, y_train, X_val, y_val, num_classes, n_features
        )
        results["GRU"] = acc
    except Exception as e:
        print(f"❌ GRU hatası: {e}")
        results["GRU"] = 0

    # Özet
    print("\n" + "=" * 60)
    print("📊 EĞİTİM SONUÇLARI")
    print("=" * 60)
    for model_name, acc in results.items():
        status = "✅" if acc > 0.9 else "⚠️" if acc > 0 else "❌"
        print(f"   {status} {model_name}: {acc*100:.2f}%")
    print("=" * 60)


if __name__ == "__main__":
    main()
