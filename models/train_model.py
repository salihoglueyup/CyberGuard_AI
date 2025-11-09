"""
Model Trainer - CyberGuard AI
Database'den veri çekip modeli eğitir
"""

import sys
import os

# Proje kök dizinini Python path'e ekle
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir) if 'src' in current_dir else current_dir
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import sqlite3
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Import'ları dene
try:
    from src.utils.feature_extractor import FeatureExtractor
    from src.models.random_forest_model import CyberAttackModel
except ImportError:
    # Eğer src.utils çalışmazsa direkt import dene
    import importlib.util

    # feature_extractor
    spec = importlib.util.spec_from_file_location(
        "feature_extractor",
        os.path.join(project_root, "src", "utils", "feature_extractor.py")
    )
    feature_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(feature_module)
    FeatureExtractor = feature_module.FeatureExtractor

    # random_forest_model
    spec = importlib.util.spec_from_file_location(
        "random_forest_model",
        os.path.join(project_root, "src", "models", "random_forest_model.py")
    )
    model_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(model_module)
    CyberAttackModel = model_module.CyberAttackModel

class ModelTrainer:
    """Model eğitim yöneticisi"""

    def __init__(self, db_path: str = 'cyberguard.db'):
        """
        Args:
            db_path (str): Database yolu
        """
        self.db_path = db_path
        self.feature_extractor = FeatureExtractor()
        self.model = CyberAttackModel(n_estimators=100)

    def load_data_from_db(self, limit: int = None) -> pd.DataFrame:
        """
        Database'den saldırı verilerini yükle

        Args:
            limit (int): Maksimum kayıt sayısı

        Returns:
            DataFrame: Saldırı verileri
        """
        print(f"📂 Veriler yükleniyor: {self.db_path}")

        conn = sqlite3.connect(self.db_path)

        query = "SELECT * FROM attacks"
        if limit:
            query += f" LIMIT {limit}"

        df = pd.read_sql_query(query, conn)
        conn.close()

        print(f"✅ {len(df)} kayıt yüklendi")
        print(f"📊 Saldırı türleri: {df['attack_type'].nunique()} farklı tür")
        print(f"📊 Severity dağılımı:\n{df['severity'].value_counts()}")

        return df

    def prepare_data(self, df: pd.DataFrame,
                    test_size: float = 0.2,
                    random_state: int = 42) -> tuple:
        """
        Verileri hazırla ve böl

        Args:
            df (DataFrame): Ham veri
            test_size (float): Test set oranı
            random_state (int): Random seed

        Returns:
            tuple: X_train, X_test, y_train, y_test
        """
        print("\n🔄 Veri hazırlanıyor...")

        # Özellikler ve etiketler
        X = self.feature_extractor.prepare_features(df, fit=True)
        y = self.feature_extractor.prepare_labels(df, fit=True)

        print(f"✅ Özellik boyutu: {X.shape}")
        print(f"✅ Etiket boyutu: {y.shape}")
        print(f"✅ Sınıf sayısı: {len(np.unique(y))}")

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )

        print(f"\n📊 Veri bölündü:")
        print(f"  Train: {len(X_train)} kayıt ({(1-test_size)*100:.0f}%)")
        print(f"  Test:  {len(X_test)} kayıt ({test_size*100:.0f}%)")

        return X_train, X_test, y_train, y_test

    def train_model(self, X_train: np.ndarray, y_train: np.ndarray):
        """
        Modeli eğit

        Args:
            X_train: Eğitim özellikleri
            y_train: Eğitim etiketleri
        """
        print("\n" + "="*60)
        print("🎯 MODEL EĞİTİMİ BAŞLIYOR")
        print("="*60)

        self.model.train(X_train, y_train)

        print("="*60)

    def evaluate_model(self, X_test: np.ndarray, y_test: np.ndarray):
        """
        Modeli değerlendir

        Args:
            X_test: Test özellikleri
            y_test: Test etiketleri
        """
        print("\n" + "="*60)
        print("📊 MODEL DEĞERLENDİRMESİ")
        print("="*60)

        metrics = self.model.evaluate(X_test, y_test)

        # Confusion matrix
        class_names = self.feature_extractor.attack_encoder.classes_
        self.model.plot_confusion_matrix(
            X_test, y_test,
            class_names=class_names,
            save_path='models/confusion_matrix.png'
        )

        # Feature importance - GÜNCELLENDİ (8 özellik)
        feature_names = [
            'source_ip_last',      # Son oktet
            'dest_ip_last',        # Son oktet
            'port_normalized',     # Normalize port
            'severity',            # Severity encoded
            'blocked',             # 0/1
            'hour',                # Saat
            'is_night',            # Gece mi?
            'is_weekend'           # Hafta sonu mu?
        ]

        importance_df = self.model.get_feature_importance(feature_names)
        print("\n📈 En Önemli Özellikler:")
        print(importance_df.head(5).to_string(index=False))

        self.model.plot_feature_importance(
            feature_names=feature_names,
            top_n=10,
            save_path='models/feature_importance.png'
        )

        print("="*60)

        return metrics

    def save_models(self):
        """Model ve feature extractor'ı kaydet"""
        print("\n💾 Modeller kaydediliyor...")

        self.model.save('models/rf_model.pkl')
        self.feature_extractor.save('models/feature_extractor.pkl')

        print("✅ Tüm modeller kaydedildi!")

    def run_full_training(self, limit: int = None):
        """
        Tam eğitim pipeline'ı çalıştır

        Args:
            limit (int): Maksimum veri sayısı
        """
        print("\n" + "🚀 CYBERGUARD AI - MODEL EĞİTİMİ".center(60))
        print("="*60 + "\n")

        # 1. Veri yükle
        df = self.load_data_from_db(limit=limit)

        if len(df) == 0:
            print("❌ Database'de veri yok!")
            print("💡 Önce mock_data_generator.py'yi çalıştırın")
            return

        # 2. Veri hazırla
        X_train, X_test, y_train, y_test = self.prepare_data(df)

        # 3. Model eğit
        self.train_model(X_train, y_train)

        # 4. Değerlendir
        metrics = self.evaluate_model(X_test, y_test)

        # 5. Kaydet
        self.save_models()

        print("\n" + "="*60)
        print("🎉 EĞİTİM TAMAMLANDI!")
        print("="*60)
        print(f"\n📈 Final Accuracy: {metrics['accuracy']*100:.2f}%")
        print(f"📁 Model dosyaları 'models/' klasöründe")
        print("\n✅ Artık dashboard'da tahmin yapabilirsiniz!")


def main():
    """Ana fonksiyon"""

    # Trainer oluştur
    trainer = ModelTrainer(db_path='cyberguard.db')

    # Tam eğitim pipeline'ı çalıştır
    trainer.run_full_training(limit=None)  # Tüm veriyi kullan


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️ Eğitim durduruldu!")
    except Exception as e:
        print(f"\n❌ HATA: {e}")
        import traceback
        traceback.print_exc()