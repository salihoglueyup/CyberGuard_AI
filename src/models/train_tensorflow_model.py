"""
TensorFlow Model Training Script - CyberGuard AI
Mock data kullanarak TensorFlow modelini eğitir (Model Manager Entegreli)
"""

import sys
import os

# Proje kök dizinini ekle
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import sqlite3
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import json
import pickle

# Import models
try:
    from src.models.tensorflow_model import CyberThreatNeuralNetwork
    from src.models.model_evaluator import ModelEvaluator
    from src.models.model_manager import ModelManager
except ImportError:
    print("⚠️  Import hatası - Lokal import deneniyor...")
    import importlib.util

    # TensorFlow model
    spec = importlib.util.spec_from_file_location(
        "tensorflow_model",
        os.path.join(project_root, "src", "models", "tensorflow_model.py")
    )
    tf_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tf_module)
    CyberThreatNeuralNetwork = tf_module.CyberThreatNeuralNetwork

    # Model evaluator
    spec = importlib.util.spec_from_file_location(
        "model_evaluator",
        os.path.join(project_root, "src", "models", "model_evaluator.py")
    )
    eval_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(eval_module)
    ModelEvaluator = eval_module.ModelEvaluator

    # Model manager
    spec = importlib.util.spec_from_file_location(
        "model_manager",
        os.path.join(project_root, "src", "models", "model_manager.py")
    )
    mgr_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mgr_module)
    ModelManager = mgr_module.ModelManager


class TensorFlowTrainer:
    """TensorFlow model eğitim yöneticisi (Model Manager entegreli)"""

    def __init__(
            self,
            db_path: str = None,
            model_name: str = None,
            description: str = None,
            table: str = 'attacks'
    ):
        """
        Args:
            db_path: Database yolu (None ise src/database/cyberguard.db kullanılır)
            model_name: Custom model adı
            description: Model açıklaması
            table: Eğitim tablosu ('attacks' veya 'defences')
        """
        # Database yolunu src/database/ klasörüne göre ayarla
        if db_path is None:
            self.db_path = os.path.join(project_root, 'src', 'database', 'cyberguard.db')
        elif not os.path.isabs(db_path):
            # Relative path ise project_root'a göre çözümle
            self.db_path = os.path.join(project_root, db_path)
        else:
            self.db_path = db_path
        
        self.table = table  # attacks veya defences
        self.model_name = model_name or f"CyberThreat_{datetime.now().strftime('%Y%m%d')}"
        self.description = description or f"TensorFlow Model for {table.title()} Analysis"

        self.model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.class_names = None

        # Model Manager
        self.model_manager = ModelManager(base_dir='models')
        self.model_id = None
        self.model_dir = None

        print("🧠 TensorFlow Trainer başlatıldı")
        print(f"📁 Database: {self.db_path}")
        print(f"🗂️  Tablo: {self.table}")
        print(f"🏷️  Model Name: {self.model_name}")

    def load_data_from_db(self, limit: int = None, random_sample: bool = False) -> pd.DataFrame:
        """Database'den verileri yükle"""
        print(f"\n📂 Veriler yükleniyor: {self.db_path} (Tablo: {self.table})")

        if not os.path.exists(self.db_path):
            raise FileNotFoundError(f"❌ Database bulunamadı: {self.db_path}")

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(f"SELECT COUNT(*) FROM {self.table}")
        total_records = cursor.fetchone()[0]

        print(f"📊 Toplam kayıt ({self.table}): {total_records:,}")

        # CRITICAL FIX: Veri yoksa hata ver
        if total_records == 0:
            conn.close()
            raise ValueError(f"❌ {self.table} tablosunda hiç kayıt yok! Mock data generator'ı çalıştırın.")

        if random_sample and limit and limit < total_records:
            query = f"SELECT * FROM {self.table} ORDER BY RANDOM() LIMIT {limit}"
        elif limit:
            query = f"SELECT * FROM {self.table} LIMIT {limit}"
        else:
            query = f"SELECT * FROM {self.table}"

        df = pd.read_sql_query(query, conn)
        conn.close()

        print(f"✅ {len(df):,} kayıt yüklendi")
        return df

    def prepare_features(self, df: pd.DataFrame) -> np.ndarray:
        """Özellik mühendisliği - Mevcut DB kolonlarına göre"""
        print("\n🔄 Özellikler hazırlanıyor...")
        print(f"📋 Mevcut kolonlar: {df.columns.tolist()}")

        # Timestamp özellikleri (ISO8601 format: 2025-11-11T23:28:41.216414)
        df['timestamp'] = pd.to_datetime(df['timestamp'], format='ISO8601')
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        df['is_night'] = ((df['hour'] >= 22) | (df['hour'] <= 6)).astype(int)

        # IP özellikleri (son oktet)
        df['source_ip_last'] = df['source_ip'].str.split('.').str[-1].astype(int)
        df['dest_ip_last'] = df['destination_ip'].str.split('.').str[-1].astype(int)

        # Port normalizasyonu
        df['port_normalized'] = df['port'] / 65535.0

        # Severity encoding
        severity_map = {'low': 0, 'medium': 1, 'high': 2, 'critical': 3}
        df['severity_encoded'] = df['severity'].map(severity_map).fillna(1)  # Default: medium

        # Blocked (0 veya 1)
        df['blocked_flag'] = df['blocked'].astype(int)

        # Opsiyonel kolonlar - varsa ekle
        optional_features = []

        # Protocol encoding (varsa)
        if 'protocol' in df.columns:
            protocol_map = {'TCP': 0, 'UDP': 1, 'ICMP': 2}
            df['protocol_encoded'] = df['protocol'].map(protocol_map).fillna(0)
            optional_features.append('protocol_encoded')

        # Packet size (varsa)
        if 'packet_size' in df.columns:
            df['packet_size_normalized'] = df['packet_size'] / 65535.0
            optional_features.append('packet_size_normalized')

        # Feature matrix oluştur
        base_features = [
            'source_ip_last',
            'dest_ip_last',
            'port_normalized',
            'severity_encoded',
            'blocked_flag',
            'hour',
            'is_night',
            'is_weekend'
        ]

        features = base_features + optional_features

        X = df[features].values

        print(f"✅ Özellik boyutu: {X.shape}")
        print(f"✅ Kullanılan özellikler ({len(features)}): {features}")

        return X

    def prepare_labels(self, df: pd.DataFrame) -> np.ndarray:
        """Etiketleri hazırla"""
        y = self.label_encoder.fit_transform(df['attack_type'])
        self.class_names = self.label_encoder.classes_.tolist()
        print(f"✅ Sınıflar: {self.class_names}")
        return y

    def initialize_model_tracking(self, hyperparameters: Dict, training_config: Dict):
        """Model tracking başlat"""
        self.model_id = self.model_manager.generate_model_id(
            model_name=self.model_name,
            model_type='neural_network'
        )

        self.model_dir = self.model_manager.create_model_directory(
            model_id=self.model_id,
            model_info={
                'name': self.model_name,
                'description': self.description,
                'type': 'Deep Neural Network',
                'framework': 'TensorFlow/Keras'
            }
        )

        self.model_manager.register_model(
            model_id=self.model_id,
            model_name=self.model_name,
            model_type='neural_network',
            framework='tensorflow',
            hyperparameters=hyperparameters,
            training_config=training_config,
            tags=['deep_learning', 'cyber_security'],
            description=self.description
        )

        self.model_manager.update_model_status(self.model_id, 'training')

    def train_model(
            self,
            X_train: np.ndarray,
            y_train: np.ndarray,
            X_val: np.ndarray,
            y_val: np.ndarray,
            epochs: int = 100,
            batch_size: int = 32,
            hidden_layers: List[int] = [256, 128, 64, 32],
            dropout_rate: float = 0.3,
            learning_rate: float = 0.001
    ):
        """Modeli eğit"""
        hyperparameters = {
            'epochs': epochs,
            'batch_size': batch_size,
            'hidden_layers': hidden_layers,
            'dropout_rate': dropout_rate,
            'learning_rate': learning_rate
        }

        training_config = {
            'train_samples': len(X_train),
            'val_samples': len(X_val),
            'optimizer': 'adam'
        }

        self.initialize_model_tracking(hyperparameters, training_config)

        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)

        self.model = CyberThreatNeuralNetwork(
            input_dim=X_train.shape[1],
            hidden_layers=hidden_layers,
            dropout_rate=dropout_rate,
            learning_rate=learning_rate
        )

        history = self.model.train(
            X_train_scaled, y_train,
            X_val_scaled, y_val,
            epochs=epochs,
            batch_size=batch_size,
            class_names=self.class_names
        )

        self.model_manager.update_model_status(self.model_id, 'trained')
        return history

    def evaluate_model(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict:
        """Modeli değerlendir"""
        X_test_scaled = self.scaler.transform(X_test)
        y_pred = self.model.predict(X_test_scaled)
        y_pred_proba = self.model.predict(X_test_scaled, return_proba=True)

        evaluator = ModelEvaluator(class_names=self.class_names)
        eval_dir = self.model_dir / 'evaluation'

        results = evaluator.generate_evaluation_report(
            y_test, y_pred, y_pred_proba,
            model_name=self.model_name,
            output_dir=str(eval_dir)
        )

        plot_dir = self.model_dir / 'plots'
        self.model.plot_training_history(save_path=str(plot_dir / 'training_history.png'))

        self.model_manager.update_model_metrics(self.model_id, results['summary'])
        self.model_manager.update_model_status(self.model_id, 'deployed')

        return results

    def save_model(self):
        """Modeli kaydet"""
        artifact_dir = self.model_dir / 'artifacts'
        model_path = artifact_dir / f'{self.model_name}.h5'
        self.model.save(str(model_path))

        with open(artifact_dir / 'scaler.pkl', 'wb') as f:
            pickle.dump(self.scaler, f)

        with open(artifact_dir / 'label_encoder.pkl', 'wb') as f:
            pickle.dump(self.label_encoder, f)

        print(f"✅ Model kaydedildi: {model_path}")

    def run_full_pipeline(
            self,
            limit: int = None,
            random_sample: bool = True,
            test_size: float = 0.2,
            val_size: float = 0.1,
            epochs: int = 100,
            batch_size: int = 32,
            hidden_layers: List[int] = [256, 128, 64, 32],
            dropout_rate: float = 0.3,
            learning_rate: float = 0.001
    ):
        """Tam eğitim pipeline'ı"""
        df = self.load_data_from_db(limit=limit, random_sample=random_sample)
        X = self.prepare_features(df)
        y = self.prepare_labels(df)

        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )

        val_size_adjusted = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size_adjusted, random_state=42, stratify=y_temp
        )

        self.train_model(
            X_train, y_train, X_val, y_val,
            epochs, batch_size, hidden_layers, dropout_rate, learning_rate
        )

        results = self.evaluate_model(X_test, y_test)
        self.save_model()

        print(f"\n🎉 Model ID: {self.model_id}")
        print(f"📈 Accuracy: {results['summary']['accuracy'] * 100:.2f}%")

        return self.model_id, results


def show_menu():
    """İnteraktif menü göster"""
    print("\n" + "=" * 60)
    print("🛡️  CYBERGUARD AI - MODEL EĞİTİM SİSTEMİ")
    print("=" * 60)
    print("\n📋 Seçenekler:")
    print("  1. 🆕 Yeni Model Oluştur")
    print("  2. 📋 Mevcut Modelleri Listele")
    print("  3. 🗑️  Model Sil")
    print("  4. 📊 Veritabanı Bilgisi")
    print("  5. ❌ Çıkış")
    print("-" * 60)
    return input("Seçiminiz (1-5): ").strip()


def get_training_params():
    """Eğitim parametrelerini kullanıcıdan al"""
    print("\n📝 Eğitim Parametreleri:")
    print("-" * 40)
    
    # Tablo seçimi
    print("\n🗂️  Hangi tabloyu eğitmek istiyorsunuz?")
    print("  1. 🗡️  Saldırılar (attacks)")
    print("  2. 🛡️  Savunmalar (defences)")
    table_choice = input("Seçiminiz (1-2) [1]: ").strip()
    table = 'defences' if table_choice == '2' else 'attacks'
    print(f"✅ Seçilen tablo: {table}")
    
    # Veri sayısı
    limit_input = input("\n📊 Kaç kayıt kullanılsın? (boş=tümü, örn: 50000): ").strip()
    limit = int(limit_input) if limit_input else None
    
    # Random sample
    if limit:
        random_input = input("🎲 Rastgele seçim yapılsın mı? (E/H) [E]: ").strip().upper()
        random_sample = random_input != 'H'
    else:
        random_sample = False
    
    # Epoch sayısı
    epochs_input = input("🔄 Epoch sayısı? [50]: ").strip()
    epochs = int(epochs_input) if epochs_input else 50
    
    # Batch size
    batch_input = input("📦 Batch size? [32]: ").strip()
    batch_size = int(batch_input) if batch_input else 32
    
    # Model adı
    name_input = input("🏷️  Model adı? (boş=otomatik): ").strip()
    model_name = name_input if name_input else None
    
    return {
        'table': table,
        'limit': limit,
        'random_sample': random_sample,
        'epochs': epochs,
        'batch_size': batch_size,
        'model_name': model_name
    }


def create_model():
    """Yeni model oluştur"""
    params = get_training_params()
    
    print("\n" + "=" * 60)
    print("🚀 Model eğitimi başlatılıyor...")
    print("=" * 60)
    print(f"   🗂️  Tablo: {params['table']}")
    print(f"   📊 Veri limiti: {params['limit'] or 'Tümü'}")
    print(f"   🎲 Rastgele seçim: {'Evet' if params['random_sample'] else 'Hayır'}")
    print(f"   🔄 Epochs: {params['epochs']}")
    print(f"   📦 Batch size: {params['batch_size']}")
    print("=" * 60)
    
    confirm = input("\n▶️  Devam etmek istiyor musunuz? (E/H): ").strip().upper()
    if confirm != 'E':
        print("❌ İptal edildi.")
        return
    
    trainer = TensorFlowTrainer(
        model_name=params['model_name'],
        table=params['table']
    )
    
    model_id, results = trainer.run_full_pipeline(
        limit=params['limit'],
        random_sample=params['random_sample'],
        epochs=params['epochs'],
        batch_size=params['batch_size']
    )
    
    print("\n" + "=" * 60)
    print("🎉 MODEL BAŞARIYLA OLUŞTURULDU!")
    print("=" * 60)
    print(f"   🆔 Model ID: {model_id}")
    print(f"   📈 Accuracy: {results['summary']['accuracy'] * 100:.2f}%")
    print("=" * 60)


def list_models():
    """Mevcut modelleri listele"""
    mm = ModelManager(base_dir='models')
    models = mm.list_models()
    
    print("\n" + "=" * 60)
    print("📋 MEVCUT MODELLER")
    print("=" * 60)
    
    if not models:
        print("❌ Henüz model yok!")
        return
    
    for i, model in enumerate(models, 1):
        name = model.get('name', 'Unknown')
        status = model.get('status', 'unknown')
        acc = model.get('metrics', {}).get('accuracy', 0)
        created = model.get('created_at', 'N/A')[:10]
        
        status_icon = "✅" if status == 'deployed' else "🔄" if status == 'training' else "📦"
        
        print(f"\n  {i}. {status_icon} {name}")
        print(f"      Status: {status} | Accuracy: {acc*100:.1f}% | Tarih: {created}")
    
    print("\n" + "=" * 60)
    print(f"Toplam: {len(models)} model")


def delete_model():
    """Model sil"""
    mm = ModelManager(base_dir='models')
    models = mm.list_models()
    
    if not models:
        print("❌ Silinecek model yok!")
        return
    
    list_models()
    
    model_num = input("\n🗑️  Silmek istediğiniz model numarası: ").strip()
    try:
        idx = int(model_num) - 1
        if 0 <= idx < len(models):
            model = models[idx]
            confirm = input(f"⚠️  '{model['name']}' silinecek. Emin misiniz? (EVET yaz): ").strip()
            if confirm == "EVET":
                mm.delete_model(model['id'], remove_files=True)
                print("✅ Model silindi!")
            else:
                print("❌ İptal edildi.")
        else:
            print("❌ Geçersiz numara!")
    except ValueError:
        print("❌ Geçersiz giriş!")


def show_db_info():
    """Veritabanı bilgisi göster"""
    db_path = os.path.join(project_root, 'src', 'database', 'cyberguard.db')
    
    if not os.path.exists(db_path):
        print("❌ Veritabanı bulunamadı!")
        return
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.execute("SELECT COUNT(*) FROM attacks")
    total = cursor.fetchone()[0]
    
    cursor.execute("SELECT attack_type, COUNT(*) FROM attacks GROUP BY attack_type")
    by_type = cursor.fetchall()
    
    conn.close()
    
    print("\n" + "=" * 60)
    print("📊 VERİTABANI BİLGİSİ")
    print("=" * 60)
    print(f"   📁 Dosya: {db_path}")
    print(f"   📈 Toplam kayıt: {total:,}")
    print("\n   🎯 Saldırı Türleri:")
    for attack_type, count in by_type:
        print(f"      • {attack_type}: {count:,}")
    print("=" * 60)


def main():
    """Ana fonksiyon - Command line veya İnteraktif menü"""
    import argparse
    
    parser = argparse.ArgumentParser(description='TensorFlow Model Training')
    parser.add_argument('--db', type=str, default=None, help='Database path')
    parser.add_argument('--name', type=str, default=None, help='Model name')
    parser.add_argument('--desc', type=str, default=None, help='Description')
    parser.add_argument('--table', type=str, default='attacks', choices=['attacks', 'defences'], help='Table to train on')
    parser.add_argument('--limit', type=int, default=50000, help='Data limit')
    parser.add_argument('--epochs', type=int, default=50, help='Epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--random', action='store_true', help='Random sample')
    parser.add_argument('--interactive', '-i', action='store_true', help='Interactive mode')
    
    args = parser.parse_args()
    
    # Argüman varsa CLI modunda çalış
    if args.name:
        print("\n" + "=" * 60)
        print("🚀 CLI MODE - TensorFlow Model Training")
        print("=" * 60)
        
        try:
            trainer = TensorFlowTrainer(
                db_path=args.db,
                model_name=args.name,
                description=args.desc or "TensorFlow Model",
                table=args.table
            )
            
            model_id, results = trainer.run_full_pipeline(
                limit=args.limit,
                random_sample=args.random,
                epochs=args.epochs,
                batch_size=args.batch_size
            )
            
            print("\n" + "=" * 60)
            print(f"✅ Model eğitimi tamamlandı!")
            print(f"📁 Model ID: {model_id}")
            if results and 'summary' in results:
                print(f"🎯 Accuracy: {results['summary'].get('accuracy', 0):.4f}")
            print("=" * 60 + "\n")
            
        except Exception as e:
            print(f"\n❌ Training hatası: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
    else:
        # İnteraktif menü modu
        while True:
            choice = show_menu()
            
            if choice == '1':
                create_model()
            elif choice == '2':
                list_models()
            elif choice == '3':
                delete_model()
            elif choice == '4':
                show_db_info()
            elif choice == '5':
                print("\n👋 Güle güle!")
                break
            else:
                print("❌ Geçersiz seçim!")
            
            input("\n⏎ Devam etmek için Enter'a basın...")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️ Program durduruldu!")
    except Exception as e:
        print(f"\n❌ HATA: {e}")
        import traceback
        traceback.print_exc()
