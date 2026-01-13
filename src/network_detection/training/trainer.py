"""
Network Model Training - CyberGuard AI
Network anomaly detection model eğitim scripti

Dosya Yolu: src/network_detection/train.py
"""

import os
import sys
import numpy as np
from datetime import datetime
from typing import Dict

# Proje root
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    from sklearn.model_selection import train_test_split
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

from src.network_detection.model import NetworkAnomalyModel
from src.network_detection.evaluator import NetworkEvaluator


class NetworkTrainer:
    """
    Network anomaly detection model eğitimi
    """
    
    ATTACK_TYPES = ['Normal', 'DDoS', 'SQL Injection', 'XSS', 'Port Scan', 'Brute Force']
    
    def __init__(self, model_dir: str = 'models/network'):
        """
        Args:
            model_dir: Model kayıt dizini
        """
        self.model_dir = model_dir
        self.model = None
        self.evaluator = NetworkEvaluator()
        
        os.makedirs(model_dir, exist_ok=True)
        
        print("🌐 Network Trainer başlatıldı")
        print(f"📁 Model dizini: {model_dir}")
    
    def generate_mock_data(self, n_samples: int = 2000) -> tuple:
        """
        Mock ağ trafiği verisi oluştur
        
        Features: [src_ip, dst_ip, src_port, dst_port, protocol, 
                   packet_size, is_dangerous_port, is_private, hour, is_night, is_weekend]
        """
        print(f"\n🎲 Mock veri oluşturuluyor ({n_samples} örnek)...")
        
        np.random.seed(42)
        
        samples_per_class = n_samples // len(self.ATTACK_TYPES)
        X_list = []
        y_list = []
        
        for class_id, attack_type in enumerate(self.ATTACK_TYPES):
            if attack_type == 'Normal':
                # Normal trafik
                features = np.column_stack([
                    np.random.uniform(0.4, 0.6, samples_per_class),  # Private IP range
                    np.random.uniform(0.0, 0.3, samples_per_class),  # Public dest
                    np.random.uniform(0.5, 1.0, samples_per_class),  # High src port
                    np.random.uniform(0.0, 0.02, samples_per_class),  # Low dest port (80, 443)
                    np.random.choice([0, 0.33], samples_per_class),   # TCP/UDP
                    np.random.uniform(0.01, 0.1, samples_per_class),  # Normal size
                    np.zeros(samples_per_class),                       # Not dangerous
                    np.ones(samples_per_class),                        # Private
                    np.random.uniform(0.3, 0.7, samples_per_class),    # Work hours
                    np.zeros(samples_per_class),                       # Not night
                    np.zeros(samples_per_class)                        # Not weekend
                ])
            elif attack_type == 'DDoS':
                features = np.column_stack([
                    np.random.uniform(0.0, 1.0, samples_per_class),    # Random src
                    np.random.uniform(0.4, 0.5, samples_per_class),    # Same dest
                    np.random.uniform(0.5, 1.0, samples_per_class),
                    np.random.uniform(0.0, 0.02, samples_per_class),   # 80, 443
                    np.zeros(samples_per_class),                        # TCP
                    np.random.uniform(0.8, 1.0, samples_per_class),     # Large packets
                    np.ones(samples_per_class),                         # Dangerous
                    np.zeros(samples_per_class),                        # Public
                    np.random.uniform(0.0, 1.0, samples_per_class),
                    np.random.choice([0, 1], samples_per_class),
                    np.random.choice([0, 1], samples_per_class)
                ])
            elif attack_type == 'Port Scan':
                features = np.column_stack([
                    np.random.uniform(0.5, 0.6, samples_per_class),    # Same src
                    np.random.uniform(0.4, 0.5, samples_per_class),    # Same dest
                    np.random.uniform(0.5, 1.0, samples_per_class),
                    np.random.uniform(0.0, 1.0, samples_per_class),    # Many ports
                    np.zeros(samples_per_class),
                    np.random.uniform(0.001, 0.01, samples_per_class), # Small packets
                    np.ones(samples_per_class),
                    np.zeros(samples_per_class),
                    np.random.uniform(0.8, 1.0, samples_per_class),    # Night time
                    np.ones(samples_per_class),
                    np.random.choice([0, 1], samples_per_class)
                ])
            else:  # SQL Injection, XSS, Brute Force
                features = np.column_stack([
                    np.random.uniform(0.0, 1.0, samples_per_class),
                    np.random.uniform(0.4, 0.6, samples_per_class),
                    np.random.uniform(0.5, 1.0, samples_per_class),
                    np.random.uniform(0.0, 0.1, samples_per_class),
                    np.zeros(samples_per_class),
                    np.random.uniform(0.02, 0.2, samples_per_class),
                    np.random.choice([0, 1], samples_per_class),
                    np.random.choice([0, 1], samples_per_class),
                    np.random.uniform(0.0, 1.0, samples_per_class),
                    np.random.choice([0, 1], samples_per_class),
                    np.random.choice([0, 1], samples_per_class)
                ])
            
            X_list.append(features)
            y_list.extend([class_id] * samples_per_class)
        
        X = np.vstack(X_list)
        y = np.array(y_list)
        
        # Shuffle
        indices = np.random.permutation(len(X))
        X = X[indices]
        y = y[indices]
        
        print(f"✅ Veri oluşturuldu: {X.shape}")
        for i, attack in enumerate(self.ATTACK_TYPES):
            count = np.sum(y == i)
            print(f"   {attack}: {count}")
        
        return X, y
    
    def train(
        self,
        X: np.ndarray = None,
        y: np.ndarray = None,
        n_samples: int = 2000,
        test_size: float = 0.2,
        model_type: str = 'random_forest',
        epochs: int = 50
    ) -> Dict:
        """Model eğit"""
        if not SKLEARN_AVAILABLE:
            raise RuntimeError("scikit-learn yüklü değil!")
        
        # Veri hazırla
        if X is None or y is None:
            X, y = self.generate_mock_data(n_samples)
        
        # Split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        print(f"\n📊 Train: {len(X_train)} | Test: {len(X_test)}")
        
        # Model
        self.model = NetworkAnomalyModel(model_type=model_type)
        train_results = self.model.train(X_train, y_train, epochs=epochs)
        
        # Değerlendir
        print("\n📊 Test değerlendirmesi...")
        y_pred = self.model.predict(X_test)
        y_proba = self.model.predict(X_test, return_proba=True)
        
        eval_metrics = self.evaluator.evaluate(y_test, y_pred, y_proba)
        self.evaluator.print_report(eval_metrics)
        
        # Kaydet
        model_path = self.model.save(self.model_dir)
        
        return {
            'train_results': train_results,
            'eval_metrics': eval_metrics,
            'model_path': model_path
        }


def main():
    """Ana fonksiyon"""
    print("\n" + "=" * 60)
    print("🌐 CYBERGUARD AI - NETWORK MODEL EĞİTİM")
    print("=" * 60)
    
    print("\n📋 Seçenekler:")
    print("  1. Yeni model eğit")
    print("  2. Mevcut modeli test et")
    print("  3. Çıkış")
    
    choice = input("\nSeçiminiz (1-3): ").strip()
    
    if choice == '1':
        n_samples = int(input("Örnek sayısı [2000]: ").strip() or "2000")
        
        trainer = NetworkTrainer()
        results = trainer.train(n_samples=n_samples)
        
        print(f"\n🎉 Accuracy: {results['eval_metrics']['accuracy']:.4f}")
        
    elif choice == '2':
        model_dir = input("Model dizini [models/network]: ").strip() or "models/network"
        
        if os.path.exists(model_dir):
            model = NetworkAnomalyModel.load(model_dir)
            
            test_features = [0.5, 0.5, 0.8, 0.01, 0, 0.9, 1, 0, 0.5, 0, 0]
            result = model.predict_single(test_features)
            print(f"\n📊 Tahmin: {result['prediction']}")
        else:
            print("❌ Model bulunamadı!")
            
    elif choice == '3':
        print("\n👋 Çıkış...")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️ İptal edildi!")
    except Exception as e:
        print(f"\n❌ HATA: {e}")
        import traceback
        traceback.print_exc()
