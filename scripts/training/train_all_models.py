"""
Train All Models - CyberGuard AI
Tüm modelleri tek seferde eğit

Dosya Yolu: scripts/train_all_models.py
"""

import os
import sys
from datetime import datetime
from typing import Dict, List

# Proje root
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)


class AllModelsTrainer:
    """
    Tüm CyberGuard AI modellerini tek seferde eğit
    
    Modeller:
    1. TensorFlow Cyber Threat Model (ana model)
    2. Malware Detection Model
    3. Network Anomaly Detection Model
    """
    
    def __init__(self):
        """Trainer başlat"""
        self.results: Dict[str, Dict] = {}
        self.start_time = None
        
        print("🚀 All Models Trainer başlatıldı")
        print(f"📁 Proje: {project_root}")
    
    def train_tensorflow_model(
        self,
        limit: int = 50000,
        epochs: int = 50,
        batch_size: int = 32
    ) -> Dict:
        """TensorFlow modeli eğit"""
        print("\n" + "=" * 60)
        print("🧠 TENSORFLOW CYBER THREAT MODEL")
        print("=" * 60)
        
        try:
            from src.models.train_tensorflow_model import TensorFlowTrainer
            
            trainer = TensorFlowTrainer()
            model_id, results = trainer.run_full_pipeline(
                limit=limit,
                random_sample=True,
                epochs=epochs,
                batch_size=batch_size
            )
            
            self.results['tensorflow'] = {
                'status': 'success',
                'model_id': model_id,
                'accuracy': results['summary']['accuracy']
            }
            
            print(f"✅ TensorFlow model tamamlandı: {model_id}")
            return self.results['tensorflow']
            
        except Exception as e:
            self.results['tensorflow'] = {
                'status': 'error',
                'error': str(e)
            }
            print(f"❌ TensorFlow model hatası: {e}")
            return self.results['tensorflow']
    
    def train_malware_model(
        self,
        n_samples: int = 2000,
        model_type: str = 'gradient_boosting'
    ) -> Dict:
        """Malware detection modeli eğit"""
        print("\n" + "=" * 60)
        print("🦠 MALWARE DETECTION MODEL")
        print("=" * 60)
        
        try:
            from src.malware_detection.train import MalwareTrainer
            
            trainer = MalwareTrainer()
            results = trainer.train(
                n_samples=n_samples,
                model_type=model_type
            )
            
            self.results['malware'] = {
                'status': 'success',
                'model_path': results['model_path'],
                'accuracy': results['eval_metrics']['accuracy'],
                'f1_score': results['eval_metrics']['f1_score']
            }
            
            print(f"✅ Malware model tamamlandı")
            return self.results['malware']
            
        except Exception as e:
            self.results['malware'] = {
                'status': 'error',
                'error': str(e)
            }
            print(f"❌ Malware model hatası: {e}")
            return self.results['malware']
    
    def train_network_model(
        self,
        n_samples: int = 2000,
        model_type: str = 'random_forest'
    ) -> Dict:
        """Network anomaly detection modeli eğit"""
        print("\n" + "=" * 60)
        print("🌐 NETWORK ANOMALY DETECTION MODEL")
        print("=" * 60)
        
        try:
            from src.network_detection.train import NetworkTrainer
            
            trainer = NetworkTrainer()
            results = trainer.train(
                n_samples=n_samples,
                model_type=model_type
            )
            
            self.results['network'] = {
                'status': 'success',
                'model_path': results['model_path'],
                'accuracy': results['eval_metrics']['accuracy'],
                'f1_score': results['eval_metrics']['f1_macro']
            }
            
            print(f"✅ Network model tamamlandı")
            return self.results['network']
            
        except Exception as e:
            self.results['network'] = {
                'status': 'error',
                'error': str(e)
            }
            print(f"❌ Network model hatası: {e}")
            return self.results['network']
    
    def train_all(
        self,
        tensorflow_limit: int = 50000,
        tensorflow_epochs: int = 50,
        malware_samples: int = 2000,
        network_samples: int = 2000
    ) -> Dict:
        """
        Tüm modelleri sırayla eğit
        
        Args:
            tensorflow_limit: TensorFlow veri limiti
            tensorflow_epochs: TensorFlow epoch sayısı
            malware_samples: Malware mock data sayısı
            network_samples: Network mock data sayısı
            
        Returns:
            Tüm sonuçlar
        """
        self.start_time = datetime.now()
        
        print("\n" + "=" * 60)
        print("🚀 TÜM MODELLER EĞİTİLİYOR")
        print("=" * 60)
        print(f"⏰ Başlangıç: {self.start_time.strftime('%H:%M:%S')}")
        
        # 1. TensorFlow
        self.train_tensorflow_model(
            limit=tensorflow_limit,
            epochs=tensorflow_epochs
        )
        
        # 2. Malware
        self.train_malware_model(n_samples=malware_samples)
        
        # 3. Network
        self.train_network_model(n_samples=network_samples)
        
        # Sonuç özeti
        end_time = datetime.now()
        duration = (end_time - self.start_time).total_seconds()
        
        self.print_summary(duration)
        
        return self.results
    
    def print_summary(self, duration: float) -> None:
        """Eğitim özetini yazdır"""
        print("\n" + "=" * 60)
        print("📊 EĞİTİM ÖZETİ")
        print("=" * 60)
        
        success_count = sum(1 for r in self.results.values() if r['status'] == 'success')
        total_count = len(self.results)
        
        print(f"\n⏱️  Toplam süre: {duration:.1f} saniye ({duration/60:.1f} dakika)")
        print(f"✅ Başarılı: {success_count}/{total_count} model")
        
        print("\n📋 Model Sonuçları:")
        for model_name, result in self.results.items():
            icon = "✅" if result['status'] == 'success' else "❌"
            print(f"\n   {icon} {model_name.upper()}")
            
            if result['status'] == 'success':
                if 'accuracy' in result:
                    print(f"      Accuracy: {result['accuracy']:.4f}")
                if 'f1_score' in result:
                    print(f"      F1-Score: {result['f1_score']:.4f}")
                if 'model_id' in result:
                    print(f"      Model ID: {result['model_id']}")
                if 'model_path' in result:
                    print(f"      Path: {result['model_path']}")
            else:
                print(f"      Hata: {result.get('error', 'Unknown')}")
        
        print("\n" + "=" * 60)


def main():
    """Ana fonksiyon - İnteraktif menü"""
    
    print("\n" + "=" * 60)
    print("🚀 CYBERGUARD AI - TÜM MODELLERİ EĞİT")
    print("=" * 60)
    
    print("\n📋 Seçenekler:")
    print("  1. Tüm modelleri eğit (varsayılan ayarlar)")
    print("  2. Tüm modelleri eğit (özel ayarlar)")
    print("  3. Sadece TensorFlow model")
    print("  4. Sadece Malware model")
    print("  5. Sadece Network model")
    print("  6. Çıkış")
    
    choice = input("\nSeçiminiz (1-6): ").strip()
    
    trainer = AllModelsTrainer()
    
    if choice == '1':
        print("\n⚡ Varsayılan ayarlarla tüm modeller eğitiliyor...")
        print("   TensorFlow: 50K veri, 50 epoch")
        print("   Malware: 2K mock data")
        print("   Network: 2K mock data")
        
        confirm = input("\n▶️  Devam? (E/H): ").strip().upper()
        if confirm == 'E':
            trainer.train_all()
            
    elif choice == '2':
        tf_limit = int(input("TensorFlow veri limiti [50000]: ").strip() or "50000")
        tf_epochs = int(input("TensorFlow epoch [50]: ").strip() or "50")
        mal_samples = int(input("Malware örnekleri [2000]: ").strip() or "2000")
        net_samples = int(input("Network örnekleri [2000]: ").strip() or "2000")
        
        trainer.train_all(
            tensorflow_limit=tf_limit,
            tensorflow_epochs=tf_epochs,
            malware_samples=mal_samples,
            network_samples=net_samples
        )
        
    elif choice == '3':
        limit = int(input("Veri limiti [50000]: ").strip() or "50000")
        epochs = int(input("Epoch [50]: ").strip() or "50")
        trainer.train_tensorflow_model(limit=limit, epochs=epochs)
        
    elif choice == '4':
        samples = int(input("Örnek sayısı [2000]: ").strip() or "2000")
        trainer.train_malware_model(n_samples=samples)
        
    elif choice == '5':
        samples = int(input("Örnek sayısı [2000]: ").strip() or "2000")
        trainer.train_network_model(n_samples=samples)
        
    elif choice == '6':
        print("\n👋 Çıkış...")
    else:
        print("❌ Geçersiz seçim!")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️ Eğitim durduruldu!")
    except Exception as e:
        print(f"\n❌ HATA: {e}")
        import traceback
        traceback.print_exc()
