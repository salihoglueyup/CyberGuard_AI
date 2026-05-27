"""
Train Model - CyberGuard AI
Tek bir model eğitimi için basit script

Dosya Yolu: scripts/train_model.py
"""

import os
import sys
import argparse
from datetime import datetime

# Proje root
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)


def train_tensorflow():
    """TensorFlow modelini eğit"""
    print("\n🧠 TensorFlow Cyber Threat Model Eğitimi\n")
    
    try:
        from src.models.train_tensorflow_model import TensorFlowTrainer
        
        # Parametreler
        limit = int(input("Veri limiti [50000]: ").strip() or "50000")
        epochs = int(input("Epoch sayısı [50]: ").strip() or "50")
        batch_size = int(input("Batch size [32]: ").strip() or "32")
        model_name = input("Model adı (boş = otomatik): ").strip() or None
        
        trainer = TensorFlowTrainer(model_name=model_name)
        model_id, results = trainer.run_full_pipeline(
            limit=limit,
            epochs=epochs,
            batch_size=batch_size
        )
        
        print(f"\n🎉 Model eğitildi: {model_id}")
        print(f"   Accuracy: {results['summary']['accuracy']:.4f}")
        
    except ImportError as e:
        print(f"❌ Import hatası: {e}")
    except Exception as e:
        print(f"❌ Hata: {e}")
        import traceback
        traceback.print_exc()


def train_malware():
    """Malware detection modelini eğit"""
    print("\n🦠 Malware Detection Model Eğitimi\n")
    
    try:
        from src.malware_detection.train import MalwareTrainer
        
        n_samples = int(input("Örnek sayısı [2000]: ").strip() or "2000")
        model_type = input("Model tipi (gradient_boosting/random_forest) [gradient_boosting]: ").strip() or "gradient_boosting"
        
        trainer = MalwareTrainer()
        results = trainer.train(
            n_samples=n_samples,
            model_type=model_type
        )
        
        print(f"\n🎉 Model eğitildi!")
        print(f"   Accuracy: {results['eval_metrics']['accuracy']:.4f}")
        print(f"   F1-Score: {results['eval_metrics']['f1_score']:.4f}")
        
    except ImportError as e:
        print(f"❌ Import hatası: {e}")
    except Exception as e:
        print(f"❌ Hata: {e}")
        import traceback
        traceback.print_exc()


def train_network():
    """Network anomaly detection modelini eğit"""
    print("\n🌐 Network Anomaly Detection Model Eğitimi\n")
    
    try:
        from src.network_detection.train import NetworkTrainer
        
        n_samples = int(input("Örnek sayısı [2000]: ").strip() or "2000")
        model_type = input("Model tipi (random_forest/isolation_forest) [random_forest]: ").strip() or "random_forest"
        
        trainer = NetworkTrainer()
        results = trainer.train(
            n_samples=n_samples,
            model_type=model_type
        )
        
        print(f"\n🎉 Model eğitildi!")
        print(f"   Accuracy: {results['eval_metrics']['accuracy']:.4f}")
        print(f"   F1-Score: {results['eval_metrics']['f1_macro']:.4f}")
        
    except ImportError as e:
        print(f"❌ Import hatası: {e}")
    except Exception as e:
        print(f"❌ Hata: {e}")
        import traceback
        traceback.print_exc()


def show_model_info():
    """Model bilgilerini göster"""
    print("\n📋 Mevcut Modeller\n")
    
    models_dir = os.path.join(project_root, 'models')
    
    # TensorFlow modelleri
    print("🧠 TensorFlow Modelleri:")
    tf_models = [d for d in os.listdir(models_dir) 
                 if os.path.isdir(os.path.join(models_dir, d)) 
                 and d.startswith('neural_network')]
    
    if tf_models:
        for model in tf_models[:5]:  # Son 5 model
            print(f"   • {model}")
    else:
        print("   (yok)")
    
    # Malware modeli
    print("\n🦠 Malware Model:")
    malware_dir = os.path.join(models_dir, 'malware')
    if os.path.exists(malware_dir):
        print(f"   • {malware_dir}")
    else:
        print("   (yok)")
    
    # Network modeli
    print("\n🌐 Network Model:")
    network_dir = os.path.join(models_dir, 'network')
    if os.path.exists(network_dir):
        print(f"   • {network_dir}")
    else:
        print("   (yok)")


def main():
    """Ana fonksiyon - İnteraktif menü"""
    
    print("\n" + "=" * 60)
    print("🎯 CYBERGUARD AI - MODEL EĞİTİM ARACI")
    print("=" * 60)
    
    print("\n📋 Hangi modeli eğitmek istiyorsunuz?")
    print("  1. TensorFlow Cyber Threat Model")
    print("  2. Malware Detection Model")
    print("  3. Network Anomaly Detection Model")
    print("  4. Model bilgilerini göster")
    print("  5. Çıkış")
    
    choice = input("\nSeçiminiz (1-5): ").strip()
    
    if choice == '1':
        train_tensorflow()
    elif choice == '2':
        train_malware()
    elif choice == '3':
        train_network()
    elif choice == '4':
        show_model_info()
    elif choice == '5':
        print("\n👋 Çıkış...")
    else:
        print("❌ Geçersiz seçim!")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️ İptal edildi!")
    except Exception as e:
        print(f"\n❌ HATA: {e}")
        import traceback
        traceback.print_exc()