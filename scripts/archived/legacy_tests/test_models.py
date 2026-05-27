"""
Model Test Script - CyberGuard AI
Eğitilmiş modelleri test eder ve performanslarını karşılaştırır

Sadece proje ana dizinindeki models/ klasöründen modelleri alır.

Dosya Yolu: scripts/tests/test_models.py
"""

import sys
import os

# Proje kök dizinini Python path'e ekle
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import numpy as np
import pandas as pd
import json
import pickle
from datetime import datetime
from typing import Dict, List, Optional

try:
    from sklearn.metrics import (
        accuracy_score, precision_recall_fscore_support,
        confusion_matrix, classification_report,
        roc_auc_score
    )
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

import warnings
warnings.filterwarnings('ignore')


class ModelTester:
    """
    Model test ve karşılaştırma sınıfı
    
    Sadece proje ana dizinindeki models/ klasöründen modelleri alır.
    
    Özellikler:
    - TensorFlow Model testing
    - Model Registry entegrasyonu
    - Performance metrics
    - Report generation
    """
    
    def __init__(self):
        """
        Tester başlat - sadece ana dizindeki models/ kullanır
        """
        # Proje ana dizinindeki models klasörü
        self.models_dir = os.path.join(project_root, 'models')
        self.registry_path = os.path.join(self.models_dir, 'model_registry.json')
        
        self.loaded_models = {}
        self.test_results = {}
        
        print("🧪 Model Tester başlatıldı")
        print(f"📁 Model dizini: {self.models_dir}")
        
        if not os.path.exists(self.models_dir):
            print(f"⚠️  Model dizini bulunamadı: {self.models_dir}")
    
    def list_available_models(self) -> List[Dict]:
        """Model registry'den mevcut modelleri listele"""
        print("\n" + "=" * 60)
        print("📋 MEVCUT MODELLER")
        print("=" * 60)
        
        models = []
        
        if os.path.exists(self.registry_path):
            with open(self.registry_path, 'r') as f:
                registry = json.load(f)
            
            # models bir liste olarak geliyor
            models_list = registry.get('models', [])
            
            for model_entry in models_list:
                model_id = model_entry.get('id', model_entry.get('model_id', 'Unknown'))
                model_info = {
                    'id': model_id,
                    'name': model_entry.get('name', model_entry.get('model_name', 'Unknown')),
                    'type': model_entry.get('type', model_entry.get('model_type', 'Unknown')),
                    'created': model_entry.get('created_at', 'Unknown'),
                    'status': model_entry.get('status', 'Unknown'),
                    'accuracy': model_entry.get('metrics', {}).get('accuracy', 0)
                }
                models.append(model_info)
                
                print(f"\n🔹 {model_id}")
                print(f"   Ad: {model_info['name']}")
                print(f"   Tür: {model_info['type']}")
                print(f"   Status: {model_info['status']}")
                if model_info['accuracy']:
                    print(f"   Accuracy: {model_info['accuracy']:.4f}")
        else:
            print("⚠️  Model registry bulunamadı")
        
        # Ayrıca klasörleri de listele
        print("\n📁 Model Klasörleri:")
        if os.path.exists(self.models_dir):
            for item in os.listdir(self.models_dir):
                item_path = os.path.join(self.models_dir, item)
                if os.path.isdir(item_path) and item.startswith('neural_network'):
                    print(f"   • {item}")
        
        print("=" * 60)
        return models
    
    def load_tensorflow_model(self, model_id: str) -> Optional[object]:
        """
        TensorFlow modeli yükle
        
        Args:
            model_id: Model ID veya klasör adı
            
        Returns:
            Yüklenen model veya None
        """
        try:
            from src.models.tensorflow_model import CyberThreatNeuralNetwork
            
            # Model klasörünü bul
            model_dir = os.path.join(self.models_dir, model_id)
            
            if not os.path.exists(model_dir):
                print(f"❌ Model klasörü bulunamadı: {model_dir}")
                return None
            
            # artifacts klasöründe .h5 dosyasını bul
            artifacts_dir = os.path.join(model_dir, 'artifacts')
            if os.path.exists(artifacts_dir):
                h5_files = [f for f in os.listdir(artifacts_dir) if f.endswith('.h5')]
                if h5_files:
                    model_path = os.path.join(artifacts_dir, h5_files[0])
                    model = CyberThreatNeuralNetwork.load(model_path)
                    print(f"✅ Model yüklendi: {model_path}")
                    return model
            
            print(f"❌ .h5 dosyası bulunamadı: {artifacts_dir}")
            return None
            
        except ImportError as e:
            print(f"❌ Import hatası: {e}")
            return None
        except Exception as e:
            print(f"❌ Model yükleme hatası: {e}")
            return None
    
    def test_model(
        self,
        model,
        X_test: np.ndarray,
        y_test: np.ndarray,
        model_name: str = "test_model",
        class_names: List[str] = None
    ) -> Dict:
        """
        Modeli test et
        
        Args:
            model: Yüklenen model
            X_test: Test verileri
            y_test: Test etiketleri
            model_name: Model adı
            class_names: Sınıf isimleri
            
        Returns:
            Test sonuçları
        """
        if not SKLEARN_AVAILABLE:
            print("❌ sklearn yüklü değil!")
            return {}
        
        print(f"\n🧪 Model test ediliyor: {model_name}")
        print("-" * 60)
        
        start_time = datetime.now()
        
        # Tahmin
        try:
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict(X_test, return_proba=True)
        except Exception as e:
            print(f"❌ Tahmin hatası: {e}")
            return {}
        
        end_time = datetime.now()
        inference_time = (end_time - start_time).total_seconds()
        
        # Metrikler
        accuracy = accuracy_score(y_test, y_pred)
        precision, recall, f1, support = precision_recall_fscore_support(
            y_test, y_pred, average='weighted', zero_division=0
        )
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        # Classification report
        report = classification_report(
            y_test, y_pred,
            target_names=class_names,
            digits=4,
            zero_division=0
        )
        
        # ROC-AUC
        try:
            roc_auc = roc_auc_score(
                y_test, y_pred_proba,
                multi_class='ovr',
                average='weighted'
            )
        except Exception:
            roc_auc = None
        
        # Sonuçları göster
        print(f"📊 Accuracy:  {accuracy * 100:.2f}%")
        print(f"📊 Precision: {precision * 100:.2f}%")
        print(f"📊 Recall:    {recall * 100:.2f}%")
        print(f"📊 F1-Score:  {f1 * 100:.2f}%")
        if roc_auc:
            print(f"📊 ROC-AUC:   {roc_auc * 100:.2f}%")
        print(f"⏱️  Inference: {inference_time:.4f} saniye ({len(X_test)} sample)")
        
        results = {
            'model_name': model_name,
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'roc_auc': float(roc_auc) if roc_auc else None,
            'confusion_matrix': cm.tolist(),
            'classification_report': report,
            'inference_time': inference_time,
            'test_samples': len(X_test),
            'tested_at': datetime.now().isoformat()
        }
        
        self.test_results[model_name] = results
        return results
    
    def save_report(self, output_path: str = None) -> str:
        """Test raporunu kaydet"""
        if not self.test_results:
            print("⚠️  Henüz test sonucu yok!")
            return ""
        
        if output_path is None:
            output_path = os.path.join(self.models_dir, 'test_report.json')
        
        report = {
            'test_date': datetime.now().isoformat(),
            'models_tested': list(self.test_results.keys()),
            'results': self.test_results
        }
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\n✅ Test raporu kaydedildi: {output_path}")
        return output_path


def main():
    """Ana fonksiyon - İnteraktif menü"""
    
    print("\n" + "=" * 60)
    print("🧪 CYBERGUARD AI - MODEL TEST SUITE")
    print("=" * 60)
    
    tester = ModelTester()
    
    print("\n📋 Seçenekler:")
    print("  1. Mevcut modelleri listele")
    print("  2. Model test et (mock data)")
    print("  3. Çıkış")
    
    choice = input("\nSeçiminiz (1-3): ").strip()
    
    if choice == '1':
        tester.list_available_models()
        
    elif choice == '2':
        # Modelleri listele
        models = tester.list_available_models()
        
        if not models and not os.path.exists(tester.models_dir):
            print("\n❌ Model bulunamadı!")
            return
        
        # Model seç
        model_id = input("\nTest edilecek model ID/klasör adı: ").strip()
        
        if model_id:
            model = tester.load_tensorflow_model(model_id)
            
            if model:
                # Mock test data
                print("\n🎲 Mock test verisi oluşturuluyor...")
                np.random.seed(42)
                X_test = np.random.rand(200, 8)
                y_test = np.random.randint(0, 5, 200)
                class_names = ['DDoS', 'SQL Injection', 'XSS', 'Port Scan', 'Brute Force']
                
                results = tester.test_model(
                    model, X_test, y_test,
                    model_name=model_id,
                    class_names=class_names
                )
                
                # Rapor kaydet
                if results:
                    tester.save_report()
        else:
            print("❌ Model ID girilmedi!")
            
    elif choice == '3':
        print("\n👋 Çıkış...")
    else:
        print("❌ Geçersiz seçim!")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️ Test durduruldu!")
    except Exception as e:
        print(f"\n❌ HATA: {e}")
        import traceback
        traceback.print_exc()