"""
ML Prediction API - CyberGuard AI
ML modelleri ile gerçek zamanlı saldırı tahmini (TensorFlow + Scikit-learn)

Dosya Yolu: src/api/ml_prediction.py
"""

import os
import sys
import numpy as np
from typing import Dict, List, Optional, Any
from datetime import datetime

# Path düzeltmesi
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, project_root)

from src.models.model_manager import ModelManager
from src.utils import Logger


class MLPredictionAPI:
    """ML Model Prediction API - TensorFlow ve Scikit-learn desteği"""

    def __init__(self, default_model_id: Optional[str] = None):
        self.logger = Logger("MLPredictionAPI")
        self.model_manager = ModelManager()
        self.default_model_id = default_model_id
        self.loaded_models = {}
        self.logger.info("✅ ML Prediction API initialized")

    def predict(self, features: Dict[str, Any], model_id: Optional[str] = None) -> Dict:
        """Tekli tahmin"""
        try:
            model_id = model_id or self.default_model_id or self._get_best_model_id()
            if not model_id:
                return self._error_response("Model bulunamadı")

            model_data = self._load_model(model_id)
            if 'error' in model_data:
                return model_data

            X = self._prepare_features(features, model_data)
            model = model_data['model']

            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba(X)[0]
                prediction = int(np.argmax(proba))
                confidence = float(np.max(proba))
            else:
                pred = model.predict(X)[0]
                if hasattr(pred, '__len__'):
                    prediction = int(np.argmax(pred))
                    confidence = float(np.max(pred))
                else:
                    prediction = int(pred)
                    confidence = 0.8

            result = {
                'success': True,
                'model_id': model_id,
                'model_name': model_data['info']['name'],
                'prediction': 'malicious' if prediction == 1 else 'benign',
                'prediction_value': prediction,
                'confidence': round(confidence, 4),
                'confidence_percentage': round(confidence * 100, 2),
                'risk_level': self._calculate_risk_level(prediction, confidence),
                'features_used': features,
                'timestamp': datetime.now().isoformat()
            }

            result['explanation'] = self._generate_explanation(result)
            self.logger.info(f"✅ Prediction: {result['prediction']} ({confidence:.2%})")
            return result

        except Exception as e:
            self.logger.error(f"Prediction error: {e}")
            import traceback
            traceback.print_exc()
            return self._error_response(str(e))

    def _load_model(self, model_id: str) -> Dict:
        """Model yükle - TensorFlow veya Scikit-learn (Keras 3 uyumlu)"""
        if model_id in self.loaded_models:
            return self.loaded_models[model_id]

        try:
            model_info = self.model_manager.load_model(model_id)
            model_path = model_info['model_path']
            file_ext = os.path.splitext(model_path)[1].lower()

            if file_ext in ['.h5', '.keras']:
                model = self._load_keras_model(model_path)
                if model is None:
                    return self._error_response("Keras model yüklenemedi")
                self.logger.info(f"✅ Keras model loaded")
            elif file_ext in ['.pkl', '.joblib']:
                import joblib
                model = joblib.load(model_path)
                self.logger.info(f"✅ Joblib model loaded")
            else:
                return self._error_response(f"Desteklenmeyen format: {file_ext}")

            self.loaded_models[model_id] = {'info': model_info, 'model': model}
            return self.loaded_models[model_id]

        except Exception as e:
            self.logger.error(f"Model load error: {e}")
            import traceback
            traceback.print_exc()
            return self._error_response(f"Model yüklenemedi: {str(e)}")
    
    def _load_keras_model(self, model_path: str):
        """Keras modeli yükle - Keras 2/3 uyumlu"""
        import tensorflow as tf
        
        # Yöntem 1: Standart yükleme (compile=False)
        try:
            model = tf.keras.models.load_model(model_path, compile=False)
            self.logger.info("Model loaded with compile=False")
            return model
        except Exception as e1:
            self.logger.warning(f"Standard load failed: {e1}")
        
        # Yöntem 2: Safe mode kapalı
        try:
            model = tf.keras.models.load_model(model_path, compile=False, safe_mode=False)
            self.logger.info("Model loaded with safe_mode=False")
            return model
        except Exception as e2:
            self.logger.warning(f"Safe mode off failed: {e2}")
        
        # Yöntem 3: Weights only
        try:
            # Metadata'dan model config al
            import json
            metadata_path = model_path.replace('.h5', '_metadata.json').replace('.keras', '_metadata.json')
            
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                
                # Basit model oluştur
                input_dim = metadata.get('input_dim', 8)
                num_classes = metadata.get('num_classes', 10)
                
                model = tf.keras.Sequential([
                    tf.keras.layers.Input(shape=(input_dim,)),
                    tf.keras.layers.Dense(256, activation='relu'),
                    tf.keras.layers.Dense(128, activation='relu'),
                    tf.keras.layers.Dense(64, activation='relu'),
                    tf.keras.layers.Dense(num_classes, activation='softmax')
                ])
                
                # Weights yükle
                model.load_weights(model_path)
                self.logger.info("Model rebuilt and weights loaded")
                return model
        except Exception as e3:
            self.logger.warning(f"Weight loading failed: {e3}")
        
        # Yöntem 4: Legacy format
        try:
            import keras
            model = keras.saving.load_model(model_path, compile=False)
            self.logger.info("Model loaded with keras.saving")
            return model
        except Exception as e4:
            self.logger.error(f"All methods failed: {e4}")
        
        return None

    def _prepare_features(self, features: Dict, model_data: Dict) -> np.ndarray:
        """Feature hazırlama - Model metadata'sına göre"""
        import json
        
        # Model info'dan input_dim al
        model_info = model_data.get('info', {})
        model_path = model_info.get('model_path', '')
        
        # Metadata dosyasını bul ve oku
        input_dim = 10  # Default
        try:
            # Model directory'den metadata oku
            model_dir = os.path.dirname(model_path)
            artifacts_dir = os.path.join(model_dir, 'artifacts') if 'artifacts' not in model_path else model_dir
            
            # Metadata dosyalarını ara
            for root, dirs, files in os.walk(os.path.dirname(model_path)):
                for f in files:
                    if f.endswith('_metadata.json'):
                        with open(os.path.join(root, f), 'r') as mf:
                            metadata = json.load(mf)
                            input_dim = metadata.get('input_dim', 10)
                            self.logger.info(f"Metadata found: input_dim={input_dim}")
                            break
        except Exception as e:
            self.logger.warning(f"Metadata okunamadı: {e}")
        
        # 10 feature oluştur (model beklentisine göre)
        feature_values = []
        
        # 1. source_ip_last (son oktet)
        ip = features.get('source_ip', '0.0.0.0')
        try:
            feature_values.append(float(ip.split('.')[-1]))
        except:
            feature_values.append(0.0)
        
        # 2. dest_ip_last
        dest_ip = features.get('destination_ip', '0.0.0.0')
        try:
            feature_values.append(float(dest_ip.split('.')[-1]))
        except:
            feature_values.append(0.0)
        
        # 3. port_normalized
        feature_values.append(features.get('port', 80) / 10000.0)
        
        # 4. severity_encoded
        sev_map = {'low': 0, 'medium': 1, 'high': 2, 'critical': 3}
        feature_values.append(float(sev_map.get(features.get('severity', 'medium').lower(), 1)))
        
        # 5. blocked
        feature_values.append(float(features.get('blocked', 0)))
        
        # 6. hour
        feature_values.append(float(datetime.now().hour))
        
        # 7. day_of_week
        feature_values.append(float(datetime.now().weekday()))
        
        # 8. is_night
        h = datetime.now().hour
        feature_values.append(1.0 if (h >= 22 or h <= 6) else 0.0)
        
        # 9. is_weekend
        feature_values.append(1.0 if datetime.now().weekday() >= 5 else 0.0)
        
        # 10. status_encoded (ek feature)
        status_map = {'active': 0, 'blocked': 1, 'allowed': 2, 'detected': 3}
        feature_values.append(float(status_map.get(features.get('status', 'detected').lower(), 3)))
        
        # Eğer input_dim farklıysa, padding veya truncate yap
        while len(feature_values) < input_dim:
            feature_values.append(0.0)
        
        feature_values = feature_values[:input_dim]
        
        X = np.array(feature_values, dtype=np.float32).reshape(1, -1)
        self.logger.info(f"Features: shape={X.shape}, input_dim={input_dim}")
        return X

    def _encode_ip(self, ip: str) -> int:
        try:
            p = ip.split('.')
            return int(p[0])*256**3 + int(p[1])*256**2 + int(p[2])*256 + int(p[3])
        except:
            return 0

    def _calculate_risk_level(self, pred: int, conf: float) -> str:
        if pred == 1:
            if conf >= 0.9: return 'CRITICAL'
            elif conf >= 0.7: return 'HIGH'
            elif conf >= 0.5: return 'MEDIUM'
            else: return 'LOW'
        return 'SAFE'

    def _generate_explanation(self, result: Dict) -> str:
        pred = result['prediction']
        conf = result['confidence_percentage']
        risk = result['risk_level']

        if pred == 'malicious':
            exp = f"⚠️ ZARARLI tespit edildi. %{conf} güven. Risk: {risk}. "
            if conf >= 90: exp += "Acil müdahale!"
            elif conf >= 70: exp += "Detaylı inceleme."
            else: exp += "İzleme önerilir."
        else:
            exp = f"✅ GÜVENLİ. %{conf} güven. Normal trafik."
        return exp

    def _get_best_model_id(self) -> Optional[str]:
        try:
            models = self.model_manager.list_models()
            if not models: return None
            deployed = [m for m in models if m.get('status') == 'deployed'] or models
            best = max(deployed, key=lambda m: m.get('metrics', {}).get('accuracy', 0))
            return best.get('id') or best.get('model_id')
        except:
            return None

    def _error_response(self, msg: str) -> Dict:
        return {'success': False, 'error': msg, 'timestamp': datetime.now().isoformat()}

    def get_available_models(self) -> List[Dict]:
        try:
            models = self.model_manager.list_models()
            return [{'id': m.get('id', 'unknown'), 'name': m.get('name', 'Unknown'),
                    'type': m.get('model_type', 'Unknown'), 'status': m.get('status', 'unknown'),
                    'accuracy': m.get('metrics', {}).get('accuracy', 0)} for m in models]
        except:
            return []


# Test
if __name__ == "__main__":
    print("🧪 ML Prediction API Test\n" + "="*60)
    api = MLPredictionAPI()

    print("\n📋 Available Models:")
    for m in api.get_available_models():
        print(f"  • {m['name']} (acc: {m['accuracy']:.2%}) - {m['status']}")

    print("\n🎯 Single Prediction:")
    result = api.predict({'source_ip': '192.168.1.100', 'port': 80, 'severity': 'high'})

    if result.get('success'):
        print(f"Prediction: {result['prediction']}")
        print(f"Confidence: {result['confidence_percentage']}%")
        print(f"Risk: {result['risk_level']}")
        print(f"Explanation: {result['explanation']}")
    else:
        print(f"Error: {result.get('error')}")

    print("\n" + "="*60 + "\n✅ Test tamamlandı!")