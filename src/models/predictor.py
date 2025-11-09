"""
Predictor - CyberGuard AI
Eğitilmiş modeli kullanarak tahmin yapar
"""

import os
import sys
import pickle
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from datetime import datetime

# Path ayarları
current_dir = os.path.dirname(os.path.abspath(__file__))
# src/models/ içindeysek, 2 seviye yukarı çık
project_root = os.path.dirname(os.path.dirname(current_dir))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

class AttackPredictor:
    """Saldırı tahmini yapan sınıf"""

    def __init__(self,
                 model_path: str = 'models/rf_model.pkl',
                 extractor_path: str = 'models/feature_extractor.pkl'):
        """
        Args:
            model_path (str): Model dosya yolu
            extractor_path (str): Feature extractor yolu
        """
        self.model_path = model_path
        self.extractor_path = extractor_path
        self.model = None
        self.feature_extractor = None
        self.is_loaded = False

    def load_models(self) -> bool:
        """
        Model ve feature extractor'ı yükle

        Returns:
            bool: Yükleme başarılı mı?
        """
        try:
            # Model yükle
            if not os.path.exists(self.model_path):
                print(f"❌ Model bulunamadı: {self.model_path}")
                return False

            with open(self.model_path, 'rb') as f:
                model_data = pickle.load(f)
                self.model = model_data['model']

            # Feature extractor yükle
            if not os.path.exists(self.extractor_path):
                print(f"❌ Feature extractor bulunamadı: {self.extractor_path}")
                return False

            with open(self.extractor_path, 'rb') as f:
                extractor_data = pickle.load(f)

                # Feature extractor sınıfını yeniden oluştur
                try:
                    from src.utils.feature_extractor import FeatureExtractor
                except ImportError:
                    # Alternatif: Direkt import (proje root'undan)
                    import importlib.util
                    # project_root zaten CyberGuard_AI/
                    extractor_file = os.path.join(project_root, 'src', 'utils', 'feature_extractor.py')

                    if not os.path.exists(extractor_file):
                        raise FileNotFoundError(f"Feature extractor bulunamadı: {extractor_file}")

                    spec = importlib.util.spec_from_file_location("feature_extractor", extractor_file)
                    feature_module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(feature_module)
                    FeatureExtractor = feature_module.FeatureExtractor

                self.feature_extractor = FeatureExtractor()
                self.feature_extractor.attack_encoder = extractor_data['attack_encoder']
                self.feature_extractor.severity_encoder = extractor_data['severity_encoder']
                self.feature_extractor.status_encoder = extractor_data['status_encoder']
                self.feature_extractor.scaler = extractor_data['scaler']
                self.feature_extractor.is_fitted = extractor_data['is_fitted']

            self.is_loaded = True
            print("✅ Model ve feature extractor yüklendi!")
            return True

        except Exception as e:
            print(f"❌ Yükleme hatası: {e}")
            import traceback
            traceback.print_exc()
            return False

    def predict_single(self, attack_data: Dict) -> Dict:
        """
        Tek bir saldırı için tahmin yap

        Args:
            attack_data (dict): Saldırı bilgileri
                {
                    'source_ip': '192.168.1.100',
                    'destination_ip': '192.168.0.10',
                    'port': 80,
                    'severity': 'high',
                    'blocked': 1,
                    'timestamp': '2024-10-29 14:30:00'
                }

        Returns:
            dict: Tahmin sonucu
                {
                    'predicted_type': 'DDoS',
                    'confidence': 0.95,
                    'probabilities': {'DDoS': 0.95, 'Port Scan': 0.03, ...},
                    'risk_level': 'high'
                }
        """
        if not self.is_loaded:
            if not self.load_models():
                return {'error': 'Model yüklenemedi'}

        try:
            # DataFrame'e çevir
            df = pd.DataFrame([attack_data])

            # Eksik alanları doldur
            if 'timestamp' not in df.columns:
                df['timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            if 'status' not in df.columns:
                df['status'] = 'detected'
            if 'attack_type' not in df.columns:
                df['attack_type'] = 'Unknown'  # Dummy değer

            # Özellikler çıkar
            X = self.feature_extractor.prepare_features(df, fit=False)

            # Tahmin yap
            prediction = self.model.predict(X)[0]
            probabilities = self.model.predict_proba(X)[0]

            # Sınıf isimlerine çevir
            predicted_type = self.feature_extractor.get_attack_type_name(prediction)
            confidence = float(np.max(probabilities))

            # Tüm olasılıklar
            all_probs = {}
            for idx, prob in enumerate(probabilities):
                attack_name = self.feature_extractor.get_attack_type_name(idx)
                all_probs[attack_name] = float(prob)

            # Risk seviyesi belirle
            risk_level = self._calculate_risk_level(
                predicted_type,
                confidence,
                attack_data.get('severity', 'medium')
            )

            result = {
                'predicted_type': predicted_type,
                'confidence': confidence,
                'probabilities': all_probs,
                'risk_level': risk_level,
                'top_3_predictions': self._get_top_predictions(all_probs, 3)
            }

            return result

        except Exception as e:
            print(f"❌ Tahmin hatası: {e}")
            import traceback
            traceback.print_exc()
            return {'error': str(e)}

    def predict_batch(self, attacks: List[Dict]) -> List[Dict]:
        """
        Birden fazla saldırı için tahmin yap

        Args:
            attacks (list): Saldırı listesi

        Returns:
            list: Tahmin sonuçları
        """
        results = []
        for attack in attacks:
            result = self.predict_single(attack)
            results.append(result)

        return results

    def _calculate_risk_level(self, attack_type: str,
                             confidence: float,
                             severity: str) -> str:
        """
        Risk seviyesi hesapla

        Args:
            attack_type (str): Saldırı türü
            confidence (float): Güven skoru
            severity (str): Severity

        Returns:
            str: Risk seviyesi
        """
        # Tehlikeli saldırı türleri
        critical_types = ['DDoS', 'SQL Injection', 'Malware']
        high_types = ['Brute Force', 'XSS']

        # Severity skorları
        severity_scores = {
            'low': 1,
            'medium': 2,
            'high': 3,
            'critical': 4
        }

        # Risk hesapla
        type_score = 3 if attack_type in critical_types else (2 if attack_type in high_types else 1)
        severity_score = severity_scores.get(severity.lower(), 2)
        confidence_score = confidence * 3

        total_score = (type_score + severity_score + confidence_score) / 3

        if total_score >= 3.5:
            return 'critical'
        elif total_score >= 2.5:
            return 'high'
        elif total_score >= 1.5:
            return 'medium'
        else:
            return 'low'

    def _get_top_predictions(self, probabilities: Dict, n: int = 3) -> List[Tuple[str, float]]:
        """
        En yüksek n tahmini al

        Args:
            probabilities (dict): Olasılıklar
            n (int): Kaç tane

        Returns:
            list: (attack_type, probability) listesi
        """
        sorted_probs = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)
        return sorted_probs[:n]

    def get_model_info(self) -> Dict:
        """
        Model bilgilerini al

        Returns:
            dict: Model bilgileri
        """
        if not self.is_loaded:
            return {'loaded': False}

        try:
            attack_types = self.feature_extractor.attack_encoder.classes_.tolist()

            return {
                'loaded': True,
                'model_type': 'Random Forest',
                'n_classes': len(attack_types),
                'attack_types': attack_types,
                'n_features': 8
            }
        except:
            return {'loaded': True, 'error': 'Bilgi alınamadı'}


# Test ve örnek kullanım
if __name__ == "__main__":
    print("🧪 AttackPredictor Test\n")

    # Predictor oluştur
    predictor = AttackPredictor()

    # Model bilgisi
    info = predictor.get_model_info()
    print(f"📊 Model Bilgisi: {info}\n")

    # Test saldırısı
    test_attack = {
        'source_ip': '192.168.1.105',      # DDoS IP aralığı
        'destination_ip': '192.168.0.10',
        'port': 80,                         # Web portu
        'severity': 'critical',             # Yüksek severity
        'blocked': 1,
        'timestamp': '2024-10-29 14:30:00'
    }

    print("🎯 Test Saldırısı:")
    print(f"  IP: {test_attack['source_ip']}")
    print(f"  Port: {test_attack['port']}")
    print(f"  Severity: {test_attack['severity']}")

    # Tahmin yap
    result = predictor.predict_single(test_attack)

    if 'error' in result:
        print(f"\n❌ Hata: {result['error']}")
    else:
        print(f"\n✅ Tahmin Sonucu:")
        print(f"  Saldırı Türü: {result['predicted_type']}")
        print(f"  Güven: {result['confidence']*100:.2f}%")
        print(f"  Risk Seviyesi: {result['risk_level'].upper()}")
        print(f"\n  Top 3 Tahmin:")
        for attack, prob in result['top_3_predictions']:
            print(f"    - {attack}: {prob*100:.2f}%")