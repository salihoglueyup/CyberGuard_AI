"""
Network Model Evaluator - CyberGuard AI
Ağ anomali tespit modeli değerlendirmesi

Dosya Yolu: src/network_detection/evaluator.py
"""

import numpy as np
from typing import Dict, List, Optional
from datetime import datetime

try:
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
        confusion_matrix, classification_report
    )
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


class NetworkEvaluator:
    """
    Network anomaly detection model değerlendirici
    
    Multi-class metrikler ve saldırı türü analizi
    """
    
    ATTACK_TYPES = ['Normal', 'DDoS', 'SQL Injection', 'XSS', 'Port Scan', 'Brute Force']
    
    def __init__(self):
        """Evaluator başlat"""
        self.evaluation_history: List[Dict] = []
        print("📊 Network Evaluator başlatıldı")
    
    def evaluate(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_pred_proba: Optional[np.ndarray] = None
    ) -> Dict:
        """
        Model performansını değerlendir
        
        Args:
            y_true: Gerçek etiketler
            y_pred: Tahminler
            y_pred_proba: Tahmin olasılıkları
            
        Returns:
            Metrik dictionary
        """
        if not SKLEARN_AVAILABLE:
            raise RuntimeError("scikit-learn yüklü değil!")
        
        metrics = {
            'timestamp': datetime.now().isoformat(),
            'sample_count': len(y_true),
            'num_classes': len(np.unique(y_true))
        }
        
        # Genel metrikler
        metrics['accuracy'] = float(accuracy_score(y_true, y_pred))
        metrics['precision_macro'] = float(precision_score(y_true, y_pred, average='macro', zero_division=0))
        metrics['recall_macro'] = float(recall_score(y_true, y_pred, average='macro', zero_division=0))
        metrics['f1_macro'] = float(f1_score(y_true, y_pred, average='macro', zero_division=0))
        
        # Weighted metrikler
        metrics['precision_weighted'] = float(precision_score(y_true, y_pred, average='weighted', zero_division=0))
        metrics['recall_weighted'] = float(recall_score(y_true, y_pred, average='weighted', zero_division=0))
        metrics['f1_weighted'] = float(f1_score(y_true, y_pred, average='weighted', zero_division=0))
        
        # Per-class metrikler
        precision_per = precision_score(y_true, y_pred, average=None, zero_division=0)
        recall_per = recall_score(y_true, y_pred, average=None, zero_division=0)
        f1_per = f1_score(y_true, y_pred, average=None, zero_division=0)
        
        metrics['per_class'] = {}
        for i, attack_type in enumerate(self.ATTACK_TYPES[:len(precision_per)]):
            metrics['per_class'][attack_type] = {
                'precision': float(precision_per[i]),
                'recall': float(recall_per[i]),
                'f1_score': float(f1_per[i])
            }
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        metrics['confusion_matrix'] = cm.tolist()
        
        # Attack detection rate (normal dışındakiler)
        attack_true = (y_true > 0).astype(int)
        attack_pred = (y_pred > 0).astype(int)
        metrics['attack_detection_rate'] = float(recall_score(attack_true, attack_pred, zero_division=0))
        metrics['false_alarm_rate'] = float(1 - precision_score(attack_true, attack_pred, zero_division=0))
        
        # Geçmişe ekle
        self.evaluation_history.append(metrics)
        
        return metrics
    
    def print_report(self, metrics: Dict) -> None:
        """Raporu yazdır"""
        print("\n" + "=" * 60)
        print("📊 NETWORK ANOMALY DETECTION - DEĞERLENDIRME RAPORU")
        print("=" * 60)
        
        print(f"\n📈 Genel Metrikler:")
        print(f"   Accuracy:           {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
        print(f"   Precision (macro):  {metrics['precision_macro']:.4f}")
        print(f"   Recall (macro):     {metrics['recall_macro']:.4f}")
        print(f"   F1-Score (macro):   {metrics['f1_macro']:.4f}")
        
        print(f"\n🎯 Saldırı Tespiti:")
        print(f"   Detection Rate:     {metrics['attack_detection_rate']:.4f}")
        print(f"   False Alarm Rate:   {metrics['false_alarm_rate']:.4f}")
        
        print(f"\n📋 Sınıf Bazlı Performans:")
        for attack_type, scores in metrics.get('per_class', {}).items():
            print(f"   {attack_type:15s} | P: {scores['precision']:.3f} | R: {scores['recall']:.3f} | F1: {scores['f1_score']:.3f}")
        
        print("=" * 60)


# Test
if __name__ == "__main__":
    if SKLEARN_AVAILABLE:
        print("🧪 Network Evaluator Test\n")
        
        np.random.seed(42)
        y_true = np.random.randint(0, 6, 100)
        y_pred = y_true.copy()
        y_pred[:15] = np.random.randint(0, 6, 15)  # %15 hata
        
        evaluator = NetworkEvaluator()
        metrics = evaluator.evaluate(y_true, y_pred)
        evaluator.print_report(metrics)
    else:
        print("❌ Test için scikit-learn gerekli!")
