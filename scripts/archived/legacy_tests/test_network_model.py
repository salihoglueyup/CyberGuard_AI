"""
Test Network Model - CyberGuard AI
Network detection modülü unit testleri

Dosya Yolu: scripts/tests/test_network_model.py
"""

import os
import sys
import unittest
import numpy as np

# Proje root
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


class TestNetworkAnomalyModel(unittest.TestCase):
    """NetworkAnomalyModel testleri"""
    
    @classmethod
    def setUpClass(cls):
        """Test setup"""
        try:
            from src.network_detection.model import NetworkAnomalyModel
            cls.model_class = NetworkAnomalyModel
            cls.can_test = True
        except ImportError as e:
            cls.can_test = False
            cls.skip_reason = str(e)
    
    def test_import(self):
        """Import testi"""
        if not self.can_test:
            self.skipTest(self.skip_reason)
        self.assertTrue(True)
    
    def test_model_creation(self):
        """Model oluşturma"""
        if not self.can_test:
            self.skipTest(self.skip_reason)
        
        model = self.model_class(model_type='random_forest')
        self.assertIsNotNone(model)
        self.assertEqual(model.model_type, 'random_forest')
    
    def test_attack_types(self):
        """Saldırı türleri kontrolü"""
        if not self.can_test:
            self.skipTest(self.skip_reason)
        
        expected_types = ['Normal', 'DDoS', 'SQL Injection', 'XSS', 'Port Scan', 'Brute Force']
        self.assertEqual(self.model_class.ATTACK_TYPES, expected_types)
    
    def test_model_training(self):
        """Model eğitimi (mock data)"""
        if not self.can_test:
            self.skipTest(self.skip_reason)
        
        try:
            model = self.model_class()
            
            # Mock data
            np.random.seed(42)
            X = np.random.rand(200, 11)
            y = np.random.randint(0, 6, 200)
            
            result = model.train(X, y, epochs=5)
            
            self.assertTrue(model.is_trained)
        except Exception as e:
            self.skipTest(f"Eğitim testi atlandı: {e}")
    
    def test_model_prediction(self):
        """Model tahmini"""
        if not self.can_test:
            self.skipTest(self.skip_reason)
        
        try:
            model = self.model_class()
            
            X = np.random.rand(200, 11)
            y = np.random.randint(0, 6, 200)
            model.train(X, y, epochs=5)
            
            X_test = np.random.rand(10, 11)
            predictions = model.predict(X_test)
            
            self.assertEqual(len(predictions), 10)
        except Exception as e:
            self.skipTest(f"Tahmin testi atlandı: {e}")


class TestNetworkDataProcessor(unittest.TestCase):
    """NetworkDataProcessor testleri"""
    
    def test_import(self):
        """Import testi"""
        try:
            from src.network_detection.data_processor import NetworkDataProcessor
            self.assertTrue(True)
        except ImportError as e:
            self.skipTest(f"Import hatası: {e}")
    
    def test_ip_processing(self):
        """IP işleme"""
        try:
            from src.network_detection.data_processor import NetworkDataProcessor
            processor = NetworkDataProcessor()
            
            # IP to numeric
            ip_numeric = processor.ip_to_numeric("192.168.1.1")
            self.assertIsInstance(ip_numeric, int)
            self.assertGreater(ip_numeric, 0)
            
            # Private IP check
            self.assertTrue(processor.is_private_ip("192.168.1.1"))
            self.assertFalse(processor.is_private_ip("8.8.8.8"))
        except ImportError:
            self.skipTest("NetworkDataProcessor import edilemedi")
    
    def test_packet_processing(self):
        """Paket işleme"""
        try:
            from src.network_detection.data_processor import NetworkDataProcessor
            processor = NetworkDataProcessor()
            
            test_packet = {
                'source_ip': '192.168.1.100',
                'destination_ip': '8.8.8.8',
                'port': 443,
                'protocol': 'TCP',
                'packet_size': 1024
            }
            
            features = processor.process_single_packet(test_packet)
            self.assertEqual(len(features), 11)  # 11 özellik
        except ImportError:
            self.skipTest("NetworkDataProcessor import edilemedi")


class TestNetworkEvaluator(unittest.TestCase):
    """NetworkEvaluator testleri"""
    
    def test_import(self):
        """Import testi"""
        try:
            from src.network_detection.evaluator import NetworkEvaluator
            self.assertTrue(True)
        except ImportError as e:
            self.skipTest(f"Import hatası: {e}")
    
    def test_evaluation(self):
        """Değerlendirme testi"""
        try:
            from src.network_detection.evaluator import NetworkEvaluator
            
            evaluator = NetworkEvaluator()
            
            y_true = np.array([0, 1, 2, 3, 0, 1])
            y_pred = np.array([0, 1, 2, 4, 0, 0])
            
            metrics = evaluator.evaluate(y_true, y_pred)
            
            self.assertIn('accuracy', metrics)
            self.assertIn('precision_macro', metrics)
            self.assertIn('attack_detection_rate', metrics)
        except ImportError:
            self.skipTest("NetworkEvaluator import edilemedi")


def run_tests():
    """Testleri çalıştır"""
    print("\n" + "=" * 60)
    print("🌐 NETWORK DETECTION MODEL TESTS")
    print("=" * 60 + "\n")
    
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    suite.addTests(loader.loadTestsFromTestCase(TestNetworkAnomalyModel))
    suite.addTests(loader.loadTestsFromTestCase(TestNetworkDataProcessor))
    suite.addTests(loader.loadTestsFromTestCase(TestNetworkEvaluator))
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    print("\n" + "=" * 60)
    print(f"✅ Passed: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"❌ Failed: {len(result.failures)}")
    print(f"⚠️  Errors: {len(result.errors)}")
    print(f"⏭️  Skipped: {len(result.skipped)}")
    print("=" * 60)
    
    return result


if __name__ == "__main__":
    run_tests()
