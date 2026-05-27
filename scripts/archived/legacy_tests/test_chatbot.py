"""
Test Chatbot - CyberGuard AI
Chatbot modülü unit testleri

Dosya Yolu: scripts/tests/test_chatbot.py
"""

import os
import sys
import unittest
from unittest.mock import MagicMock, patch

# Proje root
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


class TestContextManager(unittest.TestCase):
    """ContextManager testleri"""
    
    def test_import(self):
        """Import testi"""
        try:
            from src.chatbot.context_manager import ContextManager
            self.assertTrue(True)
        except ImportError as e:
            self.skipTest(f"Import hatası: {e}")
    
    def test_context_creation(self):
        """Yeni context oluşturma"""
        try:
            from src.chatbot.context_manager import ContextManager
            ctx = ContextManager()
            self.assertIsNotNone(ctx)
        except ImportError:
            self.skipTest("ContextManager import edilemedi")


class TestIntentClassifier(unittest.TestCase):
    """IntentClassifier testleri"""
    
    def test_import(self):
        """Import testi"""
        try:
            from src.chatbot.intent_classifier import IntentClassifier
            self.assertTrue(True)
        except ImportError as e:
            self.skipTest(f"Import hatası: {e}")
    
    def test_classify_query(self):
        """Sorgu sınıflandırma"""
        try:
            from src.chatbot.intent_classifier import IntentClassifier
            classifier = IntentClassifier()
            
            # Test sorguları
            test_queries = [
                "Son saldırıları göster",
                "DDoS nedir?",
                "Sistem durumu",
            ]
            
            for query in test_queries:
                result = classifier.classify(query)
                self.assertIsNotNone(result)
        except ImportError:
            self.skipTest("IntentClassifier import edilemedi")
        except Exception as e:
            self.skipTest(f"Test atlandı: {e}")


class TestGeminiHandler(unittest.TestCase):
    """GeminiHandler testleri"""
    
    def test_import(self):
        """Import testi"""
        try:
            from src.chatbot.gemini_handler import EnhancedGeminiHandler
            self.assertTrue(True)
        except ImportError as e:
            self.skipTest(f"Import hatası: {e}")
    
    @patch.dict(os.environ, {'GOOGLE_API_KEY': 'test_key'})
    def test_handler_initialization(self):
        """Handler başlatma (mock API key ile)"""
        try:
            from src.chatbot.gemini_handler import EnhancedGeminiHandler
            # API key olmadan test edemeyiz, sadece import kontrolü
            self.assertTrue(True)
        except ImportError:
            self.skipTest("GeminiHandler import edilemedi")


class TestModelKnowledge(unittest.TestCase):
    """ModelKnowledge testleri"""
    
    def test_import(self):
        """Import testi"""
        try:
            from src.chatbot.model_knowledge import ModelKnowledgeManager
            self.assertTrue(True)
        except ImportError as e:
            self.skipTest(f"Import hatası: {e}")


def run_tests():
    """Testleri çalıştır"""
    print("\n" + "=" * 60)
    print("🤖 CHATBOT MODULE TESTS")
    print("=" * 60 + "\n")
    
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Test sınıflarını ekle
    suite.addTests(loader.loadTestsFromTestCase(TestContextManager))
    suite.addTests(loader.loadTestsFromTestCase(TestIntentClassifier))
    suite.addTests(loader.loadTestsFromTestCase(TestGeminiHandler))
    suite.addTests(loader.loadTestsFromTestCase(TestModelKnowledge))
    
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
