"""
Random Forest Model - CyberGuard AI
Saldırı tespiti için Random Forest modeli
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)
import pickle
import os
from typing import Tuple, Dict
import matplotlib.pyplot as plt
import seaborn as sns


class CyberAttackModel:
    """Siber saldırı tespit modeli"""

    def __init__(self, n_estimators: int = 100, random_state: int = 42):
        """
        Args:
            n_estimators (int): Ağaç sayısı
            random_state (int): Random seed
        """
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=random_state,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            n_jobs=-1  # Tüm CPU'ları kullan
        )
        self.is_trained = False
        self.feature_importance = None
        self.classes = None

    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        """
        Modeli eğit

        Args:
            X_train: Eğitim özellikleri
            y_train: Eğitim etiketleri
        """
        print("🔄 Model eğitiliyor...")
        self.model.fit(X_train, y_train)
        self.is_trained = True
        self.feature_importance = self.model.feature_importances_
        self.classes = self.model.classes_
        print("✅ Model eğitimi tamamlandı!")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Tahmin yap

        Args:
            X: Özellikler

        Returns:
            np.ndarray: Tahminler
        """
        if not self.is_trained:
            raise ValueError("Model henüz eğitilmedi!")

        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Olasılıkları tahmin et

        Args:
            X: Özellikler

        Returns:
            np.ndarray: Olasılıklar
        """
        if not self.is_trained:
            raise ValueError("Model henüz eğitilmedi!")

        return self.model.predict_proba(X)

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict:
        """
        Modeli değerlendir

        Args:
            X_test: Test özellikleri
            y_test: Test etiketleri

        Returns:
            Dict: Metrikler
        """
        print("📊 Model değerlendiriliyor...")

        y_pred = self.predict(X_test)

        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
            'f1_score': f1_score(y_test, y_pred, average='weighted', zero_division=0)
        }

        print("\n📈 Model Performansı:")
        print(f"  Accuracy:  {metrics['accuracy']:.4f} ({metrics['accuracy'] * 100:.2f}%)")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall:    {metrics['recall']:.4f}")
        print(f"  F1-Score:  {metrics['f1_score']:.4f}")

        return metrics

    def get_confusion_matrix(self, X_test: np.ndarray,
                             y_test: np.ndarray,
                             class_names: list = None) -> np.ndarray:
        """
        Confusion matrix oluştur

        Args:
            X_test: Test özellikleri
            y_test: Test etiketleri
            class_names: Sınıf isimleri

        Returns:
            np.ndarray: Confusion matrix
        """
        y_pred = self.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)

        return cm

    def plot_confusion_matrix(self, X_test: np.ndarray,
                              y_test: np.ndarray,
                              class_names: list = None,
                              save_path: str = None):
        """
        Confusion matrix görselleştir

        Args:
            X_test: Test özellikleri
            y_test: Test etiketleri
            class_names: Sınıf isimleri
            save_path: Kayıt yolu
        """
        cm = self.get_confusion_matrix(X_test, y_test)

        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=class_names,
                    yticklabels=class_names)
        plt.title('Confusion Matrix - Saldırı Tespiti', fontsize=16, pad=20)
        plt.ylabel('Gerçek Değer', fontsize=12)
        plt.xlabel('Tahmin', fontsize=12)
        plt.tight_layout()

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ Confusion matrix kaydedildi: {save_path}")
        else:
            plt.show()

        plt.close()

    def get_feature_importance(self, feature_names: list = None) -> pd.DataFrame:
        """
        Feature importance'ları al

        Args:
            feature_names: Özellik isimleri

        Returns:
            DataFrame: Feature importance
        """
        if not self.is_trained:
            raise ValueError("Model henüz eğitilmedi!")

        if feature_names is None:
            feature_names = [f'feature_{i}' for i in range(len(self.feature_importance))]

        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': self.feature_importance
        }).sort_values('importance', ascending=False)

        return importance_df

    def plot_feature_importance(self, feature_names: list = None,
                                top_n: int = 10,
                                save_path: str = None):
        """
        Feature importance'ları görselleştir

        Args:
            feature_names: Özellik isimleri
            top_n: İlk kaç özellik
            save_path: Kayıt yolu
        """
        importance_df = self.get_feature_importance(feature_names)
        top_features = importance_df.head(top_n)

        plt.figure(figsize=(10, 6))
        sns.barplot(data=top_features, x='importance', y='feature', palette='viridis')
        plt.title(f'Top {top_n} Önemli Özellikler', fontsize=14, pad=15)
        plt.xlabel('Önem Skoru', fontsize=12)
        plt.ylabel('Özellik', fontsize=12)
        plt.tight_layout()

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ Feature importance kaydedildi: {save_path}")
        else:
            plt.show()

        plt.close()

    def save(self, filepath: str = 'models/rf_model.pkl'):
        """
        Modeli kaydet

        Args:
            filepath (str): Kayıt yolu
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        with open(filepath, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'is_trained': self.is_trained,
                'feature_importance': self.feature_importance,
                'classes': self.classes
            }, f)

        print(f"✅ Model kaydedildi: {filepath}")

    def load(self, filepath: str = 'models/rf_model.pkl'):
        """
        Modeli yükle

        Args:
            filepath (str): Dosya yolu
        """
        with open(filepath, 'rb') as f:
            data = pickle.load(f)

        self.model = data['model']
        self.is_trained = data['is_trained']
        self.feature_importance = data['feature_importance']
        self.classes = data['classes']

        print(f"✅ Model yüklendi: {filepath}")


# Test
if __name__ == "__main__":
    print("🧪 Random Forest Model Test")
    print("=" * 50)

    # Dummy data
    from sklearn.datasets import make_classification

    X, y = make_classification(n_samples=1000, n_features=10,
                               n_classes=5, n_informative=8,
                               random_state=42)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Model oluştur ve eğit
    model = CyberAttackModel(n_estimators=50)
    model.train(X_train, y_train)

    # Değerlendir
    metrics = model.evaluate(X_test, y_test)

    # Feature importance
    print("\n📊 Feature Importance:")
    importance_df = model.get_feature_importance()
    print(importance_df.head())

    # Kaydet
    model.save('models/rf_model_test.pkl')

    print("\n✅ Test tamamlandı!")