"""
Model Evaluator - CyberGuard AI
Detaylı model değerlendirme ve metrik hesaplama
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
    roc_curve, auc, roc_auc_score,
    precision_recall_curve, average_precision_score,
    matthews_corrcoef, cohen_kappa_score
)
from sklearn.preprocessing import label_binarize
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, List, Tuple, Optional
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


class ModelEvaluator:
    """
    Comprehensive model evaluation toolkit

    Özellikler:
    - Çoklu metrik hesaplama
    - Görselleştirme
    - Per-class analiz
    - Cross-validation
    - Model karşılaştırma
    - Detaylı raporlama
    """

    def __init__(self, class_names: List[str] = None):
        """
        Args:
            class_names: Sınıf isimleri
        """
        self.class_names = class_names
        self.evaluation_results = {}

        print("📊 Model Evaluator başlatıldı")

    def calculate_all_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_pred_proba: np.ndarray = None
    ) -> Dict:
        """
        Tüm metrikleri hesapla

        Args:
            y_true: Gerçek etiketler
            y_pred: Tahmin edilen etiketler
            y_pred_proba: Tahmin olasılıkları

        Returns:
            Metrik dictionary
        """
        print("\n📊 Metrikler hesaplanıyor...")

        metrics = {}

        # Basic metrics
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['precision_weighted'] = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        metrics['recall_weighted'] = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        metrics['f1_weighted'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)

        # Macro averages
        metrics['precision_macro'] = precision_score(y_true, y_pred, average='macro', zero_division=0)
        metrics['recall_macro'] = recall_score(y_true, y_pred, average='macro', zero_division=0)
        metrics['f1_macro'] = f1_score(y_true, y_pred, average='macro', zero_division=0)

        # Micro averages
        metrics['precision_micro'] = precision_score(y_true, y_pred, average='micro', zero_division=0)
        metrics['recall_micro'] = recall_score(y_true, y_pred, average='micro', zero_division=0)
        metrics['f1_micro'] = f1_score(y_true, y_pred, average='micro', zero_division=0)

        # Per-class metrics
        precision_per_class = precision_score(y_true, y_pred, average=None, zero_division=0)
        recall_per_class = recall_score(y_true, y_pred, average=None, zero_division=0)
        f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)

        metrics['precision_per_class'] = precision_per_class.tolist()
        metrics['recall_per_class'] = recall_per_class.tolist()
        metrics['f1_per_class'] = f1_per_class.tolist()

        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        metrics['confusion_matrix'] = cm.tolist()

        # Matthews Correlation Coefficient
        try:
            metrics['mcc'] = matthews_corrcoef(y_true, y_pred)
        except:
            metrics['mcc'] = None

        # Cohen's Kappa
        try:
            metrics['cohen_kappa'] = cohen_kappa_score(y_true, y_pred)
        except:
            metrics['cohen_kappa'] = None

        # ROC-AUC (multi-class)
        if y_pred_proba is not None:
            try:
                # One-vs-Rest ROC-AUC
                metrics['roc_auc_ovr'] = roc_auc_score(
                    y_true, y_pred_proba,
                    multi_class='ovr',
                    average='weighted'
                )

                # One-vs-One ROC-AUC
                metrics['roc_auc_ovo'] = roc_auc_score(
                    y_true, y_pred_proba,
                    multi_class='ovo',
                    average='weighted'
                )

                # Per-class ROC-AUC
                n_classes = y_pred_proba.shape[1]
                y_true_bin = label_binarize(y_true, classes=range(n_classes))

                roc_auc_per_class = []
                for i in range(n_classes):
                    try:
                        auc_score = roc_auc_score(y_true_bin[:, i], y_pred_proba[:, i])
                        roc_auc_per_class.append(auc_score)
                    except:
                        roc_auc_per_class.append(0.0)

                metrics['roc_auc_per_class'] = roc_auc_per_class

            except Exception as e:
                print(f"⚠️  ROC-AUC hesaplanamadı: {e}")
                metrics['roc_auc_ovr'] = None
                metrics['roc_auc_ovo'] = None
                metrics['roc_auc_per_class'] = None

        # Classification report
        report = classification_report(
            y_true, y_pred,
            target_names=self.class_names,
            digits=4,
            zero_division=0,
            output_dict=True
        )
        metrics['classification_report'] = report

        print("✅ Metrikler hesaplandı")

        return metrics

    def calculate_confusion_matrix_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> Dict:
        """
        Confusion matrix'ten detaylı metrikler

        Args:
            y_true: Gerçek etiketler
            y_pred: Tahminler

        Returns:
            Metrik dictionary
        """
        cm = confusion_matrix(y_true, y_pred)

        metrics = {}
        n_classes = len(cm)

        # Her sınıf için
        for i in range(n_classes):
            class_name = self.class_names[i] if self.class_names else f"Class_{i}"

            tp = cm[i, i]
            fp = cm[:, i].sum() - tp
            fn = cm[i, :].sum() - tp
            tn = cm.sum() - tp - fp - fn

            # Metrics
            accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

            metrics[class_name] = {
                'tp': int(tp),
                'fp': int(fp),
                'fn': int(fn),
                'tn': int(tn),
                'accuracy': float(accuracy),
                'precision': float(precision),
                'recall': float(recall),
                'specificity': float(specificity),
                'f1_score': float(f1)
            }

        return metrics

    def plot_confusion_matrix(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        save_path: str = 'confusion_matrix.png',
        figsize: Tuple[int, int] = (12, 10),
        normalize: bool = False
    ):
        """
        Confusion matrix görselleştir

        Args:
            y_true: Gerçek etiketler
            y_pred: Tahminler
            save_path: Kayıt yolu
            figsize: Figure boyutu
            normalize: Normalize et
        """
        cm = confusion_matrix(y_true, y_pred)

        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            fmt = '.2%'
            title = 'Normalized Confusion Matrix'
        else:
            fmt = 'd'
            title = 'Confusion Matrix'

        plt.figure(figsize=figsize)
        sns.heatmap(
            cm,
            annot=True,
            fmt=fmt,
            cmap='Blues',
            xticklabels=self.class_names if self.class_names else range(len(cm)),
            yticklabels=self.class_names if self.class_names else range(len(cm)),
            cbar_kws={'label': 'Count' if not normalize else 'Proportion'}
        )

        plt.title(title, fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('Predicted Label', fontsize=12, fontweight='bold')
        plt.ylabel('True Label', fontsize=12, fontweight='bold')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Confusion matrix kaydedildi: {save_path}")
        plt.close()

    def plot_roc_curves(
        self,
        y_true: np.ndarray,
        y_pred_proba: np.ndarray,
        save_path: str = 'roc_curves.png',
        figsize: Tuple[int, int] = (12, 10)
    ):
        """
        ROC curves görselleştir (multi-class)

        Args:
            y_true: Gerçek etiketler
            y_pred_proba: Tahmin olasılıkları
            save_path: Kayıt yolu
            figsize: Figure boyutu
        """
        n_classes = y_pred_proba.shape[1]
        y_true_bin = label_binarize(y_true, classes=range(n_classes))

        plt.figure(figsize=figsize)

        # Her sınıf için ROC curve
        for i in range(n_classes):
            fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_pred_proba[:, i])
            roc_auc = auc(fpr, tpr)

            class_name = self.class_names[i] if self.class_names else f"Class {i}"

            plt.plot(
                fpr, tpr,
                label=f'{class_name} (AUC = {roc_auc:.3f})',
                linewidth=2
            )

        # Diagonal line
        plt.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Random (AUC = 0.500)')

        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12, fontweight='bold')
        plt.ylabel('True Positive Rate', fontsize=12, fontweight='bold')
        plt.title('ROC Curves - Multi-Class', fontsize=16, fontweight='bold', pad=20)
        plt.legend(loc='lower right', fontsize=10)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ ROC curves kaydedildi: {save_path}")
        plt.close()

    def plot_precision_recall_curves(
        self,
        y_true: np.ndarray,
        y_pred_proba: np.ndarray,
        save_path: str = 'precision_recall_curves.png',
        figsize: Tuple[int, int] = (12, 10)
    ):
        """
        Precision-Recall curves

        Args:
            y_true: Gerçek etiketler
            y_pred_proba: Tahmin olasılıkları
            save_path: Kayıt yolu
            figsize: Figure boyutu
        """
        n_classes = y_pred_proba.shape[1]
        y_true_bin = label_binarize(y_true, classes=range(n_classes))

        plt.figure(figsize=figsize)

        for i in range(n_classes):
            precision, recall, _ = precision_recall_curve(y_true_bin[:, i], y_pred_proba[:, i])
            avg_precision = average_precision_score(y_true_bin[:, i], y_pred_proba[:, i])

            class_name = self.class_names[i] if self.class_names else f"Class {i}"

            plt.plot(
                recall, precision,
                label=f'{class_name} (AP = {avg_precision:.3f})',
                linewidth=2
            )

        plt.xlabel('Recall', fontsize=12, fontweight='bold')
        plt.ylabel('Precision', fontsize=12, fontweight='bold')
        plt.title('Precision-Recall Curves', fontsize=16, fontweight='bold', pad=20)
        plt.legend(loc='best', fontsize=10)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Precision-Recall curves kaydedildi: {save_path}")
        plt.close()

    def plot_metric_comparison(
        self,
        metrics: Dict,
        save_path: str = 'metric_comparison.png',
        figsize: Tuple[int, int] = (14, 8)
    ):
        """
        Per-class metrik karşılaştırması

        Args:
            metrics: Metrik dictionary
            save_path: Kayıt yolu
            figsize: Figure boyutu
        """
        if not self.class_names:
            print("⚠️  class_names tanımlı değil!")
            return

        # Veri hazırla
        precision = metrics['precision_per_class']
        recall = metrics['recall_per_class']
        f1 = metrics['f1_per_class']

        x = np.arange(len(self.class_names))
        width = 0.25

        fig, ax = plt.subplots(figsize=figsize)

        bars1 = ax.bar(x - width, precision, width, label='Precision', color='#667eea', alpha=0.8)
        bars2 = ax.bar(x, recall, width, label='Recall', color='#764ba2', alpha=0.8)
        bars3 = ax.bar(x + width, f1, width, label='F1-Score', color='#f093fb', alpha=0.8)

        # Değerleri bar üzerine yaz
        def autolabel(bars):
            for bar in bars:
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}',
                    ha='center', va='bottom', fontsize=8
                )

        autolabel(bars1)
        autolabel(bars2)
        autolabel(bars3)

        ax.set_xlabel('Attack Type', fontsize=12, fontweight='bold')
        ax.set_ylabel('Score', fontsize=12, fontweight='bold')
        ax.set_title('Per-Class Performance Metrics', fontsize=16, fontweight='bold', pad=20)
        ax.set_xticks(x)
        ax.set_xticklabels(self.class_names, rotation=45, ha='right')
        ax.legend(fontsize=10)
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim([0, 1.1])

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Metric comparison kaydedildi: {save_path}")
        plt.close()

    def generate_evaluation_report(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_pred_proba: np.ndarray = None,
        model_name: str = "Model",
        output_dir: str = "models/evaluation"
    ) -> Dict:
        """
        Kapsamlı değerlendirme raporu oluştur

        Args:
            y_true: Gerçek etiketler
            y_pred: Tahminler
            y_pred_proba: Tahmin olasılıkları
            model_name: Model adı
            output_dir: Çıktı dizini

        Returns:
            Evaluation sonuçları
        """
        import os
        os.makedirs(output_dir, exist_ok=True)

        print("\n" + "="*70)
        print(f"📊 {model_name.upper()} - KAPSAMLI DEĞERLENDİRME")
        print("="*70)

        # 1. Tüm metrikleri hesapla
        metrics = self.calculate_all_metrics(y_true, y_pred, y_pred_proba)

        # 2. Confusion matrix metrics
        cm_metrics = self.calculate_confusion_matrix_metrics(y_true, y_pred)

        # 3. Görselleştirmeler
        print("\n📊 Görselleştirmeler oluşturuluyor...")

        # Confusion Matrix
        self.plot_confusion_matrix(
            y_true, y_pred,
            save_path=f"{output_dir}/{model_name}_confusion_matrix.png"
        )

        # Normalized Confusion Matrix
        self.plot_confusion_matrix(
            y_true, y_pred,
            save_path=f"{output_dir}/{model_name}_confusion_matrix_normalized.png",
            normalize=True
        )

        # Metric Comparison
        self.plot_metric_comparison(
            metrics,
            save_path=f"{output_dir}/{model_name}_metric_comparison.png"
        )

        if y_pred_proba is not None:
            # ROC Curves
            self.plot_roc_curves(
                y_true, y_pred_proba,
                save_path=f"{output_dir}/{model_name}_roc_curves.png"
            )

            # Precision-Recall Curves
            self.plot_precision_recall_curves(
                y_true, y_pred_proba,
                save_path=f"{output_dir}/{model_name}_pr_curves.png"
            )

        # 4. Sonuçları kaydet
        results = {
            'model_name': model_name,
            'evaluation_date': datetime.now().isoformat(),
            'metrics': metrics,
            'confusion_matrix_metrics': cm_metrics,
            'summary': {
                'accuracy': metrics['accuracy'],
                'precision': metrics['precision_weighted'],
                'recall': metrics['recall_weighted'],
                'f1_score': metrics['f1_weighted'],
                'roc_auc': metrics.get('roc_auc_ovr', None)
            }
        }

        # JSON olarak kaydet
        with open(f"{output_dir}/{model_name}_evaluation_report.json", 'w') as f:
            json.dump(results, f, indent=4)

        # Özet yazdır
        print("\n" + "="*70)
        print("📋 DEĞERLENDİRME ÖZETİ")
        print("="*70)
        print(f"Accuracy:  {metrics['accuracy']*100:.2f}%")
        print(f"Precision: {metrics['precision_weighted']*100:.2f}%")
        print(f"Recall:    {metrics['recall_weighted']*100:.2f}%")
        print(f"F1-Score:  {metrics['f1_weighted']*100:.2f}%")
        if metrics.get('roc_auc_ovr'):
            print(f"ROC-AUC:   {metrics['roc_auc_ovr']*100:.2f}%")
        if metrics.get('mcc'):
            print(f"MCC:       {metrics['mcc']:.4f}")
        if metrics.get('cohen_kappa'):
            print(f"Cohen's κ: {metrics['cohen_kappa']:.4f}")
        print("="*70)

        print(f"\n✅ Değerlendirme raporu kaydedildi: {output_dir}/")

        return results


# Test
if __name__ == "__main__":
    print("🧪 Model Evaluator Test\n")

    # Örnek veri
    np.random.seed(42)
    y_true = np.random.randint(0, 5, 200)
    y_pred = y_true.copy()
    # %10 hata ekle
    errors = np.random.choice(200, 20, replace=False)
    y_pred[errors] = np.random.randint(0, 5, 20)

    # Probability predictions
    n_classes = 5
    y_pred_proba = np.random.rand(200, n_classes)
    y_pred_proba = y_pred_proba / y_pred_proba.sum(axis=1, keepdims=True)

    class_names = ['DDoS', 'SQL Injection', 'XSS', 'Port Scan', 'Brute Force']

    # Evaluator
    evaluator = ModelEvaluator(class_names=class_names)

    # Rapor oluştur
    results = evaluator.generate_evaluation_report(
        y_true, y_pred, y_pred_proba,
        model_name="TestModel",
        output_dir="test_evaluation"
    )

    print("\n✅ Test tamamlandı!")