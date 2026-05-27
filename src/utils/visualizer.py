# src/utils/visualizer.py

"""
Visualization utilities
Grafik ve görselleştirme fonksiyonları
"""

import warnings
from datetime import datetime, timedelta

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Plotly'yi opsiyonel yap
try:
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    warnings.warn("Plotly not available. Using Matplotlib only.")


class Visualizer:
    """
    CyberGuard AI Visualizer

    Özellikleri:
    - Matplotlib grafikleri (her zaman çalışır)
    - Plotly grafikleri (eğer kuruluysa)
    - Attack trend charts
    - Model performance metrics
    - Real-time dashboard components
    """

    # Renk paletleri
    COLORS = {
        'primary': '#1f77b4',
        'success': '#2ca02c',
        'warning': '#ff7f0e',
        'danger': '#d62728',
        'info': '#17becf',
        'dark': '#2c3e50',
        'light': '#ecf0f1',
    }

    # Severity renkleri
    SEVERITY_COLORS = {
        'LOW': '#3498db',  # Mavi
        'MEDIUM': '#f39c12',  # Turuncu
        'HIGH': '#e74c3c',  # Kırmızı
        'CRITICAL': '#8e44ad'  # Mor
    }

    # Attack type renkleri
    ATTACK_COLORS = {
        'DDoS': '#e74c3c',
        'Port Scan': '#3498db',
        'SQL Injection': '#9b59b6',
        'XSS': '#e67e22',
        'Brute Force': '#c0392b',
        'Malware': '#8e44ad',
        'Phishing': '#16a085',
        'MITM': '#d35400',
    }

    def __init__(self, theme: str = 'plotly_dark', use_plotly: bool = True):
        """
        Visualizer'ı başlat

        Args:
            theme (str): Plotly tema ('plotly', 'plotly_dark', 'seaborn', etc.)
            use_plotly (bool): Plotly kullan (False ise sadece Matplotlib)
        """
        self.theme = theme
        self.use_plotly = use_plotly and PLOTLY_AVAILABLE

        # Matplotlib style
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette("husl")

        if not self.use_plotly:
            print("⚠️  Using Matplotlib only (Plotly not available)")

    # ========================================
    # MATPLOTLIB VERSIONS
    # ========================================

    def plot_attack_timeline_mpl(self, attacks: list[dict],
                                 title: str = "Attack Timeline",
                                 figsize: tuple = (12, 6)) -> plt.figure:
        """
        Saldırı zaman çizelgesi (Matplotlib)

        Args:
            attacks: Saldırı listesi
            title: Grafik başlığı
            figsize: Figure boyutu

        Returns:
            matplotlib Figure
        """

        if not attacks:
            return self._create_empty_figure_mpl("No attack data available", figsize)

        # DataFrame'e çevir
        df = pd.DataFrame(attacks)
        df['timestamp'] = pd.to_datetime(df['timestamp'])

        # Saatlik gruplama
        df_hourly = df.groupby(df['timestamp'].dt.floor('H')).size().reset_index()
        df_hourly.columns = ['timestamp', 'count']

        # Grafik oluştur
        fig, ax = plt.subplots(figsize=figsize)

        ax.plot(df_hourly['timestamp'], df_hourly['count'],
                linewidth=2, marker='o', markersize=4,
                color=self.COLORS['danger'])
        ax.fill_between(df_hourly['timestamp'], df_hourly['count'],
                        alpha=0.3, color=self.COLORS['danger'])

        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.set_xlabel("Time", fontsize=12)
        ax.set_ylabel("Attack Count", fontsize=12)
        ax.grid(True, alpha=0.3)

        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()

        return fig

    def plot_attack_distribution_mpl(self, attacks: list[dict],
                                     by: str = 'attack_type',
                                     title: str = "Attack Distribution",
                                     figsize: tuple = (10, 8)) -> plt.figure:
        """
        Saldırı dağılımı (Matplotlib pie chart)

        Args:
            attacks: Saldırı listesi
            by: Gruplama anahtarı
            title: Grafik başlığı
            figsize: Figure boyutu

        Returns:
            matplotlib Figure
        """

        if not attacks:
            return self._create_empty_figure_mpl("No attack data available", figsize)

        # DataFrame'e çevir
        df = pd.DataFrame(attacks)

        # Gruplama
        counts = df[by].value_counts()

        # Renkler
        if by == 'severity':
            colors = [self.SEVERITY_COLORS.get(x, self.COLORS['primary'])
                      for x in counts.index]
        elif by == 'attack_type':
            colors = [self.ATTACK_COLORS.get(x, self.COLORS['primary'])
                      for x in counts.index]
        else:
            colors = sns.color_palette("husl", len(counts))

        # Pie chart
        fig, ax = plt.subplots(figsize=figsize)

        wedges, texts, autotexts = ax.pie(
            counts.values,
            labels=counts.index,
            autopct='%1.1f%%',
            colors=colors,
            startangle=90,
            textprops={'fontsize': 10}
        )

        # Daha iyi görünüm
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')

        ax.set_title(title, fontsize=16, fontweight='bold', pad=20)

        plt.tight_layout()

        return fig

    def plot_top_attackers_mpl(self, attacks: list[dict],
                               top_n: int = 10,
                               title: str = "Top Attacker IPs",
                               figsize: tuple = (10, 6)) -> plt.figure:
        """
        En çok saldıran IP'ler (Matplotlib bar chart)

        Args:
            attacks: Saldırı listesi
            top_n: İlk N IP
            title: Grafik başlığı
            figsize: Figure boyutu

        Returns:
            matplotlib Figure
        """

        if not attacks:
            return self._create_empty_figure_mpl("No attack data available", figsize)

        # DataFrame'e çevir
        df = pd.DataFrame(attacks)

        # IP'lere göre gruplama
        top_ips = df['source_ip'].value_counts().head(top_n)

        # Bar chart
        fig, ax = plt.subplots(figsize=figsize)

        bars = ax.barh(range(len(top_ips)), top_ips.values,
                       color=sns.color_palette("Reds_r", len(top_ips)))
        ax.set_yticks(range(len(top_ips)))
        ax.set_yticklabels(top_ips.index)

        # Değerleri barların üzerine yaz
        for i, (bar, value) in enumerate(zip(bars, top_ips.values)):
            ax.text(value, i, f' {value}',
                    va='center', fontweight='bold')

        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.set_xlabel("Attack Count", fontsize=12)
        ax.set_ylabel("Source IP", fontsize=12)
        ax.grid(axis='x', alpha=0.3)
        ax.invert_yaxis()

        plt.tight_layout()

        return fig

    def plot_confusion_matrix_mpl(self, cm: np.ndarray,
                                  labels: list[str],
                                  title: str = "Confusion Matrix",
                                  figsize: tuple = (10, 8)) -> plt.figure:
        """
        Confusion matrix (Matplotlib)

        Args:
            cm: Confusion matrix
            labels: Class labels
            title: Grafik başlığı
            figsize: Figure boyutu

        Returns:
            matplotlib Figure
        """

        fig, ax = plt.subplots(figsize=figsize)

        # Normalize
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm_normalized = np.nan_to_num(cm_normalized)

        # Heatmap
        im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
        ax.figure.colorbar(im, ax=ax)

        # Axes
        ax.set(xticks=np.arange(cm.shape[1]),
               yticks=np.arange(cm.shape[0]),
               xticklabels=labels,
               yticklabels=labels,
               title=title,
               ylabel='Actual',
               xlabel='Predicted')

        # Rotate the tick labels
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                 rotation_mode="anchor")

        # Text annotations
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, f"{cm[i, j]}\n({cm_normalized[i, j]:.1%})",
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black",
                        fontsize=9)

        plt.tight_layout()

        return fig

    def plot_training_history_mpl(self, history: dict,
                                  metrics: list[str] | None = None,
                                  title: str = "Training History",
                                  figsize: tuple = (14, 5)) -> plt.figure:
        if metrics is None:
            metrics = ['loss', 'accuracy']
        """
        Model eğitim geçmişi (Matplotlib)

        Args:
            history: Training history dict
            metrics: Gösterilecek metrikler
            title: Grafik başlığı
            figsize: Figure boyutu

        Returns:
            matplotlib Figure
        """

        fig, axes = plt.subplots(1, len(metrics), figsize=figsize)

        if len(metrics) == 1:
            axes = [axes]

        for ax, metric in zip(axes, metrics):
            # Training
            if metric in history:
                ax.plot(history[metric], label=f'Train {metric}',
                        linewidth=2, marker='o', markersize=3)

            # Validation
            val_metric = f'val_{metric}'
            if val_metric in history:
                ax.plot(history[val_metric], label=f'Val {metric}',
                        linewidth=2, marker='s', markersize=3, linestyle='--')

            ax.set_title(metric.capitalize(), fontsize=14, fontweight='bold')
            ax.set_xlabel("Epoch", fontsize=11)
            ax.set_ylabel(metric.capitalize(), fontsize=11)
            ax.legend()
            ax.grid(True, alpha=0.3)

        fig.suptitle(title, fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()

        return fig

    def _create_empty_figure_mpl(self, message: str = "No data",
                                 figsize: tuple = (10, 6)) -> plt.figure:
        """Boş grafik oluştur (Matplotlib)"""

        fig, ax = plt.subplots(figsize=figsize)

        ax.text(0.5, 0.5, message,
                ha='center', va='center',
                fontsize=20, color='gray',
                transform=ax.transAxes)

        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)

        return fig

    # ========================================
    # UNIFIED API (Matplotlib veya Plotly)
    # ========================================

    def plot_attack_timeline(self, attacks: list[dict], **kwargs):
        """Attack timeline (otomatik backend seçer)"""
        return self.plot_attack_timeline_mpl(attacks, **kwargs)

    def plot_attack_distribution(self, attacks: list[dict], **kwargs):
        """Attack distribution (otomatik backend seçer)"""
        return self.plot_attack_distribution_mpl(attacks, **kwargs)

    def plot_top_attackers(self, attacks: list[dict], **kwargs):
        """Top attackers (otomatik backend seçer)"""
        return self.plot_top_attackers_mpl(attacks, **kwargs)

    def plot_confusion_matrix(self, cm: np.ndarray, labels: list[str], **kwargs):
        """Confusion matrix (otomatik backend seçer)"""
        return self.plot_confusion_matrix_mpl(cm, labels, **kwargs)

    def plot_training_history(self, history: dict, **kwargs):
        """Training history (otomatik backend seçer)"""
        return self.plot_training_history_mpl(history, **kwargs)

    # ========================================
    # UTILITY FUNCTIONS
    # ========================================

    def save_figure(self, fig, filename: str, dpi: int = 300):
        """
        Grafiği dosyaya kaydet

        Args:
            fig: Matplotlib figure
            filename: Dosya adı
            dpi: DPI (resolution)
        """

        fig.savefig(filename, dpi=dpi, bbox_inches='tight')
        print(f"✅ Figure saved: {filename}")

    def close_all(self):
        """Tüm figürleri kapat"""
        plt.close('all')


# ========================================
# CONVENIENCE FUNCTIONS
# ========================================

def create_visualizer(theme: str = 'plotly_dark', use_plotly: bool = True) -> Visualizer:
    """
    Visualizer instance oluştur

    Args:
        theme: Plotly tema (kullanılmıyorsa göz ardı edilir)
        use_plotly: Plotly kullan (False ise sadece Matplotlib)

    Returns:
        Visualizer instance
    """
    return Visualizer(theme=theme, use_plotly=use_plotly)


# ========================================
# TEST
# ========================================

if __name__ == "__main__":
    print("🧪 Visualizer Test\n")
    print("=" * 60)

    # Visualizer oluştur
    viz = Visualizer(use_plotly=False)  # Matplotlib only

    # Test verisi oluştur
    print("\n📊 Test verileri oluşturuluyor...")

    # Saldırı verisi
    attacks = []
    attack_types = ['DDoS', 'Port Scan', 'SQL Injection', 'XSS', 'Brute Force']
    severities = ['LOW', 'MEDIUM', 'HIGH', 'CRITICAL']

    for i in range(100):
        attacks.append({
            'timestamp': datetime.now() - timedelta(hours=i),
            'attack_type': np.random.choice(attack_types),
            'source_ip': f"192.168.1.{np.random.randint(1, 255)}",
            'severity': np.random.choice(severities),
            'confidence': np.random.uniform(0.7, 0.99)
        })

    # Test 1: Attack Timeline
    print("\n1️⃣  Attack Timeline")
    fig1 = viz.plot_attack_timeline(attacks)
    plt.show()

    # Test 2: Attack Distribution
    print("\n2️⃣  Attack Distribution")
    fig2 = viz.plot_attack_distribution(attacks, by='attack_type')
    plt.show()

    # Test 3: Top Attackers
    print("\n3️⃣  Top Attackers")
    fig3 = viz.plot_top_attackers(attacks, top_n=5)
    plt.show()

    # Test 4: Confusion Matrix
    print("\n4️⃣  Confusion Matrix")
    cm = np.array([[50, 2, 1], [3, 45, 2], [1, 2, 48]])
    labels = ['Normal', 'Attack', 'Suspicious']
    fig4 = viz.plot_confusion_matrix(cm, labels)
    plt.show()

    # Test 5: Training History
    print("\n5️⃣  Training History")
    history = {
        'loss': [0.5, 0.4, 0.3, 0.25, 0.2],
        'accuracy': [0.7, 0.75, 0.8, 0.85, 0.9],
        'val_loss': [0.55, 0.45, 0.35, 0.3, 0.25],
        'val_accuracy': [0.65, 0.7, 0.75, 0.8, 0.85]
    }
    fig5 = viz.plot_training_history(history)
    plt.show()

    print("\n" + "=" * 60)
    print("✅ Visualizer test tamamlandı!")
