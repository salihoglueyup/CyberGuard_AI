"""
Makale Sonuçları Görselleştirme Scripti
CyberGuard AI - SSA-LSTMIDS Implementasyonu

Bu script referans makale (Scientific Reports 2025) ile
elde ettiğimiz sonuçları karşılaştırmalı görselleştirir.
"""

import matplotlib.pyplot as plt
import numpy as np
import json
import os

# Türkçe karakter desteği
plt.rcParams["font.family"] = "DejaVu Sans"

# ==================== VERİLER ====================

# Referans Makale Sonuçları (Scientific Reports 2025 - SSA-LSTMIDS)
MAKALE_SONUCLARI = {
    "NSL-KDD": {"accuracy": 99.36, "precision": 99.37, "recall": 99.36, "f1": 99.36},
    "CICIDS2017": {"accuracy": 99.88, "precision": 99.89, "recall": 99.88, "f1": 99.88},
    "BoT-IoT": {"accuracy": 99.99, "precision": 99.99, "recall": 99.99, "f1": 99.99},
}

# Bizim Sonuçlarımız (CyberGuard AI - Gerçek Eğitim Sonuçları)
BIZIM_SONUCLARIMIZ = {
    "NSL-KDD": {"accuracy": 94.76, "precision": 94.10, "recall": 94.76, "f1": 94.39},
    "CICIDS2017": {"accuracy": 99.78, "precision": 99.77, "recall": 99.78, "f1": 99.75},
    "BoT-IoT": {"accuracy": 99.97, "precision": 99.97, "recall": 99.97, "f1": 99.97},
}

# Model Karşılaştırması (Projemizdeki Tüm Modeller)
MODEL_KARSILASTIRMA = {
    "SSA-LSTMIDS": {"accuracy": 99.78, "f1": 99.75},
    "BiLSTM+Attention": {"accuracy": 99.77, "f1": 99.74},
    "Transformer IDS": {"accuracy": 99.78, "f1": 99.75},
    "GRU IDS": {"accuracy": 99.76, "f1": 99.73},
}

# ==================== GRAFİKLER ====================


def plot_accuracy_comparison():
    """Accuracy karşılaştırma bar chart"""
    datasets = list(MAKALE_SONUCLARI.keys())
    makale_acc = [MAKALE_SONUCLARI[d]["accuracy"] for d in datasets]
    bizim_acc = [BIZIM_SONUCLARIMIZ[d]["accuracy"] for d in datasets]

    x = np.arange(len(datasets))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(
        x - width / 2, makale_acc, width, label="Referans Makale", color="#3498db"
    )
    bars2 = ax.bar(
        x + width / 2, bizim_acc, width, label="CyberGuard AI", color="#e74c3c"
    )

    ax.set_xlabel("Veri Kümesi", fontsize=12)
    ax.set_ylabel("Doğruluk (%)", fontsize=12)
    ax.set_title(
        "SSA-LSTMIDS: Makale vs Bizim Sonuçlar - Doğruluk Karşılaştırması", fontsize=14
    )
    ax.set_xticks(x)
    ax.set_xticklabels(datasets)
    ax.set_ylim(90, 101)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    # Değerleri bar üzerine yaz
    for bar in bars1:
        height = bar.get_height()
        ax.annotate(
            f"{height:.2f}%",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            fontsize=9,
        )
    for bar in bars2:
        height = bar.get_height()
        ax.annotate(
            f"{height:.2f}%",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            fontsize=9,
        )

    plt.tight_layout()
    plt.savefig("article/gorsel_1_accuracy_karsilastirma.png", dpi=150)
    print("✅ Kaydedildi: gorsel_1_accuracy_karsilastirma.png")
    plt.close()


def plot_f1_comparison():
    """F1-Score karşılaştırma bar chart"""
    datasets = list(MAKALE_SONUCLARI.keys())
    makale_f1 = [MAKALE_SONUCLARI[d]["f1"] for d in datasets]
    bizim_f1 = [BIZIM_SONUCLARIMIZ[d]["f1"] for d in datasets]

    x = np.arange(len(datasets))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(
        x - width / 2, makale_f1, width, label="Referans Makale", color="#2ecc71"
    )
    bars2 = ax.bar(
        x + width / 2, bizim_f1, width, label="CyberGuard AI", color="#9b59b6"
    )

    ax.set_xlabel("Veri Kümesi", fontsize=12)
    ax.set_ylabel("F1-Skoru (%)", fontsize=12)
    ax.set_title(
        "SSA-LSTMIDS: Makale vs Bizim Sonuçlar - F1-Skoru Karşılaştırması", fontsize=14
    )
    ax.set_xticks(x)
    ax.set_xticklabels(datasets)
    ax.set_ylim(90, 101)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    for bar in bars1:
        height = bar.get_height()
        ax.annotate(
            f"{height:.2f}%",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            fontsize=9,
        )
    for bar in bars2:
        height = bar.get_height()
        ax.annotate(
            f"{height:.2f}%",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            fontsize=9,
        )

    plt.tight_layout()
    plt.savefig("article/gorsel_2_f1_karsilastirma.png", dpi=150)
    print("✅ Kaydedildi: gorsel_2_f1_karsilastirma.png")
    plt.close()


def plot_model_comparison():
    """Projemizdeki modellerin karşılaştırması"""
    models = list(MODEL_KARSILASTIRMA.keys())
    accuracy = [MODEL_KARSILASTIRMA[m]["accuracy"] for m in models]
    f1 = [MODEL_KARSILASTIRMA[m]["f1"] for m in models]

    x = np.arange(len(models))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width / 2, accuracy, width, label="Accuracy", color="#1abc9c")
    bars2 = ax.bar(x + width / 2, f1, width, label="F1-Score", color="#f39c12")

    ax.set_xlabel("Model", fontsize=12)
    ax.set_ylabel("Performans (%)", fontsize=12)
    ax.set_title(
        "CyberGuard AI: Model Performans Karşılaştırması (CICIDS2017)", fontsize=14
    )
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=15)
    ax.set_ylim(99.5, 100)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    for bar in bars1:
        height = bar.get_height()
        ax.annotate(
            f"{height:.2f}%",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            fontsize=9,
        )
    for bar in bars2:
        height = bar.get_height()
        ax.annotate(
            f"{height:.2f}%",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            fontsize=9,
        )

    plt.tight_layout()
    plt.savefig("article/gorsel_3_model_karsilastirma.png", dpi=150)
    print("✅ Kaydedildi: gorsel_3_model_karsilastirma.png")
    plt.close()


def plot_metrics_radar():
    """Radar chart - Tüm metrikler"""
    categories = ["Accuracy", "Precision", "Recall", "F1-Score"]

    # CICIDS2017 için değerler
    makale_values = [99.88, 99.89, 99.88, 99.88]
    bizim_values = [99.78, 99.77, 99.78, 99.75]

    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]
    makale_values += makale_values[:1]
    bizim_values += bizim_values[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection="polar"))

    ax.plot(
        angles,
        makale_values,
        "o-",
        linewidth=2,
        label="Referans Makale",
        color="#3498db",
    )
    ax.fill(angles, makale_values, alpha=0.25, color="#3498db")
    ax.plot(
        angles, bizim_values, "o-", linewidth=2, label="CyberGuard AI", color="#e74c3c"
    )
    ax.fill(angles, bizim_values, alpha=0.25, color="#e74c3c")

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=11)
    ax.set_ylim(99, 100)
    ax.set_title(
        "CICIDS2017: Performans Metrikleri Karşılaştırması", fontsize=14, pad=20
    )
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.0))

    plt.tight_layout()
    plt.savefig("article/gorsel_4_radar_chart.png", dpi=150)
    print("✅ Kaydedildi: gorsel_4_radar_chart.png")
    plt.close()


def print_comparison_table():
    """Karşılaştırma tablosunu yazdır"""
    print("\n" + "=" * 80)
    print("SSA-LSTMIDS SONUÇ KARŞILAŞTIRMASI")
    print("=" * 80)
    print(
        f"{'Veri Kümesi':<15} {'Metrik':<12} {'Makale':>12} {'Bizim':>12} {'Fark':>10}"
    )
    print("-" * 80)

    for dataset in MAKALE_SONUCLARI.keys():
        for metric in ["accuracy", "f1"]:
            makale = MAKALE_SONUCLARI[dataset][metric]
            bizim = BIZIM_SONUCLARIMIZ[dataset][metric]
            fark = bizim - makale
            metric_name = "Accuracy" if metric == "accuracy" else "F1-Score"
            print(
                f"{dataset:<15} {metric_name:<12} {makale:>11.2f}% {bizim:>11.2f}% {fark:>+9.2f}%"
            )
        print("-" * 80)


if __name__ == "__main__":
    print("🎨 Makale Sonuçları Görselleştirme Başlıyor...")
    print()

    # Grafikleri oluştur
    plot_accuracy_comparison()
    plot_f1_comparison()
    plot_model_comparison()
    plot_metrics_radar()

    # Karşılaştırma tablosunu yazdır
    print_comparison_table()

    print("\n✅ Tüm görseller 'article/' klasörüne kaydedildi!")
    print("📋 Bu görselleri Word dosyasına yapıştırabilirsiniz.")
