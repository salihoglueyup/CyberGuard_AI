"""
Real-time IDS Test Script
Eğitilmiş modeli kullanarak canlı IDS testleri

Kullanım:
    python scripts/test_realtime_ids.py --model models/ssa_lstmids_nsl_kdd_*.h5
    python scripts/test_realtime_ids.py --simulate --packets 1000
"""

import os
import sys
import argparse
import time
import numpy as np
from pathlib import Path
from datetime import datetime

# Proje yolu
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


def find_latest_model():
    """En son eğitilmiş modeli bul"""
    models_dir = PROJECT_ROOT / "model_artifacts"

    # SSA-LSTMIDS modelleri
    models = list(models_dir.glob("ssa_lstmids_*.h5"))
    if not models:
        # Diğer modeller
        models = list(models_dir.glob("*.h5"))

    if not models:
        return None

    # En son oluşturulanı seç
    return max(models, key=lambda x: x.stat().st_mtime)


def alert_callback(alert):
    """Alert callback fonksiyonu"""
    severity_colors = {
        "critical": "\033[91m",  # Red
        "high": "\033[93m",  # Yellow
        "medium": "\033[94m",  # Blue
        "low": "\033[92m",  # Green
    }
    reset = "\033[0m"

    color = severity_colors.get(alert.severity.value, "")
    print(f"\n{color}🚨 ALERT [{alert.severity.value.upper()}]{reset}")
    print(f"   ID: {alert.id}")
    print(f"   Type: {alert.attack_type.name}")
    print(f"   Confidence: {alert.confidence*100:.1f}%")
    print(f"   Time: {alert.timestamp}")


def test_with_simulation(ids, num_packets: int = 1000, attack_ratio: float = 0.3):
    """Simüle edilmiş trafik ile test"""
    from src.network_detection.realtime_ids import NetworkTrafficSimulator

    print(f"\n📊 Simülasyon başlıyor...")
    print(f"   Paket sayısı: {num_packets}")
    print(f"   Saldırı oranı: {attack_ratio*100:.0f}%")

    # Model input shape'den feature sayısını al
    input_shape = ids.model.input_shape
    num_features = input_shape[-1]

    simulator = NetworkTrafficSimulator(
        num_features=num_features, attack_ratio=attack_ratio
    )

    # Test verisi üret
    X, y = simulator.generate(num_packets)

    print(f"\n🔄 {num_packets} paket işleniyor...")

    start_time = time.time()
    alerts = []

    for i, features in enumerate(X):
        metadata = {
            "source_ip": f"192.168.1.{np.random.randint(1, 255)}",
            "dest_ip": f"10.0.0.{np.random.randint(1, 255)}",
        }

        alert = ids.process(features, metadata)
        if alert:
            alerts.append(alert)

        # Progress
        if (i + 1) % (num_packets // 10) == 0:
            progress = (i + 1) / num_packets * 100
            print(f"   {progress:.0f}% tamamlandı...")

    elapsed = time.time() - start_time

    # Sonuçlar
    print("\n" + "=" * 60)
    print("📊 TEST SONUÇLARI")
    print("=" * 60)

    metrics = ids.get_metrics()

    print(f"\n⏱️ Performans:")
    print(f"   Toplam süre: {elapsed:.2f} saniye")
    print(f"   Paket/saniye: {num_packets/elapsed:.1f}")
    print(f"   Ortalama işlem: {metrics['processing_time_avg_ms']:.3f} ms")

    print(f"\n📈 İstatistikler:")
    print(f"   Toplam paket: {metrics['total_packets']}")
    print(f"   Normal: {metrics['normal_packets']}")
    print(f"   Saldırı: {metrics['attack_packets']}")
    print(f"   Alert sayısı: {len(alerts)}")

    # Alert dağılımı
    if alerts:
        print(f"\n🚨 Alert Dağılımı:")
        severity_counts = {}
        attack_counts = {}

        for alert in alerts:
            sev = alert.severity.value
            attack = alert.attack_type.name

            severity_counts[sev] = severity_counts.get(sev, 0) + 1
            attack_counts[attack] = attack_counts.get(attack, 0) + 1

        print(f"   Severity:")
        for sev, count in sorted(severity_counts.items()):
            print(f"      {sev}: {count}")

        print(f"   Attack Type:")
        for attack, count in sorted(attack_counts.items()):
            print(f"      {attack}: {count}")

    # Başarı oranı
    expected_attacks = int(num_packets * attack_ratio)
    detected_attacks = metrics["attack_packets"]
    detection_rate = (
        detected_attacks / expected_attacks * 100 if expected_attacks > 0 else 0
    )

    print(f"\n📊 Detection Rate: {detection_rate:.1f}%")
    print(f"   Beklenen saldırı: ~{expected_attacks}")
    print(f"   Tespit edilen: {detected_attacks}")

    # Drift check
    drift_detected, drift_score = ids.check_drift()
    print(f"\n🔍 Concept Drift:")
    print(f"   Detected: {'⚠️ Evet' if drift_detected else '✅ Hayır'}")
    print(f"   Score: {drift_score:.4f}")

    return metrics, alerts


def test_with_dataset(ids, dataset_path: Path, max_samples: int = 10000):
    """Gerçek dataset ile test"""
    print(f"\n📂 Dataset yükleniyor: {dataset_path}")

    # NSL-KDD yükle
    from scripts.train_ssa_lstmids import load_nsl_kdd, prepare_data

    X, y, class_names = load_nsl_kdd(dataset_path, max_samples)
    if X is None:
        print("❌ Dataset yüklenemedi!")
        return None, None

    # Sequence hazırla
    X_train, X_test, y_train, y_test, scaler = prepare_data(X, y)

    print(f"   Test samples: {len(X_test)}")

    # Test
    print(f"\n🔄 Test başlıyor...")

    start_time = time.time()
    alerts = []
    correct = 0
    total = 0

    for i, (features_seq, true_label) in enumerate(zip(X_test, y_test)):
        # Her sequence'ın son feature'ını al
        for features in features_seq:
            alert = ids.process(features)

            # Prediction kontrolü
            prediction = 1 if alert else 0
            true_attack = 1 if true_label > 0 else 0

            if prediction == true_attack:
                correct += 1
            total += 1

            if alert:
                alerts.append(alert)

        if (i + 1) % (len(X_test) // 10) == 0:
            print(f"   {(i+1)/len(X_test)*100:.0f}% tamamlandı...")

    elapsed = time.time() - start_time
    accuracy = correct / total * 100 if total > 0 else 0

    print("\n" + "=" * 60)
    print("📊 DATASET TEST SONUÇLARI")
    print("=" * 60)

    print(f"\n📈 Accuracy: {accuracy:.2f}%")
    print(f"   Correct: {correct}/{total}")
    print(f"   Alerts: {len(alerts)}")
    print(f"   Time: {elapsed:.2f}s")

    metrics = ids.get_metrics()
    return metrics, alerts


def main():
    parser = argparse.ArgumentParser(description="Real-time IDS Test")
    parser.add_argument("--model", type=str, help="Model dosya yolu")
    parser.add_argument("--simulate", action="store_true", help="Simülasyon modu")
    parser.add_argument(
        "--packets", type=int, default=1000, help="Simülasyon paket sayısı"
    )
    parser.add_argument("--attack_ratio", type=float, default=0.3, help="Saldırı oranı")
    parser.add_argument(
        "--dataset", type=str, help="Dataset ile test (nsl_kdd, bot_iot)"
    )
    parser.add_argument(
        "--threshold", type=float, default=0.5, help="Detection threshold"
    )
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("🛡️ Real-time IDS Test Suite")
    print("=" * 60)

    # Model bul
    if args.model:
        model_path = Path(args.model)
    else:
        model_path = find_latest_model()

    if not model_path or not model_path.exists():
        print("❌ Model bulunamadı!")
        print("   Önce model eğitin: python scripts/train_ssa_lstmids.py")
        return

    print(f"\n📦 Model: {model_path.name}")

    # IDS başlat
    from src.network_detection.realtime_ids import RealTimeIDS

    ids = RealTimeIDS(
        model_path=str(model_path),
        threshold=args.threshold,
        window_size=10,
        alert_callback=alert_callback,
        verbose=True,
    )

    ids.start()

    try:
        if args.dataset:
            # Dataset ile test
            dataset_path = PROJECT_ROOT / "data" / "raw" / args.dataset
            test_with_dataset(ids, dataset_path)
        else:
            # Simülasyon
            test_with_simulation(ids, args.packets, args.attack_ratio)
    finally:
        ids.stop()

    print("\n✅ Test tamamlandı!")


if __name__ == "__main__":
    main()
