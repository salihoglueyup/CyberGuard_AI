"""
Continuous Learning Pipeline
==============================

Model drift detection and automatic retraining pipeline.
Monitors model performance and triggers retraining when needed.

Kullanım:
    python scripts/continuous_learning.py
"""

import os
import sys
import json
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import deque

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
from tensorflow import keras

print(f"✅ TensorFlow {tf.__version__}")

MODELS_DIR = PROJECT_ROOT / "models"


class DriftDetector:
    """Model drift detection using statistical methods"""

    def __init__(self, window_size=100, threshold=0.1):
        self.window_size = window_size
        self.threshold = threshold
        self.baseline_accuracy = None
        self.recent_accuracies = deque(maxlen=window_size)

    def set_baseline(self, accuracy):
        """Set baseline accuracy from initial evaluation"""
        self.baseline_accuracy = accuracy
        print(f"   📊 Baseline accuracy set: {accuracy*100:.2f}%")

    def update(self, accuracy):
        """Update with new accuracy measurement"""
        self.recent_accuracies.append(accuracy)

    def detect_drift(self):
        """Check if model has drifted"""
        if (
            self.baseline_accuracy is None
            or len(self.recent_accuracies) < self.window_size // 2
        ):
            return False, 0.0

        recent_mean = np.mean(list(self.recent_accuracies))
        drift_magnitude = self.baseline_accuracy - recent_mean

        is_drift = drift_magnitude > self.threshold

        return is_drift, drift_magnitude


class ModelRetrainer:
    """Handles model retraining logic"""

    def __init__(self, model_builder, input_shape, num_classes):
        self.model_builder = model_builder
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.training_data_buffer = []
        self.max_buffer_size = 10000

    def add_training_data(self, X, y):
        """Add new data to training buffer"""
        for i in range(len(X)):
            self.training_data_buffer.append((X[i], y[i]))
            if len(self.training_data_buffer) > self.max_buffer_size:
                self.training_data_buffer.pop(0)

    def retrain(self, epochs=20, batch_size=128):
        """Retrain model with buffered data"""
        if len(self.training_data_buffer) < 100:
            print("   ⚠️ Not enough data for retraining")
            return None

        print(f"\n   🔄 Retraining with {len(self.training_data_buffer)} samples...")

        # Prepare data
        X = np.array([x for x, y in self.training_data_buffer])
        y = np.array([y for x, y in self.training_data_buffer])

        # Build new model
        model = self.model_builder(self.input_shape, self.num_classes)

        # Train
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor="loss", patience=5, restore_best_weights=True
            )
        ]

        model.fit(
            X, y, epochs=epochs, batch_size=batch_size, callbacks=callbacks, verbose=0
        )

        return model


class ContinuousLearningPipeline:
    """Main continuous learning orchestrator"""

    def __init__(self, model, model_builder, input_shape, num_classes):
        self.current_model = model
        self.drift_detector = DriftDetector(window_size=50, threshold=0.05)
        self.retrainer = ModelRetrainer(model_builder, input_shape, num_classes)
        self.version = 1
        self.retraining_history = []

    def evaluate_batch(self, X, y):
        """Evaluate model on a batch and check for drift"""
        y_pred = np.argmax(self.current_model.predict(X, verbose=0), axis=1)
        accuracy = np.mean(y_pred == y)

        self.drift_detector.update(accuracy)

        return accuracy

    def check_and_retrain(self, X_new, y_new):
        """Check for drift and retrain if needed"""
        is_drift, magnitude = self.drift_detector.detect_drift()

        if is_drift:
            print(f"\n   ⚠️ Drift detected! Magnitude: {magnitude*100:.2f}%")

            # Add new data to buffer
            self.retrainer.add_training_data(X_new, y_new)

            # Retrain
            new_model = self.retrainer.retrain()

            if new_model is not None:
                self.current_model = new_model
                self.version += 1

                # Reset drift detector
                accuracy = self.evaluate_batch(X_new, y_new)
                self.drift_detector.set_baseline(accuracy)

                self.retraining_history.append(
                    {
                        "timestamp": datetime.now().isoformat(),
                        "version": self.version,
                        "drift_magnitude": magnitude,
                        "new_accuracy": accuracy,
                    }
                )

                print(f"   ✅ Model retrained! Version: {self.version}")
                return True

        # Add data to buffer for future retraining
        self.retrainer.add_training_data(X_new, y_new)
        return False

    def save_model(self):
        """Save current model"""
        model_path = (
            MODELS_DIR
            / f"continuous_v{self.version}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.keras"
        )
        self.current_model.save(model_path)
        return model_path


def build_model(input_shape, num_classes):
    """Model builder function"""
    inputs = keras.layers.Input(shape=input_shape)
    x = keras.layers.Conv1D(32, 3, padding="same", activation="relu")(inputs)
    x = keras.layers.MaxPooling1D(2, padding="same")(x)
    x = keras.layers.LSTM(64)(x)
    x = keras.layers.Dense(64, activation="relu")(x)
    x = keras.layers.Dropout(0.3)(x)
    outputs = keras.layers.Dense(num_classes, activation="softmax")(x)

    model = keras.Model(inputs, outputs)
    model.compile(
        optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
    )
    return model


def simulate_data_stream(n_batches=20, batch_size=100, drift_at=10):
    """Simulate streaming data with concept drift"""
    print("\n" + "=" * 60)
    print("📊 Simulating Data Stream with Concept Drift")
    print("=" * 60)

    # Initial data distribution
    input_shape = (10, 4)  # timesteps, features
    num_classes = 2

    batches = []

    for i in range(n_batches):
        X = np.random.rand(batch_size, *input_shape).astype(np.float32)

        if i < drift_at:
            # Normal distribution
            y = (X[:, 0, 0] > 0.5).astype(int)
        else:
            # Drifted distribution (different decision boundary)
            y = (X[:, 0, 0] + X[:, 0, 1] > 1.0).astype(int)

        batches.append((X, y))

    print(f"   Total batches: {n_batches}")
    print(f"   Batch size: {batch_size}")
    print(f"   Drift at batch: {drift_at}")

    return batches, input_shape, num_classes


def main():
    print("\n" + "=" * 70)
    print("🎓 Continuous Learning Pipeline")
    print("=" * 70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Simulate data stream
    batches, input_shape, num_classes = simulate_data_stream(
        n_batches=20, batch_size=200, drift_at=12
    )

    # Build initial model
    print("\n" + "=" * 60)
    print("🧠 Building Initial Model")
    print("=" * 60)

    initial_model = build_model(input_shape, num_classes)

    # Train on first few batches
    X_initial = np.concatenate([b[0] for b in batches[:5]])
    y_initial = np.concatenate([b[1] for b in batches[:5]])

    initial_model.fit(X_initial, y_initial, epochs=10, batch_size=64, verbose=0)

    # Initialize pipeline
    pipeline = ContinuousLearningPipeline(
        initial_model, build_model, input_shape, num_classes
    )

    # Set baseline
    baseline_acc = pipeline.evaluate_batch(X_initial, y_initial)
    pipeline.drift_detector.set_baseline(baseline_acc)

    # Process stream
    print("\n" + "=" * 60)
    print("🔄 Processing Data Stream")
    print("=" * 60)

    stream_results = []

    for i, (X, y) in enumerate(batches[5:], start=6):
        accuracy = pipeline.evaluate_batch(X, y)
        retrained = pipeline.check_and_retrain(X, y)

        stream_results.append(
            {"batch": i, "accuracy": accuracy, "retrained": retrained}
        )

        print(f"   Batch {i}: Accuracy={accuracy*100:.2f}% {'🔄' if retrained else ''}")

    # Final summary
    print("\n" + "=" * 60)
    print("📊 Continuous Learning Summary")
    print("=" * 60)

    print(f"\n   Model versions: {pipeline.version}")
    print(f"   Retraining events: {len(pipeline.retraining_history)}")

    accuracies = [r["accuracy"] for r in stream_results]
    print(f"   Mean accuracy: {np.mean(accuracies)*100:.2f}%")
    print(f"   Min accuracy: {np.min(accuracies)*100:.2f}%")
    print(f"   Max accuracy: {np.max(accuracies)*100:.2f}%")

    # Save final model
    model_path = pipeline.save_model()
    print(f"\n💾 Final model saved: {model_path}")

    # Save results
    results = {
        "total_batches": len(batches),
        "model_versions": pipeline.version,
        "retraining_history": pipeline.retraining_history,
        "stream_results": stream_results,
        "created_at": datetime.now().isoformat(),
    }

    results_path = MODELS_DIR / "continuous_learning_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"💾 Results saved: {results_path}")

    print("\n" + "=" * 70)
    print("✅ Continuous Learning Pipeline Complete!")
    print("=" * 70)
    print(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
