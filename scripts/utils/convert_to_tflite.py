"""
Edge Deployment Script - TensorFlow Lite Conversion
=====================================================

Modelleri IoT cihazlarında çalışabilecek hafif formata dönüştürür.
- TFLite conversion
- Quantization (INT8, Float16)
- Model size optimization

Kullanım:
    python scripts/convert_to_tflite.py
"""

import os
import sys
import json
import numpy as np
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
from tensorflow import keras

print(f"✅ TensorFlow {tf.__version__}")

MODELS_DIR = PROJECT_ROOT / "models"


def find_keras_models():
    """Mevcut .keras modellerini bul"""
    models = list(MODELS_DIR.glob("*.keras"))
    return models


def convert_to_tflite(model_path, quantize=False):
    """Model TFLite'a dönüştür"""
    print(f"\n   Converting: {model_path.name}")

    # Load model
    model = keras.models.load_model(model_path)

    # Base converter
    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    if quantize:
        # INT8 quantization
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]

    # Convert
    tflite_model = converter.convert()

    # Save
    suffix = "_quant" if quantize else ""
    output_path = model_path.with_suffix(f"{suffix}.tflite")

    with open(output_path, "wb") as f:
        f.write(tflite_model)

    # Size comparison
    original_size = model_path.stat().st_size / (1024 * 1024)  # MB
    tflite_size = output_path.stat().st_size / (1024 * 1024)  # MB
    reduction = (1 - tflite_size / original_size) * 100

    print(f"   Original: {original_size:.2f} MB")
    print(f"   TFLite: {tflite_size:.2f} MB")
    print(f"   Reduction: {reduction:.1f}%")

    return output_path, tflite_size


def test_tflite_inference(tflite_path, test_input_shape):
    """TFLite model inference test"""
    print(f"\n   Testing inference: {tflite_path.name}")

    # Load interpreter
    interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
    interpreter.allocate_tensors()

    # Get I/O details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Create random test input
    input_shape = input_details[0]["shape"]
    test_input = np.random.rand(*input_shape).astype(np.float32)

    # Run inference
    start_time = datetime.now()
    interpreter.set_tensor(input_details[0]["index"], test_input)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]["index"])
    inference_time = (datetime.now() - start_time).total_seconds() * 1000

    print(f"   Input shape: {input_shape}")
    print(f"   Output shape: {output.shape}")
    print(f"   Inference time: {inference_time:.2f} ms")

    return inference_time


def main():
    print("\n" + "=" * 70)
    print("🎓 Edge Deployment - TFLite Conversion")
    print("=" * 70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Find models
    models = find_keras_models()
    print(f"\n   Found {len(models)} .keras models")

    if not models:
        print("   No models to convert!")
        return

    # Convert each model
    results = []

    for model_path in models[:5]:  # Convert first 5 models
        print(f"\n" + "=" * 60)
        print(f"📦 Processing: {model_path.name}")
        print("=" * 60)

        try:
            # Standard conversion
            tflite_path, size = convert_to_tflite(model_path, quantize=False)

            # Quantized conversion
            tflite_quant_path, quant_size = convert_to_tflite(model_path, quantize=True)

            # Test inference
            inference_time = test_tflite_inference(tflite_path, None)

            results.append(
                {
                    "model": model_path.name,
                    "tflite_path": str(tflite_path),
                    "tflite_quant_path": str(tflite_quant_path),
                    "tflite_size_mb": size,
                    "tflite_quant_size_mb": quant_size,
                    "inference_time_ms": inference_time,
                }
            )

        except Exception as e:
            print(f"   ❌ Error: {e}")

    # Summary
    print("\n" + "=" * 60)
    print("📊 Conversion Summary")
    print("=" * 60)

    for r in results:
        print(f"\n   {r['model']}")
        print(f"      TFLite: {r['tflite_size_mb']:.2f} MB")
        print(f"      Quantized: {r['tflite_quant_size_mb']:.2f} MB")
        print(f"      Inference: {r['inference_time_ms']:.2f} ms")

    # Save results
    results_path = MODELS_DIR / "tflite_conversion_results.json"
    with open(results_path, "w") as f:
        json.dump(
            {"conversions": results, "created_at": datetime.now().isoformat()},
            f,
            indent=2,
        )

    print(f"\n💾 Results saved: {results_path}")

    print("\n" + "=" * 70)
    print("✅ Edge Deployment Complete!")
    print("=" * 70)
    print(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
