"""
Model Quantization & ONNX / TFLite Export Pipeline
==================================================
Converts trained Keras transfer learning models into ONNX and quantized TFLite
formats for low-latency web inference and edge diagnostic deployment.
"""

import os
import sys
import time
import argparse
from pathlib import Path
from typing import Dict, Any

import numpy as np
import tensorflow as tf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
from src.models.model_factory import ModelFactory


def convert_to_tflite(
    keras_model: tf.keras.Model,
    output_path: Path,
    quantize: bool = False,
) -> int:
    """Convert Keras model to TFLite format with optional dynamic range quantization."""
    converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
    if quantize:
        converter.optimizations = [tf.lite.Optimize.DEFAULT]

    tflite_model = converter.convert()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        f.write(tflite_model)
    return len(tflite_model)


def convert_to_onnx(
    keras_model: tf.keras.Model,
    output_path: Path,
) -> bool:
    """Convert Keras model to ONNX format using tf2onnx if available."""
    try:
        import tf2onnx
        input_signature = [
            tf.TensorSpec(keras_model.input_shape, tf.float32, name="input_image")
        ]
        output_path.parent.mkdir(parents=True, exist_ok=True)
        onnx_model, _ = tf2onnx.convert.from_keras(
            keras_model,
            input_signature=input_signature,
            opset=13,
            output_path=str(output_path),
        )
        return True
    except ImportError:
        print("  ⚠ tf2onnx is not installed. Skipping direct ONNX serialization.")
        return False
    except Exception as e:
        print(f"  ⚠ ONNX conversion encountered an issue: {e}")
        return False


def benchmark_latency(
    model: tf.keras.Model,
    input_shape: tuple = (1, 224, 224, 3),
    n_runs: int = 20,
) -> Dict[str, float]:
    """Measure inference latency per CT slice."""
    dummy_input = np.random.randn(*input_shape).astype(np.float32)

    # Warmup
    for _ in range(3):
        _ = model.predict(dummy_input, verbose=0)

    start = time.perf_counter()
    for _ in range(n_runs):
        _ = model.predict(dummy_input, verbose=0)
    elapsed = time.perf_counter() - start

    avg_ms = (elapsed / n_runs) * 1000.0
    throughput = n_runs / elapsed
    return {
        "avg_latency_ms": round(avg_ms, 2),
        "throughput_fps": round(throughput, 2),
    }


def export_and_quantize(
    model_name: str = "resnet50",
    output_dir: Path = None,
) -> Dict[str, Any]:
    """Build, export, quantize, and benchmark model inference formats."""
    if output_dir is None:
        output_dir = config.MODELS_DIR / "exported"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(f"  Exporting & Quantizing Model: {model_name.upper()}")
    print("=" * 60)

    # Load or instantiate model
    model_obj = ModelFactory.create(model_name, build=True, compile=True)
    keras_model = model_obj.model

    # 1. TFLite FP32
    tflite_fp32_path = output_dir / f"{model_name}.tflite"
    fp32_size = convert_to_tflite(keras_model, tflite_fp32_path, quantize=False)
    print(f"  ✓ TFLite FP32 saved: {tflite_fp32_path.name} ({fp32_size / (1024*1024):.2f} MB)")

    # 2. TFLite INT8 Quantized
    tflite_int8_path = output_dir / f"{model_name}_quantized.tflite"
    int8_size = convert_to_tflite(keras_model, tflite_int8_path, quantize=True)
    print(f"  ✓ Quantized TFLite saved: {tflite_int8_path.name} ({int8_size / (1024*1024):.2f} MB)")

    # 3. ONNX Export
    onnx_path = output_dir / f"{model_name}.onnx"
    onnx_success = convert_to_onnx(keras_model, onnx_path)
    if onnx_success:
        print(f"  ✓ ONNX Runtime format saved: {onnx_path.name}")

    # Latency Benchmark
    bench = benchmark_latency(keras_model, input_shape=(1, *model_obj.input_shape))
    print(f"  ✓ Baseline Keras Latency: {bench['avg_latency_ms']} ms/slice ({bench['throughput_fps']} FPS)")

    return {
        "model": model_name,
        "tflite_fp32_size_mb": round(fp32_size / (1024 * 1024), 2),
        "tflite_quantized_size_mb": round(int8_size / (1024 * 1024), 2),
        "compression_ratio": round(fp32_size / max(1, int8_size), 2),
        "latency_benchmark": bench,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export model to ONNX & Quantized TFLite")
    parser.add_argument("--model", type=str, default="resnet50")
    parser.add_argument("--output-dir", type=str, default=str(config.MODELS_DIR / "exported"))
    args = parser.parse_args()

    res = export_and_quantize(model_name=args.model, output_dir=Path(args.output_dir))
    print("\nExport & Quantization Complete:")
    print(f"  Compression: {res['compression_ratio']}x smaller with INT8 Quantization")
