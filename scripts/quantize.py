#!/usr/bin/env python3
"""Quantize ONNX models to INT8 for faster CPU inference.

Creates a quantized model alongside the original — the engine auto-detects
and uses the quantized version when available.

Usage:
    python scripts/quantize.py --model-dir models/en-meta
    python scripts/quantize.py --model-dir models/hinglish-loans --method dynamic
    python scripts/quantize.py --all
"""

import argparse
import os
import sys
import time
from pathlib import Path


def quantize_model(model_dir: str, method: str = "dynamic") -> str:
    from onnxruntime.quantization import quantize_dynamic, QuantType

    model_path = os.path.join(model_dir, "model.onnx")
    output_path = os.path.join(model_dir, "model_int8.onnx")

    if not os.path.isfile(model_path):
        print(f"  SKIP: {model_path} not found")
        return ""

    if os.path.isfile(output_path):
        orig_mb = os.path.getsize(model_path) / 1024 / 1024
        quant_mb = os.path.getsize(output_path) / 1024 / 1024
        print(f"  Already quantized: {output_path} ({quant_mb:.0f} MB, {quant_mb/orig_mb*100:.0f}%)")
        return output_path

    data_path = os.path.join(model_dir, "model.onnx.data")
    has_external = os.path.isfile(data_path)

    orig_mb = os.path.getsize(model_path) / 1024 / 1024
    if has_external:
        orig_mb += os.path.getsize(data_path) / 1024 / 1024

    print(f"  Quantizing: {model_path} ({orig_mb:.0f} MB)")
    print(f"  Method: {method}")

    t0 = time.time()

    if method == "dynamic":
        quantize_dynamic(
            model_input=model_path,
            model_output=output_path,
            weight_type=QuantType.QInt8,
            extra_options={"MatMulConstBOnly": True},
            use_external_data_format=has_external,
        )
    else:
        print(f"  Unknown method: {method}")
        return ""

    elapsed = time.time() - t0
    quant_mb = os.path.getsize(output_path) / 1024 / 1024
    quant_data = os.path.join(model_dir, "model_int8.onnx.data")
    if os.path.isfile(quant_data):
        quant_mb += os.path.getsize(quant_data) / 1024 / 1024

    print(f"  Done: {quant_mb:.0f} MB ({quant_mb/orig_mb*100:.0f}% of original) in {elapsed:.1f}s")
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Quantize ONNX models to INT8")
    parser.add_argument("--model-dir", help="Path to model directory")
    parser.add_argument("--method", default="dynamic", choices=["dynamic"],
                        help="Quantization method (default: dynamic)")
    parser.add_argument("--all", action="store_true",
                        help="Quantize all downloaded models")
    args = parser.parse_args()

    if args.all:
        models_dir = Path(__file__).parent.parent / "models"
        if not models_dir.exists():
            print("No models/ directory found")
            sys.exit(1)
        for d in sorted(models_dir.iterdir()):
            if d.is_dir() and (d / "model.onnx").exists():
                print(f"\n{'='*50}")
                print(f"  {d.name}")
                print(f"{'='*50}")
                quantize_model(str(d), args.method)
    elif args.model_dir:
        quantize_model(args.model_dir, args.method)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
