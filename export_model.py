"""
Model Export Utility
Export trained PyTorch model to ONNX with quantization
"""

import torch
import onnx
import onnxruntime as ort
import time
import argparse
from pathlib import Path

from model import BeautyPredictor


def export_to_onnx(
    pytorch_model_path: str,
    onnx_output_path: str,
    opset_version: int = 12,
    verify: bool = True
):
    """
    Export PyTorch model to ONNX format
    
    Args:
        pytorch_model_path: Path to PyTorch model (.pth)
        onnx_output_path: Output path for ONNX model
        opset_version: ONNX opset version
        verify: Verify exported model
    """
    print(f"\n{'='*70}")
    print("ONNX Export Utility")
    print(f"{'='*70}\n")
    
    # Load PyTorch model
    print(f"Loading PyTorch model from {pytorch_model_path}...")
    device = torch.device('cpu')
    model = BeautyPredictor(pretrained=False)
    
    if Path(pytorch_model_path).exists():
        checkpoint = torch.load(pytorch_model_path, map_location=device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        print("✓ Model loaded successfully")
    else:
        print("⚠ Model file not found, using untrained weights")
    
    model.eval()
    
    # Create dummy input
    dummy_input = torch.randn(1, 3, 224, 224)
    
    # Export to ONNX
    print(f"\nExporting to ONNX...")
    torch.onnx.export(
        model,
        dummy_input,
        onnx_output_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )
    print(f"✓ Exported to {onnx_output_path}")
    
    # Verify ONNX model
    if verify:
        print("\nVerifying ONNX model...")
        onnx_model = onnx.load(onnx_output_path)
        onnx.checker.check_model(onnx_model)
        print("✓ ONNX model verification passed")
    
    # Get file size
    file_size_mb = Path(onnx_output_path).stat().st_size / (1024 ** 2)
    print(f"✓ Model size: {file_size_mb:.2f} MB")
    
    # Benchmark
    print("\nBenchmarking ONNX model...")
    session = ort.InferenceSession(onnx_output_path)
    test_input = torch.randn(1, 3, 224, 224).numpy()
    
    # Warm-up
    for _ in range(5):
        session.run(None, {'input': test_input})
    
    # Measure
    times = []
    for _ in range(100):
        start = time.time()
        session.run(None, {'input': test_input})
        times.append((time.time() - start) * 1000)
    
    avg_time = sum(times) / len(times)
    print(f"✓ Average inference time: {avg_time:.2f}ms")
    
    return onnx_output_path


def quantize_onnx_model(
    onnx_model_path: str,
    quantized_output_path: str
):
    """
    Apply dynamic Int8 quantization to ONNX model
    
    Args:
        onnx_model_path: Path to ONNX model
        quantized_output_path: Output path for quantized model
    """
    from onnxruntime.quantization import quantize_dynamic, QuantType
    
    print(f"\n{'='*70}")
    print("ONNX Quantization")
    print(f"{'='*70}\n")
    
    print(f"Quantizing {onnx_model_path}...")
    
    quantize_dynamic(
        model_input=onnx_model_path,
        model_output=quantized_output_path,
        weight_type=QuantType.QInt8
    )
    
    # Compare sizes
    original_size = Path(onnx_model_path).stat().st_size / (1024 ** 2)
    quantized_size = Path(quantized_output_path).stat().st_size / (1024 ** 2)
    reduction = (1 - quantized_size / original_size) * 100
    
    print(f"\n✓ Original size: {original_size:.2f} MB")
    print(f"✓ Quantized size: {quantized_size:.2f} MB")
    print(f"✓ Size reduction: {reduction:.1f}%")
    
    # Benchmark quantized model
    print("\nBenchmarking quantized model...")
    session = ort.InferenceSession(quantized_output_path)
    test_input = torch.randn(1, 3, 224, 224).numpy()
    
    # Warm-up
    for _ in range(5):
        session.run(None, {'input': test_input})
    
    # Measure
    times = []
    for _ in range(100):
        start = time.time()
        session.run(None, {'input': test_input})
        times.append((time.time() - start) * 1000)
    
    avg_time = sum(times) / len(times)
    print(f"✓ Average inference time: {avg_time:.2f}ms")
    
    return quantized_output_path


def main():
    parser = argparse.ArgumentParser(description='Export and quantize beauty prediction model')
    parser.add_argument('--model', type=str, default='checkpoints/best_model.pth',
                        help='Path to PyTorch model checkpoint')
    parser.add_argument('--onnx-output', type=str, default='models/beauty_model.onnx',
                        help='Output path for ONNX model')
    parser.add_argument('--quantized-output', type=str, default='models/beauty_model_int8.onnx',
                        help='Output path for quantized ONNX model')
    parser.add_argument('--skip-quantization', action='store_true',
                        help='Skip quantization step')
    
    args = parser.parse_args()
    
    # Create output directory
    Path(args.onnx_output).parent.mkdir(parents=True, exist_ok=True)
    
    # Export to ONNX
    onnx_path = export_to_onnx(
        args.model,
        args.onnx_output
    )
    
    # Quantize
    if not args.skip_quantization:
        quantize_onnx_model(
            onnx_path,
            args.quantized_output
        )
    
    print(f"\n{'='*70}")
    print("Export Complete!")
    print(f"{'='*70}\n")
    print(f"ONNX Model: {args.onnx_output}")
    if not args.skip_quantization:
        print(f"Quantized Model: {args.quantized_output}")
    print()


if __name__ == '__main__':
    main()
