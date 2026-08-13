"""Weight-only quantization of browser-supported ConvNeXt linear layers."""

from pathlib import Path
import json

from onnxruntime.quantization import QuantType, quantize_dynamic


ROOT = Path(__file__).resolve().parents[1]
source = ROOT / "research/modern-convnext/model-fp32.onnx"
target = ROOT / "research/modern-convnext/model-linear-int8.onnx"
quantize_dynamic(
    source,
    target,
    weight_type=QuantType.QInt8,
    per_channel=True,
    op_types_to_quantize=["MatMul", "Gemm"],
)
print(json.dumps({"path": str(target), "bytes": target.stat().st_size}))
