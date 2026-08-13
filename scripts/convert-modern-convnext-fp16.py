"""Convert the modern ConvNeXt ONNX weights to float16 with float32 I/O."""

from pathlib import Path
import json

import onnx
from onnxconverter_common import float16


ROOT = Path(__file__).resolve().parents[1]
source = ROOT / "research/modern-convnext/model-fp32.onnx"
target = ROOT / "research/modern-convnext/model-fp16.onnx"
model = onnx.load(source)
converted = float16.convert_float_to_float16(model, keep_io_types=True, disable_shape_infer=False)
onnx.save(converted, target)
print(json.dumps({"path": str(target), "bytes": target.stat().st_size}))
