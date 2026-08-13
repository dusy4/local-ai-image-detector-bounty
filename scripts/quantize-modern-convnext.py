"""Static QDQ quantization for browser-compatible modern ConvNeXt inference."""

from pathlib import Path
import json

import numpy as np
from onnxruntime.quantization import CalibrationDataReader, CalibrationMethod, QuantFormat, QuantType, quantize_static
from PIL import Image
from torchvision import transforms


ROOT = Path(__file__).resolve().parents[1]
preprocess = transforms.Compose([
    transforms.Resize(288),
    transforms.CenterCrop(256),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])
paths = sorted((ROOT / "benchmark-aigibench/real").glob("*"))[:20]
paths += sorted((ROOT / "benchmark-aigibench/fake").glob("*"))[:20]
paths += sorted((ROOT / "benchmark-corebench/fake").glob("*"))


class Reader(CalibrationDataReader):
    def __init__(self):
        self.iterator = iter(paths)

    def get_next(self):
        try:
            path = next(self.iterator)
        except StopIteration:
            return None
        tensor = preprocess(Image.open(path).convert("RGB")).unsqueeze(0).numpy().astype(np.float32)
        return {"pixel_values": tensor}


source = ROOT / "research/modern-convnext/model-fp32.onnx"
target = ROOT / "research/modern-convnext/model-qdq-int8.onnx"
quantize_static(
    source,
    target,
    Reader(),
    quant_format=QuantFormat.QDQ,
    activation_type=QuantType.QUInt8,
    weight_type=QuantType.QInt8,
    per_channel=True,
    calibrate_method=CalibrationMethod.MinMax,
)
print(json.dumps({"path": str(target), "bytes": target.stat().st_size}))
