"""Write upstream SAFE DWT output and matching RGBA input for port validation."""

from pathlib import Path
import importlib.util
import sys

import numpy as np
from PIL import Image
import torch


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "research" / "safe"
spec = importlib.util.spec_from_file_location("safe_resnet", SOURCE / "models" / "resnet.py")
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(module)

image = Image.open(sys.argv[1]).convert("RGB")
canvas = Image.new("RGB", (max(256, image.width), max(256, image.height)))
canvas.paste(image, ((canvas.width - image.width) // 2, (canvas.height - image.height) // 2))
left, top = (canvas.width - 256) // 2, (canvas.height - 256) // 2
pixels = np.asarray(canvas.crop((left, top, left + 256, top + 256))).copy()
rgba = np.concatenate((pixels, np.full((256, 256, 1), 255, dtype=np.uint8)), axis=2)
tensor = torch.from_numpy(pixels.transpose(2, 0, 1)).float().unsqueeze(0) / 255
model = module.resnet50(num_classes=2)
with torch.inference_mode():
    band = model._preprocess_dwt(tensor).numpy().astype(np.float32)
(ROOT / "research" / "safe-input.rgba").write_bytes(rgba.tobytes())
(ROOT / "research" / "safe-expected.f32").write_bytes(band.tobytes())
