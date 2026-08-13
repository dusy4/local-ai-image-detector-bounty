"""Decode one image with the upstream invisible-watermark implementation."""

from pathlib import Path
import json
import sys

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "research" / "invisible-watermark"))
from imwatermark import WatermarkDecoder  # noqa: E402


message = 0b101100111110110010010000011110111011000110011110
expected = np.array([int(bit) for bit in bin(message)[2:]])
image = cv2.imread(sys.argv[1])
if image is None:
    raise RuntimeError(f"Cannot read {sys.argv[1]}")
height = image.shape[0] // 8 * 8
width = image.shape[1] // 8 * 8
image = image[:height, :width]
decoded = WatermarkDecoder("bits", 48).decode(image, "dwtDct")
rgba = cv2.cvtColor(image, cv2.COLOR_BGR2RGBA)
target = ROOT / "research" / "sdxl-reference.rgba"
target.write_bytes(rgba.tobytes())
print(json.dumps({"width": width, "height": height, "matches": int(np.sum(decoded == expected)), "bits": "".join(map(lambda bit: str(int(bit)), decoded)), "raw": str(target)}))
