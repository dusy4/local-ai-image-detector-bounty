"""Measure the public SDXL watermark decoder false-positive rate locally."""

from pathlib import Path
import json
import sys

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "research" / "invisible-watermark"))
from imwatermark import WatermarkDecoder  # noqa: E402


message = 0xB3EC907BB19E
expected = np.array([int(bit) for bit in bin(message)[2:]])
decoder = WatermarkDecoder("bits", 48)
rows = []
for folder in ("real", "fake"):
    for path in sorted((ROOT / "benchmark" / folder).glob("*")):
        image = cv2.imread(str(path))
        if image is None or image.shape[0] * image.shape[1] < 256 * 256:
            continue
        decoded = decoder.decode(image, "dwtDct")
        rows.append({"file": f"{folder}/{path.name}", "matches": int(np.sum(decoded == expected))})
print(json.dumps({
    "images": len(rows),
    "likelyWatermarked": sum(row["matches"] >= 34 for row in rows),
    "maximumMatches": max(row["matches"] for row in rows),
    "rows": rows,
}, indent=2))
