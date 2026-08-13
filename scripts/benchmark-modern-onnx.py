"""Evaluate a modern ConvNeXt ONNX variant with the exact browser preprocessing."""

from pathlib import Path
import json
import sys

import numpy as np
import onnxruntime as ort
from PIL import Image
from torchvision import transforms


ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / (sys.argv[1] if len(sys.argv) > 1 else "research/modern-convnext/model-qdq-int8.onnx")
DATASET = ROOT / (sys.argv[2] if len(sys.argv) > 2 else "benchmark-corebench")
LIMIT_PER_CLASS = int(sys.argv[3]) if len(sys.argv) > 3 else 0
OUTPUT = ROOT / (sys.argv[4] if len(sys.argv) > 4 else "reports/modern-convnext-onnx.json")
preprocess = transforms.Compose([
    transforms.Resize(288), transforms.CenterCrop(256), transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])
manifest_path = DATASET / "manifest.json"
sources = {}
if manifest_path.exists():
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    sources = {Path(item["path"]).name: item.get("source", "unknown") for item in manifest["samples"]}
paths = []
for folder, truth in (("real", 0), ("fake", 1)):
    folder_paths = sorted((DATASET / folder).glob("*"))
    if LIMIT_PER_CLASS:
        folder_paths = folder_paths[:LIMIT_PER_CLASS]
    paths += [(path, truth, sources.get(path.name, folder)) for path in folder_paths]
session = ort.InferenceSession(MODEL, providers=["CPUExecutionProvider"])
rows = []
for path, truth, source in paths:
    inputs = preprocess(Image.open(path).convert("RGB")).unsqueeze(0).numpy().astype(np.float32)
    logits = session.run(None, {"pixel_values": inputs})[0][0]
    exponential = np.exp(logits - np.max(logits))
    score = float(exponential[1] / exponential.sum())
    rows.append({"file": str(path.relative_to(ROOT)), "truth": truth, "source": source, "score": score})
positives = [row for row in rows if row["truth"]]
negatives = [row for row in rows if not row["truth"]]
tpr = sum(row["score"] >= 0.65 for row in positives) / len(positives) if positives else None
tnr = sum(row["score"] < 0.65 for row in negatives) / len(negatives) if negatives else None
report = {"model": str(MODEL.relative_to(ROOT)), "images": len(rows), "threshold": 0.65, "tpr": tpr, "tnr": tnr, "rows": rows}
OUTPUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
print(json.dumps({key: value for key, value in report.items() if key != "rows"}, indent=2))
