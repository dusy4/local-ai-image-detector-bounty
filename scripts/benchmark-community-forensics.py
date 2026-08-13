"""Evaluate official Community Forensics 224px on the local real/fake tree."""

from pathlib import Path
import importlib.util
import json
import sys

from PIL import Image
import numpy as np
from safetensors.torch import load_file
import torch
from torchvision import transforms


ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / (sys.argv[1] if len(sys.argv) > 1 else "benchmark")
LIMIT_PER_CLASS = int(sys.argv[2]) if len(sys.argv) > 2 else 0
OUTPUT = ROOT / sys.argv[3] if len(sys.argv) > 3 else None
SOURCE = ROOT / "research" / "community-forensics-official"
spec = importlib.util.spec_from_file_location("commfor_models", SOURCE / "models.py")
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(module)
model = module.ViTClassifier(model_size="small", input_size=224, patch_size=16, device="cpu")
model.load_state_dict(load_file(ROOT / "research" / "commfor-224.safetensors"))
model.eval()
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])
rows = []
manifest_path = DATASET / "manifest.json"
sources = {}
if manifest_path.exists():
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    sources = {Path(item["path"]).name: item.get("source", "unknown") for item in manifest["samples"]}
for folder, truth in (("real", 0), ("fake", 1)):
    paths = sorted((DATASET / folder).glob("*"))
    if LIMIT_PER_CLASS:
        paths = paths[:LIMIT_PER_CLASS]
    for start in range(0, len(paths), 8):
        batch_paths = paths[start:start + 8]
        inputs = torch.stack([preprocess(Image.open(path).convert("RGB")) for path in batch_paths])
        with torch.inference_mode():
            scores = torch.sigmoid(model(inputs)).flatten().tolist()
        rows.extend({"file": f"{folder}/{path.name}", "truth": truth, "source": sources.get(path.name, folder), "score": score} for path, score in zip(batch_paths, scores))


def metrics(threshold):
    positives = [row for row in rows if row["truth"] == 1]
    negatives = [row for row in rows if row["truth"] == 0]
    tpr = sum(row["score"] >= threshold for row in positives) / len(positives) if positives else None
    tnr = sum(row["score"] < threshold for row in negatives) / len(negatives) if negatives else None
    values = [value for value in (tpr, tnr) if value is not None]
    return {"threshold": threshold, "tpr": tpr, "tnr": tnr, "balancedAccuracy": sum(values) / len(values)}


best = max((metrics(threshold) for threshold in sorted({row["score"] for row in rows})), key=lambda item: item["balancedAccuracy"])
report = {"images": len(rows), "at0.5": metrics(0.5), "at0.65": metrics(0.65), "bestInSample": best, "rows": rows}
rendered = json.dumps(report, indent=2)
if OUTPUT:
    OUTPUT.write_text(rendered, encoding="utf-8")
print(json.dumps({"images": report["images"], "at0.65": report["at0.65"], "bestInSample": report["bestInSample"]}, indent=2))
