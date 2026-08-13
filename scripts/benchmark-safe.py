"""Evaluate the official SAFE checkpoint on the local real/fake tree."""

from pathlib import Path
import importlib.util
import json

import numpy as np
from PIL import Image
import torch


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "research" / "safe"
DATASET = ROOT / "benchmark"
spec = importlib.util.spec_from_file_location("safe_resnet", SOURCE / "models" / "resnet.py")
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(module)


def preprocess(path: Path):
    image = Image.open(path).convert("RGB")
    canvas = Image.new("RGB", (max(256, image.width), max(256, image.height)))
    canvas.paste(image, ((canvas.width - image.width) // 2, (canvas.height - image.height) // 2))
    left, top = (canvas.width - 256) // 2, (canvas.height - 256) // 2
    pixels = np.asarray(canvas.crop((left, top, left + 256, top + 256))).copy()
    return torch.from_numpy(pixels.transpose(2, 0, 1)).float() / 255


model = module.resnet50(num_classes=2)
model.load_state_dict(torch.load(SOURCE / "checkpoint" / "checkpoint-best.pth", map_location="cpu", weights_only=True)["model"])
model.eval()
rows = []
for folder, truth in (("real", 0), ("fake", 1)):
    paths = sorted((DATASET / folder).glob("*"))
    for start in range(0, len(paths), 8):
        batch_paths = paths[start:start + 8]
        inputs = torch.stack([preprocess(path) for path in batch_paths])
        with torch.inference_mode():
            scores = torch.softmax(model(inputs), dim=1)[:, 1].tolist()
        rows.extend({"file": f"{folder}/{path.name}", "truth": truth, "score": score} for path, score in zip(batch_paths, scores))


def metrics(threshold):
    positives = [row for row in rows if row["truth"] == 1]
    negatives = [row for row in rows if row["truth"] == 0]
    tpr = sum(row["score"] >= threshold for row in positives) / len(positives)
    tnr = sum(row["score"] < threshold for row in negatives) / len(negatives)
    return {"threshold": threshold, "tpr": tpr, "tnr": tnr, "balancedAccuracy": (tpr + tnr) / 2}


best = max((metrics(threshold) for threshold in sorted({row["score"] for row in rows})), key=lambda item: item["balancedAccuracy"])
print(json.dumps({"images": len(rows), "at0.5": metrics(0.5), "at0.65": metrics(0.65), "bestInSample": best, "rows": rows}, indent=2))
