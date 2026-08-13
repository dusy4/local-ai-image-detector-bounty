"""Evaluate the official FerretNet checkpoint on a local real/fake tree."""

from pathlib import Path
import importlib.util
import json

import numpy as np
from PIL import Image
import torch


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "research" / "ferretnet"
DATASET = ROOT / "benchmark"
MEAN = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
STD = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


Ferret = load_module("ferretnet_model", SOURCE / "src" / "model" / "ferretnet.py").Ferret
get_lpd_dict = load_module("ferretnet_lpd", SOURCE / "src" / "model" / "lpd.py").get_lpd_dict


def preprocess(path: Path):
    image = Image.open(path).convert("RGB")
    canvas = Image.new("RGB", (max(256, image.width), max(256, image.height)))
    canvas.paste(image, ((canvas.width - image.width) // 2, (canvas.height - image.height) // 2))
    left, top = (canvas.width - 256) // 2, (canvas.height - 256) // 2
    pixels = np.asarray(canvas.crop((left, top, left + 256, top + 256)), dtype=np.float32) / 255.0
    pixels = (pixels - MEAN) / STD
    return torch.from_numpy(pixels.transpose(2, 0, 1))


model = Ferret(3, 1, 96, [2, 2], "median", 3, get_lpd_dict())
checkpoint = torch.load(
    SOURCE / "checkpoints" / "4cls_ckpt" / "ferretnet-b-median-3.pth",
    map_location="cpu",
    weights_only=True,
)["model"]
model.load_state_dict({key.removeprefix("module."): value for key, value in checkpoint.items()})
model.eval()

rows = []
for folder, truth in (("real", 0), ("fake", 1)):
    paths = sorted((DATASET / folder).glob("*"))
    for start in range(0, len(paths), 8):
        batch_paths = paths[start:start + 8]
        inputs = torch.stack([preprocess(path) for path in batch_paths])
        with torch.inference_mode():
            scores = torch.sigmoid(model(inputs)).flatten().tolist()
        rows.extend({"file": f"{folder}/{path.name}", "truth": truth, "score": score} for path, score in zip(batch_paths, scores))


def metrics(threshold):
    positives = [row for row in rows if row["truth"] == 1]
    negatives = [row for row in rows if row["truth"] == 0]
    tpr = sum(row["score"] >= threshold for row in positives) / len(positives)
    tnr = sum(row["score"] < threshold for row in negatives) / len(negatives)
    return {"threshold": threshold, "tpr": tpr, "tnr": tnr, "balancedAccuracy": (tpr + tnr) / 2}


candidates = sorted({row["score"] for row in rows})
best = max((metrics(threshold) for threshold in candidates), key=lambda item: item["balancedAccuracy"])
print(json.dumps({"images": len(rows), "at0.5": metrics(0.5), "at0.65": metrics(0.65), "bestInSample": best, "rows": rows}, indent=2))
