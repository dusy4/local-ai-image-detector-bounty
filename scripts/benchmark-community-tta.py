"""Evaluate multi-crop max TTA on the exact active Community Forensics graph."""

from pathlib import Path
import json
import sys

import numpy as np
from PIL import Image, ImageOps
from safetensors.torch import load_file
import torch
import importlib.util


ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / (sys.argv[1] if len(sys.argv) > 1 else "benchmark")
OUTPUT = ROOT / (sys.argv[2] if len(sys.argv) > 2 else "reports/community-tta.json")
LIMIT_PER_CLASS = int(sys.argv[3]) if len(sys.argv) > 3 else 0
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
spec = importlib.util.spec_from_file_location("commfor_models", ROOT / "research/community-forensics-official/models.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
model = module.ViTClassifier(model_size="small", input_size=224, patch_size=16, device="cpu")
model.load_state_dict(load_file(ROOT / "research/commfor-224.safetensors"))
model.eval()


def crops(path):
    image = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    output = []
    for resize, center_only in ((256, False), (288, True)):
        scale = resize / min(image.size)
        resized = image.resize((round(image.width * scale), round(image.height * scale)), Image.Resampling.BICUBIC)
        width, height = resized.size
        origins = [((width - 224) // 2, (height - 224) // 2)] if center_only else [
            ((width - 224) // 2, (height - 224) // 2), (0, 0), (width - 224, 0), (0, height - 224), (width - 224, height - 224)
        ]
        for left, top in origins:
            crop = resized.crop((left, top, left + 224, top + 224))
            array = np.asarray(crop, dtype=np.float32) / 255
            output.append(((array - MEAN) / STD).transpose(2, 0, 1))
    return np.stack(output)


manifest = json.loads((DATASET / "manifest.json").read_text(encoding="utf-8")) if (DATASET / "manifest.json").exists() else {"samples": []}
source_by_name = {Path(row["path"]).name: row.get("source", "unknown") for row in manifest["samples"]}
paths = []
for folder, truth in (("real", 0), ("fake", 1)):
    class_paths = sorted((DATASET / folder).glob("*"))
    if LIMIT_PER_CLASS: class_paths = class_paths[:LIMIT_PER_CLASS]
    paths.extend((path, truth, source_by_name.get(path.name, folder)) for path in class_paths)
rows = []
for path, truth, source in paths:
    with torch.inference_mode():
        scores = torch.sigmoid(model(torch.from_numpy(crops(path)))).flatten().numpy()
    rows.append({"file": str(path.relative_to(ROOT)), "truth": truth, "source": source, "center": float(scores[0]), "max6": float(scores.max())})


def metrics(key):
    positive = [row for row in rows if row["truth"]]
    negative = [row for row in rows if not row["truth"]]
    return {"tpr": sum(row[key] >= .65 for row in positive) / len(positive) if positive else None, "tnr": sum(row[key] < .65 for row in negative) / len(negative) if negative else None}


report = {"dataset": str(DATASET.relative_to(ROOT)), "images": len(rows), "center": metrics("center"), "max6": metrics("max6"), "rows": rows}
OUTPUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
print(json.dumps({key: value for key, value in report.items() if key != "rows"}, indent=2))
