"""Evaluate the pinned Flux EfficientNet specialist at browser-feasible resolutions."""

from pathlib import Path
import json
import sys

from PIL import Image
import torch
from torchvision import transforms


ROOT = Path(__file__).resolve().parents[1]
SIZE = int(sys.argv[1]) if len(sys.argv) > 1 else 256
REAL_LIMIT = int(sys.argv[2]) if len(sys.argv) > 2 else 25
model = torch.jit.load(ROOT / "research/flux-detector/model.pt", map_location="cpu").eval()
preprocess = transforms.Compose([
    transforms.Resize(SIZE + 32),
    transforms.CenterCrop(SIZE),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

paths = [(path, 1, "FLUX.1-Krea-dev") for path in sorted((ROOT / "benchmark-corebench/fake").glob("*"))]
paths += [(path, 0, "real") for path in sorted((ROOT / "benchmark-aigibench/real").glob("*"))[:REAL_LIMIT]]
rows = []
for path, truth, source in paths:
    tensor = preprocess(Image.open(path).convert("RGB")).unsqueeze(0)
    with torch.inference_mode():
        real_probability = torch.sigmoid(model(tensor)).item()
    rows.append({"file": str(path.relative_to(ROOT)), "truth": truth, "source": source, "score": 1 - real_probability})

positives = [row for row in rows if row["truth"]]
negatives = [row for row in rows if not row["truth"]]
tpr = sum(row["score"] >= 0.65 for row in positives) / len(positives) if positives else None
tnr = sum(row["score"] < 0.65 for row in negatives) / len(negatives) if negatives else None
report = {"size": SIZE, "images": len(rows), "threshold": 0.65, "tpr": tpr, "tnr": tnr, "rows": rows}
(ROOT / f"reports/flux-detector-{SIZE}.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
print(json.dumps({key: value for key, value in report.items() if key != "rows"}, indent=2))
