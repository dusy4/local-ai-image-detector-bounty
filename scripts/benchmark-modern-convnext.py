"""Evaluate the pinned modern-generator ConvNeXt checkpoint safely."""

from pathlib import Path
import json
import sys

from PIL import Image
import timm
import torch
from torchvision import transforms


ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / (sys.argv[1] if len(sys.argv) > 1 else "benchmark-corebench")
LIMIT_PER_CLASS = int(sys.argv[2]) if len(sys.argv) > 2 else 0
OUTPUT = ROOT / (sys.argv[3] if len(sys.argv) > 3 else "reports/modern-convnext-smoke.json")
model = timm.create_model("convnextv2_base", pretrained=False, num_classes=2)
checkpoint = torch.load(ROOT / "research/modern-convnext/checkpoint_phase2.pth", map_location="cpu", weights_only=True)
model.load_state_dict(checkpoint["model"])
model.eval()
preprocess = transforms.Compose([
    transforms.Resize(288),
    transforms.CenterCrop(256),
    transforms.ToTensor(),
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
rows = []
for start in range(0, len(paths), 4):
    batch = paths[start:start + 4]
    inputs = torch.stack([preprocess(Image.open(path).convert("RGB")) for path, _, _ in batch])
    with torch.inference_mode():
        scores = torch.softmax(model(inputs), dim=1)[:, 1].tolist()
    rows.extend({"file": str(path.relative_to(ROOT)), "truth": truth, "source": source, "score": score} for (path, truth, source), score in zip(batch, scores))

positives = [row for row in rows if row["truth"]]
negatives = [row for row in rows if not row["truth"]]
tpr = sum(row["score"] >= 0.65 for row in positives) / len(positives) if positives else None
tnr = sum(row["score"] < 0.65 for row in negatives) / len(negatives) if negatives else None
report = {"images": len(rows), "threshold": 0.65, "tpr": tpr, "tnr": tnr, "rows": rows}
OUTPUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
print(json.dumps({key: value for key, value in report.items() if key != "rows"}, indent=2))
