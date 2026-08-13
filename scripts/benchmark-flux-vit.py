"""Evaluate the Apache-2.0 FLUX.1 specialist on newer held-out providers."""

from pathlib import Path
import json

from PIL import Image
import torch
from transformers import AutoModelForImageClassification, ViTImageProcessor


ROOT = Path(__file__).resolve().parents[1]
model = AutoModelForImageClassification.from_pretrained("ash12321/flux-detector-vit").eval()
processor = ViTImageProcessor(size={"height": 224, "width": 224}, do_resize=True, do_rescale=True, do_normalize=True)
manifest = json.loads((ROOT / "benchmark/manifest.json").read_text(encoding="utf-8"))["samples"]
targets = {"flux.2-klein-9b", "midjourney-7", "recraft-v3", "gpt-image-1.5"}
selected = [item for item in manifest if item["truth"] == 0 or item["source"] in targets]
selected = [item for item in selected if item["truth"] == 1] + [item for item in selected if item["truth"] == 0][:50]
labels = {int(index): str(label).lower() for index, label in model.config.id2label.items()}
fake_index = next(
    (index for index, label in labels.items() if any(token in label for token in ("fake", "ai", "generated"))),
    1,
)
rows = []
for start in range(0, len(selected), 8):
    batch = selected[start:start + 8]
    images = [Image.open(ROOT / item["path"]).convert("RGB") for item in batch]
    inputs = processor(images=images, return_tensors="pt")
    with torch.inference_mode():
        probabilities = torch.softmax(model(**inputs).logits, dim=1)
    for item, values in zip(batch, probabilities):
        rows.append({**item, "score": float(values[fake_index])})
by_source = {}
for source in sorted({row["source"] for row in rows}):
    part = [row for row in rows if row["source"] == source]
    by_source[source] = {"n": len(part), "truth": "AI" if part[0]["truth"] else "real", "accuracy": sum((row["score"] >= 0.65) == bool(row["truth"]) for row in part) / len(part)}
report = {"model": "ash12321/flux-detector-vit", "threshold": 0.65, "labels": model.config.id2label, "bySource": by_source, "rows": rows}
(ROOT / "reports/flux-vit-providers.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
print(json.dumps({"labels": report["labels"], "bySource": by_source}, indent=2))
