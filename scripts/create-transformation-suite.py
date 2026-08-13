"""Create web-realistic sibling variants without crossing source-image splits."""

from pathlib import Path
import io
import json

from PIL import Image, ImageFilter, ImageEnhance


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "benchmark-aigibench"
TARGET = ROOT / "benchmark-transforms"


def jpeg_roundtrip(image, quality):
    buffer = io.BytesIO()
    image.save(buffer, "JPEG", quality=quality)
    buffer.seek(0)
    return Image.open(buffer).convert("RGB")


def contain(image, longest):
    copy = image.copy()
    copy.thumbnail((longest, longest), Image.Resampling.LANCZOS)
    return copy


def center_crop(image, fraction):
    width, height = image.size
    new_width, new_height = int(width * fraction), int(height * fraction)
    left, top = (width - new_width) // 2, (height - new_height) // 2
    return image.crop((left, top, left + new_width, top + new_height))


source_manifest = json.loads((SOURCE / "manifest.json").read_text(encoding="utf-8"))
metadata = {Path(item["path"]).name: item for item in source_manifest["samples"]}
manifest = []
for folder, truth in (("real", 0), ("fake", 1)):
    (TARGET / folder).mkdir(parents=True, exist_ok=True)
    for index, path in enumerate(sorted((SOURCE / folder).glob("*"))[:25]):
        image = Image.open(path).convert("RGB")
        source = metadata[path.name]
        variants = {
            "jpeg80": jpeg_roundtrip(image, 80),
            "resize512": contain(image, 512),
            "crop90": center_crop(image, 0.9).resize(image.size, Image.Resampling.BICUBIC),
            "blur": image.filter(ImageFilter.GaussianBlur(0.6)),
            "contrast": ImageEnhance.Contrast(image).enhance(1.08),
        }
        for transform, variant in variants.items():
            file = f"{index:03d}-{source['row']}-{transform}.jpg"
            variant.save(TARGET / folder / file, "JPEG", quality=92)
            manifest.append({"path": f"benchmark-transforms/{folder}/{file}", "truth": truth, "source": source["source"], "row": source["row"], "transform": transform})
(TARGET / "manifest.json").write_text(json.dumps({"dataset": source_manifest["dataset"], "sampling": "25 source images per class; siblings kept together", "samples": manifest}, indent=2), encoding="utf-8")
print(json.dumps({"sourceImages": 50, "variants": len(manifest)}))
