"""Test whether an independently derived spectral signature transfers to held-out OpenAI images."""

from pathlib import Path
import json

import numpy as np
from PIL import Image, ImageFilter


ROOT = Path(__file__).resolve().parents[1]
SIZE = 512
TRAIN = sorted((ROOT / "fixtures/openai-fresh/fake").glob("lowentropy-*.png"))
POSITIVE = sorted(path for path in (ROOT / "fixtures/openai-fresh/fake").glob("*.png") if path not in TRAIN)
NEGATIVE = sorted((ROOT / "benchmark/real").glob("*"))[:100]


def spectrum(path: Path):
    image = Image.open(path).convert("RGB").resize((SIZE, SIZE), Image.Resampling.LANCZOS)
    blurred = image.filter(ImageFilter.GaussianBlur(radius=2.0))
    residual = np.asarray(image, dtype=np.float32) - np.asarray(blurred, dtype=np.float32)
    window = np.hanning(SIZE).astype(np.float32)
    residual *= window[:, None, None] * window[None, :, None]
    return np.fft.rfft2(residual, axes=(0, 1))


train = np.stack([spectrum(path) for path in TRAIN])
unit = train / np.maximum(np.abs(train), 1e-12)
mean_unit = unit.mean(axis=0)
coherence = np.abs(mean_unit)
phase = np.angle(mean_unit)
magnitude = np.log1p(np.abs(train).mean(axis=0))
fy = np.minimum(np.arange(SIZE), SIZE - np.arange(SIZE))[:, None, None]
fx = np.arange(SIZE // 2 + 1)[None, :, None]
band = (np.hypot(fy, fx) >= 4) & (np.hypot(fy, fx) <= 80)
quality = np.where(band, coherence * magnitude, -np.inf)
flat = np.argpartition(quality.ravel(), -128)[-128:]
indices = np.array(np.unravel_index(flat, quality.shape)).T
indices = indices[np.argsort(quality[tuple(indices.T)])[::-1]]


def score(path: Path):
    current = spectrum(path)
    values = []
    for y, x, channel in indices:
        distance = abs(np.arctan2(np.sin(np.angle(current[y, x, channel]) - phase[y, x, channel]), np.cos(np.angle(current[y, x, channel]) - phase[y, x, channel])))
        values.append(1 - distance / np.pi)
    return float(np.mean(values))


positive_rows = [{"file": str(path.relative_to(ROOT)), "score": score(path)} for path in POSITIVE]
negative_rows = [{"file": str(path.relative_to(ROOT)), "score": score(path)} for path in NEGATIVE]
report = {
    "method": "independent RGB residual phase coherence",
    "imageSize": SIZE,
    "trainingImages": [str(path.relative_to(ROOT)) for path in TRAIN],
    "heldOutPositive": positive_rows,
    "negative": negative_rows,
    "summary": {
        "positiveMean": float(np.mean([row["score"] for row in positive_rows])),
        "positiveMin": float(np.min([row["score"] for row in positive_rows])),
        "negativeMean": float(np.mean([row["score"] for row in negative_rows])),
        "negativeP95": float(np.percentile([row["score"] for row in negative_rows], 95)),
        "negativeMax": float(np.max([row["score"] for row in negative_rows])),
    },
    "carriers": [{"fy": int(y if y <= SIZE // 2 else y - SIZE), "fx": int(x), "channel": int(channel), "coherence": float(coherence[y, x, channel]), "phase": float(phase[y, x, channel])} for y, x, channel in indices],
}
(ROOT / "reports/openai-synthid-spectral-experiment.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
print(json.dumps(report["summary"], indent=2))
