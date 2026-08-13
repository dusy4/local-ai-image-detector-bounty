"""Export the pinned official Community Forensics 224px checkpoint."""

from pathlib import Path
import importlib.util

from safetensors.torch import load_file
import torch


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "research" / "community-forensics-official"
spec = importlib.util.spec_from_file_location("commfor_models", SOURCE / "models.py")
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(module)

model = module.ViTClassifier(model_size="small", input_size=224, patch_size=16, device="cpu")
model.load_state_dict(load_file(ROOT / "research" / "commfor-224.safetensors"), strict=True)
model.eval()
target = ROOT / "public" / "models" / "community-forensics-official" / "model.onnx"
target.parent.mkdir(parents=True, exist_ok=True)
torch.onnx.export(
    model,
    torch.zeros(1, 3, 224, 224),
    target,
    input_names=["pixel_values"],
    output_names=["logit"],
    opset_version=17,
    do_constant_folding=True,
    dynamo=False,
)
print(target)
