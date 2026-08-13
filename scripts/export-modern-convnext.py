"""Export the pinned MIT modern-generator ConvNeXt checkpoint to ONNX."""

from pathlib import Path
import json

import timm
import torch


ROOT = Path(__file__).resolve().parents[1]
model = timm.create_model("convnextv2_base", pretrained=False, num_classes=2)
checkpoint = torch.load(ROOT / "research/modern-convnext/checkpoint_phase2.pth", map_location="cpu", weights_only=True)
model.load_state_dict(checkpoint["model"])
model.eval()
target = ROOT / "research/modern-convnext/model-fp32.onnx"
torch.onnx.export(
    model,
    torch.zeros(1, 3, 256, 256),
    target,
    input_names=["pixel_values"],
    output_names=["logits"],
    opset_version=17,
    do_constant_folding=True,
    dynamo=False,
)
print(json.dumps({"path": str(target), "bytes": target.stat().st_size}))
