"""Export the official SAFE checkpoint, including its bior1.3 DWT preprocessing."""

from pathlib import Path
import importlib.util

import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "research" / "safe"
spec = importlib.util.spec_from_file_location("safe_resnet", SOURCE / "models" / "resnet.py")
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(module)

model = module.resnet50(num_classes=2)
checkpoint = torch.load(SOURCE / "checkpoint" / "checkpoint-best.pth", map_location="cpu", weights_only=True)["model"]
model.load_state_dict(checkpoint, strict=True)
model.eval()


class SafeBody(nn.Module):
    def __init__(self, source):
        super().__init__()
        self.conv1 = source.conv1
        self.bn1 = source.bn1
        self.relu = source.relu
        self.maxpool = source.maxpool
        self.layer1 = source.layer1
        self.layer2 = source.layer2
        self.avgpool = source.avgpool
        self.fc1 = source.fc1

    def forward(self, diagonal_band):
        x = self.maxpool(self.relu(self.bn1(self.conv1(diagonal_band))))
        x = self.layer2(self.layer1(x))
        return self.fc1(torch.flatten(self.avgpool(x), 1))


exported = SafeBody(model).eval()

target = ROOT / "public" / "models" / "safe" / "model.onnx"
target.parent.mkdir(parents=True, exist_ok=True)
torch.onnx.export(
    exported,
    torch.zeros(1, 3, 256, 256),
    target,
    input_names=["diagonal_band"],
    output_names=["logits"],
    opset_version=17,
    do_constant_folding=True,
    dynamo=False,
)
print(target)
