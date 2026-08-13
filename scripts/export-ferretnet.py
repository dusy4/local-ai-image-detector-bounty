"""Export the official FerretNet-B checkpoint without its unsupported median op."""

from pathlib import Path
import importlib.util
import sys

import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "research" / "ferretnet"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


Ferret = load_module("ferretnet_model", SOURCE / "src" / "model" / "ferretnet.py").Ferret
get_lpd_dict = load_module("ferretnet_lpd", SOURCE / "src" / "model" / "lpd.py").get_lpd_dict


class ResidualFerret(nn.Module):
    """CNN body accepting the precomputed x - masked_median(x) residual."""

    def __init__(self, model: Ferret):
        super().__init__()
        self.cbr1 = model.cbr1
        self.cbr2 = model.cbr2
        self.feature = model.feature
        self.avg_pool = model.avg_pool
        self.logit = model.logit

    def forward(self, residual):
        x = self.cbr1(residual)
        x = self.cbr2(x)
        x = self.feature(x)
        x = self.avg_pool(x)
        return self.logit(torch.flatten(x, 1))


model = Ferret(
    in_channels=3,
    num_classes=1,
    dim=96,
    depths=[2, 2],
    lpd_func="median",
    window_size=3,
    lpd_dict=get_lpd_dict(),
)
checkpoint = torch.load(
    SOURCE / "checkpoints" / "4cls_ckpt" / "ferretnet-b-median-3.pth",
    map_location="cpu",
    weights_only=True,
)["model"]
checkpoint = {key.removeprefix("module."): value for key, value in checkpoint.items()}
model.load_state_dict(checkpoint, strict=True)
exported = ResidualFerret(model).eval()

target = ROOT / "public" / "models" / "ferretnet" / "model.onnx"
target.parent.mkdir(parents=True, exist_ok=True)
torch.onnx.export(
    exported,
    torch.zeros(1, 3, 256, 256),
    target,
    input_names=["residual"],
    output_names=["logit"],
    opset_version=17,
    do_constant_folding=True,
    dynamo=False,
)
print(target)
