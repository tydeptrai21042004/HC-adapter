# models/tuning_modules/residual_adapter.py
import torch
import torch.nn as nn
from typing import Iterable

# ---- small helpers -----------------------------------------------------------
_ACTS = {
    "relu": nn.ReLU(inplace=True),
    "gelu": nn.GELU(),
    "silu": nn.SiLU(inplace=True),
    "none": nn.Identity(),
}
_NORMS = {
    "bn":   lambda c: nn.BatchNorm2d(c),
    # Channel-wise layer-norm style for NCHW; GroupNorm(1, C) is a simple stand-in.
    "ln":   lambda c: nn.GroupNorm(1, c),
    "none": lambda c: nn.Identity(),
}

def _make_core(channels: int, reduction: int, norm: str, act: str) -> nn.Sequential:
    """
    Standard adapter bottleneck: 1x1 -> norm -> act -> 1x1.
    Using bias=False avoids redundant biases when norms are present.
    """
    hidden = max(1, channels // max(1, int(reduction)))
    return nn.Sequential(
        nn.Conv2d(channels, hidden, kernel_size=1, bias=False),
        _NORMS[norm](hidden),
        _ACTS[act],
        nn.Conv2d(hidden, channels, kernel_size=1, bias=False),
    )

# ---- adapter wrappers --------------------------------------------------------
class ParallelResidualAdapter(nn.Module):
    """
    PARALLEL residual adapter attached at a block output:
        y = Block(x)
        out = y + gate * Core(x)

    This "block-level" parallel form is a practical post-hoc adapter for frozen CNNs.
    """
    def __init__(self, block: nn.Module, channels: int,
                 reduction: int = 16, norm: str = "bn",
                 act: str = "relu", gate_init: float = 0.1):
        super().__init__()
        self.block = block
        self.core  = _make_core(channels, reduction, norm, act)
        self.gate  = nn.Parameter(torch.tensor(float(gate_init)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.block(x)
        a = self.core(x)
        return y + self.gate * a

class SeriesResidualAdapter(nn.Module):
    """
    SERIES residual adapter attached AFTER the block (light extra layer):
        y = Block(x)
        out = y + gate * Core(y)
    """
    def __init__(self, block: nn.Module, channels: int,
                 reduction: int = 16, norm: str = "bn",
                 act: str = "relu", gate_init: float = 0.1):
        super().__init__()
        self.block = block
        self.core  = _make_core(channels, reduction, norm, act)
        self.gate  = nn.Parameter(torch.tensor(float(gate_init)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.block(x)
        a = self.core(y)
        return y + self.gate * a

# ---- builders/attach utilities ----------------------------------------------
def build_residual_adapter(block: nn.Module, channels: int, mode: str = "parallel",
                           reduction: int = 16, norm: str = "bn",
                           act: str = "relu", gate_init: float = 0.1) -> nn.Module:
    if mode == "parallel":
        return ParallelResidualAdapter(
            block, channels, reduction=reduction, norm=norm, act=act, gate_init=gate_init
        )
    elif mode == "series":
        return SeriesResidualAdapter(
            block, channels, reduction=reduction, norm=norm, act=act, gate_init=gate_init
        )
    else:
        raise ValueError(f"Unknown residual adapter mode: {mode}")

def _wrap_resnet_layer(layer: nn.Sequential, mode: str,
                       reduction: int, norm: str, act: str, gate_init: float):
    """
    Replace each block in a torchvision ResNet layer with an adapter-wrapped block.
    Supports BasicBlock / Bottleneck.
    """
    for i, blk in enumerate(layer):
        # Infer output channel width per block
        if hasattr(blk, "conv3"):      # Bottleneck (ResNet-50/101)
            c_out = blk.conv3.out_channels
        else:                          # BasicBlock (ResNet-18/34)
            c_out = blk.conv2.out_channels

        layer[i] = build_residual_adapter(
            blk, c_out, mode=mode,
            reduction=reduction, norm=norm, act=act, gate_init=gate_init
        )

def attach_residual_adapters_resnet(model: nn.Module,
                                    stages: Iterable[int] = (1, 2, 3, 4),
                                    mode: str = "parallel",
                                    reduction: int = 16,
                                    norm: str = "bn",
                                    act: str = "relu",
                                    gate_init: float = 0.1) -> nn.Module:
    """
    Attach residual adapters to torchvision-style ResNet (layer1..layer4).
    """
    if 1 in stages: _wrap_resnet_layer(model.layer1, mode, reduction, norm, act, gate_init)
    if 2 in stages: _wrap_resnet_layer(model.layer2, mode, reduction, norm, act, gate_init)
    if 3 in stages: _wrap_resnet_layer(model.layer3, mode, reduction, norm, act, gate_init)
    if 4 in stages: _wrap_resnet_layer(model.layer4, mode, reduction, norm, act, gate_init)
    return model
