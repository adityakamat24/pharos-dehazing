"""Core building blocks for PharosNet (DESIGN.md §3).

RepNAFBlock: reparameterizable NAFNet-style block. The spatial mixing conv is a
multi-branch (3x3 + 1x1 + identity, RepVGG/DEA-Net style) depthwise conv at train
time that folds to a single depthwise 3x3 conv at inference. Depthwise is used
because a full conv at the expanded width would blow the 1.5-3M parameter budget.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNorm2d(nn.Module):
    """Channel-wise LayerNorm over C for NCHW tensors (NAFNet style).

    Statistics are computed in float32 for AMP safety, then cast back.
    """

    def __init__(self, channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dt = x.dtype
        xf = x.float()
        u = xf.mean(1, keepdim=True)
        s = (xf - u).pow(2).mean(1, keepdim=True)
        xn = ((xf - u) * torch.rsqrt(s + self.eps)).to(dt)
        return xn * self.weight.to(dt).view(1, -1, 1, 1) + self.bias.to(dt).view(1, -1, 1, 1)


class SimpleGate(nn.Module):
    """Channel-split multiplicative gate (NAFNet): [a, b] -> a * b."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a, b = x.chunk(2, dim=1)
        return a * b


class RepConv(nn.Module):
    """Reparameterizable convolution (RepVGG/DEA-Net style).

    Train time: parallel k x k, 1x1 and (when in==out, stride==1) identity
    branches, each with its own BatchNorm. `reparameterize()` folds all branches
    (BN included) into a single k x k conv with bias; numerically equivalent in
    eval mode to atol < 1e-4. Supports grouped/depthwise convs.
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel: int = 3,
        stride: int = 1,
        groups: int = 1,
        use_bn: bool = True,
    ) -> None:
        super().__init__()
        assert kernel % 2 == 1, "kernel must be odd"
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.kernel = kernel
        self.stride = stride
        self.groups = groups
        self.use_bn = use_bn
        self.deployed = False
        pad = kernel // 2

        self.conv_k = nn.Conv2d(in_ch, out_ch, kernel, stride, pad, groups=groups, bias=not use_bn)
        self.bn_k = nn.BatchNorm2d(out_ch) if use_bn else None
        self.conv_1 = nn.Conv2d(in_ch, out_ch, 1, stride, 0, groups=groups, bias=not use_bn)
        self.bn_1 = nn.BatchNorm2d(out_ch) if use_bn else None
        self.has_identity = in_ch == out_ch and stride == 1
        self.bn_id = nn.BatchNorm2d(out_ch) if (use_bn and self.has_identity) else None
        self.reparam_conv: nn.Conv2d | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.deployed:
            return self.reparam_conv(x)
        out = self._branch(self.conv_k, self.bn_k, x) + self._branch(self.conv_1, self.bn_1, x)
        if self.has_identity:
            out = out + (self.bn_id(x) if self.bn_id is not None else x)
        return out

    @staticmethod
    def _branch(conv: nn.Conv2d, bn: nn.BatchNorm2d | None, x: torch.Tensor) -> torch.Tensor:
        y = conv(x)
        return bn(y) if bn is not None else y

    # -- reparameterization ------------------------------------------------
    def _fold(
        self, weight: torch.Tensor, conv_bias: torch.Tensor | None, bn: nn.BatchNorm2d | None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if bn is None:
            b = conv_bias if conv_bias is not None else torch.zeros(self.out_ch, device=weight.device)
            return weight, b
        std = torch.sqrt(bn.running_var + bn.eps)
        t = (bn.weight / std).reshape(-1, 1, 1, 1)
        w = weight * t
        base = conv_bias if conv_bias is not None else torch.zeros(self.out_ch, device=weight.device)
        b = bn.bias + (base - bn.running_mean) * bn.weight / std
        return w, b

    def _pad_to_k(self, w: torch.Tensor) -> torch.Tensor:
        if w.shape[-1] == self.kernel:
            return w
        p = (self.kernel - w.shape[-1]) // 2
        return F.pad(w, [p, p, p, p])

    def _identity_kernel(self, device: torch.device) -> torch.Tensor:
        ci = self.in_ch // self.groups
        co = self.out_ch // self.groups
        c = self.kernel // 2
        w = torch.zeros(self.out_ch, ci, self.kernel, self.kernel, device=device)
        for o in range(self.out_ch):
            w[o, o % co, c, c] = 1.0  # in==out guaranteed by has_identity, so ci==co
        return w

    @torch.no_grad()
    def reparameterize(self) -> None:
        if self.deployed:
            return
        device = self.conv_k.weight.device
        wk, bk = self._fold(self.conv_k.weight, self.conv_k.bias, self.bn_k)
        w1, b1 = self._fold(self.conv_1.weight, self.conv_1.bias, self.bn_1)
        w = wk + self._pad_to_k(w1)
        b = bk + b1
        if self.has_identity:
            wid, bid = self._fold(self._identity_kernel(device), None, self.bn_id)
            w = w + wid
            b = b + bid
        conv = nn.Conv2d(
            self.in_ch, self.out_ch, self.kernel, self.stride, self.kernel // 2,
            groups=self.groups, bias=True,
        )
        conv.weight.data.copy_(w)
        conv.bias.data.copy_(b)
        self.reparam_conv = conv.to(device)
        for name in ("conv_k", "bn_k", "conv_1", "bn_1", "bn_id"):
            if getattr(self, name, None) is not None:
                delattr(self, name)
        self.deployed = True


class RepNAFBlock(nn.Module):
    """NAFNet block with a reparameterizable depthwise spatial conv.

    Flow: LN -> 1x1 expand -> RepConv (depthwise 3x3, multi-branch) -> SimpleGate
    -> simplified channel attention -> 1x1 project (+residual); then an FFN with
    SimpleGate. `beta`/`gamma` residual scales init at 0 so the block starts as an
    identity map (stable training, near-identity net at init).
    """

    def __init__(self, c: int, dw_expand: int = 2, ffn_expand: int = 2) -> None:
        super().__init__()
        dw = c * dw_expand
        self.norm1 = LayerNorm2d(c)
        self.conv1 = nn.Conv2d(c, dw, 1)
        self.rep = RepConv(dw, dw, kernel=3, groups=dw, use_bn=True)  # depthwise reparam conv
        self.sg = SimpleGate()
        self.sca = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(dw // 2, dw // 2, 1))
        self.conv3 = nn.Conv2d(dw // 2, c, 1)

        self.norm2 = LayerNorm2d(c)
        ffn = c * ffn_expand
        self.conv4 = nn.Conv2d(c, ffn, 1)
        self.conv5 = nn.Conv2d(ffn // 2, c, 1)

        self.beta = nn.Parameter(torch.zeros(1, c, 1, 1))
        self.gamma = nn.Parameter(torch.zeros(1, c, 1, 1))

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        x = self.norm1(inp)
        x = self.conv1(x)
        x = self.rep(x)
        x = self.sg(x)
        x = x * self.sca(x)
        x = self.conv3(x)
        y = inp + self.beta * x
        x = self.conv4(self.norm2(y))
        x = self.sg(x)
        x = self.conv5(x)
        return y + self.gamma * x

    def reparameterize(self) -> None:
        self.rep.reparameterize()


class HaarDownsample(nn.Module):
    """Wavelet (Haar) downsampling: 4 subbands per channel then a 1x1 merge conv.

    The Haar filter bank is a fixed orthonormal buffer applied depthwise with
    stride 2 (LL, LH, HL, HH -> 4*C), then merged to `out_ch`. Odd spatial sizes
    are replicate-padded to even internally.
    """

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.in_ch = in_ch
        f = 0.5 * torch.tensor(
            [
                [[1.0, 1.0], [1.0, 1.0]],    # LL
                [[1.0, 1.0], [-1.0, -1.0]],  # LH
                [[1.0, -1.0], [1.0, -1.0]],  # HL
                [[1.0, -1.0], [-1.0, 1.0]],  # HH
            ],
            dtype=torch.float32,
        )
        w = f.unsqueeze(1).repeat(in_ch, 1, 1, 1)  # [4*in_ch, 1, 2, 2], grouped per input channel
        self.register_buffer("haar", w)
        self.merge = nn.Conv2d(4 * in_ch, out_ch, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, c, h, w = x.shape
        if h % 2 or w % 2:
            x = F.pad(x, [0, w % 2, 0, h % 2], mode="replicate")
        y = F.conv2d(x, self.haar.to(x.dtype), stride=2, groups=c)
        return self.merge(y)


class FiLM(nn.Module):
    """Feature-wise linear modulation: per-channel scale/shift from a vector.

    fc is zero-initialized so the module starts as identity (gamma=0, beta=0).
    """

    def __init__(self, channels: int, cond_dim: int) -> None:
        super().__init__()
        self.fc = nn.Linear(cond_dim, 2 * channels)
        nn.init.zeros_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        g, b = self.fc(cond).chunk(2, dim=1)
        return x * (1 + g[:, :, None, None]) + b[:, :, None, None]
