"""Prediction heads for PharosNet (DESIGN.md §3.2, §3.4, §3.5).

- DegradationHead: pooled deep features -> {beta, airlight, sigma, domain_logits}
  plus a conditioning embedding for FiLM.
- ConfidenceHead: low-res log-variance -> upsampled calibrated confidence in (0,1].
- DetailBranch: magnitude-bounded residual detail from concat(I, J0).
- TransmissionHead: auxiliary low-res transmission (sigmoid), training signal only.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import RepConv


class DegradationHead(nn.Module):
    """Pooled features -> continuous degradation estimate + FiLM conditioning.

    beta (density) and sigma (non-homogeneity) use softplus (>=0); airlight uses
    sigmoid (RGB color in [0,1]); domain_logits are raw (softmax at loss time).
    The conditioning vector concatenates a learned embedding with the estimated
    physical quantities and a small domain embedding (DESIGN: continuous, not a
    discrete classifier). Its width is exposed as `cond_dim` for the FiLM modules.
    """

    def __init__(self, in_ch: int, hidden: int = 128, embed: int = 32, domain_embed: int = 8) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_ch, hidden), nn.GELU(), nn.Linear(hidden, hidden), nn.GELU()
        )
        self.head_beta = nn.Linear(hidden, 1)
        self.head_air = nn.Linear(hidden, 3)
        self.head_sigma = nn.Linear(hidden, 1)
        self.head_domain = nn.Linear(hidden, 3)
        self.embed = nn.Linear(hidden, embed)
        self.domain_embed = nn.Linear(3, domain_embed)
        self.cond_dim = embed + 1 + 3 + 1 + domain_embed

    def forward(self, feat: torch.Tensor) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        v = F.adaptive_avg_pool2d(feat, 1).flatten(1)
        h = self.mlp(v)
        beta = F.softplus(self.head_beta(h))
        airlight = torch.sigmoid(self.head_air(h))
        sigma = F.softplus(self.head_sigma(h))
        domain_logits = self.head_domain(h)
        cond = torch.cat(
            [self.embed(h), beta, airlight, sigma, self.domain_embed(domain_logits)], dim=1
        )
        deg = {"beta": beta, "airlight": airlight, "sigma": sigma, "domain_logits": domain_logits}
        return deg, cond


class ConfidenceHead(nn.Module):
    """Low-res log-variance -> full-res calibrated confidence in (0,1].

    Map: conf = exp(-relu(logvar)). It is monotone non-increasing in the predicted
    log-variance: logvar <= 0 -> conf = 1 (fully trusted), large logvar -> conf -> 0.
    Returns (confidence, logvar) both at full res; logvar feeds the heteroscedastic
    NLL loss (|J-GT|/sigma + log sigma) downstream.
    """

    def __init__(self, in_ch: int, mid: int = 16) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, mid, 3, 1, 1), nn.GELU(), nn.Conv2d(mid, 1, 3, 1, 1)
        )

    def forward(self, feat: torch.Tensor, out_hw: tuple[int, int]) -> tuple[torch.Tensor, torch.Tensor]:
        logvar_lr = self.net(feat)
        logvar = F.interpolate(logvar_lr, size=out_hw, mode="bilinear", align_corners=False)
        conf = torch.exp(-F.relu(logvar))
        return conf, logvar


class DetailBranch(nn.Module):
    """Magnitude-bounded residual detail branch (full res).

    Input concat(I, J0) (6 channels), `layers` reparameterizable 3x3 convs at
    `channels` width, output r = s * tanh(f). The per-channel scale `s` is learned
    and initialized small (0.05) so hallucination is bounded by construction.
    """

    def __init__(self, channels: int = 12, layers: int = 4, scale_init: float = 0.05) -> None:
        super().__init__()
        assert layers >= 2
        self.in_conv = RepConv(6, channels, 3, use_bn=True)
        self.mids = nn.ModuleList(
            [RepConv(channels, channels, 3, use_bn=True) for _ in range(layers - 2)]
        )
        self.out_conv = RepConv(channels, 3, 3, use_bn=True)
        self.act = nn.GELU()
        self.scale = nn.Parameter(torch.full((1, 3, 1, 1), scale_init))

    def forward(self, image: torch.Tensor, j0: torch.Tensor) -> torch.Tensor:
        x = self.act(self.in_conv(torch.cat([image, j0], dim=1)))
        for m in self.mids:
            x = self.act(m(x))
        f = self.out_conv(x)
        return self.scale * torch.tanh(f)

    def reparameterize(self) -> None:
        self.in_conv.reparameterize()
        for m in self.mids:
            m.reparameterize()
        self.out_conv.reparameterize()


class TransmissionHead(nn.Module):
    """Auxiliary low-res transmission map (B,1,h,w in [0,1]); training-only signal."""

    def __init__(self, in_ch: int, mid: int = 16) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, mid, 3, 1, 1), nn.GELU(), nn.Conv2d(mid, 1, 3, 1, 1)
        )

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.net(feat))
