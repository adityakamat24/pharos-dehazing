"""Discriminators for the vivid-mode perceptual/adversarial fine-tune (WS-vivid).

Two spectral-norm architectures, both taking a 3-channel RGB crop in [0, 1] (the
restored ``J`` as *fake*, the clean GT as *real*) and emitting a logits map:

- :class:`PatchDiscriminator` — the default 70x70-PatchGAN lineage: a stack of
  strided spectral-norm convs + LeakyReLU, no BatchNorm (spectral norm supplies
  the Lipschitz control). ~1.7M params at the defaults.
- :class:`UNetDiscriminator` — a compact Real-ESRGAN-style U-Net with skip
  connections; per-pixel logits for sharper texture gradients. Behind the
  ``disc.type: unet`` config flag.

These are *training-only* modules owned by :class:`~pharos.losses.vivid_losses.VividLoss`
(they never touch the deployed PharosNet or its inference cost). Everything here
is CPU-importable and float32/AMP-safe.
"""
from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import spectral_norm


def _get(cfg: Any, key: str, default: Any) -> Any:
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def _sn(module: nn.Module) -> nn.Module:
    """Spectral-normalise a conv (parametrizations API; state_dict round-trips)."""
    return spectral_norm(module)


class PatchDiscriminator(nn.Module):
    """PatchGAN with spectral norm (DESIGN.md N4 photography variant).

    ``in_ch`` RGB -> a map of patch logits (higher = more "real"). Channels grow
    ``base_ch * 2**i`` capped at ``max_ch``; the penultimate conv is stride 1 so
    the receptive field grows without collapsing the map to 1x1.
    """

    def __init__(
        self,
        in_ch: int = 3,
        base_ch: int = 64,
        n_layers: int = 3,
        max_ch: int = 256,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = [
            _sn(nn.Conv2d(in_ch, base_ch, 4, 2, 1)),
            nn.LeakyReLU(0.2, inplace=True),
        ]
        ch = base_ch
        for i in range(1, n_layers):  # strided downsampling stack
            nxt = min(base_ch * (2 ** i), max_ch)
            layers += [_sn(nn.Conv2d(ch, nxt, 4, 2, 1)), nn.LeakyReLU(0.2, inplace=True)]
            ch = nxt
        nxt = min(base_ch * (2 ** n_layers), max_ch)  # stride-1 penultimate (wider RF)
        layers += [_sn(nn.Conv2d(ch, nxt, 4, 1, 1)), nn.LeakyReLU(0.2, inplace=True)]
        layers += [_sn(nn.Conv2d(nxt, 1, 4, 1, 1))]  # -> patch logits
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class UNetDiscriminator(nn.Module):
    """Compact U-Net discriminator with spectral norm (Real-ESRGAN style).

    Per-pixel real/fake logits via an encoder/decoder with additive skips. Inputs
    should have spatial dims divisible by 8 so the skip tensors align (restoration
    crops are 256, so this always holds in training).
    """

    def __init__(self, in_ch: int = 3, base_ch: int = 32) -> None:
        super().__init__()
        c = base_ch
        self.conv0 = nn.Conv2d(in_ch, c, 3, 1, 1)
        self.down1 = _sn(nn.Conv2d(c, c * 2, 4, 2, 1))
        self.down2 = _sn(nn.Conv2d(c * 2, c * 4, 4, 2, 1))
        self.down3 = _sn(nn.Conv2d(c * 4, c * 8, 4, 2, 1))
        self.up1 = _sn(nn.Conv2d(c * 8, c * 4, 3, 1, 1))
        self.up2 = _sn(nn.Conv2d(c * 4, c * 2, 3, 1, 1))
        self.up3 = _sn(nn.Conv2d(c * 2, c, 3, 1, 1))
        self.tail0 = _sn(nn.Conv2d(c, c, 3, 1, 1))
        self.tail1 = _sn(nn.Conv2d(c, c, 3, 1, 1))
        self.out = nn.Conv2d(c, 1, 3, 1, 1)

    @staticmethod
    def _up(x: torch.Tensor) -> torch.Tensor:
        return F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a = F.leaky_relu(self.conv0(x), 0.2, inplace=True)
        d1 = F.leaky_relu(self.down1(a), 0.2, inplace=True)
        d2 = F.leaky_relu(self.down2(d1), 0.2, inplace=True)
        d3 = F.leaky_relu(self.down3(d2), 0.2, inplace=True)
        u1 = F.leaky_relu(self.up1(self._up(d3)), 0.2, inplace=True) + d2
        u2 = F.leaky_relu(self.up2(self._up(u1)), 0.2, inplace=True) + d1
        u3 = F.leaky_relu(self.up3(self._up(u2)), 0.2, inplace=True) + a
        t = F.leaky_relu(self.tail0(u3), 0.2, inplace=True)
        t = F.leaky_relu(self.tail1(t), 0.2, inplace=True)
        return self.out(t)


def build_discriminator(cfg: Any = None) -> nn.Module:
    """Construct a discriminator from a ``disc`` config dict.

    ``type``: ``"patch"`` (default) or ``"unet"``. Remaining keys are the
    per-arch constructor args (``base_ch``, ``n_layers``, ``max_ch``, ``in_ch``).
    """
    dtype = str(_get(cfg, "type", "patch")).lower()
    in_ch = int(_get(cfg, "in_ch", 3))
    if dtype in ("unet", "unet_sn", "unetdiscriminator"):
        return UNetDiscriminator(in_ch=in_ch, base_ch=int(_get(cfg, "base_ch", 32)))
    return PatchDiscriminator(
        in_ch=in_ch,
        base_ch=int(_get(cfg, "base_ch", 64)),
        n_layers=int(_get(cfg, "n_layers", 3)),
        max_ch=int(_get(cfg, "max_ch", 256)),
    )


def count_params(module: nn.Module) -> int:
    """Total parameter count (helper for tests / reporting)."""
    return sum(p.numel() for p in module.parameters())
