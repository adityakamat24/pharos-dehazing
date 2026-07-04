"""Staleness compositor for RevealNet (DESIGN.md §9d.3).

Per-pixel arbitration between the current restoration ``J_t`` and remembered scene
content ``M.rgb``: memory wins where its *decayed* trust exceeds the current-frame
confidence. The blend weight is ``sigmoid(k*(trust*age_decay(age) - conf_t))``.

Alongside the blended frame the compositor emits a **staleness** map — seconds (or
frames) since each pixel was last directly confirmed — upsampled to full resolution
and masked to where memory actually contributed. No analog exists in prior
restoration work (nearest is robotics occupancy-grid decay); it is a user-facing
honesty signal rendered as an overlay in the demo.
"""
from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F


def age_decay(age: torch.Tensor, half_life: float) -> torch.Tensor:
    """Exponential trust decay with a configurable half-life: 0.5 ** (age/half_life)."""
    hl = max(float(half_life), 1e-6)
    return torch.exp(-math.log(2.0) * age / hl)


def composite(
    rgb_t: torch.Tensor,
    conf_t: torch.Tensor,
    memory: Any,
    cfg: Any,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Blend current restoration with memory; return (out B,3,H,W, staleness B,1,H,W).

    ``rgb_t``: current restoration (full res). ``conf_t``: full-res confidence in
    (0,1]. ``memory``: a ``RevealMemory`` at mid res. ``cfg`` provides ``comp_k`` and
    ``half_life``. Memory buffers are upsampled to full res and cast to ``rgb_t``'s
    dtype for the blend; staleness stays in age units, masked by the blend weight.
    """
    h, w = rgb_t.shape[-2], rgb_t.shape[-1]
    dt = rgb_t.dtype
    mem_rgb = _up(memory.rgb, (h, w)).to(dt)
    mem_trust = _up(memory.trust, (h, w)).to(dt)
    mem_age = _up(memory.age, (h, w))                      # keep float32 age
    conf_t = conf_t.to(dt)

    k = float(_get(cfg, "comp_k", 8.0))
    half_life = float(_get(cfg, "half_life", 30.0))
    decayed = mem_trust * age_decay(mem_age, half_life).to(dt)   # effective memory trust
    weight = torch.sigmoid(k * (decayed - conf_t))              # B,1,H,W, memory share
    out = weight * mem_rgb + (1.0 - weight) * rgb_t
    staleness = mem_age * weight.to(mem_age.dtype)             # age where memory contributed
    return out.clamp(0.0, 1.0), staleness


def _up(x: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    if x.shape[-2:] == (size[0], size[1]):
        return x
    return F.interpolate(x, size=size, mode="bilinear", align_corners=False)


def _get(cfg: Any, key: str, default: Any) -> Any:
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)
