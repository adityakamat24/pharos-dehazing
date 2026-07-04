"""Reveal memory for RevealNet (DESIGN.md §9d.2).

``RevealMemory`` is the core mid-res registered buffer: a dict of three aligned
tensors ``{rgb: B,3,h,w ; trust: B,1,h,w ; age: B,1,h,w}``. It is threaded through
``PharosOutput.state`` (an opaque, per-sequence object — not a network parameter, so
truncated-BPTT across frames works via ``detach``). All updates are elementwise /
gather-free (``torch.where`` + lerp), so cost is O(pixels) per frame and trivially
real-time. ``age`` counts elapsed time since a pixel was last directly confirmed
(``dt`` units — frames by default, seconds if a real timestep is supplied).

Robust merge (burst-photography / Super-Res-Zoom robustness-mask pattern): the
effective observation weight is ``w = conf_t * align_trust``; where ``w`` exceeds the
merge threshold the memory RGB is lerped toward the current restoration, trust is
raised and age reset; elsewhere trust decays and age grows.
"""
from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from .aligner import invert_homography, warp_grid


class RevealMemory:
    """Mid-res registered scene memory: rgb + per-pixel trust + per-pixel age."""

    def __init__(self, rgb: torch.Tensor, trust: torch.Tensor, age: torch.Tensor) -> None:
        self.rgb = rgb      # B,3,h,w in [0,1], float32
        self.trust = trust  # B,1,h,w in [0,1], float32
        self.age = age      # B,1,h,w >= 0, float32 (dt units)

    # -- construction ---------------------------------------------------------
    @classmethod
    def seed(cls, rgb0: torch.Tensor, seed_trust: float) -> "RevealMemory":
        """First-frame seed: memory = current restoration at low initial trust, age 0."""
        rgb = rgb0.detach().to(torch.float32).clamp(0.0, 1.0)
        trust = torch.full_like(rgb[:, :1], float(seed_trust))
        age = torch.zeros_like(rgb[:, :1])
        return cls(rgb, trust, age)

    @property
    def buffers(self) -> dict[str, torch.Tensor]:
        """Named-buffer view (the {rgb, trust, age} dict of the spec)."""
        return {"rgb": self.rgb, "trust": self.trust, "age": self.age}

    def detach(self) -> "RevealMemory":
        """Return a graph-detached copy (for truncated BPTT over long horizons)."""
        return RevealMemory(self.rgb.detach(), self.trust.detach(), self.age.detach())

    def reset(self) -> None:
        """Clear the memory: zero rgb/trust, age 0 (tracking-loss re-anchor)."""
        self.rgb = torch.zeros_like(self.rgb)
        self.trust = torch.zeros_like(self.trust)
        self.age = torch.zeros_like(self.age)

    # -- geometric registration ----------------------------------------------
    def warp(self, homography: torch.Tensor) -> None:
        """Warp all channels by ``homography`` (anchor->current), in place.

        ``homography``: B,3,3 mapping the memory's (anchor) coords to the current
        view. The sampling grid needs the inverse (output->input). RGB uses border
        padding (no black seams); trust is multiplied by an in-bounds validity mask
        so newly-revealed regions fade to zero trust (border-fade at invalid areas).
        """
        h, w = self.rgb.shape[-2], self.rgb.shape[-1]
        inv = invert_homography(homography)
        grid = warp_grid(inv, (h, w)).to(self.rgb.dtype)
        valid = F.grid_sample(
            torch.ones_like(self.trust), grid, mode="bilinear",
            padding_mode="zeros", align_corners=True,
        )
        self.rgb = F.grid_sample(
            self.rgb, grid, mode="bilinear", padding_mode="border", align_corners=True
        )
        self.trust = F.grid_sample(
            self.trust, grid, mode="bilinear", padding_mode="border", align_corners=True
        ) * valid
        self.age = F.grid_sample(
            self.age, grid, mode="bilinear", padding_mode="border", align_corners=True
        ) * valid

    # -- robust temporal merge ------------------------------------------------
    def update(
        self,
        rgb_t: torch.Tensor,
        conf_t: torch.Tensor,
        align_trust: torch.Tensor,
        cfg: Any,
        dt: float = 1.0,
    ) -> None:
        """Merge the current restoration into memory (burst robust-merge), in place.

        ``rgb_t``: B,3,h,w restoration ; ``conf_t``/``align_trust``: B,1,h,w in [0,1].
        ``cfg`` provides ``merge_thresh``, ``decay_keep``, ``decay_miss``.
        Where ``w = conf_t*align_trust > merge_thresh``: rgb lerps toward ``rgb_t`` by
        ``w``, trust <- max(trust*decay_keep, w), age <- 0. Elsewhere: trust *=
        decay_miss, age += dt. Fully elementwise (torch.where), no gather.
        """
        rgb_t = rgb_t.to(self.rgb.dtype).clamp(0.0, 1.0)
        conf_t = conf_t.to(self.rgb.dtype)
        align_trust = align_trust.to(self.rgb.dtype)
        w = (conf_t * align_trust).clamp(0.0, 1.0)                 # B,1,h,w
        merge = w > float(_get(cfg, "merge_thresh", 0.1))         # bool mask

        merged_rgb = self.rgb * (1.0 - w) + rgb_t * w
        self.rgb = torch.where(merge.expand_as(self.rgb), merged_rgb, self.rgb)

        kept = torch.maximum(self.trust * float(_get(cfg, "decay_keep", 0.98)), w)
        missed = self.trust * float(_get(cfg, "decay_miss", 0.9))
        self.trust = torch.where(merge, kept, missed).clamp(0.0, 1.0)

        aged = self.age + float(dt)
        self.age = torch.where(merge, torch.zeros_like(self.age), aged)


def _get(cfg: Any, key: str, default: Any) -> Any:
    """Read an optional key from a Config/dict with a fallback default."""
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)
