"""Tiered aligner for RevealNet (DESIGN.md §9d.1).

A small CNN on the low-res (current frame, memory-anchor) pair predicts a global
homography (as 4-point corner offsets composed to a 3x3 H) plus a content-aware
trust signal: a scalar (global alignment confidence) and a spatial mask (per-region
trust). Tier logic: when the scalar trust falls below ``t_lo`` the warp collapses to
identity and trust to zero (memory freeze) so a lost track never corrupts memory.

An optional external motion prior (gyro / codec motion-vectors, expressed as 4-point
normalized corner offsets) is fused *additively* onto the predicted offsets — a hook
only, no hardware code. Everything is CPU-importable, AMP-safe (homography math runs
in float32) and uses ONNX-friendly ops in ``warp_grid`` (meshgrid + matmul + divide).
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# Unit-square corners in normalized [-1, 1] grid_sample coordinates: TL, TR, BR, BL.
_CORNERS = ((-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0))


def four_point_to_homography(offsets: torch.Tensor) -> torch.Tensor:
    """Compose a 3x3 homography from 4 corner offsets via the DLT (normalized coords).

    ``offsets``: B,4,2 displacements (in [-1,1] units) of the unit-square corners.
    Returns H (B,3,3, float32) mapping *source* (anchor) -> *destination* (current)
    normalized coordinates: ``p_dst ~ H @ p_src``. Solved as a batched 8x8 linear
    system; math is forced to float32 for AMP safety (linalg is float-only).
    """
    b = offsets.shape[0]
    dtype, device = torch.float32, offsets.device
    src = torch.tensor(_CORNERS, dtype=dtype, device=device).unsqueeze(0).expand(b, 4, 2)
    dst = src + offsets.to(dtype)
    x, y = src[..., 0], src[..., 1]          # B,4
    u, v = dst[..., 0], dst[..., 1]          # B,4
    z = torch.zeros_like(x)
    o = torch.ones_like(x)
    # Two rows per correspondence: [x y 1 0 0 0 -ux -uy]=u ; [0 0 0 x y 1 -vx -vy]=v
    row_u = torch.stack([x, y, o, z, z, z, -u * x, -u * y], dim=-1)  # B,4,8
    row_v = torch.stack([z, z, z, x, y, o, -v * x, -v * y], dim=-1)  # B,4,8
    a = torch.cat([row_u, row_v], dim=1)                              # B,8,8
    rhs = torch.cat([u, v], dim=1).unsqueeze(-1)                      # B,8,1
    h = torch.linalg.solve(a, rhs).squeeze(-1)                        # B,8
    ones = torch.ones(b, 1, dtype=dtype, device=device)
    return torch.cat([h, ones], dim=1).reshape(b, 3, 3)


def invert_homography(h: torch.Tensor) -> torch.Tensor:
    """Batched 3x3 inverse in float32 (AMP-safe)."""
    return torch.linalg.inv(h.to(torch.float32))


def warp_grid(mat: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    """Sampling grid for ``F.grid_sample`` from a homography (ONNX-friendly ops only).

    ``mat``: B,3,3 mapping *output* normalized coords -> *input* (source) normalized
    coords. ``size``: (h, w) of the output. Returns grid B,h,w,2 (x, y in [-1,1]).
    Uses meshgrid + matmul + perspective divide — no gather, no 5D grid_sample.
    """
    b = mat.shape[0]
    h, w = int(size[0]), int(size[1])
    dtype, device = torch.float32, mat.device
    ys = torch.linspace(-1.0, 1.0, h, dtype=dtype, device=device)
    xs = torch.linspace(-1.0, 1.0, w, dtype=dtype, device=device)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    base = torch.stack([xx, yy, torch.ones_like(xx)], dim=-1)  # h,w,3
    base = base.reshape(1, h * w, 3).expand(b, h * w, 3)
    warped = base @ mat.to(dtype).transpose(1, 2)             # B,hw,3
    denom = warped[..., 2:3]
    sign = torch.where(denom < 0, -torch.ones_like(denom), torch.ones_like(denom))
    denom = denom + sign * 1e-6                                # keep sign, avoid /0
    grid = warped[..., :2] / denom
    return grid.reshape(b, h, w, 2)


class TieredAligner(nn.Module):
    """Content-aware global homography + trust from a low-res (current, anchor) pair.

    The offset regressor is zero-initialized so an untrained / uncertain aligner
    predicts the identity homography (graceful degradation to identity), and the
    trust heads start near 0.5. ``t_lo`` is the freeze threshold applied here.
    """

    def __init__(self, in_ch: int = 3, width: int = 24, t_lo: float = 0.2) -> None:
        super().__init__()
        self.t_lo = float(t_lo)
        c1, c2 = width, width * 2
        self.body = nn.Sequential(
            nn.Conv2d(2 * in_ch, c1, 3, 2, 1), nn.GELU(),
            nn.Conv2d(c1, c2, 3, 2, 1), nn.GELU(),
            nn.Conv2d(c2, c2, 3, 2, 1), nn.GELU(),
        )
        self.trust_map = nn.Conv2d(c2, 1, 1)          # spatial (per-region) trust
        self.to_offset = nn.Linear(c2, 8)             # 4 corner (x, y) offsets
        self.to_scalar = nn.Linear(c2, 1)             # global alignment trust
        nn.init.zeros_(self.to_offset.weight)
        nn.init.zeros_(self.to_offset.bias)           # identity homography at init

    def forward(
        self,
        cur_lr: torch.Tensor,
        anchor_lr: torch.Tensor,
        motion_prior: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (H B,3,3, trust_map B,1,h,w in (0,1), scalar_trust B,1 in [0,1)).

        Tier logic is applied here: samples whose scalar trust < ``t_lo`` get the
        identity homography and zeroed trust (memory freeze). ``motion_prior`` (B,8
        normalized 4-point offsets) is added to the predicted offsets when given.
        """
        h, w = cur_lr.shape[-2], cur_lr.shape[-1]
        feat = self.body(torch.cat([cur_lr, anchor_lr], dim=1))
        pooled = F.adaptive_avg_pool2d(feat, 1).flatten(1)
        offset = self.to_offset(pooled)                       # B,8
        if motion_prior is not None:
            offset = offset + motion_prior.to(offset.dtype)   # additive sensor prior
        scalar = torch.sigmoid(self.to_scalar(pooled))        # B,1 in (0,1)
        tmap = torch.sigmoid(self.trust_map(feat))            # B,1,hf,wf
        tmap = F.interpolate(tmap, size=(h, w), mode="bilinear", align_corners=False)

        homog = four_point_to_homography(offset.reshape(-1, 4, 2))  # B,3,3 float32

        # Tier: freeze samples below t_lo -> identity warp + zero trust.
        frozen = (scalar < self.t_lo).view(-1, 1, 1)
        eye = torch.eye(3, dtype=homog.dtype, device=homog.device).expand_as(homog)
        homog = torch.where(frozen, eye, homog)
        keep = (~frozen.view(-1, 1)).to(scalar.dtype)
        scalar = scalar * keep
        tmap = tmap * keep.view(-1, 1, 1, 1)
        return homog, tmap, scalar

    # exposed helpers (ONNX-friendly warp + homography algebra)
    warp_grid = staticmethod(warp_grid)
    four_point_to_homography = staticmethod(four_point_to_homography)
    invert_homography = staticmethod(invert_homography)
