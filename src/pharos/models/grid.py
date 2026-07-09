"""Bilateral affine grid: prediction, guidance, slicing, application (DESIGN.md §3.3).

The network predicts a low-res bilateral grid G of per-cell 3x4 affine transforms.
A full-res guidance map g in [0,1] gives the third (range) coordinate; trilinear
slicing yields a per-pixel affine (M, b), applied to the input to get the coarse
restoration J0 = M*I + b. Slicing uses F.grid_sample so it is fully differentiable.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class BilateralGridHead(nn.Module):
    """Encoder features -> bilateral grid G of shape B,coeffs,D,Gh,Gw.

    coeffs=12 packs a 3x4 affine (9 for M, 3 for b). The output conv is
    initialized to emit the identity affine everywhere (M=I, b=0) so J0 ~= I at
    init (near-identity net; helps the severity gate and stable training).
    Channel layout after view is coeff-major: channel = coeff*D + d.
    """

    def __init__(self, in_ch: int, depth: int = 8, size: int = 16, coeffs: int = 12) -> None:
        super().__init__()
        self.depth = depth
        self.size = size
        self.coeffs = coeffs
        self.body = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(in_ch, in_ch, 3, 1, 1),
            nn.GELU(),
        )
        self.to_grid = nn.Conv2d(in_ch, coeffs * depth, 1)
        nn.init.zeros_(self.to_grid.weight)
        bias = torch.zeros(coeffs * depth)
        for coeff in (0, 4, 8):  # diagonal of the 3x3 M -> identity
            bias[coeff * depth : (coeff + 1) * depth] = 1.0
        self.to_grid.bias.data.copy_(bias)

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        x = self.body(feat)
        # bilinear resize, not adaptive_avg_pool2d: ONNX cannot express adaptive
        # pooling for non-divisible input sizes and the numerics are equivalent
        # for a smooth feature map feeding a learned 1x1.
        x = F.interpolate(x, size=(self.size, self.size), mode="bilinear", align_corners=False)
        x = self.to_grid(x)
        b = x.shape[0]
        return x.view(b, self.coeffs, self.depth, self.size, self.size)


class GuidanceNet(nn.Module):
    """3-conv guidance network on the full-res frame -> g in [0,1] (B,1,H,W)."""

    def __init__(self, mid: int = 16) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, mid, 1),
            nn.GELU(),
            nn.Conv2d(mid, mid, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(mid, 1, 1),
        )

    def forward(self, frame: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.net(frame))


def slice_grid(grid: torch.Tensor, guidance: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Trilinearly slice a bilateral grid at every full-res pixel.

    grid: B,12,D,Gh,Gw ; guidance: B,1,H,W in [0,1].
    Because the spatial sample points form a regular mesh, trilinear slicing
    decomposes exactly into (a) bilinear spatial upsampling of every depth
    slice (align_corners=True == grid_sample on a linspace(-1,1) mesh) and
    (b) a per-pixel lerp between the two neighbouring depth slices selected by
    the guidance value. Identical numerics to 5D grid_sample with border
    padding, but expressible in ONNX (5D GridSample is not exportable).
    Returns per-pixel affine M (B,9,H,W) and b (B,3,H,W).
    """
    b, c, d, gh, gw = grid.shape
    _, _, h, w = guidance.shape
    up = F.interpolate(grid.reshape(b, c * d, gh, gw), size=(h, w), mode="bilinear", align_corners=True)
    up = up.view(b, c, d, h, w)
    z = guidance.clamp(0.0, 1.0).squeeze(1) * (d - 1)  # B,H,W in [0, D-1]
    z0 = z.floor().long().clamp(0, d - 1)
    z1 = (z0 + 1).clamp(max=d - 1)
    wz = (z - z0.to(z.dtype)).view(b, 1, h, w)
    idx0 = z0.view(b, 1, 1, h, w).expand(b, c, 1, h, w)
    idx1 = z1.view(b, 1, 1, h, w).expand(b, c, 1, h, w)
    v0 = up.gather(2, idx0).squeeze(2)  # B,c,H,W
    v1 = up.gather(2, idx1).squeeze(2)
    sampled = v0 * (1.0 - wz) + v1 * wz
    return sampled[:, :9], sampled[:, 9:12]


def apply_affine(image: torch.Tensor, m: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Apply per-pixel affine: J0[i] = sum_j M[i,j]*I[j] + b[i].

    image: B,3,H,W ; m: B,9,H,W (row-major 3x3) ; b: B,3,H,W -> J0: B,3,H,W.
    """
    bsz, _, h, w = image.shape
    m = m.view(bsz, 3, 3, h, w)
    return torch.einsum("bijhw,bjhw->bihw", m, image) + b
