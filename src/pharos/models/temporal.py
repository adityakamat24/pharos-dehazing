"""Causal temporal state for video mode (DESIGN.md §3.6).

A ConvGRU refines the bilateral grid over time, fed the flattened grid plus a
projection of the low-res features. A confidence-weighted EMA of the refined grid
adds stability, and a scene-cut detector (low-res histogram L1 distance) resets
the state on hard cuts. Strictly causal (no future frames).
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvGRUCell(nn.Module):
    """Standard convolutional GRU cell."""

    def __init__(self, in_ch: int, hidden_ch: int, kernel: int = 3) -> None:
        super().__init__()
        pad = kernel // 2
        self.conv_zr = nn.Conv2d(in_ch + hidden_ch, 2 * hidden_ch, kernel, 1, pad)
        self.conv_h = nn.Conv2d(in_ch + hidden_ch, hidden_ch, kernel, 1, pad)

    def forward(self, x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        z, r = torch.sigmoid(self.conv_zr(torch.cat([x, h], dim=1))).chunk(2, dim=1)
        q = torch.tanh(self.conv_h(torch.cat([x, r * h], dim=1)))
        return (1 - z) * h + z * q


class TemporalModule(nn.Module):
    """ConvGRU + confidence-weighted EMA over the bilateral grid, with scene-cut reset.

    State is a dict {"h": gru hidden, "ema": smoothed grid (flat), "hist": last
    low-res histogram}. On each frame the current grid (depth folded into channels)
    is concatenated with a projection of the low-res features and fed to the GRU;
    the EMA gain per grid cell is proportional to confidence so trusted frames
    follow the new estimate while uncertain frames lean on history.
    """

    def __init__(
        self,
        grid_ch: int,
        feat_ch: int,
        feat_proj: int = 32,
        ema_gain: float = 0.7,
        scene_thresh: float = 0.5,
        hist_bins: int = 16,
    ) -> None:
        super().__init__()
        self.grid_ch = grid_ch
        self.ema_gain = ema_gain
        self.scene_thresh = scene_thresh
        self.hist_bins = hist_bins
        self.feat_proj = nn.Conv2d(feat_ch, feat_proj, 1)
        self.gru = ConvGRUCell(grid_ch + feat_proj, grid_ch)

    def _histogram(self, frame: torch.Tensor) -> torch.Tensor:
        # frame B,3,h,w in [0,1] -> normalized per-channel histogram B,3*bins
        b = frame.shape[0]
        idx = torch.clamp((frame * self.hist_bins).long(), 0, self.hist_bins - 1)
        hists = [F.one_hot(idx[:, c].reshape(b, -1), self.hist_bins).float().sum(1) for c in range(3)]
        h = torch.cat(hists, dim=1)
        return h / (h.sum(1, keepdim=True) + 1e-6)

    def forward(
        self,
        grid: torch.Tensor,
        feat: torch.Tensor,
        conf: torch.Tensor,
        frame_lr: torch.Tensor,
        state: Optional[dict],
    ) -> tuple[torch.Tensor, dict]:
        b, c, d, gh, gw = grid.shape
        g_flat = grid.reshape(b, c * d, gh, gw)
        feat_r = self.feat_proj(F.interpolate(feat, size=(gh, gw), mode="bilinear", align_corners=False))
        conf_r = F.interpolate(conf, size=(gh, gw), mode="bilinear", align_corners=False)
        hist = self._histogram(frame_lr)

        if state is None:
            h = torch.zeros(b, self.grid_ch, gh, gw, device=grid.device, dtype=grid.dtype)
            ema = g_flat
        else:
            h, ema = state["h"], state["ema"]
            dist = (hist - state["hist"]).abs().sum(1)  # B
            cut = (dist > self.scene_thresh).to(grid.dtype).view(b, 1, 1, 1)
            h = h * (1 - cut)
            ema = ema * (1 - cut) + g_flat * cut

        h_new = self.gru(torch.cat([g_flat, feat_r], dim=1), h)
        gain = (self.ema_gain * conf_r).clamp(0.0, 1.0)
        ema_new = (1 - gain) * ema + gain * h_new
        smoothed = ema_new.reshape(b, c, d, gh, gw)
        return smoothed, {"h": h_new, "ema": ema_new, "hist": hist}
