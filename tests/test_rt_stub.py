"""Shared test fixtures for the WS-E (rt) tests.

Defines a tiny ``StubModel`` implementing the ``pharos.contracts.PharosModel`` protocol
(valid ``PharosOutput``, ``.reparameterize()``, recurrent state) and a small ``Config``
factory. This file is import-only (its name matches ``test_rt_*`` so it lives with the
suite); it contains no tests of its own.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from pharos.config import Config
from pharos.contracts import PharosOutput


class StubModel(nn.Module):
    """Minimal, ONNX-friendly stand-in for PharosNet (convs + interpolate only).

    Behaviour is deterministic-ish and cheap: a bounded residual on the input, a sigmoid
    confidence map in (0, 1), a broadcast bilateral grid, a degradation dict, and a flat
    single-tensor recurrent state (a frame counter) so state threading is observable and
    video-mode export could in principle succeed. ``last_state_in`` records what the last
    forward received so tests can assert state is passed through.
    """

    def __init__(self, grid_depth: int = 8, grid_size: int = 16) -> None:
        super().__init__()
        self.grid_depth = grid_depth
        self.grid_size = grid_size
        self.body = nn.Conv2d(3, 3, 3, padding=1)
        self.conf = nn.Conv2d(3, 1, 3, padding=1)
        self.grid_head = nn.Conv2d(3, 12, 1)
        self.deg_fc = nn.Linear(3, 8)
        self.reparameterized = False
        self.last_state_in: object = "UNSET"

    def reparameterize(self) -> None:
        self.reparameterized = True

    def forward(self, frame, state=None, cond=None) -> PharosOutput:  # type: ignore[override]
        self.last_state_in = state
        b = frame.shape[0]
        residual = 0.02 * torch.tanh(self.body(frame))
        output = torch.clamp(frame + residual, 0.0, 1.0)
        confidence = torch.sigmoid(self.conf(frame))  # strictly in (0, 1)

        gs = self.grid_size
        low = F.interpolate(frame, size=(gs, gs), mode="bilinear", align_corners=False)
        g = self.grid_head(low)  # b,12,gs,gs
        grid = g.unsqueeze(2).expand(b, 12, self.grid_depth, self.grid_size, self.grid_size).contiguous()

        pooled = frame.mean(dim=(2, 3))  # b,3
        feats = self.deg_fc(pooled)      # b,8
        deg = {
            "beta": torch.sigmoid(feats[:, 0:1]),
            "airlight": torch.sigmoid(feats[:, 1:4]),
            "sigma": torch.sigmoid(feats[:, 4:5]),
            "domain_logits": feats[:, 5:8],
        }
        new_state = frame.new_zeros((b, 1)) if state is None else state + 1.0
        gate_alpha = deg["beta"]
        return PharosOutput(
            output=output,
            confidence=confidence,
            grid=grid,
            state=new_state,
            deg=deg,
            t_hat=None,
            aux={"gate_alpha": gate_alpha},
        )


class DictStateModel(StubModel):
    """StubModel whose recurrent state is a dict — used to exercise ExportUnsupported."""

    def forward(self, frame, state=None, cond=None) -> PharosOutput:  # type: ignore[override]
        out = super().forward(frame, state, cond)
        out.state = {"h": out.state}  # opaque, non-tensor state
        return out


def make_config(out_root: str, resolutions=None, frames: int = 5) -> Config:
    """Small config with the keys WS-E reads (bench.*, model.gate, out_root)."""
    return Config(
        {
            "out_root": str(out_root),
            "model": {
                "lowres": 64,
                "gate": {"beta_lo": 0.15, "beta_hi": 0.45},
                "grid": {"depth": 8, "size": 16},
            },
            "bench": {"resolutions": resolutions or [[64, 48], [96, 64]], "frames": frames},
        }
    )
