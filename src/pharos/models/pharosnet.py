"""PharosNet: the deployed student network (DESIGN.md §3).

Assembles the low-res encoder (RepNAF blocks + Haar downsampling), degradation /
FiLM conditioning, bilateral affine grid + full-res slicing, magnitude-bounded
detail branch, confidence + auxiliary transmission heads, optional causal ConvGRU
temporal state, and a continuous severity gate. Everything is CPU-importable,
float32/AMP-safe, and handles arbitrary (non-divisible) input sizes.
"""
from __future__ import annotations

from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..contracts import PharosOutput
from .blocks import FiLM, HaarDownsample, RepConv, RepNAFBlock
from .grid import BilateralGridHead, GuidanceNet, apply_affine, slice_grid
from .heads import ConfidenceHead, DegradationHead, DetailBranch, TransmissionHead
from .temporal import TemporalModule


def _get(cfg: Any, key: str, default: Any) -> Any:
    """Read an optional key from a Config/dict with a fallback default."""
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


class PharosNet(nn.Module):
    """Real-time, confidence-aware, unified dehazing & desmoking network.

    Constructed from the `model` section of the config. `forward(frame, state,
    cond) -> PharosOutput`. Video mode threads `state`; with `temporal=True` a
    fresh state is created even when `state is None` and returned. With
    `temporal=False` state stays None (image mode).
    """

    def __init__(self, cfg: Any) -> None:
        super().__init__()
        self.lowres = int(_get(cfg, "lowres", 256))
        enc = list(_get(cfg, "enc_channels", [24, 48, 96, 96]))
        assert len(enc) == 4, "enc_channels must have 4 stages"
        grid_cfg = _get(cfg, "grid", {"depth": 8, "size": 16})
        self.grid_depth = int(_get(grid_cfg, "depth", 8))
        self.grid_size = int(_get(grid_cfg, "size", 16))
        detail_cfg = _get(cfg, "detail", {"channels": 12, "layers": 4, "scale_init": 0.05})
        gate_cfg = _get(cfg, "gate", {"beta_lo": 0.15, "beta_hi": 0.45})
        self.beta_lo = float(_get(gate_cfg, "beta_lo", 0.15))
        self.beta_hi = float(_get(gate_cfg, "beta_hi", 0.45))
        self.film_stages = list(_get(cfg, "film_stages", [2, 3]))
        self.temporal = bool(_get(cfg, "temporal", True))
        blocks = list(_get(cfg, "blocks", [2, 2, 6, 6]))
        assert len(blocks) == 4

        # -- encoder (low-res) ------------------------------------------------
        self.stem = nn.Conv2d(3, enc[0], 3, 1, 1)
        self.stage0 = self._make_stage(enc[0], blocks[0])
        self.down1 = HaarDownsample(enc[0], enc[1])
        self.stage1 = self._make_stage(enc[1], blocks[1])
        self.down2 = HaarDownsample(enc[1], enc[2])
        self.stage2 = self._make_stage(enc[2], blocks[2])
        self.down3 = HaarDownsample(enc[2], enc[3])
        self.stage3 = self._make_stage(enc[3], blocks[3])
        self._stage_ch = enc

        # -- degradation + FiLM conditioning ----------------------------------
        self.deg_head = DegradationHead(enc[2])  # pooled "stage-3" (index 2) features
        cond_dim = self.deg_head.cond_dim
        self.films = nn.ModuleDict(
            {str(i): FiLM(enc[i], cond_dim) for i in self.film_stages}
        )
        ext_cond = int(_get(cfg, "cond_dim", 0))  # optional external conditioning width
        self.cond_proj = nn.Linear(ext_cond, cond_dim) if ext_cond > 0 else None

        # -- grid / guidance / detail / heads ---------------------------------
        self.grid_head = BilateralGridHead(enc[3], self.grid_depth, self.grid_size)
        self.guidance = GuidanceNet()
        self.detail = DetailBranch(
            int(_get(detail_cfg, "channels", 12)),
            int(_get(detail_cfg, "layers", 4)),
            float(_get(detail_cfg, "scale_init", 0.05)),
        )
        self.conf_head = ConfidenceHead(enc[3])
        self.trans_head = TransmissionHead(enc[3])

        # -- temporal ---------------------------------------------------------
        if self.temporal:
            self.temporal_mod = TemporalModule(
                grid_ch=12 * self.grid_depth,
                feat_ch=enc[3],
                ema_gain=float(_get(cfg, "ema_gain", 0.7)),
                scene_thresh=float(_get(cfg, "scene_thresh", 0.5)),
            )
        else:
            self.temporal_mod = None

    @staticmethod
    def _make_stage(channels: int, n: int) -> nn.Sequential:
        return nn.Sequential(*[RepNAFBlock(channels) for _ in range(max(1, n))])

    # -- helpers --------------------------------------------------------------
    def _lowres(self, frame: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Resize long side to `lowres`, then pad to a multiple of 8 (3 downsamples).

        Shapes go through int() so the ONNX tracer constant-folds them at a fixed
        export resolution (traced .shape entries are Tensors; round() breaks).
        """
        h, w = int(frame.shape[-2]), int(frame.shape[-1])
        scale = self.lowres / max(h, w)
        h0 = max(1, int(h * scale + 0.5))
        w0 = max(1, int(w * scale + 0.5))
        lr = F.interpolate(frame, size=(h0, w0), mode="bilinear", align_corners=False)
        ph, pw = (-h0) % 8, (-w0) % 8
        lr_p = F.pad(lr, [0, pw, 0, ph], mode="replicate") if (ph or pw) else lr
        return lr, lr_p

    @staticmethod
    def _smoothstep(x: torch.Tensor, lo: float, hi: float) -> torch.Tensor:
        t = ((x - lo) / (hi - lo)).clamp(0.0, 1.0)
        return t * t * (3 - 2 * t)

    @staticmethod
    def severity_gate(restored: torch.Tensor, frame: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
        """out = alpha*restored + (1-alpha)*frame; alpha broadcast from B,1."""
        a = alpha.view(-1, 1, 1, 1)
        return a * restored + (1 - a) * frame

    # -- forward --------------------------------------------------------------
    def forward(
        self,
        frame: torch.Tensor,
        state: Optional[Any] = None,
        cond: Optional[torch.Tensor] = None,
    ) -> PharosOutput:
        b, _, h, w = frame.shape
        lr, lr_p = self._lowres(frame)

        # encoder
        x = self.stem(lr_p)
        x = self.stage0(x)
        x = self.stage1(self.down1(x))
        f2 = self.stage2(self.down2(x))

        # degradation + conditioning
        deg, cond_vec = self.deg_head(f2)
        if cond is not None and self.cond_proj is not None:
            cond_vec = cond_vec + self.cond_proj(cond)
        if 2 in self.film_stages:
            f2 = self.films["2"](f2, cond_vec)
        f3 = self.stage3(self.down3(f2))
        if 3 in self.film_stages:
            f3 = self.films["3"](f3, cond_vec)

        # heads on deep low-res features
        grid = self.grid_head(f3)
        conf, logvar = self.conf_head(f3, (h, w))
        t_hat = self.trans_head(f3)

        # temporal smoothing of the grid (video mode)
        if self.temporal_mod is not None:
            grid, new_state = self.temporal_mod(grid, f3, conf, lr, state)
        else:
            new_state = None

        # full-res slicing + affine
        g = self.guidance(frame)
        m, bias = slice_grid(grid, g)
        j0 = apply_affine(frame, m, bias)

        # bounded detail + clamp
        r = self.detail(frame, j0)
        j = torch.clamp(j0 + r, 0.0, 1.0)

        # continuous severity gate
        alpha = self._smoothstep(deg["beta"], self.beta_lo, self.beta_hi)
        out = self.severity_gate(j, frame, alpha)

        return PharosOutput(
            output=out,
            confidence=conf,
            grid=grid,
            state=new_state,
            deg=deg,
            t_hat=t_hat,
            aux={
                "j0": j0,
                "j": j,
                "detail": r,
                "alpha": alpha,
                "logvar": logvar,
                "guidance": g,
            },
        )

    def reparameterize(self) -> None:
        """Fold every reparameterizable conv into a single conv (in place, eval only)."""
        for module in list(self.modules()):
            if isinstance(module, RepConv):
                module.reparameterize()
