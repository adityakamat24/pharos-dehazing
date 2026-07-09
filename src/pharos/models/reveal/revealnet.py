"""RevealNet: confidence-weighted reveal accumulation over PharosNet (DESIGN.md §9d).

RevealNet wraps a built ``PharosNet`` and adds a long-horizon, confidence-weighted
scene memory: per frame it (1) runs the inner single-frame restoration, (2) aligns
the accumulated memory into the current view (tiered homography aligner), (3) merges
the reliable pixels of the current restoration into memory (burst robust-merge), and
(4) composites current restoration with remembered content, arbitrated by confidence
vs decayed memory trust, emitting a staleness map.

Contract: ``forward(frame, state, cond) -> PharosOutput`` with an opaque ``state`` and
a ``reparameterize()`` passthrough, so existing engine / rt code runs it unmodified
(it satisfies ``contracts.PharosModel``). Everything is CPU-importable, AMP-safe and
strictly causal (no future frames); memory cost is O(pixels) per frame.
"""
from __future__ import annotations

from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...contracts import PharosOutput
from ..pharosnet import PharosNet
from .aligner import TieredAligner
from .compositor import composite
from .memory import RevealMemory

_DEFAULTS: dict[str, Any] = {
    "mem_res": 128,        # mid-res memory buffer, long side
    "align_res": 64,       # low-res aligner input, long side
    "t_lo": 0.2,           # aligner freeze threshold (scalar trust)
    "merge_thresh": 0.1,   # effective-weight threshold to merge into memory
    "decay_keep": 0.98,    # trust decay applied before max() on a merge
    "decay_miss": 0.9,     # trust decay on a miss
    "half_life": 30.0,     # age-decay half-life (dt units) for compositing
    "dt": 1.0,             # timestep added to age per frame (frames or seconds)
    "comp_k": 8.0,         # compositor sigmoid sharpness
    "seed_trust": 0.1,     # first-frame memory trust
    "aligner_width": 24,   # aligner base channel width
    "reanchor_px": 0.35,   # re-anchor when cumulative corner drift exceeds this frac
    "rebase_decay": 0.9,   # trust multiplier applied on a re-anchor
    "anchor_margin": 1.5,  # anchor buffer size as a multiple of the mid-res view
}


def _get(cfg: Any, key: str, default: Any) -> Any:
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


class RevealNet(nn.Module):
    """PharosNet + tiered aligner + reveal memory + staleness compositor."""

    def __init__(self, inner: PharosNet, cfg: Optional[Any] = None) -> None:
        super().__init__()
        self.inner = inner
        self.cfg = {k: _get(cfg, k, v) for k, v in _DEFAULTS.items()}
        self.mem_res = int(self.cfg["mem_res"])
        self.align_res = int(self.cfg["align_res"])
        self.seed_trust = float(self.cfg["seed_trust"])
        self.dt = float(self.cfg["dt"])
        self.anchor_margin = float(self.cfg["anchor_margin"])
        self.aligner = TieredAligner(
            in_ch=3, width=int(self.cfg["aligner_width"]), t_lo=float(self.cfg["t_lo"])
        )

    # -- helpers --------------------------------------------------------------
    @staticmethod
    def _scaled(h: int, w: int, long_side: int) -> tuple[int, int]:
        """Long side capped at ``long_side`` (never upsampled), aspect preserved."""
        scale = min(1.0, long_side / max(h, w))
        return max(1, int(h * scale + 0.5)), max(1, int(w * scale + 0.5))

    @staticmethod
    def _resize(x: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
        if x.shape[-2:] == (size[0], size[1]):
            return x
        return F.interpolate(x, size=size, mode="bilinear", align_corners=False)

    # -- forward --------------------------------------------------------------
    def forward(
        self,
        frame: torch.Tensor,
        state: Optional[Any] = None,
        cond: Optional[torch.Tensor] = None,
        motion_prior: Optional[torch.Tensor] = None,
    ) -> PharosOutput:
        inner_state = state.get("inner") if isinstance(state, dict) else None
        inner_out = self.inner(frame, inner_state, cond)
        j, conf = inner_out.output, inner_out.confidence
        h, w = frame.shape[-2], frame.shape[-1]

        mh, mw = self._scaled(h, w, self.mem_res)
        ah, aw = self._scaled(h, w, self.align_res)
        cur_lr = self._resize(frame, (ah, aw))
        j_mid = self._resize(j, (mh, mw))
        conf_mid = self._resize(conf, (mh, mw))

        reanchors = 0
        if not isinstance(state, dict):
            # First frame: seed memory from the current restoration at low trust.
            memory = RevealMemory.seed(j_mid, self.seed_trust, margin=self.anchor_margin)
            align_full = torch.zeros_like(conf)
            homog = None
            keyframe = cur_lr
        else:
            memory: RevealMemory = state["memory"]
            keyframe = state.get("keyframe", state["anchor"])
            # Track-to-KEYFRAME: estimate the cumulative anchor->current homography
            # directly against the anchor-epoch keyframe. Per-step errors do not
            # compose (frame-to-frame odometry drift measurably killed T=32 recall:
            # -1.5 dB with a proven-exact memory).
            h_cum, tmap, scalar = self.aligner(cur_lr, keyframe, motion_prior)
            prev_cum = memory.H.detach()
            memory.set_cumulative(h_cum)
            reanchors = memory.maybe_reanchor(self.cfg, scalar)    # rare single-resample rebase
            if reanchors:
                keyframe = cur_lr                                  # new anchor epoch
            # expose the frame-to-frame equivalent for L_align supervision
            # (loss compares against relative GT warps): H_t = H_cum @ inv(H_prev_cum).
            homog = h_cum @ torch.linalg.inv(
                prev_cum.float() + 1e-8 * torch.eye(3, device=prev_cum.device)
            ).to(h_cum.dtype)
            align_mid = self._resize(tmap, (mh, mw)) * scalar.view(-1, 1, 1, 1)
            memory.update(j_mid, conf_mid, align_mid, self.cfg, dt=self.dt)
            align_full = self._resize(tmap * scalar.view(-1, 1, 1, 1), (h, w))

        view = memory.read_view()                                  # anchor -> current view (read)
        out, staleness = composite(j, conf, view, self.cfg)

        aux = dict(inner_out.aux)
        aux["staleness"] = staleness
        aux["memory_trust"] = self._resize(view.trust, (h, w))     # current-view read-path trust
        aux["align_trust"] = align_full
        aux["j_restored"] = j
        aux["reanchors"] = torch.full(
            (frame.shape[0],), float(reanchors), device=frame.device
        )  # aux stat: re-anchor count this frame (per-batch scalar)
        if homog is not None:
            aux["align_H"] = homog  # estimated 3x3 homography (RevealLoss L_align supervision)
        new_state = {"inner": inner_out.state, "memory": memory, "anchor": cur_lr,
                     "keyframe": keyframe}
        return PharosOutput(
            output=out,
            confidence=conf,
            grid=inner_out.grid,
            state=new_state,
            deg=inner_out.deg,
            t_hat=inner_out.t_hat,
            aux=aux,
        )

    def reparameterize(self) -> None:
        """Fold the inner PharosNet's reparameterizable convs (aligner has none)."""
        self.inner.reparameterize()
