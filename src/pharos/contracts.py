"""Frozen interface contracts for Pharos workstreams.

Implementers: do NOT modify this file. If a contract blocks you, note it in your
final report instead. Everything here must import with no GPU, no datasets, and
no optional teacher dependencies installed.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Protocol

import torch

# Domain ids used everywhere (batch dicts, conditioning, synthesis).
DOMAIN_HAZE = 0
DOMAIN_SMOKE = 1
DOMAIN_SATELLITE = 2
DOMAIN_NAMES = {DOMAIN_HAZE: "haze", DOMAIN_SMOKE: "smoke", DOMAIN_SATELLITE: "satellite"}


@dataclass
class PharosOutput:
    """Return type of PharosNet.forward.

    Shapes are for batch B and full input resolution H x W. Low-res internals use
    the config value model.lowres (default 256) on the long side.
    """

    output: torch.Tensor            # B,3,H,W in [0,1] — restored frame (after severity gate)
    confidence: torch.Tensor        # B,1,H,W in (0,1] — calibrated trust map (1 = trustworthy)
    grid: torch.Tensor              # B,12,D,Gh,Gw — bilateral affine grid (pre-slicing)
    state: Optional[Any]            # opaque recurrent state; pass back on next frame; None = image mode
    deg: dict[str, torch.Tensor]    # {"beta": B,1 ; "airlight": B,3 ; "sigma": B,1 ; "domain_logits": B,3}
    t_hat: Optional[torch.Tensor] = None   # B,1,h,w aux transmission (low-res), may be None at inference
    aux: dict[str, torch.Tensor] = field(default_factory=dict)  # anything extra (J0, detail, gate alpha...)


class PharosModel(Protocol):
    """Protocol implemented by pharos.models.pharosnet.PharosNet."""

    def forward(
        self,
        frame: torch.Tensor,                 # B,3,H,W in [0,1]
        state: Optional[Any] = None,         # recurrent state from previous frame (video mode)
        cond: Optional[torch.Tensor] = None, # optional B,E external conditioning embedding
    ) -> PharosOutput: ...

    def reparameterize(self) -> None:
        """Collapse multi-branch training blocks into single convs (in place, inference only)."""


# ---------------------------------------------------------------------------
# Batch dict contract (all datasets in pharos/data yield exactly this shape).
# Images: torch.float32 in [0,1], CHW. Clips add a leading T dim per sample,
# i.e. batches are B,T,3,H,W and batch["clip"] is True.
# ---------------------------------------------------------------------------
BATCH_KEYS = {
    "hazy": "FloatTensor B,3,H,W (or B,T,3,H,W for clips) — degraded input",
    "clean": "FloatTensor same shape, or None for unpaired data",
    "domain": "LongTensor B — DOMAIN_* id",
    "clip": "bool — True if temporal clip batch",
    "meta": "dict — dataset name, paths, synthesis params (beta/airlight/sigma when synthetic)",
}


class TeacherBundle(Protocol):
    """Training-time-only priors (pharos/teachers). Each lazily loads on first use.

    Every teacher is optional: if disabled in config or its dependency is missing,
    the attribute is None and losses must skip the corresponding term.
    """

    depth: Optional[Any]     # callable(img B,3,H,W) -> B,1,h,w relative depth (higher = farther)
    detector: Optional[Any]  # callable(img B,3,H,W) -> list of feature maps (frozen detector)
    flow: Optional[Any]      # callable(a, b) -> B,2,H,W flow a->b (clean frames, training only)


class LossFn(Protocol):
    """pharos.losses.PharosLoss."""

    def __call__(
        self, out: PharosOutput, batch: dict, teachers: TeacherBundle
    ) -> tuple[torch.Tensor, dict[str, float]]: ...
