"""RevealNet (v2) dataset: temporally-coherent drifting-smoke clips with GT.

:class:`RevealVideoDataset` wraps a clean image-or-video pool and, per sample, calls
:func:`pharos.data.reveal_synthesis.synthesize_reveal_clip` to render a dense,
drifting, camera-jittered smoke clip whose clean background is known at every pixel
and frame. It yields the standard Pharos batch contract with ``clip=True`` (hazy /
clean are ``(T, 3, H, W)``) and a ``meta`` dict carrying the reveal supervision
(``smoke_density``, ``transmission``, ``cam_H``) plus the usual ``full_lowres``
global-context stream.

This module reuses the clean-pool resolution and sequence-window plumbing from
:mod:`pharos.data.datasets` by import (nothing is copied) and does not modify it. A
:func:`build_reveal_dataset` factory (``name='reveal_video'``) mirrors
``datasets.build_dataset`` so the engine can wire it in later without editing
``datasets.py``.

Notes / deviations
------------------
* Random flips are **not** applied: a flip would need a matching conjugation of
  ``cam_H`` (and a flip of ``smoke_density``), and the value of the reveal signal
  comes from a large synthetic-scene variety, not from mirror augmentation. ``crop``
  IS supported and ``cam_H`` is conjugated exactly by the crop offset so it stays a
  valid warp in crop-local pixel coordinates.
* ``crop`` never up-samples: the effective crop is ``min(crop, H, W)`` so the pixel
  scale (and hence ``cam_H`` / ``smoke_density``) stays consistent with the render.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from ..contracts import DOMAIN_SMOKE
from .datasets import (
    SynthVideoDataset,
    _PharosDataset,
    _cfg_get,
    _clip_windows,
    _discover_sequences,
    _resolve_clean_pool,
)
from .degradations import RobustnessPipeline
from .reveal_synthesis import synthesize_reveal_clip
from .transforms import _rand_int, list_images_recursive, make_lowres

__all__ = ["RevealVideoDataset", "build_reveal_dataset"]


def _conjugate_translate(cam_H: torch.Tensor, left: int, top: int) -> torch.Tensor:
    """Re-express per-frame homographies in crop-local pixel coordinates.

    A crop at offset ``(left, top)`` maps crop coord ``c`` to full coord ``A @ c`` with
    ``A = translate(left, top)``. The warp in crop coordinates is therefore
    ``H' = A^{-1} @ H @ A`` (exact, since the crop window is identical for every frame).
    """
    dtype, device = cam_H.dtype, cam_H.device
    a = torch.tensor([[1.0, 0.0, left], [0.0, 1.0, top], [0.0, 0.0, 1.0]], dtype=dtype, device=device)
    a_inv = torch.tensor(
        [[1.0, 0.0, -left], [0.0, 1.0, -top], [0.0, 0.0, 1.0]], dtype=dtype, device=device
    )
    return a_inv @ cam_H @ a  # broadcast over the leading T dimension


class RevealVideoDataset(_PharosDataset):
    """Drifting-smoke reveal clips over a clean image-or-video pool.

    ``clean_root`` may contain per-sequence subfolders of clean frames (real video)
    or a flat folder of stills (each still becomes a static clip over which the smoke
    drifts and the camera jitters). Every sample is domain ``smoke``.
    """

    def __init__(
        self,
        clean_root: str | Path,
        clip_len: int = 8,
        crop: int = 0,
        augment: bool = False,
        lowres: int = 256,
        seed: int | None = None,
        robustness: RobustnessPipeline | None = None,
        name: str = "reveal_video",
        **synth_kwargs: Any,
    ) -> None:
        self.clip_len = int(clip_len)
        self.crop, self.augment, self.lowres, self.seed, self.name = crop, augment, lowres, seed, name
        self.robustness = robustness
        self.synth_kwargs = synth_kwargs
        self.seqs = _discover_sequences(clean_root)
        if self.seqs:
            self.stills: list[Path] = []
            self.windows = _clip_windows(self.seqs, clip_len)
        else:
            self.stills = list_images_recursive(clean_root)
            self.windows = [(i, 0) for i in range(len(self.stills))]  # type: ignore[misc]

    def __len__(self) -> int:
        return len(self.windows)

    # reuse the clean-clip loader (stills -> static clip; real seq -> window)
    _clean_clip = SynthVideoDataset._clean_clip

    def __getitem__(self, idx: int) -> dict:
        si, start = self.windows[idx]
        g = self._gen(idx)
        clean = self._clean_clip(si, start)  # (T, 3, H, W)
        out = synthesize_reveal_clip(clean, generator=g, **self.synth_kwargs)
        hazy = out["hazy"]
        if self.robustness is not None:
            hazy = torch.stack([self.robustness(hazy[i], g) for i in range(hazy.shape[0])])
        return self._finish_reveal_clip(
            hazy, out["gt"], out["smoke_density"], out["transmission"], out["cam_H"], out, g
        )

    def _finish_reveal_clip(
        self,
        hazy: torch.Tensor,
        clean: torch.Tensor,
        density: torch.Tensor,
        transmission: torch.Tensor,
        cam_H: torch.Tensor,
        out: dict,
        generator: torch.Generator | None,
    ) -> dict:
        meta: dict[str, Any] = {
            "dataset": self.name,
            "synthetic": True,
            "reveal": True,
            "domain_name": "smoke",
            "airlight": out["airlight"].float(),
            "beta": torch.tensor([out["beta"]], dtype=torch.float32),
        }
        # global-context lowres computed from the *full* (pre-crop) hazy frames
        meta["full_lowres"] = torch.stack([make_lowres(f, self.lowres) for f in hazy])

        if self.crop and self.crop > 0:
            _, _, h, w = hazy.shape
            cs = min(self.crop, h, w)
            top = _rand_int(h - cs, generator)
            left = _rand_int(w - cs, generator)
            ys, xs = slice(top, top + cs), slice(left, left + cs)
            hazy = hazy[:, :, ys, xs]
            clean = clean[:, :, ys, xs]
            density = density[:, :, ys, xs]
            transmission = transmission[:, :, ys, xs]
            cam_H = _conjugate_translate(cam_H, left, top)

        meta["smoke_density"] = density
        meta["transmission"] = transmission
        meta["cam_H"] = cam_H
        return {"hazy": hazy, "clean": clean, "domain": DOMAIN_SMOKE, "clip": True, "meta": meta}


def build_reveal_dataset(name: str, cfg: Any, split: str = "train") -> Dataset:
    """Factory for the RevealNet dataset (mirrors ``datasets.build_dataset``).

    Only ``name='reveal_video'`` is understood. Reads ``data_root``, ``model.lowres``,
    ``train.crop`` and ``train.clip_len`` (curriculum clip length) from ``cfg`` exactly
    like the core factory, resolves the clean pool with the same logic, and enables
    crop + robustness augmentation only on the train split.
    """
    if name != "reveal_video":
        raise ValueError(f"build_reveal_dataset only handles 'reveal_video', got {name!r}")
    data_root = Path(_cfg_get(cfg, "data_root", "data"))
    lowres = int(_cfg_get(cfg, "model.lowres", 256))
    crop = int(_cfg_get(cfg, "train.crop", 256))
    clip_len = int(_cfg_get(cfg, "train.clip_len", 8))
    seed = _cfg_get(cfg, "seed", None)
    is_train = split == "train"
    clean_root = _resolve_clean_pool(cfg, data_root)
    robustness = RobustnessPipeline() if is_train else None
    return RevealVideoDataset(
        clean_root,
        clip_len=clip_len,
        crop=crop if is_train else 0,
        augment=is_train,
        lowres=lowres,
        seed=seed,
        robustness=robustness,
        name=name,
    )
