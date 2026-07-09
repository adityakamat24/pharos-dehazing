"""Reveal memory for RevealNet (DESIGN.md §9d.2) — anchor-frame design.

``RevealMemory`` is the core mid-res registered buffer: three aligned tensors
``{rgb: B,3,h,w ; trust: B,1,h,w ; age: B,1,h,w}`` held in a **fixed anchor
coordinate frame** plus a cumulative homography ``H`` (``H_cur_from_anchor``,
``B,3,3``) mapping anchor-view coords -> the current view. The buffer tensors are
**never warped frame-to-frame** — that was the v2.0 bug: re-warping the buffer in
place every frame stacks N bilinear resamples over a T-frame clip, so the memory
accumulates interpolation blur regardless of alignment accuracy (trained at T=8,
deployed at T=32 -> mush). Here the stored buffer is only ever resampled on a
(rare) re-anchor; the *cumulative* warp is composed as a cheap 3x3 matmul.

Coordinate conventions (all normalized ``grid_sample`` coords, ``align_corners=True``):

* **view-normalized**: ``[-1,1]`` over the current camera view (mid-res ``view_hw``).
  This is the space the aligner and the loss (``align_H``) operate in.
* **buffer-normalized**: ``[-1,1]`` over the full anchor buffer (which may carry an
  ``anchor_margin`` so panning keeps off-screen content). The view sits centered in
  the buffer; ``S = diag(view_w/buf_w, view_h/buf_h, 1)`` maps a (anchor-)view coord
  to its buffer coord and ``S_inv`` the reverse.
* ``H`` maps **anchor-view -> current-view**: ``p_cur ~ H @ p_anchor``. Identity at
  seed. Composed per frame as ``H <- H_t @ H`` where ``H_t`` is the aligner's
  frame-to-frame (current<-previous) homography.

Read path (compositor): warp the buffer into the current view with ONE
``grid_sample`` using ``M_read = S @ inv(H)``. Write path (merge): warp the *current
observation* into the anchor frame with ``M_write = H @ S_inv`` (one resample of the
NEW observation only), then robust-merge in anchor coords. Decay/age updates touch
the whole buffer every frame (elementwise, no resampling).
"""
from __future__ import annotations

from typing import Any, NamedTuple

import torch
import torch.nn.functional as F

from .aligner import invert_homography, warp_grid

# Unit-square corners in normalized [-1,1] coords (TL, TR, BR, BL) for drift measure.
_CORNERS = ((-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0))


class ViewBuffers(NamedTuple):
    """Buffers warped into the current view (read-path output for the compositor)."""

    rgb: torch.Tensor
    trust: torch.Tensor
    age: torch.Tensor


class RevealMemory:
    """Anchor-frame scene memory: rgb + per-pixel trust + per-pixel age + cumulative H."""

    def __init__(
        self,
        rgb: torch.Tensor,
        trust: torch.Tensor,
        age: torch.Tensor,
        H: torch.Tensor | None = None,
        view_hw: tuple[int, int] | None = None,
    ) -> None:
        self.rgb = rgb      # B,3,bh,bw in [0,1], float32 (anchor frame)
        self.trust = trust  # B,1,bh,bw in [0,1], float32
        self.age = age      # B,1,bh,bw >= 0, float32 (dt units)
        b = rgb.shape[0]
        if H is None:
            H = torch.eye(3, dtype=torch.float32, device=rgb.device).unsqueeze(0).expand(b, 3, 3)
        self.H = H.to(torch.float32).contiguous()  # H_cur_from_anchor, B,3,3
        # View (mid-res) size; defaults to the buffer size (margin 1.0, view == buffer).
        self.view_hw: tuple[int, int] = view_hw or (int(rgb.shape[-2]), int(rgb.shape[-1]))

    # -- construction ---------------------------------------------------------
    @classmethod
    def seed(cls, rgb0: torch.Tensor, seed_trust: float, margin: float = 1.0) -> "RevealMemory":
        """First-frame seed: place ``rgb0`` (view coords) into a margin-padded anchor buffer.

        ``margin`` (>= 1.0) enlarges the buffer around the anchor view so panning keeps
        off-view content. The seed is written with an identity ``H``; trust is
        ``seed_trust`` where the observation lands and 0 in the margin, age 0.
        """
        rgb0 = rgb0.detach().to(torch.float32).clamp(0.0, 1.0)
        b, _, vh, vw = rgb0.shape
        bh = max(vh, int(round(float(margin) * vh)))
        bw = max(vw, int(round(float(margin) * vw)))
        dev = rgb0.device
        mem = cls(
            torch.zeros(b, 3, bh, bw, dtype=torch.float32, device=dev),
            torch.zeros(b, 1, bh, bw, dtype=torch.float32, device=dev),
            torch.zeros(b, 1, bh, bw, dtype=torch.float32, device=dev),
            H=None,
            view_hw=(vh, vw),
        )
        rgb_buf, valid = mem._obs_into_buffer(rgb0)
        mem.rgb = rgb_buf
        mem.trust = valid * float(seed_trust)
        return mem

    @property
    def buffers(self) -> dict[str, torch.Tensor]:
        """Named-buffer view (the {rgb, trust, age} dict of the spec)."""
        return {"rgb": self.rgb, "trust": self.trust, "age": self.age}

    def detach(self) -> "RevealMemory":
        """Return a graph-detached copy (buffers + cumulative H) for truncated BPTT."""
        return RevealMemory(
            self.rgb.detach(), self.trust.detach(), self.age.detach(),
            H=self.H.detach(), view_hw=self.view_hw,
        )

    def reset(self) -> None:
        """Clear the memory: zero rgb/trust/age and reset H to identity (tracking-loss)."""
        self.rgb = torch.zeros_like(self.rgb)
        self.trust = torch.zeros_like(self.trust)
        self.age = torch.zeros_like(self.age)
        b = self.rgb.shape[0]
        self.H = torch.eye(3, dtype=torch.float32, device=self.rgb.device).unsqueeze(0).expand(b, 3, 3)

    # -- geometry helpers -----------------------------------------------------
    def _scale_mats(self) -> tuple[torch.Tensor, torch.Tensor]:
        """(S, S_inv): view<->buffer normalized-coord scale (view centered in buffer).

        Under ``align_corners=True`` the exact scale mapping a centered ``view`` window
        of pixels onto the ``buffer`` grid is ``(view-1)/(buffer-1)`` (endpoints land on
        pixel centers), which is 1.0 when there is no margin (buffer == view).
        """
        vh, vw = self.view_hw
        bh, bw = int(self.rgb.shape[-2]), int(self.rgb.shape[-1])
        sx = float(vw - 1) / float(max(bw - 1, 1))
        sy = float(vh - 1) / float(max(bh - 1, 1))
        dev = self.rgb.device
        s = torch.tensor([[sx, 0.0, 0.0], [0.0, sy, 0.0], [0.0, 0.0, 1.0]], device=dev).unsqueeze(0)
        s_inv = torch.tensor(
            [[1.0 / sx, 0.0, 0.0], [0.0, 1.0 / sy, 0.0], [0.0, 0.0, 1.0]], device=dev
        ).unsqueeze(0)
        return s, s_inv

    def _resample(self, mat: torch.Tensor, size: tuple[int, int]) -> ViewBuffers:
        """One grid_sample of every buffer channel by ``mat`` (output->input norm coords)."""
        grid = warp_grid(mat, size)
        valid = F.grid_sample(
            torch.ones_like(self.trust), grid, mode="bilinear", padding_mode="zeros",
            align_corners=True,
        )
        rgb = F.grid_sample(
            self.rgb, grid, mode="bilinear", padding_mode="border", align_corners=True
        )
        trust = F.grid_sample(
            self.trust, grid, mode="bilinear", padding_mode="border", align_corners=True
        ) * valid
        age = F.grid_sample(
            self.age, grid, mode="bilinear", padding_mode="border", align_corners=True
        ) * valid
        return ViewBuffers(rgb, trust, age)

    def _obs_into_buffer(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Warp a current-view observation into the anchor buffer; return (obs_buf, valid)."""
        _, s_inv = self._scale_mats()
        mat = self.H @ s_inv                                  # buffer-norm -> view-norm
        bh, bw = int(self.rgb.shape[-2]), int(self.rgb.shape[-1])
        grid = warp_grid(mat, (bh, bw))
        obs = obs.to(torch.float32)
        obs_buf = F.grid_sample(
            obs, grid, mode="bilinear", padding_mode="zeros", align_corners=True
        )
        valid = F.grid_sample(
            torch.ones_like(obs[:, :1]), grid, mode="bilinear", padding_mode="zeros",
            align_corners=True,
        )
        return obs_buf, valid

    def _corner_disp(self) -> torch.Tensor:
        """Mean 4-point corner displacement of ``H`` in view-normalized units (per sample)."""
        dev = self.H.device
        corners = torch.tensor(_CORNERS, dtype=torch.float32, device=dev)      # 4,2
        hom = torch.cat([corners, torch.ones(4, 1, device=dev)], dim=-1)       # 4,3
        warped = (self.H @ hom.t()).transpose(1, 2)                            # B,4,3
        xy = warped[..., :2] / (warped[..., 2:3] + 1e-8)                       # B,4,2
        return (xy - corners).norm(dim=-1).mean(dim=1)                         # B

    # -- registration (frame-to-frame is a matmul; buffer is NOT resampled) ---
    def compose(self, h_t: torch.Tensor) -> None:
        """Compose the aligner's frame-to-frame ``h_t`` (cur<-prev) into cumulative ``H``."""
        self.H = (h_t.to(torch.float32) @ self.H).contiguous()

    def set_cumulative(self, h_cum: torch.Tensor) -> None:
        """Set ``H`` (cur<-anchor) directly from a track-to-keyframe estimate.

        Unlike :meth:`compose`, per-frame estimation errors do not accumulate:
        each registration is absolute w.r.t. the anchor-epoch keyframe.
        """
        self.H = h_cum.to(torch.float32).contiguous()

    def warp(self, homography: torch.Tensor) -> None:
        """Resample the whole buffer by ``homography`` (anchor->current) in place.

        Kept as the single-resample primitive used by re-anchoring (and by tests). NOT
        called per frame — the anchor-frame design composes warps instead. RGB uses
        border padding; trust/age are multiplied by an in-bounds validity mask.
        """
        bh, bw = int(self.rgb.shape[-2]), int(self.rgb.shape[-1])
        out = self._resample(invert_homography(homography), (bh, bw))
        self.rgb, self.trust, self.age = out.rgb, out.trust, out.age

    def read_view(self) -> ViewBuffers:
        """Warp buffers from the anchor frame into the current view (ONE grid_sample each)."""
        s, _ = self._scale_mats()
        mat = s @ invert_homography(self.H)                    # view-norm -> buffer-norm
        return self._resample(mat, self.view_hw)

    def maybe_reanchor(self, cfg: Any, scalar: torch.Tensor) -> int:
        """Re-anchor drifted/lost samples to the current view; return the rebase count.

        Triggers per sample when the cumulative 4-point corner displacement exceeds
        ``reanchor_px`` (as a fraction of the full normalized extent 2.0) OR the aligner
        scalar trust has collapsed below ``t_lo``. On a rebase the buffer is resampled
        ONCE into the current view, ``H`` resets to identity and trust is multiplied by
        ``rebase_decay``.
        """
        reanchor_px = float(_get(cfg, "reanchor_px", 0.35))
        t_lo = float(_get(cfg, "t_lo", 0.2))
        rebase_decay = float(_get(cfg, "rebase_decay", 0.9))
        drift = self._corner_disp() > (reanchor_px * 2.0)                      # B bool
        collapse = scalar.reshape(-1).to(drift.device) < t_lo                 # B bool
        flag = drift | collapse
        if not bool(flag.any()):
            return 0
        s, s_inv = self._scale_mats()
        mat = s @ invert_homography(self.H) @ s_inv                           # new-buf -> old-buf
        bh, bw = int(self.rgb.shape[-2]), int(self.rgb.shape[-1])
        out = self._resample(mat, (bh, bw))
        fb = flag.view(-1, 1, 1, 1)
        self.rgb = torch.where(fb.expand_as(self.rgb), out.rgb, self.rgb)
        self.trust = torch.where(fb, (out.trust * rebase_decay).clamp(0.0, 1.0), self.trust)
        self.age = torch.where(fb, out.age, self.age)
        eye = torch.eye(3, dtype=torch.float32, device=self.H.device).expand_as(self.H)
        self.H = torch.where(flag.view(-1, 1, 1), eye, self.H).contiguous()
        return int(flag.sum())

    # -- robust temporal merge (in anchor coords) -----------------------------
    def update(
        self,
        rgb_t: torch.Tensor,
        conf_t: torch.Tensor,
        align_trust: torch.Tensor,
        cfg: Any,
        dt: float = 1.0,
    ) -> None:
        """Merge the current-view restoration into the anchor buffer (robust merge).

        ``rgb_t``: B,3,vh,vw restoration ; ``conf_t``/``align_trust``: B,1,vh,vw in [0,1]
        (all in current-view coords). The observation and its weight ``w = conf*align``
        are warped ONCE into the anchor frame (the buffer itself is never resampled).
        Where ``w > merge_thresh`` and the observation is in-bounds: rgb lerps toward the
        observation by ``w``, trust <- max(trust*decay_keep, w), age <- 0. Elsewhere:
        trust *= decay_miss, age += dt (applied to the whole buffer — cheap, no resample).
        """
        _, s_inv = self._scale_mats()
        mat = self.H @ s_inv                                  # buffer-norm -> view-norm
        bh, bw = int(self.rgb.shape[-2]), int(self.rgb.shape[-1])
        grid = warp_grid(mat, (bh, bw))
        rgb_buf = F.grid_sample(
            rgb_t.to(torch.float32).clamp(0.0, 1.0), grid, mode="bilinear",
            padding_mode="zeros", align_corners=True,
        )
        w_view = (conf_t.to(torch.float32) * align_trust.to(torch.float32)).clamp(0.0, 1.0)
        w = F.grid_sample(
            w_view, grid, mode="bilinear", padding_mode="zeros", align_corners=True
        )
        valid = F.grid_sample(
            torch.ones_like(w_view), grid, mode="bilinear", padding_mode="zeros",
            align_corners=True,
        )
        w = (w * valid).clamp(0.0, 1.0)                       # B,1,bh,bw (anchor frame)
        merge = w > float(_get(cfg, "merge_thresh", 0.1))     # bool mask

        merged_rgb = self.rgb * (1.0 - w) + rgb_buf * w
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
