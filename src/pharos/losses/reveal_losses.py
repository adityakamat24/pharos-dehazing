"""RevealLoss: v2 reveal-accumulation supervision (DESIGN.md §9d.4).

RevealLoss *wraps* a :class:`~pharos.losses.PharosLoss` (compose, not copy) and
adds three reveal-specific terms on top of the full v1 loss stack:

    total = PharosLoss(out, batch, teachers)                       # all v1 terms
          + w_recall · L_recall     # reward remembering occluded-but-seen pixels
          + w_align  · L_align      # supervise the tiered aligner (4-pt / homography)
          + w_stale  · L_stale      # calibrate memory trust vs actual memory error

Every reveal term degrades to exactly 0 when its inputs are unavailable (image
batch instead of a clip, no clean GT, missing ``meta`` keys, missing model
``aux`` keys). Training therefore runs whether or not RevealNet exposes the
alignment / memory auxiliaries yet. ``__call__`` keeps the ``LossFn`` contract:
``(total, {term: float})`` with per-term values detached and ``log['total']``
overwritten with the combined total.

Shapes (clip batches, the reveal regime): ``out.output`` is ``B,T,3,H,W`` and
``batch['clean']`` matches. ``batch['meta']`` is the engine's list of per-sample
dicts (or a single pre-collated dict); spatial reveal signals are read via
:func:`_meta_spatial` and normalised with :func:`_to_bt1`.
"""
from __future__ import annotations

from typing import Any, Optional

import torch
import torch.nn.functional as F

from ..contracts import PharosOutput

# Candidate aux key names for the estimated homography / warp trust — RevealNet is
# a parallel workstream, so accept any of these (first present wins).
_H_EST_KEYS = ("align_H", "H_est", "est_H", "homography", "warp_H", "H")
# Normalised corner coordinates for the 4-point homography parameterisation.
_CORNERS = ((-1.0, -1.0), (-1.0, 1.0), (1.0, -1.0), (1.0, 1.0))


class RevealLoss:
    """Implements contracts.LossFn; wraps a PharosLoss instance."""

    def __init__(self, cfg: Any, inner: Optional[Any] = None) -> None:
        loss_cfg = _get(cfg, "loss", {}) or {}
        rev = _get(loss_cfg, "reveal", {}) or {}
        self.w = {
            "recall": float(_get(rev, "recall", 1.0)),
            "align": float(_get(rev, "align", 0.2)),
            "stale": float(_get(rev, "stale", 0.05)),
        }
        # density thresholds for the recall mask (smoke_density in [0, 1]).
        self.occ_thresh = float(_get(rev, "occ_thresh", 0.6))
        self.reveal_thresh = float(_get(rev, "reveal_thresh", 0.3))

        # compose: keep a reference to the inner v1 loss (build one if not injected).
        if inner is None:
            from .losses import PharosLoss  # lazy: keeps stub tests import-light

            inner = PharosLoss(cfg)
        self.inner = inner

        self.eps = 1e-6
        self.conf_eps = 1e-4  # mirrors PharosLoss._conf clamp for the precision form

    # ------------------------------------------------------------------
    def __call__(
        self, out: PharosOutput, batch: dict, teachers: Any
    ) -> tuple[torch.Tensor, dict[str, float]]:
        inner_total, log = self.inner(out, batch, teachers)
        log = dict(log) if isinstance(log, dict) else {}
        device = out.output.device

        terms: dict[str, torch.Tensor] = {
            "recall": self._recall(out, batch, device),
            "align": self._align(out, batch, device),
            "stale": self._stale(out, batch, device),
        }
        total = inner_total
        for name, val in terms.items():
            total = total + self.w[name] * val
        for name, val in terms.items():
            log[name] = float(val.detach())
        log["total"] = float(total.detach())
        return total, log

    # ------------------------------------------------------------------
    # L_recall — the core reveal supervision. On pixels occluded NOW (high
    # smoke_density) that were revealed at some EARLIER frame, the composited
    # output should still match the clean GT: reward the memory for remembering.
    def _recall(self, out: PharosOutput, batch: dict, device) -> torch.Tensor:
        output = out.output
        clean = batch.get("clean")
        if output is None or output.dim() != 5 or clean is None or clean.dim() != 5:
            return _z(device)  # needs a clip (time history) + clip GT
        density = _meta_spatial(batch.get("meta") or {}, "smoke_density", device)
        density = _to_bt1(density, output)
        if density is None:
            return _z(device)
        mask = revealed_recall_mask(density, self.occ_thresh, self.reveal_thresh)  # B,T,1,H,W
        denom = mask.sum()
        if float(denom) <= 0.0:
            return _z(device)
        err = (output - clean).abs().mean(dim=2, keepdim=True)  # B,T,1,H,W
        return (err * mask).sum() / (denom + self.eps)

    # L_align — supervise the aligner against the known synthetic camera warp.
    # L1 on the 4-point offsets of the estimated homography vs the GT (cam_H).
    def _align(self, out: PharosOutput, batch: dict, device) -> torch.Tensor:
        aux = out.aux if isinstance(out.aux, dict) else {}
        h_est = _first_present(aux, _H_EST_KEYS)
        h_gt = _meta_spatial(batch.get("meta") or {}, "cam_H", device)
        if h_est is None or h_gt is None:
            return _z(device)
        h_est = h_est.to(device).float()
        h_gt = h_gt.float()
        if h_est.shape[-2:] != (3, 3) or h_gt.shape[-2:] != (3, 3):
            return _z(device)
        # RevealNet estimates FRAME-TO-FRAME warps in normalized coords (T-1 per
        # clip: no estimate for frame 0); synthesis GT cam_H is CUMULATIVE from
        # frame 0 in PIXEL coords. Convert: relative GT = H_t · H_{t-1}^{-1},
        # then conjugate into normalized coords with the clip's resolution.
        if h_est.dim() == 4 and h_gt.dim() == 4 and h_est.shape[1] == h_gt.shape[1] - 1:
            hazy = batch.get("hazy")
            if hazy is None or hazy.dim() != 5:
                return _z(device)
            hh, ww = int(hazy.shape[-2]), int(hazy.shape[-1])
            # fp32 island: under AMP autocast matmuls emit Half and linalg.inv
            # rejects low-precision dtypes.
            dev_type = device.type if hasattr(device, "type") else "cuda"
            with torch.autocast(device_type=dev_type, enabled=False):
                s = torch.tensor(
                    [[2.0 / ww, 0.0, -1.0], [0.0, 2.0 / hh, -1.0], [0.0, 0.0, 1.0]], device=device
                )
                s_inv = torch.linalg.inv(s)
                rel = h_gt.float()[:, 1:] @ torch.linalg.inv(h_gt.float()[:, :-1])  # B,T-1,3,3 px
                rel_n = s @ rel @ s_inv                                             # normalized
                off_e = homography_to_4pt(h_est.float().reshape(-1, 3, 3), self.eps)
                off_g = homography_to_4pt(rel_n.reshape(-1, 3, 3), self.eps)
                off_gi = homography_to_4pt(
                    torch.linalg.inv(rel_n).reshape(-1, 3, 3), self.eps
                )
                # convention-robust: the aligner's warp direction may be either
                # prev->cur or cur->prev; supervise against the closer one.
                l_fwd = (off_e - off_g).abs().mean()
                l_bwd = (off_e - off_gi).abs().mean()
                return torch.minimum(l_fwd, l_bwd)
        he = h_est.reshape(-1, 3, 3)
        hg = h_gt.reshape(-1, 3, 3)
        if he.shape[0] != hg.shape[0]:
            # broadcast per-sample GT across frames when only one side is stacked.
            if hg.shape[0] > 0 and he.shape[0] % hg.shape[0] == 0:
                hg = hg.repeat_interleave(he.shape[0] // hg.shape[0], dim=0)
            elif he.shape[0] > 0 and hg.shape[0] % he.shape[0] == 0:
                he = he.repeat_interleave(hg.shape[0] // he.shape[0], dim=0)
            else:
                return _z(device)
        off_e = homography_to_4pt(he, self.eps)
        off_g = homography_to_4pt(hg, self.eps)
        return (off_e - off_g).abs().mean()

    # L_stale — Laplace-NLL calibration of memory trust against the real memory
    # error on memory-contributed pixels (staleness > 0). Mirrors PharosLoss._conf.
    def _stale(self, out: PharosOutput, batch: dict, device) -> torch.Tensor:
        aux = out.aux if isinstance(out.aux, dict) else {}
        trust = aux.get("memory_trust")
        stale_map = aux.get("staleness")
        output = out.output
        clean = batch.get("clean")
        if trust is None or stale_map is None or output is None or clean is None:
            return _z(device)
        if output.dim() == 5:  # flatten clip time into the batch dim (as _conf does)
            if clean.dim() != 5:
                return _z(device)
            output = output.flatten(0, 1)
            clean = clean.flatten(0, 1)
            trust = _flatten_time(trust)
            stale_map = _flatten_time(stale_map)
            logvar = _flatten_time(aux.get("memory_logvar"))
        elif clean.dim() != 4:
            return _z(device)
        else:
            logvar = aux.get("memory_logvar")
        trust = _align_map(trust, output, device)
        stale_map = _align_map(stale_map, output, device)
        if trust is None or stale_map is None:
            return _z(device)
        err = (output - clean).abs().mean(dim=1, keepdim=True)  # N,1,H,W
        mask = (stale_map > 0.0).float()
        denom = mask.sum()
        if float(denom) <= 0.0:
            return _z(device)
        if logvar is not None:
            lv = _align_map(logvar, output, device)
            if lv is not None:
                lv = lv.clamp(-6.0, 3.0)
                nll = err * torch.exp(-lv) + lv
                return (nll * mask).sum() / (denom + self.eps)
        p = trust.clamp(self.conf_eps, 1.0)  # trust read as a precision
        nll = err * p - torch.log(p)
        return (nll * mask).sum() / (denom + self.eps)


# ----------------------------------------------------------------------
# reveal-mask construction (standalone + unit-tested)
# ----------------------------------------------------------------------
def revealed_recall_mask(
    density: torch.Tensor, occ_thresh: float, reveal_thresh: float
) -> torch.Tensor:
    """Mask of pixels occluded at frame t but revealed at some earlier frame.

    ``density`` is ``B,T,1,H,W`` (smoke opacity in [0, 1]). A pixel counts as
    *revealed before* t if its density was ``<= reveal_thresh`` at any frame
    ``t' < t`` (strictly earlier — the memory must have had a chance to store it).
    Returns a float ``B,T,1,H,W`` mask in {0, 1}.
    """
    occluded = (density >= occ_thresh).float()
    low = (density <= reveal_thresh).float()
    cum = torch.cummax(low, dim=1).values  # running "ever visible up to & incl. t"
    revealed_before = torch.zeros_like(cum)
    revealed_before[:, 1:] = cum[:, :-1]  # shift by one -> strictly earlier frames
    return occluded * revealed_before


def homography_to_4pt(h: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Convert homographies ``h`` (``...,3,3``) to 4-point corner offsets (``...,4,2``).

    The offset is ``H·corner − corner`` at four normalised corners; an identity
    homography maps to all-zero offsets (the deep-homography parameterisation).
    """
    lead = h.shape[:-2]
    hh = h.reshape(-1, 3, 3)
    corners = torch.tensor(_CORNERS, device=h.device, dtype=h.dtype)  # 4,2
    ones = torch.ones(4, 1, device=h.device, dtype=h.dtype)
    hom = torch.cat([corners, ones], dim=-1)  # 4,3
    warped = torch.matmul(hh, hom.t()).transpose(1, 2)  # N,4,3
    xy = warped[..., :2] / (warped[..., 2:3] + eps)  # N,4,2
    off = xy - corners
    return off.reshape(*lead, 4, 2)


# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------
def _z(device) -> torch.Tensor:
    return torch.zeros((), device=device)


def _first_present(d: dict, keys: tuple[str, ...]) -> Optional[torch.Tensor]:
    for k in keys:
        v = d.get(k)
        if v is not None:
            return v
    return None


def _flatten_time(x: Any) -> Any:
    """Flatten a leading ``B,T`` into ``B*T`` for time-stacked aux (dim >= 5)."""
    if torch.is_tensor(x) and x.dim() >= 5:
        return x.flatten(0, 1)
    return x


def _to_bt1(x: Optional[torch.Tensor], ref: torch.Tensor) -> Optional[torch.Tensor]:
    """Normalise a clip signal to ``B,T,1,Hr,Wr`` aligned to ``ref`` (``B,T,3,H,W``).

    Accepts ``B,T,C,H,W`` (channel-averaged) or ``B,T,H,W`` (channel added), and
    bilinearly resizes the spatial dims to ``ref``. None on any shape mismatch.
    """
    if x is None:
        return None
    b, t, hr, wr = ref.shape[0], ref.shape[1], ref.shape[-2], ref.shape[-1]
    if x.dim() == 5:
        x = x.mean(dim=2, keepdim=True)
    elif x.dim() == 4:
        x = x.unsqueeze(2)
    else:
        return None
    if x.shape[0] != b or x.shape[1] != t:
        return None
    x = x.to(ref.device).float()
    if x.shape[-2] != hr or x.shape[-1] != wr:
        flat = x.reshape(b * t, 1, x.shape[-2], x.shape[-1])
        flat = F.interpolate(flat, size=(hr, wr), mode="bilinear", align_corners=False)
        x = flat.reshape(b, t, 1, hr, wr)
    return x


def _align_map(x: Optional[torch.Tensor], ref: torch.Tensor, device) -> Optional[torch.Tensor]:
    """Align a per-frame map to ``ref`` (``N,3,H,W``) as ``N,1,H,W``; None if unusable."""
    if x is None:
        return None
    x = x.to(device).float()
    if x.dim() == 3:  # N,H,W
        x = x.unsqueeze(1)
    elif x.dim() == 4 and x.shape[1] != 1:  # N,C,H,W -> collapse channels
        x = x.mean(dim=1, keepdim=True)
    if x.dim() != 4:
        return None
    if x.shape[0] != ref.shape[0]:
        if x.shape[0] == 1:
            x = x.expand(ref.shape[0], -1, -1, -1)
        else:
            return None
    if x.shape[-2:] != ref.shape[-2:]:
        x = F.interpolate(x, size=ref.shape[-2:], mode="bilinear", align_corners=False)
    return x


def _meta_spatial(meta: Any, key: str, device) -> Optional[torch.Tensor]:
    """Read a spatial/tensor synthesis signal from batch meta.

    ``meta`` is either a single pre-collated dict (value used as-is) or the
    engine's list of per-sample dicts (per-sample values stacked along a new
    batch dim). Returns a float tensor on ``device`` or None when absent/ragged.
    """
    if isinstance(meta, dict):
        val = meta.get(key)
        if val is None:
            return None
        try:
            return torch.as_tensor(val, dtype=torch.float32).to(device)
        except Exception:
            return None
    if isinstance(meta, (list, tuple)) and meta and all(isinstance(m, dict) for m in meta):
        vals = [m.get(key) for m in meta]
        if any(v is None for v in vals):
            return None
        try:
            return torch.stack([torch.as_tensor(v, dtype=torch.float32) for v in vals]).to(device)
        except Exception:
            return None
    return None


def _get(obj: Any, key: str, default: Any = None) -> Any:
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)
