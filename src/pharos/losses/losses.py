"""PharosLoss: the full training loss stack (DESIGN.md §5).

L = L_rec + λ_freq·L_freq + λ_conf·L_conf + λ_depth·L_depth
      + λ_det·L_det + λ_temp·L_temp + λ_phys·L_phys

Every term degrades gracefully to exactly 0 when its inputs are unavailable
(no clean GT, teacher disabled, missing aux keys, image-mode batch for temporal,
no synthesis params for physics). Training therefore runs with any subset of
teachers/losses enabled. `__call__` returns `(total, {term: float})` with the
per-term values detached.
"""
from __future__ import annotations

from typing import Any, Optional

import torch
import torch.nn.functional as F

from ..contracts import PharosOutput
from ..teachers.flow import flow_warp


class PharosLoss:
    """Implements contracts.LossFn."""

    def __init__(self, cfg: Any) -> None:
        loss_cfg = _get(cfg, "loss", {}) or {}
        self.w = {
            "rec": float(_get(loss_cfg, "rec", 1.0)),
            "freq": float(_get(loss_cfg, "freq", 0.1)),
            "conf": float(_get(loss_cfg, "conf", 0.05)),
            "depth": float(_get(loss_cfg, "depth", 0.1)),
            "det": float(_get(loss_cfg, "det", 0.05)),
            "temp": float(_get(loss_cfg, "temp", 0.5)),
            "phys": float(_get(loss_cfg, "phys", 0.1)),
        }
        det_cfg = _get(_get(cfg, "teachers", {}) or {}, "detector", {}) or {}
        self.det_every_n = int(_get(det_cfg, "every_n", 1) or 1)

        # tunables (not part of the frozen contract; kept as attributes)
        self.charb_eps = 1e-3
        self.conf_eps = 1e-4
        self.depth_pairs = 512
        self.depth_lowres = 64
        self.eps = 1e-8
        self._det_counter = 0

    # ------------------------------------------------------------------
    def __call__(
        self, out: PharosOutput, batch: dict, teachers: Any
    ) -> tuple[torch.Tensor, dict[str, float]]:
        device = out.output.device
        clean = _match_clean(batch.get("clean"), out.output)
        is_clip = bool(batch.get("clip", False))

        terms: dict[str, torch.Tensor] = {
            "rec": self._rec(out, clean, device),
            "freq": self._freq(out, clean, device),
            "conf": self._conf(out, clean, device),
            "depth": self._depth(out, batch, teachers, device),
            "det": self._det(out, batch, teachers, device),
            "temp": self._temporal(out, batch, teachers, device, is_clip),
            "phys": self._phys(out, batch, device),
        }
        total = torch.zeros((), device=device)
        for name, val in terms.items():
            total = total + self.w[name] * val
        log = {name: float(val.detach()) for name, val in terms.items()}
        log["total"] = float(total.detach())
        return total, log

    # ------------------------------------------------------------------
    # L_rec — Charbonnier(J, GT)
    def _rec(self, out: PharosOutput, clean: Optional[torch.Tensor], device) -> torch.Tensor:
        if clean is None:
            return _z(device)
        return _charbonnier(out.output, clean, self.charb_eps)

    # L_freq — L1 on FFT amplitude
    def _freq(self, out: PharosOutput, clean: Optional[torch.Tensor], device) -> torch.Tensor:
        if clean is None:
            return _z(device)
        fo = torch.fft.rfft2(out.output.float(), dim=(-2, -1))
        fc = torch.fft.rfft2(clean.float(), dim=(-2, -1))
        return (fo.abs() - fc.abs()).abs().mean()

    # L_conf — heteroscedastic Laplace NLL.
    # confidence in (0,1] is read as precision p; sigma = 1/p, so
    #   NLL = |err|/sigma + log sigma = |err|·p − log p   (numerically stable).
    def _conf(self, out: PharosOutput, clean: Optional[torch.Tensor], device) -> torch.Tensor:
        if clean is None or out.confidence is None:
            return _z(device)
        err = (out.output - clean).abs().mean(dim=1, keepdim=True)  # B,1,H,W
        conf = out.confidence
        if conf.shape[-2:] != err.shape[-2:]:
            conf = F.interpolate(conf, size=err.shape[-2:], mode="bilinear", align_corners=False)
        p = conf.clamp(self.conf_eps, 1.0)
        return (err * p - torch.log(p)).mean()

    # L_depth — student feature-affinity vs teacher depth-affinity on clean img.
    def _depth(self, out: PharosOutput, batch: dict, teachers: Any, device) -> torch.Tensor:
        depth_fn = _get(teachers, "depth", None)
        clean = batch.get("clean")
        if depth_fn is None or clean is None:
            return _z(device)
        clean_img = clean[:, -1] if clean.dim() == 5 else clean  # current frame for clips

        feats = _student_feats(out)
        if feats is None:
            return _z(device)
        b, c, fh, fw = feats.shape

        with torch.no_grad():
            depth = depth_fn(clean_img)  # B,1,h,w in [0,1]
        if depth is None:
            return _z(device)
        depth = F.interpolate(depth.float().to(device), size=(fh, fw), mode="bilinear", align_corners=False)

        n = fh * fw
        if n < 2:
            return _z(device)
        k = min(self.depth_pairs, n * (n - 1))
        idx_i = torch.randint(0, n, (k,), device=device)
        idx_j = torch.randint(0, n, (k,), device=device)

        d_flat = depth.view(b, n)  # B,N
        a_teacher = (d_flat[:, idx_i] - d_flat[:, idx_j]).abs()  # B,K

        f_flat = feats.view(b, c, n)
        fi = f_flat[:, :, idx_i]  # B,C,K
        fj = f_flat[:, :, idx_j]
        cos = F.cosine_similarity(fi, fj, dim=1, eps=self.eps)  # B,K
        a_student = 1.0 - cos  # cosine distance

        a_teacher = _l2norm_rows(a_teacher, self.eps)
        a_student = _l2norm_rows(a_student, self.eps)
        return (a_student - a_teacher).abs().mean()

    # L_det — L1 between detector FPN feats on J vs clean (every_n-th call only).
    def _det(self, out: PharosOutput, batch: dict, teachers: Any, device) -> torch.Tensor:
        det_fn = _get(teachers, "detector", None)
        clean = _match_clean(batch.get("clean"), out.output)
        if det_fn is None or clean is None:
            return _z(device)
        self._det_counter += 1
        if self._det_counter % self.det_every_n != 0:
            return _z(device)
        with torch.no_grad():
            feats_gt = det_fn(clean)
        feats_j = det_fn(out.output)
        if not feats_j or not feats_gt:
            return _z(device)
        loss = _z(device)
        n = min(len(feats_j), len(feats_gt))
        for i in range(n):
            loss = loss + (feats_j[i] - feats_gt[i].detach()).abs().mean()
        return loss / max(n, 1)

    # L_temp — grid smoothness + flow-warp photometric on clean-frame teacher flow.
    def _temporal(self, out: PharosOutput, batch: dict, teachers: Any, device, is_clip: bool) -> torch.Tensor:
        if not is_clip:
            return _z(device)
        loss = _z(device)

        # grid smoothness ‖G_t − G_{t-1}‖₁ (optionally scene-cut weighted)
        grids = out.aux.get("grids") if isinstance(out.aux, dict) else None
        if grids is not None and grids.dim() == 6 and grids.shape[1] > 1:
            diff = (grids[:, 1:] - grids[:, :-1]).abs()  # B,T-1,12,D,Gh,Gw
            w = _scene_cut_weight(batch, grids.shape[1] - 1, device)
            if w is not None:
                diff = diff * w.view(1, -1, 1, 1, 1, 1)
            loss = loss + diff.mean()

        # flow-warp photometric on per-frame outputs, flow from clean frames
        flow_fn = _get(teachers, "flow", None)
        outs = out.aux.get("outputs") if isinstance(out.aux, dict) else None
        clean = batch.get("clean")
        if (
            flow_fn is not None
            and outs is not None
            and outs.dim() == 5
            and outs.shape[1] > 1
            and clean is not None
            and clean.dim() == 5
            and clean.shape[1] == outs.shape[1]
        ):
            t = outs.shape[1]
            photo = _z(device)
            for ti in range(1, t):
                with torch.no_grad():
                    flow = flow_fn(clean[:, ti - 1], clean[:, ti])  # prev -> cur
                warped = flow_warp(outs[:, ti], flow.to(device))  # cur aligned to prev
                photo = photo + _charbonnier(warped, outs[:, ti - 1], self.charb_eps)
            loss = loss + photo / max(t - 1, 1)
        return loss

    # L_phys — supervised beta/airlight/sigma (L1) + domain (CE), when present.
    def _phys(self, out: PharosOutput, batch: dict, device) -> torch.Tensor:
        deg = out.deg or {}
        meta = batch.get("meta") or {}
        loss = _z(device)
        found = False

        for key in ("beta", "airlight", "sigma"):
            pred = _get(deg, key, None)
            target = _tensor_like(_meta_get(meta, key), pred, device)
            if pred is not None and target is not None:
                loss = loss + (pred - target).abs().mean()
                found = True

        logits = _get(deg, "domain_logits", None)
        domain = batch.get("domain")
        if logits is not None and domain is not None:
            dom = domain.to(device).long().view(-1)
            if dom.numel() == logits.shape[0]:
                loss = loss + F.cross_entropy(logits, dom)
                found = True

        return loss if found else _z(device)


# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------
def _z(device) -> torch.Tensor:
    return torch.zeros((), device=device)


def _charbonnier(x: torch.Tensor, y: torch.Tensor, eps: float) -> torch.Tensor:
    return torch.sqrt((x - y) ** 2 + eps * eps).mean()


def _l2norm_rows(x: torch.Tensor, eps: float) -> torch.Tensor:
    return x / (x.norm(dim=-1, keepdim=True) + eps)


def _match_clean(clean: Optional[torch.Tensor], ref: torch.Tensor) -> Optional[torch.Tensor]:
    """Align a (possibly clip) clean tensor to the single-frame output.

    Returns the current (last) frame of a clip so per-frame terms compare to the
    frame the output corresponds to. None when shapes are incompatible.
    """
    if clean is None:
        return None
    if clean.dim() == ref.dim():
        return clean
    if clean.dim() == 5 and ref.dim() == 4:
        return clean[:, -1]
    return None


def _student_feats(out: PharosOutput) -> Optional[torch.Tensor]:
    """Low-res student features for the depth-affinity loss.

    Prefers out.aux['lowres_feats'] (B,C,h,w); falls back to pooling the affine
    grid out.grid (B,12,D,Gh,Gw) over the guidance-bin dim to (B,12,Gh,Gw).
    """
    aux = out.aux if isinstance(out.aux, dict) else {}
    feats = aux.get("lowres_feats")
    if feats is not None and feats.dim() == 4:
        return feats
    grid = out.grid
    if grid is not None and grid.dim() == 5:
        return grid.mean(dim=2)  # B,12,Gh,Gw
    if grid is not None and grid.dim() == 4:
        return grid
    return None


def _meta_get(meta: Any, key: str) -> Any:
    """Read a synthesis param from batch meta.

    meta may be a single dict (pre-collated values) or a list of per-sample
    dicts (the engine's collate keeps it as a list); the list form is stacked
    into a B,* float tensor. None when the key is absent from any sample.
    """
    if isinstance(meta, dict):
        return meta.get(key)
    if isinstance(meta, (list, tuple)) and meta and all(isinstance(m, dict) for m in meta):
        vals = [m.get(key) for m in meta]
        if any(v is None for v in vals):
            return None
        try:
            return torch.stack([torch.as_tensor(v, dtype=torch.float32).reshape(-1) for v in vals])
        except Exception:
            return None
    return None


def _scene_cut_weight(batch: dict, length: int, device) -> Optional[torch.Tensor]:
    meta = batch.get("meta") or {}
    sc = _meta_get(meta, "scene_cut")
    if sc is None:
        return None
    try:
        w = torch.as_tensor(sc, device=device, dtype=torch.float32).view(-1)
    except Exception:
        return None
    if w.numel() < length:
        return None
    return (1.0 - w[:length]).clamp(0.0, 1.0)


def _tensor_like(value: Any, ref: Optional[torch.Tensor], device) -> Optional[torch.Tensor]:
    """Coerce a meta synthesis param to a tensor broadcastable onto `ref`."""
    if value is None or ref is None:
        return None
    try:
        t = torch.as_tensor(value, device=device, dtype=ref.dtype)
    except Exception:
        return None
    if t.dim() == 1 and ref.dim() == 2 and t.shape[0] == ref.shape[0]:
        t = t.view(ref.shape[0], -1)
    if t.shape == ref.shape:
        return t
    try:
        return t.expand_as(ref)
    except Exception:
        return None


def _get(obj: Any, key: str, default: Any = None) -> Any:
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)
