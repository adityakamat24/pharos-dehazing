"""RevealNet (v2) evaluation: does the model *remember* the scene through smoke?

Single-frame restoration can only report what the current frame shows. A memory
model (DESIGN §9d) should additionally reconstruct pixels that are occluded *now* but
were seen clearly at an earlier frame. These metrics quantify exactly that:

* :func:`psnr_over_time`  — per-frame PSNR of the restored clip vs GT.
* :func:`recall_curve`    — the REVEAL metric. For every frame, restrict attention to
  pixels that are occluded now but had been revealed earlier, and report the PSNR and
  the fraction correctly recovered there. A no-memory model scores ~0; a
  perfect-memory model scores ~1.
* :func:`time_to_recover` — scalar summaries of the above (mean recall PSNR / fraction
  correct, mean time-to-first-reveal, final recovered fraction).

All functions are pure torch and CPU-testable. Tensors are ``(T, C, H, W)`` (or
``(T, H, W)``) float in ``[0, 1]``; ``density`` is the smoke density where a *higher*
value means *more* occlusion (as produced by
:func:`pharos.data.reveal_synthesis.synthesize_reveal_clip`). ``thresh`` splits
occluded (``density > thresh``) from revealed (``density <= thresh``).

The metrics operate per pixel-location across time and therefore assume the frames
are already registered to a common view (e.g. via the known ``cam_H``); with only
small camera jitter they are a good approximation directly.
"""
from __future__ import annotations

import math

import torch

from .metrics import psnr

__all__ = ["psnr_over_time", "recall_curve", "time_to_recover"]

_PSNR_CAP = 100.0  # dB cap used when averaging (perfect matches give +inf per frame)


def _as_tchw(x: torch.Tensor) -> torch.Tensor:
    if x.dim() == 3:  # (T, H, W) -> (T, 1, H, W)
        return x.unsqueeze(1)
    if x.dim() == 4:
        return x
    raise ValueError(f"expected (T,C,H,W) or (T,H,W) tensor, got shape {tuple(x.shape)}")


def _as_thw(density: torch.Tensor) -> torch.Tensor:
    if density.dim() == 4:  # (T, 1, H, W) or (T, C, H, W)
        return density[:, 0] if density.shape[1] == 1 else density.mean(dim=1)
    if density.dim() == 3:
        return density
    raise ValueError(f"expected density (T,1,H,W) or (T,H,W), got shape {tuple(density.shape)}")


def _masked_psnr(out: torch.Tensor, gt: torch.Tensor, mask: torch.Tensor, max_val: float) -> float:
    """PSNR over the pixels selected by ``mask`` (H,W bool). ``out``/``gt`` are (C,H,W)."""
    n = int(mask.sum().item())
    if n == 0:
        return float("nan")
    c = out.shape[0]
    sel = mask.unsqueeze(0)  # 1,H,W
    sq = ((out - gt) ** 2) * sel
    mse = float(sq.sum().item()) / (n * c)
    if mse <= 0.0:
        return float("inf")
    return 10.0 * math.log10((max_val ** 2) / mse)


def _frac_correct(out: torch.Tensor, gt: torch.Tensor, mask: torch.Tensor, tol: float) -> float:
    """Fraction of masked pixels whose per-pixel RMS error (over channels) is <= tol."""
    n = int(mask.sum().item())
    if n == 0:
        return float("nan")
    err = ((out - gt) ** 2).mean(dim=0).sqrt()  # H,W
    correct = (err <= tol) & mask
    return float(correct.sum().item()) / n


def _finite_mean(xs: list[float], cap: float | None = None) -> float:
    """Mean over the non-NaN entries; ``inf`` is replaced by ``cap`` when given."""
    vals: list[float] = []
    for x in xs:
        if x != x:  # NaN
            continue
        if math.isinf(x):
            if cap is None:
                continue
            vals.append(cap)
        else:
            vals.append(x)
    return float(sum(vals) / len(vals)) if vals else float("nan")


def psnr_over_time(outputs: torch.Tensor, gt: torch.Tensor, max_val: float = 1.0) -> list[float]:
    """Per-frame PSNR (dB) of ``outputs`` vs ``gt`` for a clip.

    ``outputs`` / ``gt`` are ``(T, C, H, W)`` (or ``(T, H, W)``). Returns a list of
    length ``T``; a perfectly reconstructed frame contributes ``+inf`` (as in
    :func:`pharos.engine.metrics.psnr`).
    """
    o, g = _as_tchw(outputs), _as_tchw(gt)
    if o.shape != g.shape:
        raise ValueError(f"shape mismatch {tuple(o.shape)} vs {tuple(g.shape)}")
    return [psnr(o[t], g[t], max_val=max_val) for t in range(o.shape[0])]


def recall_curve(
    outputs: torch.Tensor,
    gt: torch.Tensor,
    density: torch.Tensor,
    thresh: float = 0.5,
    tol: float = 0.05,
    max_val: float = 1.0,
) -> dict[str, list[float]]:
    """The REVEAL metric as a function of frame index.

    For each frame ``t`` the *recall region* is the set of pixels that are occluded at
    ``t`` (``density[t] > thresh``) yet were revealed at some earlier frame
    (``density[s] <= thresh`` for some ``s < t``). Only a model with memory can fill
    these in, so scores here separate memory models from single-frame ones.

    Returns a dict of length-``T`` lists:
        ``psnr_recall``  masked PSNR over the recall region (``nan`` if empty).
        ``frac_correct`` fraction of the recall region reconstructed within ``tol``
                         RMS error (``nan`` if empty).
        ``frac_region``  size of the recall region as a fraction of all pixels.
    """
    o, g, d = _as_tchw(outputs), _as_tchw(gt), _as_thw(density)
    t_len = o.shape[0]
    if not (o.shape == g.shape and d.shape[0] == t_len and d.shape[-2:] == o.shape[-2:]):
        raise ValueError(
            f"incompatible shapes outputs{tuple(o.shape)} gt{tuple(g.shape)} density{tuple(d.shape)}"
        )
    occluded = d > thresh
    revealed = ~occluded
    seen_before = torch.zeros_like(revealed[0])
    psnr_recall: list[float] = []
    frac_correct: list[float] = []
    frac_region: list[float] = []
    for t in range(t_len):
        recall_mask = occluded[t] & seen_before
        psnr_recall.append(_masked_psnr(o[t], g[t], recall_mask, max_val))
        frac_correct.append(_frac_correct(o[t], g[t], recall_mask, tol))
        frac_region.append(float(recall_mask.float().mean().item()))
        seen_before = seen_before | revealed[t]
    return {"psnr_recall": psnr_recall, "frac_correct": frac_correct, "frac_region": frac_region}


def time_to_recover(
    outputs: torch.Tensor,
    gt: torch.Tensor,
    density: torch.Tensor,
    thresh: float = 0.5,
    tol: float = 0.05,
    max_val: float = 1.0,
) -> dict[str, float]:
    """Scalar summaries of the reveal dynamics.

    Returns
    -------
    dict with:
        ``mean_recall_psnr``   mean masked recall PSNR over frames (inf capped at
                               100 dB, empty frames ignored).
        ``mean_frac_correct``  mean recall fraction-correct over frames.
        ``final_frac_correct`` recall fraction-correct at the last non-empty frame.
        ``recall_region_frac_final`` size of the recall region at the last frame.
        ``reveal_time_mean``   mean frame index of first direct reveal, averaged over
                               pixels that are ever revealed (how long until the scene
                               is first seen; lower is better coverage).
    """
    rc = recall_curve(outputs, gt, density, thresh=thresh, tol=tol, max_val=max_val)
    d = _as_thw(density)
    revealed = d <= thresh  # T,H,W
    ever = revealed.any(dim=0)
    first_reveal = torch.argmax(revealed.float(), dim=0)  # H,W (0 where never revealed)
    reveal_time_mean = (
        float(first_reveal[ever].float().mean().item()) if bool(ever.any()) else float("nan")
    )

    fc = rc["frac_correct"]
    final_fc = next((v for v in reversed(fc) if v == v), float("nan"))  # last non-NaN
    return {
        "mean_recall_psnr": _finite_mean(rc["psnr_recall"], cap=_PSNR_CAP),
        "mean_frac_correct": _finite_mean(fc),
        "final_frac_correct": final_fc,
        "recall_region_frac_final": rc["frac_region"][-1] if rc["frac_region"] else float("nan"),
        "reveal_time_mean": reveal_time_mean,
    }
