"""RevealNet (v2) synthetic supervision: drifting, opaque, camera-jittered smoke.

This module extends :mod:`pharos.data.synthesis` (it imports and reuses its Perlin
primitives, smoke tints and helpers — nothing is copied) to produce the training
signal described in DESIGN §9d:

    a dense, turbulent smoke volume DRIFTING over a clean video clip, with truly
    OPAQUE cores (transmission floor of exactly 0 — some regions fully occluded for
    stretches of frames) and a small per-frame CAMERA homography jitter applied
    consistently to both the clean background and the smoke.

The headline property (why "RevealNet"): over a long clip *most* background pixels
get revealed at *some* frame, while any *single* frame keeps 30-70% of the scene
heavily occluded. A memory model can therefore accumulate the scene over time even
though no single frame shows it. The exact per-frame 3x3 homography is returned so a
downstream aligner can be supervised, and the ground-truth clean background is known
at every pixel and frame.

Everything is pure torch, CPU-friendly and reproducible given a ``torch.Generator``.
Rendered tensors are float32 in ``[0, 1]``; ``cam_H`` is float32 ``(T, 3, 3)``.
"""
from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F

from ..contracts import DOMAIN_SMOKE
from .synthesis import _SMOKE_TINTS, _u, fractal_noise

__all__ = [
    "synthesize_reveal_clip",
    "coverage",
    "warp_homography",
]


# ---------------------------------------------------------------------------
# camera homography jitter (smooth random walk of small step homographies)
# ---------------------------------------------------------------------------
def _step_homography(
    h: int,
    w: int,
    tx: float,
    ty: float,
    rot_deg: float,
    log_scale: float,
    persp_x: float,
    persp_y: float,
    device: torch.device | str,
    dtype: torch.dtype,
) -> torch.Tensor:
    """A small homography about the image centre (source->dest pixel coordinates).

    Composed of translation, rotation, isotropic scale and a touch of perspective.
    Built as ``T(c) @ (P·S·R) @ T(-c)`` so rotation/scale/perspective act around the
    frame centre rather than the origin.
    """
    cx, cy = (w - 1) / 2.0, (h - 1) / 2.0
    ang = math.radians(rot_deg)
    s = math.exp(log_scale)
    cos_a, sin_a = math.cos(ang) * s, math.sin(ang) * s

    core = torch.tensor(
        [
            [cos_a, -sin_a, tx],
            [sin_a, cos_a, ty],
            [persp_x, persp_y, 1.0],
        ],
        device=device,
        dtype=dtype,
    )
    to_center = torch.tensor(
        [[1.0, 0.0, cx], [0.0, 1.0, cy], [0.0, 0.0, 1.0]], device=device, dtype=dtype
    )
    from_center = torch.tensor(
        [[1.0, 0.0, -cx], [0.0, 1.0, -cy], [0.0, 0.0, 1.0]], device=device, dtype=dtype
    )
    return to_center @ core @ from_center


def _camera_walk(
    t: int,
    h: int,
    w: int,
    generator: torch.Generator | None,
    trans_px: float,
    rot_deg: float,
    scale: float,
    persp: float,
    momentum: float,
    device: torch.device | str,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Cumulative per-frame homographies ``H_0..H_{T-1}`` (source frame-0 -> frame-t).

    ``H_0`` is the identity. Each frame advances the camera by a small step whose
    velocity is an AR(1) process (``momentum``) so the path is a *smooth* random walk
    (hand-held shake) rather than white noise. ``H_t = dH_t @ ... @ dH_1``.
    """
    out = torch.empty(max(t, 1), 3, 3, device=device, dtype=dtype)
    out[0] = torch.eye(3, device=device, dtype=dtype)
    # AR(1) velocity per DOF: v <- momentum*v + (1-momentum)*noise
    vel = torch.zeros(6, device=device, dtype=dtype)
    accum = out[0].clone()
    for i in range(1, t):
        noise = torch.tensor(
            [
                _u(-trans_px, trans_px, generator),
                _u(-trans_px, trans_px, generator),
                _u(-rot_deg, rot_deg, generator),
                _u(-scale, scale, generator),
                _u(-persp, persp, generator),
                _u(-persp, persp, generator),
            ],
            device=device,
            dtype=dtype,
        )
        vel = momentum * vel + (1.0 - momentum) * noise
        step = _step_homography(
            h,
            w,
            tx=float(vel[0]),
            ty=float(vel[1]),
            rot_deg=float(vel[2]),
            log_scale=float(vel[3]),
            persp_x=float(vel[4]) / max(w, 1),
            persp_y=float(vel[5]) / max(h, 1),
            device=device,
            dtype=dtype,
        )
        accum = step @ accum
        out[i] = accum
    return out


def warp_homography(
    img: torch.Tensor,
    homography: torch.Tensor,
    padding_mode: str = "border",
) -> torch.Tensor:
    """Backward-warp a ``(C, H, W)`` image by a 3x3 ``homography`` (source->dest).

    Output pixel ``p`` samples the source at ``homography^{-1} @ p`` (in pixel
    coordinates, with a perspective divide). ``warp_homography(x, I)`` is the
    identity, and warps compose: ``warp(warp(x, A), B) == warp(x, B @ A)`` up to
    resampling. This is the convention used for both the returned ``cam_H`` and the
    internal application to the clean/smoke layers.
    """
    _, h, w = img.shape
    device, dtype = img.device, img.dtype
    ys, xs = torch.meshgrid(
        torch.arange(h, device=device, dtype=dtype),
        torch.arange(w, device=device, dtype=dtype),
        indexing="ij",
    )
    ones = torch.ones_like(xs)
    dest = torch.stack((xs, ys, ones), dim=0).reshape(3, -1)  # 3, H*W
    src = torch.linalg.inv(homography.to(device=device, dtype=dtype)) @ dest  # 3, H*W
    src = src[:2] / src[2:3].clamp_min(1e-8)
    gx = 2.0 * src[0] / max(w - 1, 1) - 1.0
    gy = 2.0 * src[1] / max(h - 1, 1) - 1.0
    grid = torch.stack((gx, gy), dim=-1).reshape(1, h, w, 2)
    out = F.grid_sample(
        img.unsqueeze(0), grid, mode="bilinear", padding_mode=padding_mode, align_corners=True
    )
    return out.squeeze(0)


# ---------------------------------------------------------------------------
# advection of the density field by a smooth velocity field + turbulence
# ---------------------------------------------------------------------------
def _velocity_field(
    h: int,
    w: int,
    generator: torch.Generator | None,
    drift_px: float,
    swirl_px: float,
    device: torch.device | str,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Smooth 2D velocity field ``(2, H, W)`` in pixels/frame: a global drift plus a
    low-frequency swirl so different parts of the field move differently."""
    ang = _u(0.0, 2 * math.pi, generator)
    gx = drift_px * math.cos(ang)
    gy = drift_px * math.sin(ang)
    swx = fractal_noise(h, w, octaves=2, base_res=2, generator=generator, device=device) * 2 - 1
    swy = fractal_noise(h, w, octaves=2, base_res=2, generator=generator, device=device) * 2 - 1
    vx = gx + swirl_px * swx
    vy = gy + swirl_px * swy
    return torch.stack((vx, vy), dim=0).to(dtype)


def _advect(
    base: torch.Tensor,
    frame: int,
    velocity: torch.Tensor,
    turb: torch.Tensor,
    turb_px: float,
    turb_speed: float,
    turb_phase: float,
) -> torch.Tensor:
    """Sample ``base`` (1,H,W) at coordinates displaced by ``frame`` steps of the
    velocity field plus a time-varying turbulence wobble. Reflection padding keeps
    the drifting field seamless."""
    _, h, w = base.shape
    device, dtype = base.device, base.dtype
    ys = torch.linspace(-1, 1, h, device=device, dtype=dtype).view(h, 1).expand(h, w)
    xs = torch.linspace(-1, 1, w, device=device, dtype=dtype).view(1, w).expand(h, w)

    wob = turb_px * math.sin(turb_phase + frame * turb_speed)
    off_x = frame * velocity[0] + wob * turb[0]
    off_y = frame * velocity[1] + wob * turb[1]
    # advection is a backward lookup: density_i(x) = base(x - offset)
    gx = xs - 2.0 * off_x / max(w - 1, 1)
    gy = ys - 2.0 * off_y / max(h - 1, 1)
    grid = torch.stack((gx, gy), dim=-1).unsqueeze(0)
    out = F.grid_sample(
        base.unsqueeze(0), grid, mode="bilinear", padding_mode="reflection", align_corners=True
    )
    return out.squeeze(0)


# ---------------------------------------------------------------------------
# main entry point
# ---------------------------------------------------------------------------
def synthesize_reveal_clip(
    clean_frames: torch.Tensor,
    generator: torch.Generator | None = None,
    *,
    beta: float | None = None,
    density_gamma: float = 1.8,
    core_thresh: float = 0.60,
    octaves: int | None = None,
    base_res: int = 4,
    drift_px: float = 2.6,
    swirl_px: float = 4.5,
    turb_px: float = 2.5,
    turb_speed: float = 0.8,
    cam_trans_px: float = 2.0,
    cam_rot_deg: float = 1.1,
    cam_scale: float = 0.012,
    cam_persp: float = 0.5,
    cam_momentum: float = 0.55,
    airlight: torch.Tensor | None = None,
    reveal_thresh: float = 0.5,
) -> dict[str, Any]:
    """Synthesize a drifting, opaque, camera-jittered smoke clip over ``clean_frames``.

    Parameters
    ----------
    clean_frames : (T, 3, H, W) float tensor in [0, 1]
        The clean background clip. For a *static* scene pass a repeated still; the
        returned ``cam_H`` then captures **all** background motion and
        ``warp_homography(gt[0], cam_H[t]) == gt[t]``.
    beta : optical-thickness scale (default random in [2.3, 3.0]); higher -> thicker
        smoke gradient. (Fully opaque cores come from ``core_thresh``, not ``beta``.)
    density_gamma : contrast applied to the [0,1] density (``>1`` opens up clear gaps
        while keeping cores dense), boosting both opacity and reveal frequency.
    core_thresh : density at/above which transmission is clamped to exactly 0 (opaque
        core / transmission floor 0).
    drift_px, swirl_px, turb_px, turb_speed : smoke advection (global drift, spatial
        swirl, turbulence amplitude and temporal frequency).
    cam_* : per-step camera-jitter magnitudes and AR(1) ``cam_momentum``.
    reveal_thresh : transmission above which a pixel counts as "revealed" (used only to
        populate the returned ``revealed`` mask / :func:`coverage`).

    Returns
    -------
    dict with keys:
        ``hazy``         (T, 3, H, W) degraded clip in [0, 1]
        ``gt``           (T, 3, H, W) clean background *in the same (jittered) view*
        ``smoke_density``(T, 1, H, W) drifting density field in [0, 1] (high = thick)
        ``transmission`` (T, 1, H, W) per-pixel transmission in [0, 1] (0 = opaque)
        ``revealed``     (T, 1, H, W) bool, ``transmission > reveal_thresh``
        ``cam_H``        (T, 3, 3) per-frame homography (frame-0 source -> frame-t)
        ``airlight``     (3,) frozen smoke colour, ``beta`` (float), ``reveal_thresh``
    """
    if clean_frames.dim() != 4 or clean_frames.shape[1] != 3:
        raise ValueError(
            f"expected clean_frames of shape (T,3,H,W), got {tuple(clean_frames.shape)}"
        )
    t, _, h, w = clean_frames.shape
    device = clean_frames.device
    dtype = clean_frames.dtype

    if beta is None:
        beta = _u(2.3, 3.0, generator)
    if octaves is None:
        octaves = 5

    # base density + fixed turbulence direction fields
    base_density = (
        fractal_noise(h, w, octaves=octaves, base_res=base_res, generator=generator, device=device)
        .view(1, h, w)
        .to(dtype)
    )
    turb = torch.stack(
        (
            fractal_noise(h, w, octaves=3, base_res=3, generator=generator, device=device) * 2 - 1,
            fractal_noise(h, w, octaves=3, base_res=3, generator=generator, device=device) * 2 - 1,
        ),
        dim=0,
    ).to(dtype)
    velocity = _velocity_field(h, w, generator, drift_px, swirl_px, device, dtype)
    turb_phase = _u(0.0, 2 * math.pi, generator)

    # frozen colored airlight (reuse the smoke tint palette)
    if airlight is None:
        idx = int(torch.randint(0, len(_SMOKE_TINTS), (1,), generator=generator).item())
        tint = torch.tensor(_SMOKE_TINTS[idx])
        tint = (tint + (torch.rand(3, generator=generator) - 0.5) * 0.08).clamp(0.2, 0.95)
        airlight = tint
    air = airlight.to(device=device, dtype=dtype).view(3, 1, 1)

    cam_H = _camera_walk(
        t, h, w, generator, cam_trans_px, cam_rot_deg, cam_scale, cam_persp, cam_momentum,
        device, dtype,
    )

    hazy = torch.empty(t, 3, h, w, device=device, dtype=dtype)
    gt = torch.empty(t, 3, h, w, device=device, dtype=dtype)
    density = torch.empty(t, 1, h, w, device=device, dtype=dtype)
    transmission = torch.empty(t, 1, h, w, device=device, dtype=dtype)

    for i in range(t):
        # 1) smoke drifts in the world frame
        dens_world = _advect(base_density, i, velocity, turb, turb_px, turb_speed, turb_phase)
        dens_world = dens_world.clamp(0, 1) ** density_gamma
        # 2) camera moves both the smoke and the background into the current view
        dens_view = warp_homography(dens_world, cam_H[i]).clamp(0, 1)
        clean_view = warp_homography(clean_frames[i], cam_H[i]).clamp(0, 1)
        # 3) transmission with a genuine opaque-core floor of 0
        tval = torch.exp(-beta * dens_view)
        tval = torch.where(dens_view >= core_thresh, torch.zeros_like(tval), tval)
        # 4) composite: convex blend keeps hazy in [0, 1]
        frame = clean_view * tval + air * (1.0 - tval)

        hazy[i] = frame.clamp(0, 1)
        gt[i] = clean_view
        density[i] = dens_view
        transmission[i] = tval

    revealed = transmission > reveal_thresh
    return {
        "hazy": hazy,
        "gt": gt,
        "smoke_density": density,
        "transmission": transmission,
        "revealed": revealed,
        "cam_H": cam_H,
        "airlight": air.view(3).cpu(),
        "beta": float(beta),
        "reveal_thresh": float(reveal_thresh),
        "domain": DOMAIN_SMOKE,
    }


# ---------------------------------------------------------------------------
# coverage helper
# ---------------------------------------------------------------------------
def coverage(
    transmission: torch.Tensor,
    thresh: float = 0.5,
    upto: int | None = None,
) -> list[float] | float:
    """Fraction of pixels revealed at least once up to each frame.

    A pixel is "revealed" at frame ``s`` when ``transmission[s] > thresh``. The value
    at frame ``t`` is the fraction of pixels revealed in *any* frame ``s <= t`` and is
    monotonically non-decreasing in ``t``.

    Parameters
    ----------
    transmission : (T, 1, H, W) or (T, H, W) tensor in [0, 1].
    thresh : reveal threshold on transmission.
    upto : if given, return the single scalar coverage up to and including frame
        ``upto``; otherwise return the whole length-``T`` cumulative curve as a list.

    Notes
    -----
    Coverage is measured in raw pixel coordinates (camera jitter is small relative to
    the smoke drift that drives revelation). To measure scene coverage under large
    camera motion, first align the transmission stack to a common frame with
    :func:`warp_homography` and ``cam_H``.
    """
    tr = transmission
    if tr.dim() == 4:
        tr = tr[:, 0]
    elif tr.dim() != 3:
        raise ValueError(f"expected (T,1,H,W) or (T,H,W), got {tuple(transmission.shape)}")
    revealed = tr > thresh  # T, H, W
    ever = torch.zeros_like(revealed[0])
    curve: list[float] = []
    for i in range(revealed.shape[0]):
        ever = ever | revealed[i]
        curve.append(float(ever.float().mean().item()))
    if upto is None:
        return curve
    return curve[max(0, min(upto, len(curve) - 1))]
