"""On-the-fly degradation synthesis from clean images.

Three generators, all operating on float32 CHW tensors in [0,1] and returning
`(hazy, params)` where params = {"beta", "airlight", "sigma", "domain"}:

* :func:`ground_haze`  — atmospheric scattering, depth-driven (Koschmieder ASM),
  beta ~ U[0.4, 3.0], near-gray airlight jitter.
* :func:`smoke`        — multi-octave Perlin density field (NOT depth-driven),
  colored soot/warm airlight, optional corner fire-glow.
* :func:`satellite`    — near-uniform transmission with per-channel wavelength
  bias (blue scatters most), slight low-frequency spatial variation.

Plus :func:`synthesize` (name dispatch) and :func:`synthesize_clip` which produces
a temporally-coherent clip by slowly varying beta and drifting the Perlin field.

Perlin/fractal noise is implemented here in pure torch (no extra deps) and is
seedable/reproducible via a ``torch.Generator``.
"""
from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F

from ..contracts import DOMAIN_HAZE, DOMAIN_SATELLITE, DOMAIN_SMOKE

Params = dict[str, Any]


# ---------------------------------------------------------------------------
# Perlin / fractal noise (pure torch, seedable)
# ---------------------------------------------------------------------------
def _fade(t: torch.Tensor) -> torch.Tensor:
    return 6 * t**5 - 15 * t**4 + 10 * t**3


def perlin_2d(
    shape: tuple[int, int],
    res: tuple[int, int],
    generator: torch.Generator | None = None,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Single-octave 2D Perlin (gradient) noise, roughly in [-1, 1].

    `res` = (periods_y, periods_x). `shape` must be divisible by `res`; callers
    should use :func:`fractal_noise` which handles padding for arbitrary sizes.
    """
    ry, rx = res
    h, w = shape
    dy, dx = h // ry, w // rx
    # fractional coordinate within each lattice cell
    ys = torch.arange(0, ry, 1.0 / dy, device=device, dtype=dtype)[:h]
    xs = torch.arange(0, rx, 1.0 / dx, device=device, dtype=dtype)[:w]
    gy, gx = torch.meshgrid(ys % 1, xs % 1, indexing="ij")
    grid = torch.stack((gy, gx), dim=-1)  # (h, w, 2)

    angles = 2 * math.pi * torch.rand(ry + 1, rx + 1, generator=generator, device=device, dtype=dtype)
    gradients = torch.stack((torch.cos(angles), torch.sin(angles)), dim=-1)  # (ry+1, rx+1, 2)

    def tile(sy: slice, sx: slice) -> torch.Tensor:
        g = gradients[sy, sx]
        return g.repeat_interleave(dy, 0).repeat_interleave(dx, 1)[:h, :w]

    def dot(grad: torch.Tensor, off_y: float, off_x: float) -> torch.Tensor:
        shifted = torch.stack((grid[..., 0] + off_y, grid[..., 1] + off_x), dim=-1)
        return (shifted * grad).sum(dim=-1)

    n00 = dot(tile(slice(0, -1), slice(0, -1)), 0.0, 0.0)
    n10 = dot(tile(slice(1, None), slice(0, -1)), -1.0, 0.0)
    n01 = dot(tile(slice(0, -1), slice(1, None)), 0.0, -1.0)
    n11 = dot(tile(slice(1, None), slice(1, None)), -1.0, -1.0)

    t = _fade(grid)
    n0 = n00 * (1 - t[..., 0]) + n10 * t[..., 0]
    n1 = n01 * (1 - t[..., 0]) + n11 * t[..., 0]
    return math.sqrt(2.0) * (n0 * (1 - t[..., 1]) + n1 * t[..., 1])


def fractal_noise(
    height: int,
    width: int,
    octaves: int = 4,
    base_res: int = 4,
    persistence: float = 0.5,
    lacunarity: int = 2,
    generator: torch.Generator | None = None,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
    normalize: bool = True,
) -> torch.Tensor:
    """Multi-octave fractal Perlin noise, shape (height, width).

    When ``normalize`` is True the output is min-max scaled to [0, 1]; otherwise it
    is the raw fBm sum (roughly [-1, 1]). Deterministic given ``generator``.
    """
    lacunarity = int(lacunarity)
    max_period = base_res * lacunarity ** (octaves - 1)
    ph = math.ceil(height / max_period) * max_period
    pw = math.ceil(width / max_period) * max_period

    noise = torch.zeros(ph, pw, device=device, dtype=dtype)
    amplitude, total, period = 1.0, 0.0, base_res
    for _ in range(octaves):
        noise = noise + amplitude * perlin_2d((ph, pw), (period, period), generator, device, dtype)
        total += amplitude
        amplitude *= persistence
        period *= lacunarity
    noise = noise[:height, :width] / total

    if normalize:
        lo, hi = noise.min(), noise.max()
        noise = (noise - lo) / (hi - lo + 1e-8)
    return noise


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _u(low: float, high: float, generator: torch.Generator | None) -> float:
    return float(low + (high - low) * torch.rand(1, generator=generator).item())


def fallback_depth(height: int, width: int, device: torch.device | str = "cpu") -> torch.Tensor:
    """Cheap gradient depth used when no teacher depth is supplied: farther toward
    the top of the frame (a common outdoor prior). Returns (1, H, W) in [0, 1]."""
    col = torch.linspace(1.0, 0.0, height, device=device).view(height, 1)
    return col.expand(height, width).unsqueeze(0).contiguous()


def _prep_depth(depth: torch.Tensor | None, h: int, w: int, device: torch.device | str) -> torch.Tensor:
    if depth is None:
        return fallback_depth(h, w, device)
    d = depth
    if d.dim() == 2:
        d = d.unsqueeze(0)
    if d.shape[-2:] != (h, w):
        d = F.interpolate(
            d.unsqueeze(0).float(), size=(h, w), mode="bilinear", align_corners=False
        ).squeeze(0)
    d = d.to(device).float()
    lo, hi = d.min(), d.max()
    return (d - lo) / (hi - lo + 1e-8)


# ---------------------------------------------------------------------------
# generators
# ---------------------------------------------------------------------------
def ground_haze(
    clean: torch.Tensor,
    depth: torch.Tensor | None = None,
    generator: torch.Generator | None = None,
    beta: float | None = None,
    airlight: torch.Tensor | None = None,
) -> tuple[torch.Tensor, Params]:
    """Depth-driven atmospheric scattering: I = J*t + A*(1-t), t = exp(-beta*d).

    beta ~ U[0.4, 3.0] (unless given); near-gray airlight jitter (unless given).
    """
    c, h, w = clean.shape
    device = clean.device
    d = _prep_depth(depth, h, w, device)  # (1, H, W) in [0,1], higher = farther

    if beta is None:
        beta = _u(0.4, 3.0, generator)
    if airlight is None:
        base = _u(0.7, 1.0, generator)
        jitter = (torch.rand(3, generator=generator) - 0.5) * 0.06
        airlight = (base + jitter).clamp(0.5, 1.0)
    airlight = airlight.to(device).view(3, 1, 1)

    t = torch.exp(-beta * d)  # (1, H, W)
    hazy = clean * t + airlight * (1 - t)
    hazy = hazy.clamp(0, 1)
    params = {
        "beta": float(beta),
        "airlight": airlight.view(3).cpu(),
        "sigma": 0.05,  # ground haze is near-homogeneous
        "domain": DOMAIN_HAZE,
    }
    return hazy, params


# soot-gray / brown / warm smoke tints (RGB, roughly)
_SMOKE_TINTS = (
    (0.55, 0.55, 0.55),  # neutral soot gray
    (0.62, 0.60, 0.55),  # light gray
    (0.45, 0.38, 0.30),  # dark brown soot
    (0.60, 0.50, 0.38),  # warm tan
    (0.70, 0.62, 0.50),  # dusty warm
)


def smoke(
    clean: torch.Tensor,
    generator: torch.Generator | None = None,
    beta: float | None = None,
    airlight: torch.Tensor | None = None,
    density: torch.Tensor | None = None,
    fire_glow: bool | None = None,
) -> tuple[torch.Tensor, Params]:
    """Non-homogeneous smoke: multi-octave Perlin density field (NOT depth-driven),
    colored soot/warm airlight, optional additive fire-glow in a random corner."""
    c, h, w = clean.shape
    device = clean.device

    if beta is None:
        beta = _u(1.0, 4.0, generator)
    if density is None:
        octaves = int(torch.randint(3, 6, (1,), generator=generator).item())
        density = fractal_noise(h, w, octaves=octaves, base_res=4, generator=generator, device=device)
    density = density.to(device).view(1, h, w)

    if airlight is None:
        idx = int(torch.randint(0, len(_SMOKE_TINTS), (1,), generator=generator).item())
        tint = torch.tensor(_SMOKE_TINTS[idx])
        tint = (tint + (torch.rand(3, generator=generator) - 0.5) * 0.08).clamp(0.2, 0.95)
        airlight = tint
    airlight = airlight.to(device).view(3, 1, 1)

    t = torch.exp(-beta * density)  # (1, H, W)
    hazy = clean * t + airlight * (1 - t)

    if fire_glow is None:
        fire_glow = torch.rand(1, generator=generator).item() < 0.35
    if fire_glow:
        hazy = _add_fire_glow(hazy, generator)

    hazy = hazy.clamp(0, 1)
    sigma = float(density.std().item())  # spatial non-homogeneity measure
    params = {
        "beta": float(beta),
        "airlight": airlight.view(3).cpu(),
        "sigma": sigma,
        "domain": DOMAIN_SMOKE,
    }
    return hazy, params


def _add_fire_glow(img: torch.Tensor, generator: torch.Generator | None) -> torch.Tensor:
    """Additive warm gain concentrated in one randomly-chosen corner region."""
    c, h, w = img.shape
    yy = torch.linspace(0, 1, h, device=img.device).view(h, 1)
    xx = torch.linspace(0, 1, w, device=img.device).view(1, w)
    corner = int(torch.randint(0, 4, (1,), generator=generator).item())
    cy = 0.0 if corner in (0, 1) else 1.0
    cx = 0.0 if corner in (0, 2) else 1.0
    dist2 = (yy - cy) ** 2 + (xx - cx) ** 2
    sigma = _u(0.15, 0.35, generator)
    mask = torch.exp(-dist2 / (2 * sigma**2))  # (h, w)
    gain = _u(0.15, 0.45, generator)
    warm = torch.tensor([1.0, 0.55, 0.2], device=img.device).view(3, 1, 1)
    return img + gain * mask.unsqueeze(0) * warm


def satellite(
    clean: torch.Tensor,
    generator: torch.Generator | None = None,
    beta: float | None = None,
    airlight: torch.Tensor | None = None,
) -> tuple[torch.Tensor, Params]:
    """Near-uniform thin haze with per-channel wavelength bias (blue scatters most),
    plus slight low-frequency spatial variation. No depth relation."""
    c, h, w = clean.shape
    device = clean.device

    if beta is None:
        beta = _u(0.2, 1.2, generator)
    # low-frequency spatial variation around a mean of 1.0
    var = fractal_noise(h, w, octaves=2, base_res=2, generator=generator, device=device)
    var = 0.85 + 0.30 * var  # in [0.85, 1.15]
    var = var.view(1, h, w).to(device)

    # wavelength bias: blue attenuated most -> largest per-channel beta
    band = torch.tensor([0.75, 1.0, 1.35], device=device).view(3, 1, 1)  # R, G, B
    t = torch.exp(-(beta * band) * var)  # (3, H, W)

    if airlight is None:
        base = _u(0.8, 1.0, generator)
        tint = torch.tensor([0.95, 0.98, 1.05])  # slightly bluish haze
        airlight = (base * tint + (torch.rand(3, generator=generator) - 0.5) * 0.04).clamp(0.6, 1.0)
    airlight = airlight.to(device).view(3, 1, 1)

    hazy = clean * t + airlight * (1 - t)
    hazy = hazy.clamp(0, 1)
    params = {
        "beta": float(beta),
        "airlight": airlight.view(3).cpu(),
        "sigma": float(var.std().item()),
        "domain": DOMAIN_SATELLITE,
    }
    return hazy, params


# ---------------------------------------------------------------------------
# dispatch + temporally-coherent clips
# ---------------------------------------------------------------------------
_GENERATORS = {
    "haze": ground_haze,
    DOMAIN_HAZE: ground_haze,
    "smoke": smoke,
    DOMAIN_SMOKE: smoke,
    "satellite": satellite,
    DOMAIN_SATELLITE: satellite,
}


def synthesize(
    clean: torch.Tensor,
    domain: str | int,
    generator: torch.Generator | None = None,
    depth: torch.Tensor | None = None,
    **kwargs: Any,
) -> tuple[torch.Tensor, Params]:
    """Dispatch to a generator by domain name ('haze'/'smoke'/'satellite') or id."""
    fn = _GENERATORS.get(domain)
    if fn is None:
        raise ValueError(f"unknown synthesis domain: {domain!r}")
    if fn is ground_haze:
        return fn(clean, depth=depth, generator=generator, **kwargs)
    return fn(clean, generator=generator, **kwargs)


def _drift_field(base: torch.Tensor, dy: float, dx: float) -> torch.Tensor:
    """Sub-pixel translate a (1,H,W) field by (dy, dx) pixels via grid_sample
    (reflection padding), giving a smooth temporal drift with no seams."""
    _, h, w = base.shape
    ys = torch.linspace(-1, 1, h, device=base.device)
    xs = torch.linspace(-1, 1, w, device=base.device)
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")
    gy = gy + 2 * dy / max(h - 1, 1)
    gx = gx + 2 * dx / max(w - 1, 1)
    grid = torch.stack((gx, gy), dim=-1).unsqueeze(0)
    out = F.grid_sample(
        base.unsqueeze(0), grid, mode="bilinear", padding_mode="reflection", align_corners=True
    )
    return out.squeeze(0)


def synthesize_clip(
    clean_frames: torch.Tensor,
    domain: str | int,
    generator: torch.Generator | None = None,
    depth: torch.Tensor | None = None,
    beta_jitter: float = 0.15,
    drift_px: float = 1.5,
) -> tuple[torch.Tensor, Params]:
    """Temporally-coherent synthesis over a clip.

    ``clean_frames`` is (T,3,H,W). beta varies smoothly across frames and, for smoke,
    the Perlin density field drifts a little each frame — so adjacent degraded frames
    stay close. Returns (hazy_clip (T,3,H,W), params) where params records the mean
    beta and the base airlight.
    """
    assert clean_frames.dim() == 4, "expected (T,3,H,W)"
    tt, c, h, w = clean_frames.shape
    device = clean_frames.device

    dom = domain
    if dom in ("haze", DOMAIN_HAZE):
        base_beta = _u(0.4, 3.0, generator)
        base_air = None
        base_density = None
    elif dom in ("smoke", DOMAIN_SMOKE):
        base_beta = _u(1.0, 4.0, generator)
        base_air = None
        octaves = int(torch.randint(3, 6, (1,), generator=generator).item())
        base_density = fractal_noise(h, w, octaves=octaves, base_res=4, generator=generator, device=device)
    else:
        base_beta = _u(0.2, 1.2, generator)
        base_air = None
        base_density = None

    phase = _u(0.0, 2 * math.pi, generator)
    out = torch.empty_like(clean_frames)
    params: Params = {}
    fixed_air: torch.Tensor | None = None
    for i in range(tt):
        # smooth beta variation (sinusoid) shared across the clip
        beta_i = base_beta * (1.0 + beta_jitter * math.sin(phase + i * 0.6))
        beta_i = max(0.05, beta_i)
        if dom in ("smoke", DOMAIN_SMOKE):
            dens_i = _drift_field(base_density.view(1, h, w), i * drift_px, i * drift_px * 0.5)
            frame, p = smoke(
                clean_frames[i], generator=generator, beta=beta_i, airlight=fixed_air, density=dens_i,
                fire_glow=False,
            )
        else:
            frame, p = synthesize(
                clean_frames[i], dom, generator=generator, depth=depth, beta=beta_i, airlight=fixed_air
            )
        if fixed_air is None:
            fixed_air = p["airlight"].to(device)  # freeze airlight for temporal consistency
        out[i] = frame
        params = p
    params["beta"] = float(base_beta)
    return out, params
