"""On-the-fly degradation synthesis from clean images.

Three generators, all operating on float32 CHW tensors in [0,1] and returning
`(hazy, params)` where params = {"beta", "beta_bs", "airlight", "sigma", "domain"}:

* :func:`ground_haze`  — atmospheric scattering, depth-driven (Koschmieder ASM),
  beta ~ U[0.4, 3.0], near-gray airlight jitter. With ``isp_aware`` the scattering
  is injected in linear (pre-ISP) space with Poisson shot noise, then the camera
  ISP gamma is re-applied (SynFog, CVPR'24).
* :func:`smoke`        — density field (NOT depth-driven), colored soot/warm
  airlight, optional corner fire-glow. ``smoke_mode`` selects the density model:
  ``"perlin"`` (legacy multi-octave fBm), ``"turbulent"`` (source-point curl-noise
  advection, LSD3K/STANet lineage), or ``"mix"`` (50/50 per sample).
* :func:`satellite`    — near-uniform transmission with per-channel wavelength
  bias (blue scatters most), slight low-frequency spatial variation.

All generators expose ``beta`` (attenuation coeff) and ``beta_bs`` (backscatter /
airlight-build-up coeff, Sea-thru split). They coincide in the legacy paths and
differ slightly in turbulent-smoke / ISP-aware modes (``beta_bs = beta·U(0.8,1.3)``).

Plus :func:`synthesize` (name dispatch) and :func:`synthesize_clip` which produces
a temporally-coherent clip by slowly varying beta and either drifting the Perlin
field or advancing the turbulent advection between frames.

Perlin/fractal noise and the turbulent advection are implemented here in pure
torch (no extra deps) and are seedable/reproducible via a ``torch.Generator``.
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
# Turbulent source-point smoke (curl-noise advection)
#
# Surgical-desmoking literature (LSD3K arXiv:2407.13132, STANet arXiv:2512.02780)
# rejects multi-octave Perlin for smoke: real smoke emits from 1-3 point sources
# and evolves by turbulent advection outward. We grow a density field by advecting
# emitted density along a divergence-free curl-noise flow (curl of a low-frequency
# Perlin potential) on a coarse grid, then upsample + blur — cheap and turbulent.
# ---------------------------------------------------------------------------
def _gaussian_blur2d(field: torch.Tensor, ksize: int = 5, sigma: float = 1.5) -> torch.Tensor:
    """Separable Gaussian blur on an (N,1,H,W) tensor (reflection padded)."""
    coords = torch.arange(ksize, dtype=field.dtype, device=field.device) - (ksize - 1) / 2
    k1 = torch.exp(-(coords**2) / (2 * sigma * sigma))
    k1 = k1 / k1.sum()
    pad = ksize // 2
    x = F.pad(field, (pad, pad, 0, 0), mode="reflect")
    x = F.conv2d(x, k1.view(1, 1, 1, ksize))
    x = F.pad(x, (0, 0, pad, pad), mode="reflect")
    return F.conv2d(x, k1.view(1, 1, ksize, 1))


def _curl_flow(
    coarse: int, generator: torch.Generator | None, device, dtype, buoyancy: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """Divergence-free turbulent flow = curl of a low-frequency Perlin potential.

    For a 2D scalar potential psi, ``u = (dpsi/dy, -dpsi/dx)`` is divergence-free
    (no sources/sinks -> swirling, incompressible-looking motion). A constant
    upward bias models buoyant rise of hot smoke (image y grows downward, so we
    subtract it from ``vy``). Returned unit-scaled on a (coarse, coarse) grid.
    """
    pot = fractal_noise(
        coarse, coarse, octaves=3, base_res=2, generator=generator,
        device=device, dtype=dtype, normalize=False,
    )
    gy, gx = torch.gradient(pot)  # d/d row (y), d/d col (x)
    vx = gy
    vy = -gx - buoyancy
    mag = torch.sqrt(vx * vx + vy * vy).max().clamp_min(1e-6)
    return vx / mag, vy / mag


def _emission_field(
    coarse: int, n_sources: int, generator: torch.Generator | None, device, dtype, sigma: float
) -> torch.Tensor:
    """Sum of Gaussian source blobs, biased toward the lower half / vertical edges."""
    ys = torch.linspace(0, 1, coarse, device=device, dtype=dtype).view(coarse, 1)
    xs = torch.linspace(0, 1, coarse, device=device, dtype=dtype).view(1, coarse)
    field = torch.zeros(coarse, coarse, device=device, dtype=dtype)
    for _ in range(n_sources):
        cy = _u(0.55, 0.98, generator)  # lower half (smoke rises from the ground)
        cx = _u(0.0, 1.0, generator)
        if torch.rand(1, generator=generator).item() < 0.5:  # half the time hug an edge
            cx = cx * 0.25 if cx < 0.5 else 1.0 - (1.0 - cx) * 0.25
        dist2 = (ys - cy) ** 2 + (xs - cx) ** 2
        field = field + torch.exp(-dist2 / (2 * sigma * sigma))
    return field


class _SmokeAdvector:
    """Steppable coarse-grid smoke: emit at sources, advect along curl flow, decay.

    ``step(k)`` runs k semi-Lagrangian advection steps and returns the current
    coarse density (a copy). Density that has travelled more steps is more decayed,
    so plumes fade with distance from their source. Deterministic under ``generator``.
    """

    def __init__(
        self,
        coarse: int,
        generator: torch.Generator | None,
        device,
        dtype,
        n_sources: int,
        dt: float = 1.5,
        decay: float = 0.9,
        buoyancy: float = 0.4,
        emit_sigma: float = 0.09,
    ) -> None:
        self.coarse = coarse
        self.decay = decay
        self.vx, self.vy = _curl_flow(coarse, generator, device, dtype, buoyancy)
        self.emission = _emission_field(coarse, n_sources, generator, device, dtype, emit_sigma)
        self.dens = torch.zeros(coarse, coarse, device=device, dtype=dtype)
        ys = torch.linspace(-1, 1, coarse, device=device, dtype=dtype)
        xs = torch.linspace(-1, 1, coarse, device=device, dtype=dtype)
        gy0, gx0 = torch.meshgrid(ys, xs, indexing="ij")
        scale = 2.0 / max(coarse - 1, 1)  # cells -> normalized grid_sample units
        samp_x = gx0 - self.vx * dt * scale
        samp_y = gy0 - self.vy * dt * scale
        self._grid = torch.stack((samp_x, samp_y), dim=-1).unsqueeze(0)  # (1,C,C,2)

    def step(self, k: int = 1) -> torch.Tensor:
        for _ in range(max(1, k)):
            self.dens = self.dens + self.emission
            warped = F.grid_sample(
                self.dens.view(1, 1, self.coarse, self.coarse), self._grid,
                mode="bilinear", padding_mode="border", align_corners=True,
            )
            self.dens = warped.view(self.coarse, self.coarse) * self.decay
        return self.dens.clone()


def _finalize_density(coarse_dens: torch.Tensor, h: int, w: int) -> torch.Tensor:
    """Upsample a coarse density to (h, w), blur, min-max normalize to [0,1]."""
    up = F.interpolate(
        coarse_dens.view(1, 1, *coarse_dens.shape[-2:]), size=(h, w),
        mode="bilinear", align_corners=False,
    )
    up = _gaussian_blur2d(up).view(h, w)
    lo, hi = up.min(), up.max()
    return (up - lo) / (hi - lo + 1e-8)


def _match_moments(field: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """Rescale ``field`` to ``ref``'s mean/std then clamp to [0,1].

    Keeps the turbulent structure while matching the Perlin path's density-range
    statistics, so downstream ASM composition and meta (beta/airlight/sigma) stay
    identical in distribution regardless of ``smoke_mode``.
    """
    out = (field - field.mean()) / field.std().clamp_min(1e-6) * ref.std() + ref.mean()
    return out.clamp(0.0, 1.0)


def turbulent_density(
    height: int,
    width: int,
    generator: torch.Generator | None = None,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
    coarse: int = 48,
    steps: int = 16,
    n_sources: int | None = None,
) -> torch.Tensor:
    """Source-point turbulent smoke density in [0,1], shape (height, width).

    1-3 emission sources (lower-half / edge biased) grow a density field by
    curl-noise advection (divergence-free, buoyant) over ``steps`` steps on a
    coarse grid, then upsample + blur. Deterministic under ``generator``.
    """
    if n_sources is None:
        n_sources = int(torch.randint(1, 4, (1,), generator=generator).item())
    sim = _SmokeAdvector(coarse, generator, device, dtype, n_sources)
    sim.step(steps)
    return _finalize_density(sim.dens, height, width)


def _turbulent_clip_density(
    frames: int,
    height: int,
    width: int,
    generator: torch.Generator | None,
    device,
    dtype: torch.dtype = torch.float32,
    coarse: int = 48,
    burn_in: int = 12,
) -> list[torch.Tensor]:
    """A temporally-coherent list of ``frames`` turbulent density maps in [0,1].

    The advection state is burned in, then advanced one gentle step per frame; the
    whole clip is normalized with a single (global) min-max + moment-match so the
    plume evolves smoothly with no per-frame flicker.
    """
    n_sources = int(torch.randint(1, 4, (1,), generator=generator).item())
    sim = _SmokeAdvector(coarse, generator, device, dtype, n_sources)
    sim.step(burn_in)
    stack = torch.stack([sim.step(1) for _ in range(frames)])  # (T,C,C)
    ups = F.interpolate(stack.unsqueeze(1), size=(height, width), mode="bilinear", align_corners=False)
    ups = _gaussian_blur2d(ups)  # (T,1,H,W)
    lo, hi = ups.min(), ups.max()
    ups = (ups - lo) / (hi - lo + 1e-8)
    ref = fractal_noise(height, width, octaves=4, base_res=4, generator=generator, device=device)
    ups = ((ups - ups.mean()) / ups.std().clamp_min(1e-6) * ref.std() + ref.mean()).clamp(0.0, 1.0)
    return [ups[i, 0] for i in range(frames)]


def _resolve_smoke_mode(smoke_mode: str, generator: torch.Generator | None) -> str:
    """Resolve ``"mix"`` to ``"perlin"``/``"turbulent"`` 50/50 (deterministic)."""
    if smoke_mode == "mix":
        return "turbulent" if torch.rand(1, generator=generator).item() < 0.5 else "perlin"
    return smoke_mode


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _u(low: float, high: float, generator: torch.Generator | None) -> float:
    return float(low + (high - low) * torch.rand(1, generator=generator).item())


def _shot_noise(img: torch.Tensor, photons: float, generator: torch.Generator | None) -> torch.Tensor:
    """Shot (Poisson) noise in linear space; lower ``photons`` => noisier.

    ``torch.poisson`` takes no generator, so (matching pharos.data.degradations) we
    use a seeded Gaussian whose variance equals the Poisson rate — valid for
    moderate counts and reproducible under ``generator``.
    """
    photons = max(1.0, float(photons))
    rate = img.clamp_min(0.0) * photons
    noisy = rate + torch.randn(img.shape, generator=generator) * rate.sqrt()
    return noisy / photons


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
    isp_aware: bool = False,
    beta_bs: float | None = None,
    beta_bs_ratio: tuple[float, float] = (0.8, 1.3),
    isp_gamma: float = 2.2,
    shot_photons: tuple[float, float] = (40.0, 300.0),
) -> tuple[torch.Tensor, Params]:
    """Depth-driven atmospheric scattering: I = J*t_att + A*(1-t_bs).

    beta ~ U[0.4, 3.0] (unless given); near-gray airlight jitter (unless given).
    ``t_att = exp(-beta*d)`` attenuates the scene, ``t_bs = exp(-beta_bs*d)`` builds
    up airlight/backscatter (Sea-thru split); they coincide unless ``isp_aware``.

    ``isp_aware`` (SynFog, CVPR'24): invert the ISP (sRGB->linear via gamma), apply
    ASM + Poisson shot noise in linear space (noise dominates low-contrast/foggy
    regions), then re-apply the forward ISP gamma. ``isp_aware=False`` is the exact
    legacy post-ISP ASM.
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
    if beta_bs is None:
        beta_bs = beta * _u(*beta_bs_ratio, generator) if isp_aware else beta

    t = torch.exp(-beta * d)         # attenuation
    t_bs = torch.exp(-beta_bs * d)   # airlight / backscatter build-up
    if isp_aware:
        gamma = float(isp_gamma)
        clean_lin = clean.clamp(0, 1) ** gamma
        air_lin = airlight.clamp(0, 1) ** gamma
        hazy_lin = clean_lin * t + air_lin * (1 - t_bs)
        photons = _u(*shot_photons, generator)
        hazy_lin = _shot_noise(hazy_lin, photons, generator)
        hazy = hazy_lin.clamp(0, 1) ** (1.0 / gamma)
    else:
        hazy = clean * t + airlight * (1 - t_bs)  # t_bs == t -> exact legacy ASM
    hazy = hazy.clamp(0, 1)
    params = {
        "beta": float(beta),
        "beta_bs": float(beta_bs),
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
    smoke_mode: str = "mix",
    beta_bs: float | None = None,
    beta_bs_ratio: tuple[float, float] = (0.8, 1.3),
) -> tuple[torch.Tensor, Params]:
    """Non-homogeneous smoke (NOT depth-driven), colored soot/warm airlight, optional
    additive fire-glow in a random corner.

    ``smoke_mode``: ``"perlin"`` (legacy multi-octave fBm density), ``"turbulent"``
    (source-point curl-noise advection, moment-matched to the Perlin density stats),
    or ``"mix"`` (50/50 per sample). Attenuation uses ``beta``; airlight build-up
    uses ``beta_bs`` (== beta in perlin mode, ``beta·U(0.8,1.3)`` in turbulent mode).
    """
    c, h, w = clean.shape
    device = clean.device

    if beta is None:
        beta = _u(1.0, 4.0, generator)

    mode: str | None = None
    if density is None:
        mode = _resolve_smoke_mode(smoke_mode, generator)
        if mode == "turbulent":
            turb = turbulent_density(h, w, generator=generator, device=device)
            ref = fractal_noise(h, w, octaves=4, base_res=4, generator=generator, device=device)
            density = _match_moments(turb, ref)
        else:
            octaves = int(torch.randint(3, 6, (1,), generator=generator).item())
            density = fractal_noise(h, w, octaves=octaves, base_res=4, generator=generator, device=device)
    density = density.to(device).view(1, h, w)

    if airlight is None:
        idx = int(torch.randint(0, len(_SMOKE_TINTS), (1,), generator=generator).item())
        tint = torch.tensor(_SMOKE_TINTS[idx])
        tint = (tint + (torch.rand(3, generator=generator) - 0.5) * 0.08).clamp(0.2, 0.95)
        airlight = tint
    airlight = airlight.to(device).view(3, 1, 1)

    if beta_bs is None:
        if mode is None:
            mode = _resolve_smoke_mode(smoke_mode, generator)
        beta_bs = beta * _u(*beta_bs_ratio, generator) if mode == "turbulent" else beta

    t = torch.exp(-beta * density)         # attenuation
    t_bs = torch.exp(-beta_bs * density)   # airlight / backscatter build-up
    hazy = clean * t + airlight * (1 - t_bs)

    if fire_glow is None:
        fire_glow = torch.rand(1, generator=generator).item() < 0.35
    if fire_glow:
        hazy = _add_fire_glow(hazy, generator)

    hazy = hazy.clamp(0, 1)
    sigma = float(density.std().item())  # spatial non-homogeneity measure
    params = {
        "beta": float(beta),
        "beta_bs": float(beta_bs),
        "airlight": airlight.view(3).cpu(),
        "sigma": sigma,
        "domain": DOMAIN_SMOKE,
        "smoke_mode": mode if mode is not None else smoke_mode,
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
        "beta_bs": float(beta),  # satellite is near-uniform: no attenuation/backscatter split
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

# kwargs each generator accepts — lets callers pass a superset (e.g. the whole
# ``synthesis`` config block) without leaking irrelevant keys to a generator.
_ALLOWED_KWARGS = {
    ground_haze: {
        "beta", "airlight", "isp_aware", "beta_bs", "beta_bs_ratio", "isp_gamma", "shot_photons",
    },
    smoke: {"beta", "airlight", "density", "fire_glow", "smoke_mode", "beta_bs", "beta_bs_ratio"},
    satellite: {"beta", "airlight", "beta_bs"},
}


def synthesize(
    clean: torch.Tensor,
    domain: str | int,
    generator: torch.Generator | None = None,
    depth: torch.Tensor | None = None,
    **kwargs: Any,
) -> tuple[torch.Tensor, Params]:
    """Dispatch to a generator by domain name ('haze'/'smoke'/'satellite') or id.

    Only kwargs the target generator accepts (and that are not None) are forwarded,
    so a config dict carrying every knob can be splatted in for any domain.
    """
    fn = _GENERATORS.get(domain)
    if fn is None:
        raise ValueError(f"unknown synthesis domain: {domain!r}")
    kw = {k: v for k, v in kwargs.items() if k in _ALLOWED_KWARGS[fn] and v is not None}
    if fn is ground_haze:
        return ground_haze(clean, depth=depth, generator=generator, **kw)
    return fn(clean, generator=generator, **kw)


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
    smoke_mode: str = "mix",
    isp_aware: bool = False,
    beta_bs_ratio: tuple[float, float] = (0.8, 1.3),
) -> tuple[torch.Tensor, Params]:
    """Temporally-coherent synthesis over a clip.

    ``clean_frames`` is (T,3,H,W). beta varies smoothly across frames; for smoke the
    density either drifts (perlin) or advances the turbulent advection one gentle
    step per frame (temporal coherence) — so adjacent degraded frames stay close.
    Returns (hazy_clip (T,3,H,W), params) recording the mean beta/beta_bs and airlight.
    """
    assert clean_frames.dim() == 4, "expected (T,3,H,W)"
    tt, c, h, w = clean_frames.shape
    device = clean_frames.device

    dom = domain
    base_density: torch.Tensor | None = None
    smoke_frames: list[torch.Tensor] | None = None  # per-frame turbulent density
    if dom in ("haze", DOMAIN_HAZE):
        base_beta = _u(0.4, 3.0, generator)
    elif dom in ("smoke", DOMAIN_SMOKE):
        base_beta = _u(1.0, 4.0, generator)
        mode = _resolve_smoke_mode(smoke_mode, generator)
        if mode == "turbulent":
            smoke_frames = _turbulent_clip_density(tt, h, w, generator, device)
        else:
            octaves = int(torch.randint(3, 6, (1,), generator=generator).item())
            base_density = fractal_noise(
                h, w, octaves=octaves, base_res=4, generator=generator, device=device
            )
    else:
        base_beta = _u(0.2, 1.2, generator)

    # backscatter coeff tracks beta by a fixed clip-level ratio (== 1 in legacy paths)
    split = (dom in ("smoke", DOMAIN_SMOKE) and smoke_frames is not None) or (
        dom in ("haze", DOMAIN_HAZE) and isp_aware
    )
    bs_ratio = _u(*beta_bs_ratio, generator) if split else 1.0

    phase = _u(0.0, 2 * math.pi, generator)
    out = torch.empty_like(clean_frames)
    params: Params = {}
    fixed_air: torch.Tensor | None = None
    for i in range(tt):
        # smooth beta variation (sinusoid) shared across the clip
        beta_i = max(0.05, base_beta * (1.0 + beta_jitter * math.sin(phase + i * 0.6)))
        beta_bs_i = beta_i * bs_ratio
        if dom in ("smoke", DOMAIN_SMOKE):
            dens_i = (
                smoke_frames[i] if smoke_frames is not None
                else _drift_field(base_density.view(1, h, w), i * drift_px, i * drift_px * 0.5)
            )
            frame, p = smoke(
                clean_frames[i], generator=generator, beta=beta_i, airlight=fixed_air,
                density=dens_i, fire_glow=False, beta_bs=beta_bs_i,
            )
        elif dom in ("haze", DOMAIN_HAZE):
            frame, p = ground_haze(
                clean_frames[i], depth=depth, generator=generator, beta=beta_i,
                airlight=fixed_air, isp_aware=isp_aware, beta_bs=beta_bs_i,
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
    params["beta_bs"] = float(base_beta * bs_ratio)
    return out, params
