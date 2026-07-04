"""Robustness augmentations applied to the *degraded input only* (never the GT).

Pipeline-sweep findings baked in: JPEG compression, H.264-like 8x8 DCT blockiness,
gaussian/poisson ISO noise, and exposure / white-balance jitter. Each op takes and
returns a float32 CHW tensor in [0,1]; randomness flows through a ``torch.Generator``.
:class:`RobustnessPipeline` composes them with per-op probabilities.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
import torch

from .transforms import to_numpy_u8, to_tensor


def _rand(low: float, high: float, generator: torch.Generator | None) -> float:
    return float(low + (high - low) * torch.rand(1, generator=generator).item())


# ---------------------------------------------------------------------------
# JPEG
# ---------------------------------------------------------------------------
def jpeg_compress(img: torch.Tensor, quality: int) -> torch.Tensor:
    """Round-trip through JPEG at the given quality (30-95 typical) via cv2."""
    quality = int(max(1, min(100, quality)))
    rgb = to_numpy_u8(img)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        return img.clamp(0, 1)
    dec = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    rgb2 = cv2.cvtColor(dec, cv2.COLOR_BGR2RGB)
    return to_tensor(rgb2)


# ---------------------------------------------------------------------------
# H.264-like blockiness via 8x8 DCT quantization of the luma channel
# ---------------------------------------------------------------------------
def _dct_matrix(n: int = 8) -> np.ndarray:
    k = np.arange(n)
    m = np.sqrt(2.0 / n) * np.cos(np.pi * (2 * k[None, :] + 1) * k[:, None] / (2 * n))
    m[0, :] = np.sqrt(1.0 / n)
    return m.astype(np.float32)


_DCT8 = _dct_matrix(8)


def dct_blockiness(img: torch.Tensor, q: float = 16.0) -> torch.Tensor:
    """Quantize 8x8 DCT coefficients of the luma channel (H.264-ish blockiness).

    Larger ``q`` (quant step) => coarser blocks. Chroma is left untouched.
    """
    rgb = to_numpy_u8(img).astype(np.float32)
    ycrcb = cv2.cvtColor(rgb, cv2.COLOR_RGB2YCrCb)
    y = ycrcb[:, :, 0]
    h, w = y.shape
    ph = (8 - h % 8) % 8
    pw = (8 - w % 8) % 8
    yp = np.pad(y, ((0, ph), (0, pw)), mode="edge")
    hh, ww = yp.shape
    # (nby, 8, nbx, 8) -> (nby, nbx, 8, 8)
    blocks = yp.reshape(hh // 8, 8, ww // 8, 8).transpose(0, 2, 1, 3)
    coeff = np.einsum("ij,abjk,kl->abil", _DCT8, blocks, _DCT8.T)
    coeff = np.round(coeff / q) * q
    rec = np.einsum("ij,abjk,kl->abil", _DCT8.T, coeff, _DCT8)
    yp2 = rec.transpose(0, 2, 1, 3).reshape(hh, ww)
    ycrcb[:, :, 0] = np.clip(yp2[:h, :w], 0, 255)
    out = cv2.cvtColor(ycrcb.astype(np.uint8), cv2.COLOR_YCrCb2RGB)
    return to_tensor(out)


# ---------------------------------------------------------------------------
# ISO noise
# ---------------------------------------------------------------------------
def gaussian_noise(img: torch.Tensor, sigma: float, generator: torch.Generator | None = None) -> torch.Tensor:
    """Additive white gaussian noise; ``sigma`` in [0,1] image units."""
    noise = torch.randn(img.shape, generator=generator) * sigma
    return (img + noise).clamp(0, 1)


def poisson_noise(
    img: torch.Tensor, scale: float = 30.0, generator: torch.Generator | None = None
) -> torch.Tensor:
    """Shot (poisson) noise. Lower ``scale`` => noisier (fewer photons)."""
    scale = max(1.0, scale)
    rate = (img.clamp(0, 1) * scale)
    # torch.poisson has no generator arg; approximate with a seeded gaussian whose
    # variance equals the poisson rate (valid for moderate counts) for reproducibility.
    noisy = rate + torch.randn(img.shape, generator=generator) * rate.clamp(min=0).sqrt()
    return (noisy / scale).clamp(0, 1)


def iso_noise(
    img: torch.Tensor,
    sigma: float = 0.02,
    poisson_scale: float = 60.0,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Combined read (gaussian) + shot (poisson) noise, mimicking sensor ISO gain."""
    out = poisson_noise(img, poisson_scale, generator)
    return gaussian_noise(out, sigma, generator)


# ---------------------------------------------------------------------------
# Photometric jitter
# ---------------------------------------------------------------------------
def exposure_jitter(img: torch.Tensor, gain: float) -> torch.Tensor:
    """Multiplicative exposure change."""
    return (img * gain).clamp(0, 1)


def white_balance_jitter(img: torch.Tensor, gains: torch.Tensor) -> torch.Tensor:
    """Per-channel gains (3,) for white-balance shift."""
    return (img * gains.view(3, 1, 1)).clamp(0, 1)


# ---------------------------------------------------------------------------
# Composed pipeline
# ---------------------------------------------------------------------------
@dataclass
class RobustnessPipeline:
    """Randomized composition of the robustness augmentations.

    Each op fires independently with its probability. Ranges follow the design's
    pipeline sweep (JPEG QF 30-95, etc.). Call as ``pipeline(img, generator)``.
    """

    jpeg_p: float = 0.5
    jpeg_quality: tuple[int, int] = (30, 95)
    block_p: float = 0.25
    block_q: tuple[float, float] = (8.0, 32.0)
    noise_p: float = 0.4
    noise_sigma: tuple[float, float] = (0.005, 0.04)
    poisson_scale: tuple[float, float] = (40.0, 120.0)
    exposure_p: float = 0.4
    exposure_gain: tuple[float, float] = (0.7, 1.3)
    wb_p: float = 0.3
    wb_gain: tuple[float, float] = (0.9, 1.1)

    def _fire(self, p: float, generator: torch.Generator | None) -> bool:
        return torch.rand(1, generator=generator).item() < p

    def __call__(self, img: torch.Tensor, generator: torch.Generator | None = None) -> torch.Tensor:
        out = img.clamp(0, 1)
        if self._fire(self.exposure_p, generator):
            out = exposure_jitter(out, _rand(*self.exposure_gain, generator))
        if self._fire(self.wb_p, generator):
            gains = torch.tensor([_rand(*self.wb_gain, generator) for _ in range(3)])
            out = white_balance_jitter(out, gains)
        if self._fire(self.noise_p, generator):
            out = iso_noise(
                out,
                sigma=_rand(*self.noise_sigma, generator),
                poisson_scale=_rand(*self.poisson_scale, generator),
                generator=generator,
            )
        if self._fire(self.block_p, generator):
            out = dct_blockiness(out, _rand(*self.block_q, generator))
        if self._fire(self.jpeg_p, generator):
            lo, hi = self.jpeg_quality
            out = jpeg_compress(out, int(_rand(lo, hi, generator)))
        return out.clamp(0, 1)
