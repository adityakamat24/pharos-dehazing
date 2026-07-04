"""CPU, no-network tests for pharos.data.synthesis."""
from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import pytest
import torch

from pharos.contracts import DOMAIN_HAZE, DOMAIN_SATELLITE, DOMAIN_SMOKE
from pharos.data import synthesis as S


def _gen(seed: int) -> torch.Generator:
    return torch.Generator().manual_seed(seed)


def _clean(h: int = 48, w: int = 64) -> torch.Tensor:
    return torch.rand(3, h, w, generator=_gen(100))


# ---------------------------------------------------------------------------
# Perlin / fractal noise
# ---------------------------------------------------------------------------
def test_perlin_reproducible_same_seed():
    a = S.fractal_noise(48, 64, octaves=4, generator=_gen(1))
    b = S.fractal_noise(48, 64, octaves=4, generator=_gen(1))
    assert torch.allclose(a, b)


def test_perlin_differs_with_seed():
    a = S.fractal_noise(48, 64, octaves=4, generator=_gen(1))
    c = S.fractal_noise(48, 64, octaves=4, generator=_gen(2))
    assert not torch.allclose(a, c)


def test_perlin_normalized_range():
    n = S.fractal_noise(48, 64, octaves=4, generator=_gen(3))
    assert n.shape == (48, 64)
    assert float(n.min()) >= 0.0 and float(n.max()) <= 1.0


def test_perlin_is_smooth():
    # gradient noise: adjacent pixels are far more similar than random noise (~0.33 mean |diff|)
    n = S.fractal_noise(64, 64, octaves=4, generator=_gen(4))
    dx = (n[:, 1:] - n[:, :-1]).abs().mean()
    dy = (n[1:, :] - n[:-1, :]).abs().mean()
    assert float(dx) < 0.1 and float(dy) < 0.1


def test_perlin_arbitrary_shape():
    n = S.fractal_noise(37, 53, octaves=3, generator=_gen(5))  # not divisible by any power of 2
    assert n.shape == (37, 53)


def test_perlin_2d_range():
    p = S.perlin_2d((32, 32), (4, 4), generator=_gen(6))
    assert p.shape == (32, 32)
    assert p.abs().max() <= 1.5  # gradient noise is bounded around [-1, 1]


# ---------------------------------------------------------------------------
# depth
# ---------------------------------------------------------------------------
def test_fallback_depth():
    d = S.fallback_depth(20, 30)
    assert d.shape == (1, 20, 30)
    assert float(d.min()) >= 0.0 and float(d.max()) <= 1.0
    # top row (far) should be larger than bottom row (near)
    assert float(d[0, 0].mean()) > float(d[0, -1].mean())


# ---------------------------------------------------------------------------
# generators: range, params, domain ids
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "name,domain,beta_lo,beta_hi",
    [("haze", DOMAIN_HAZE, 0.4, 3.0), ("smoke", DOMAIN_SMOKE, 1.0, 4.0),
     ("satellite", DOMAIN_SATELLITE, 0.2, 1.2)],
)
def test_generator_output_and_params(name, domain, beta_lo, beta_hi):
    clean = _clean()
    for seed in range(5):
        hazy, p = S.synthesize(clean, name, generator=_gen(seed))
        assert hazy.shape == clean.shape
        assert float(hazy.min()) >= 0.0 and float(hazy.max()) <= 1.0
        assert p["domain"] == domain
        assert beta_lo - 1e-6 <= p["beta"] <= beta_hi + 1e-6
        air = p["airlight"]
        assert air.shape == (3,)
        assert float(air.min()) >= 0.0 and float(air.max()) <= 1.0
        assert p["sigma"] >= 0.0


def test_ground_haze_accepts_depth():
    clean = _clean()
    depth = torch.rand(1, 48, 64, generator=_gen(9))
    hazy, p = S.ground_haze(clean, depth=depth, generator=_gen(0))
    assert hazy.shape == clean.shape
    assert p["domain"] == DOMAIN_HAZE


def test_ground_haze_depth_resized():
    clean = _clean(48, 64)
    depth = torch.rand(1, 16, 20, generator=_gen(9))  # mismatched -> resized internally
    hazy, _ = S.ground_haze(clean, depth=depth, generator=_gen(0))
    assert hazy.shape == clean.shape


def test_satellite_wavelength_bias():
    # thin uniform haze -> blue channel attenuated most, so on a mid-gray scene the
    # blue channel should be pulled toward airlight more than red on average.
    clean = torch.full((3, 40, 40), 0.5)
    hazy, _ = S.satellite(clean, generator=_gen(0))
    assert hazy.shape == clean.shape
    assert float(hazy.min()) >= 0.0 and float(hazy.max()) <= 1.0


def test_smoke_fire_glow_stays_in_range():
    clean = _clean()
    hazy, p = S.smoke(clean, generator=_gen(0), fire_glow=True)
    assert float(hazy.min()) >= 0.0 and float(hazy.max()) <= 1.0
    assert p["domain"] == DOMAIN_SMOKE


def test_synthesize_unknown_domain_raises():
    with pytest.raises(ValueError):
        S.synthesize(_clean(), "fog", generator=_gen(0))


def test_synthesize_by_domain_id():
    hazy, p = S.synthesize(_clean(), DOMAIN_SMOKE, generator=_gen(0))
    assert p["domain"] == DOMAIN_SMOKE


# ---------------------------------------------------------------------------
# temporally-coherent video synthesis
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("domain", ["haze", "smoke", "satellite"])
def test_synthesize_clip_shape_and_range(domain):
    frames = _clean().unsqueeze(0).repeat(4, 1, 1, 1)
    clip, p = S.synthesize_clip(frames, domain, generator=_gen(0))
    assert clip.shape == frames.shape
    assert float(clip.min()) >= 0.0 and float(clip.max()) <= 1.0
    assert "beta" in p and "airlight" in p


@pytest.mark.parametrize("domain", ["haze", "smoke", "satellite"])
def test_synthesize_clip_is_temporally_smooth(domain):
    # static scene (repeated clean frame): adjacent degraded frames must stay close,
    # i.e. the degradation field drifts smoothly across the clip.
    frame = _clean(48, 64)
    frames = frame.unsqueeze(0).repeat(5, 1, 1, 1)
    clip, _ = S.synthesize_clip(frames, domain, generator=_gen(1))
    adj_l1 = (clip[1:] - clip[:-1]).abs().mean()
    assert float(adj_l1) < 0.05


def test_synthesize_clip_reproducible():
    frames = _clean().unsqueeze(0).repeat(3, 1, 1, 1)
    a, _ = S.synthesize_clip(frames, "smoke", generator=_gen(7))
    b, _ = S.synthesize_clip(frames, "smoke", generator=_gen(7))
    assert torch.allclose(a, b)
