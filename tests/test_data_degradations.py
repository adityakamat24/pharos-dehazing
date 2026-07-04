"""CPU, no-network tests for pharos.data.degradations."""
from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import pytest
import torch

from pharos.data import degradations as D


def _gen(seed: int) -> torch.Generator:
    return torch.Generator().manual_seed(seed)


def _img(h: int = 40, w: int = 56) -> torch.Tensor:
    return torch.rand(3, h, w, generator=_gen(0))


def _check(out: torch.Tensor, ref: torch.Tensor) -> None:
    assert out.shape == ref.shape
    assert out.dtype == torch.float32
    assert float(out.min()) >= 0.0 and float(out.max()) <= 1.0


def test_jpeg_preserves_shape_range():
    img = _img()
    for q in (30, 60, 95):
        _check(D.jpeg_compress(img, q), img)


def test_dct_blockiness_preserves_shape_range():
    img = _img(37, 45)  # non-multiple of 8 to exercise padding
    for q in (8.0, 16.0, 32.0):
        _check(D.dct_blockiness(img, q), img)


def test_gaussian_noise():
    img = _img()
    _check(D.gaussian_noise(img, 0.05, _gen(1)), img)


def test_poisson_noise():
    img = _img()
    _check(D.poisson_noise(img, 40.0, _gen(1)), img)


def test_iso_noise():
    img = _img()
    _check(D.iso_noise(img, 0.02, 60.0, _gen(1)), img)


def test_exposure_jitter():
    img = _img()
    _check(D.exposure_jitter(img, 1.3), img)
    _check(D.exposure_jitter(img, 0.7), img)


def test_white_balance_jitter():
    img = _img()
    gains = torch.tensor([1.1, 1.0, 0.9])
    _check(D.white_balance_jitter(img, gains), img)


def test_pipeline_preserves_shape_range():
    img = _img()
    pipe = D.RobustnessPipeline()
    for seed in range(6):
        _check(pipe(img, _gen(seed)), img)


def test_pipeline_reproducible():
    img = _img()
    pipe = D.RobustnessPipeline()
    a = pipe(img, _gen(3))
    b = pipe(img, _gen(3))
    assert torch.allclose(a, b)


def test_pipeline_all_on_changes_image():
    img = _img()
    # force every op on so the output definitely differs from the input
    pipe = D.RobustnessPipeline(
        jpeg_p=1.0, block_p=1.0, noise_p=1.0, exposure_p=1.0, wb_p=1.0,
    )
    out = pipe(img, _gen(0))
    assert out.shape == img.shape
    assert not torch.allclose(out, img)


def test_noise_actually_perturbs():
    img = torch.full((3, 32, 32), 0.5)
    out = D.gaussian_noise(img, 0.1, _gen(0))
    assert float((out - img).abs().mean()) > 0.0
