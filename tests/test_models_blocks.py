"""Tests for pharos.models.blocks (WS-A)."""
import pathlib
import sys

import pytest
import torch

# src layout, package not installed: make `pharos` importable standalone.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from pharos.models.blocks import (  # noqa: E402
    FiLM,
    HaarDownsample,
    LayerNorm2d,
    RepConv,
    RepNAFBlock,
    SimpleGate,
)


def _train_bn(module: torch.nn.Module, shape: tuple[int, ...], steps: int = 4) -> None:
    module.train()
    for _ in range(steps):
        module(torch.rand(*shape))
    module.eval()


@pytest.mark.parametrize(
    "in_ch,out_ch,groups",
    [(16, 16, 1), (8, 16, 1), (16, 16, 16), (12, 24, 1)],  # identity/expand, full/depthwise
)
def test_repconv_reparam_equivalence(in_ch, out_ch, groups):
    torch.manual_seed(0)
    conv = RepConv(in_ch, out_ch, kernel=3, groups=groups, use_bn=True)
    _train_bn(conv, (4, in_ch, 12, 12))
    x = torch.rand(2, in_ch, 12, 12)
    y1 = conv(x)
    conv.reparameterize()
    y2 = conv(x)
    assert conv.deployed
    assert torch.allclose(y1, y2, atol=1e-4), float((y1 - y2).abs().max())
    # idempotent
    conv.reparameterize()
    assert torch.allclose(conv(x), y2, atol=1e-6)


def test_repconv_identity_branch_presence():
    assert RepConv(16, 16).has_identity is True
    assert RepConv(8, 16).has_identity is False
    assert RepConv(16, 16, stride=2).has_identity is False


def test_repnafblock_reparam_equivalence():
    torch.manual_seed(1)
    blk = RepNAFBlock(24)
    _train_bn(blk, (4, 24, 16, 16))
    x = torch.rand(2, 24, 16, 16)
    y1 = blk(x)
    blk.reparameterize()
    y2 = blk(x)
    assert torch.allclose(y1, y2, atol=1e-4), float((y1 - y2).abs().max())


def test_repnafblock_shape_preserved():
    blk = RepNAFBlock(48).eval()
    x = torch.rand(2, 48, 20, 24)
    assert blk(x).shape == x.shape


def test_haar_downsample_halves_and_handles_odd():
    down = HaarDownsample(8, 16).eval()
    y = down(torch.rand(2, 8, 32, 32))
    assert y.shape == (2, 16, 16, 16)
    # odd spatial size is padded internally to even
    y_odd = down(torch.rand(2, 8, 31, 29))
    assert y_odd.shape == (2, 16, 16, 15)


def test_haar_filters_orthonormal_energy_preserving():
    # For a per-channel constant input only the LL band is nonzero.
    down = HaarDownsample(1, 4)
    x = torch.ones(1, 1, 4, 4)
    sub = torch.nn.functional.conv2d(x, down.haar, stride=2, groups=1)
    ll, highs = sub[:, 0], sub[:, 1:]
    assert torch.allclose(highs, torch.zeros_like(highs), atol=1e-6)
    assert torch.allclose(ll, torch.full_like(ll, 2.0), atol=1e-6)  # 0.5*(1+1+1+1)


def test_film_zero_init_is_identity():
    film = FiLM(16, 10).eval()
    x = torch.rand(2, 16, 8, 8)
    cond = torch.rand(2, 10)
    assert torch.allclose(film(x, cond), x, atol=1e-6)


def test_simplegate_and_layernorm():
    sg = SimpleGate()
    x = torch.rand(2, 8, 4, 4)
    assert sg(x).shape == (2, 4, 4, 4)
    ln = LayerNorm2d(8)
    with torch.no_grad():
        y = ln(torch.rand(2, 8, 5, 5))
    assert y.shape == (2, 8, 5, 5)
    # normalized: near zero mean across channels at init (weight=1, bias=0)
    assert float(y.mean(1).abs().mean()) < 1e-4
