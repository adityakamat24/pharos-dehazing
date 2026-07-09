"""Tests for pharos.models.grid (WS-A)."""
import pathlib
import sys

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from pharos.models.grid import (  # noqa: E402
    BilateralGridHead,
    GuidanceNet,
    apply_affine,
    slice_grid,
)


def test_grid_head_shape_and_identity_init():
    head = BilateralGridHead(32, depth=8, size=16).eval()
    g = head(torch.rand(2, 32, 12, 12))
    assert g.shape == (2, 12, 8, 16, 16)
    # zero-init to_grid weight -> constant identity affine everywhere.
    m, b = slice_grid(g, torch.rand(2, 1, 20, 24))
    m = m.view(2, 3, 3, 20, 24)
    eye = torch.eye(3).view(1, 3, 3, 1, 1)
    assert torch.allclose(m, eye.expand_as(m), atol=1e-5)
    assert torch.allclose(b, torch.zeros_like(b), atol=1e-5)


def test_slice_grid_shapes():
    g = torch.rand(3, 12, 8, 16, 16)
    guidance = torch.rand(3, 1, 40, 50)
    m, b = slice_grid(g, guidance)
    assert m.shape == (3, 9, 40, 50)
    assert b.shape == (3, 3, 40, 50)


def test_apply_affine_identity_and_correctness():
    img = torch.rand(2, 3, 8, 9)
    # identity affine returns the input
    m = torch.eye(3).view(1, 9, 1, 1).expand(2, 9, 8, 9).contiguous()
    b = torch.zeros(2, 3, 8, 9)
    assert torch.allclose(apply_affine(img, m, b), img, atol=1e-6)
    # constant scale + shift
    m2 = (torch.eye(3) * 2.0).view(1, 9, 1, 1).expand(2, 9, 8, 9).contiguous()
    b2 = torch.full((2, 3, 8, 9), 0.1)
    assert torch.allclose(apply_affine(img, m2, b2), img * 2.0 + 0.1, atol=1e-6)


def test_slice_grid_differentiable():
    g = torch.rand(1, 12, 8, 16, 16, requires_grad=True)
    guidance = torch.rand(1, 1, 24, 24, requires_grad=True)
    m, b = slice_grid(g, guidance)
    (m.sum() + b.sum()).backward()
    assert g.grad is not None and torch.isfinite(g.grad).all()
    assert guidance.grad is not None and torch.isfinite(guidance.grad).all()


def test_guidance_range():
    gn = GuidanceNet().eval()
    with torch.no_grad():
        out = gn(torch.rand(2, 3, 32, 40))
    assert out.shape == (2, 1, 32, 40)
    assert float(out.min()) >= 0.0 and float(out.max()) <= 1.0
