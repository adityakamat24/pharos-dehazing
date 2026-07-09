"""CPU tests for the flow teacher and flow_warp utility (no network)."""
from __future__ import annotations

import torch

from pharos.teachers.flow import FlowTeacher, flow_warp


def test_flow_warp_identity_returns_input():
    img = torch.rand(2, 3, 16, 24)
    flow = torch.zeros(2, 2, 16, 24)
    warped = flow_warp(img, flow)
    assert warped.shape == img.shape
    assert torch.allclose(warped, img, atol=1e-5)


def test_flow_warp_constant_shift():
    # a +1px shift in x should move content by one column (border padding at edge)
    img = torch.zeros(1, 1, 4, 4)
    img[0, 0, :, 2] = 1.0
    flow = torch.zeros(1, 2, 4, 4)
    flow[0, 0] = 1.0  # sample one pixel to the right
    warped = flow_warp(img, flow)
    assert warped.shape == img.shape
    assert warped[0, 0, :, 1].mean() > 0.5  # column 2 pulled into column 1


def test_flow_teacher_unavailable_returns_zeros():
    t = FlowTeacher(device="cpu")
    t.available = False  # simulate missing weights without downloading
    a = torch.rand(2, 3, 32, 32)
    b = torch.rand(2, 3, 32, 32)
    out = t(a, b)
    assert out.shape == (2, 2, 32, 32)
    assert torch.count_nonzero(out) == 0
