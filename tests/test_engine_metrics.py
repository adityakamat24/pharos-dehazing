"""Tests for pharos.engine.metrics (psnr / ssim / warp_error)."""
from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch  # noqa: E402

from pharos.engine import metrics as M  # noqa: E402


def test_psnr_identical_is_inf():
    x = torch.rand(2, 3, 16, 16)
    assert math.isinf(M.psnr(x, x))


def test_ssim_identical_is_one():
    x = torch.rand(2, 3, 32, 32)
    assert abs(M.ssim(x, x) - 1.0) < 1e-4


def test_psnr_known_noise():
    torch.manual_seed(0)
    x = torch.rand(1, 3, 256, 256)
    std = 0.1
    noisy = x + std * torch.randn_like(x)
    # PSNR of additive noise with std s (max_val 1) ~ -20*log10(s) = 20 dB.
    val = M.psnr(x, noisy)
    assert abs(val - 20.0) < 1.5, val


def test_ssim_noise_between_zero_and_one():
    torch.manual_seed(1)
    x = torch.rand(1, 3, 64, 64)
    noisy = (x + 0.2 * torch.randn_like(x)).clamp(0, 1)
    s = M.ssim(x, noisy)
    assert 0.0 < s < 1.0


def test_warp_error_zero_flow_equals_frame_diff():
    torch.manual_seed(2)
    prev = torch.rand(1, 3, 32, 32)
    curr = torch.rand(1, 3, 32, 32)
    flow = torch.zeros(1, 2, 32, 32)
    we = M.warp_error(prev, curr, flow)
    fd = M.frame_diff(prev, curr)
    assert abs(we - fd) < 1e-4


def test_warp_error_none_flow_uses_proxy():
    x = torch.rand(1, 3, 16, 16)
    assert M.warp_error(x, x, None) == 0.0
