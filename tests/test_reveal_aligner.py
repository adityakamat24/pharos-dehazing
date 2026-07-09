"""Tests for the RevealNet tiered aligner (WS-v2A, DESIGN.md §9d.1)."""
import pathlib
import sys

import torch
import torch.nn.functional as F

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pharos.models.reveal.aligner import (  # noqa: E402
    TieredAligner,
    four_point_to_homography,
    invert_homography,
    warp_grid,
)


def _smooth_image(h: int, w: int) -> torch.Tensor:
    yy, xx = torch.meshgrid(torch.linspace(0, 1, h), torch.linspace(0, 1, w), indexing="ij")
    return torch.stack([xx, yy, 0.5 * (xx + yy)], dim=0).unsqueeze(0)


def test_zero_offset_is_identity_homography():
    h = four_point_to_homography(torch.zeros(2, 4, 2))
    eye = torch.eye(3).expand(2, 3, 3)
    assert torch.allclose(h, eye, atol=1e-5)


def test_identity_warp_grid_is_a_noop():
    img = torch.rand(1, 3, 10, 12)
    grid = warp_grid(torch.eye(3).unsqueeze(0), (10, 12))
    out = F.grid_sample(img, grid, mode="bilinear", padding_mode="border", align_corners=True)
    assert torch.allclose(out, img, atol=1e-4)


def test_homography_warp_round_trip_accuracy():
    # Warp a smooth image by H then by its inverse; interior must be recovered.
    img = _smooth_image(60, 72)
    for scale in (0.03, 0.05, 0.08):
        torch.manual_seed(3)
        off = (torch.rand(1, 4, 2) - 0.5) * scale
        h = four_point_to_homography(off)
        fwd = warp_grid(invert_homography(h), (60, 72))  # output->input needs inverse
        warped = F.grid_sample(img, fwd, mode="bilinear", padding_mode="border", align_corners=True)
        bwd = warp_grid(h, (60, 72))
        back = F.grid_sample(warped, bwd, mode="bilinear", padding_mode="border", align_corners=True)
        c = back[..., 8:-8, 8:-8], img[..., 8:-8, 8:-8]
        assert (c[0] - c[1]).abs().mean() < 1e-3, scale


def test_invert_homography_is_a_left_inverse():
    torch.manual_seed(0)
    h = four_point_to_homography((torch.rand(3, 4, 2) - 0.5) * 0.1)
    prod = invert_homography(h) @ h
    eye = torch.eye(3).expand(3, 3, 3)
    assert torch.allclose(prod, eye, atol=1e-4)


def test_aligner_shapes_and_ranges():
    al = TieredAligner().eval()
    cur, anc = torch.rand(2, 3, 32, 40), torch.rand(2, 3, 32, 40)
    with torch.no_grad():
        h, tmap, scalar = al(cur, anc)
    assert h.shape == (2, 3, 3)
    assert tmap.shape == (2, 1, 32, 40)
    assert scalar.shape == (2, 1)
    assert float(tmap.min()) >= 0.0 and float(tmap.max()) <= 1.0
    assert float(scalar.min()) >= 0.0 and float(scalar.max()) <= 1.0


def test_aligner_identity_at_init():
    # Zero-init offset head -> identity homography for graceful degradation.
    al = TieredAligner().eval()
    h, _, _ = al(torch.rand(1, 3, 24, 24), torch.rand(1, 3, 24, 24))
    assert torch.allclose(h, torch.eye(3).unsqueeze(0), atol=1e-5)


def test_alignment_failure_freezes_to_identity_and_zero_trust():
    al = TieredAligner(t_lo=0.2).eval()
    with torch.no_grad():
        al.to_scalar.bias.fill_(-30.0)  # scalar trust ~ 0 < t_lo -> freeze
        h, tmap, scalar = al(torch.rand(1, 3, 16, 16), torch.rand(1, 3, 16, 16))
    assert torch.allclose(h, torch.eye(3).unsqueeze(0), atol=1e-5)
    assert float(scalar.max()) == 0.0
    assert float(tmap.max()) == 0.0


def test_motion_prior_is_fused_additively():
    al = TieredAligner().eval()  # zero-init predicted offset -> output == prior offset
    prior = torch.zeros(1, 8)
    prior[0, 0] = 0.05  # shift the top-left corner x
    h_prior, _, _ = al(torch.rand(1, 3, 16, 16), torch.rand(1, 3, 16, 16), motion_prior=prior)
    h_ref = four_point_to_homography(prior.reshape(1, 4, 2))
    assert torch.allclose(h_prior, h_ref, atol=1e-5)


def test_aligner_param_budget():
    n = sum(p.numel() for p in TieredAligner().parameters())
    print(f"\nTieredAligner params: {n:,}")
    assert n < 0.6e6, n
