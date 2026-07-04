"""CPU, no-network tests for pharos.data.reveal_synthesis (tiny 48x64 clips, seeded)."""
from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import pytest
import torch

from pharos.contracts import DOMAIN_SMOKE
from pharos.data.reveal_synthesis import coverage, synthesize_reveal_clip, warp_homography

H, W = 48, 64


def _gen(seed: int) -> torch.Generator:
    return torch.Generator().manual_seed(seed)


def _static_clip(t: int, seed: int = 0, h: int = H, w: int = W) -> torch.Tensor:
    """A repeated still -> static scene; cam_H then captures ALL background motion."""
    frame = torch.rand(3, h, w, generator=_gen(seed))
    return frame.unsqueeze(0).repeat(t, 1, 1, 1)


def _max_consecutive(mask: torch.Tensor) -> torch.Tensor:
    """Per-pixel longest run of True along the time axis. ``mask`` is (T, H, W)."""
    best = torch.zeros(mask.shape[1:], dtype=torch.int32)
    cur = torch.zeros_like(best)
    for i in range(mask.shape[0]):
        cur = torch.where(mask[i], cur + 1, torch.zeros_like(cur))
        best = torch.maximum(best, cur)
    return best


# ---------------------------------------------------------------------------
# shapes / ranges
# ---------------------------------------------------------------------------
def test_output_shapes_and_keys():
    t = 16
    out = synthesize_reveal_clip(_static_clip(t), generator=_gen(0))
    assert out["hazy"].shape == (t, 3, H, W)
    assert out["gt"].shape == (t, 3, H, W)
    assert out["smoke_density"].shape == (t, 1, H, W)
    assert out["transmission"].shape == (t, 1, H, W)
    assert out["revealed"].shape == (t, 1, H, W)
    assert out["cam_H"].shape == (t, 3, 3)
    assert out["airlight"].shape == (3,)
    assert out["domain"] == DOMAIN_SMOKE
    assert isinstance(out["beta"], float)


def test_hazy_and_fields_in_range():
    out = synthesize_reveal_clip(_static_clip(16), generator=_gen(1))
    for key in ("hazy", "gt", "smoke_density", "transmission"):
        x = out[key]
        assert float(x.min()) >= 0.0 and float(x.max()) <= 1.0, key


def test_wrong_input_shape_raises():
    with pytest.raises(ValueError):
        synthesize_reveal_clip(torch.rand(3, H, W), generator=_gen(0))  # missing T dim
    with pytest.raises(ValueError):
        synthesize_reveal_clip(torch.rand(4, 1, H, W), generator=_gen(0))  # not 3 channels


def test_reproducible_same_seed():
    a = synthesize_reveal_clip(_static_clip(8, seed=5), generator=_gen(3))
    b = synthesize_reveal_clip(_static_clip(8, seed=5), generator=_gen(3))
    assert torch.allclose(a["hazy"], b["hazy"])
    assert torch.allclose(a["cam_H"], b["cam_H"])


# ---------------------------------------------------------------------------
# opaque cores (transmission ~0 for >=3 consecutive frames)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_opaque_cores_persist(seed):
    out = synthesize_reveal_clip(_static_clip(16, seed=seed + 10), generator=_gen(seed))
    tr = out["transmission"][:, 0]  # T,H,W
    opaque = tr < 1e-4  # genuinely occluded (transmission floor 0)
    runs = _max_consecutive(opaque)
    assert int((runs >= 3).sum()) > 0, "expected opaque cores lasting >=3 consecutive frames"
    # and there really is a hard floor of exactly 0 somewhere
    assert float(tr.min()) == 0.0


# ---------------------------------------------------------------------------
# coverage (revelation over time)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_coverage_monotone_and_high_by_16(seed):
    out = synthesize_reveal_clip(_static_clip(16, seed=seed + 20), generator=_gen(seed))
    cov = coverage(out["transmission"])  # default reveal thresh 0.5
    assert len(cov) == 16
    assert all(cov[i] <= cov[i + 1] + 1e-9 for i in range(len(cov) - 1)), "not monotone"
    assert cov[15] > 0.8, f"coverage by T=16 = {cov[15]:.3f}, expected > 0.8"


def test_coverage_upto_scalar_matches_curve():
    out = synthesize_reveal_clip(_static_clip(12), generator=_gen(7))
    cov = coverage(out["transmission"])
    assert coverage(out["transmission"], upto=5) == pytest.approx(cov[5])
    assert coverage(out["transmission"], upto=999) == pytest.approx(cov[-1])  # clamps


def test_per_frame_partial_occlusion():
    # any single frame keeps a substantial fraction occluded, but not the whole scene
    out = synthesize_reveal_clip(_static_clip(16), generator=_gen(2))
    tr = out["transmission"][:, 0]
    occ = (tr <= 0.5).float().mean(dim=(1, 2))  # per-frame occluded fraction
    assert 0.30 <= float(occ.mean()) <= 0.75, float(occ.mean())
    assert float(occ.min()) > 0.05 and float(occ.max()) < 0.95  # never fully clear / fully hidden


# ---------------------------------------------------------------------------
# camera homography (alignment supervision)
# ---------------------------------------------------------------------------
def test_cam_H_first_is_identity():
    out = synthesize_reveal_clip(_static_clip(6), generator=_gen(0))
    assert torch.allclose(out["cam_H"][0], torch.eye(3), atol=1e-6)


def test_cam_H_composes_on_static_content():
    # warping the frame-0 GT by the cumulative H must reproduce frame-t GT (static scene)
    out = synthesize_reveal_clip(_static_clip(8, seed=99), generator=_gen(3))
    gt, cam_H = out["gt"], out["cam_H"]
    for t in range(gt.shape[0]):
        approx = warp_homography(gt[0], cam_H[t])
        # compare on the interior to avoid border-padding effects at the frame edge
        err = (approx[:, 4:-4, 4:-4] - gt[t][:, 4:-4, 4:-4]).abs().mean()
        assert float(err) < 1e-3, f"frame {t}: cam_H composition error {float(err):.5f}"


def test_cam_H_actually_moves_the_view():
    # with default jitter, later frames must differ from frame 0 (non-degenerate motion)
    out = synthesize_reveal_clip(_static_clip(16, seed=1), generator=_gen(4))
    assert not torch.allclose(out["cam_H"][-1], torch.eye(3), atol=1e-3)
