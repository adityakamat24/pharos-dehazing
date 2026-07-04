"""Tests for RevealNet reveal memory + compositor (WS-v2A, DESIGN.md §9d.2-3)."""
import pathlib
import sys

import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pharos.models.reveal.aligner import four_point_to_homography  # noqa: E402
from pharos.models.reveal.compositor import age_decay, composite  # noqa: E402
from pharos.models.reveal.memory import RevealMemory  # noqa: E402

CFG = {"merge_thresh": 0.1, "decay_keep": 0.98, "decay_miss": 0.9, "comp_k": 8.0, "half_life": 30.0}


def _mem(h=8, w=8, trust=0.5):
    return RevealMemory(
        torch.rand(1, 3, h, w), torch.full((1, 1, h, w), trust), torch.zeros(1, 1, h, w)
    )


def test_seed_low_trust_zero_age():
    m = RevealMemory.seed(torch.rand(1, 3, 6, 6), seed_trust=0.1)
    assert m.rgb.shape == (1, 3, 6, 6)
    assert torch.allclose(m.trust, torch.full_like(m.trust, 0.1))
    assert float(m.age.max()) == 0.0
    assert set(m.buffers) == {"rgb", "trust", "age"}


def test_update_merge_raises_trust_resets_age():
    m = _mem(trust=0.5)
    # conf=1, align=1 -> w=1 > merge_thresh everywhere.
    m.update(torch.ones(1, 3, 8, 8), torch.ones(1, 1, 8, 8), torch.ones(1, 1, 8, 8), CFG, dt=1.0)
    assert torch.allclose(m.trust, torch.ones_like(m.trust), atol=1e-6)  # max(0.5*0.98, 1.0)=1
    assert float(m.age.max()) == 0.0
    assert torch.allclose(m.rgb, torch.ones_like(m.rgb), atol=1e-6)      # lerp with w=1 -> J_t


def test_update_miss_decays_trust_grows_age():
    m = _mem(trust=0.5)
    for i in range(3):
        # conf=0 -> w=0 < merge_thresh -> miss branch everywhere.
        m.update(torch.rand(1, 3, 8, 8), torch.zeros(1, 1, 8, 8), torch.ones(1, 1, 8, 8), CFG, dt=1.0)
        assert abs(float(m.trust.mean()) - 0.5 * 0.9 ** (i + 1)) < 1e-5
        assert abs(float(m.age.mean()) - (i + 1)) < 1e-5


def test_update_partial_merge_is_per_pixel():
    m = _mem(trust=0.3)
    conf = torch.zeros(1, 1, 8, 8)
    conf[..., :4, :] = 1.0                # top half confident -> merge, bottom miss
    m.update(torch.ones(1, 3, 8, 8), conf, torch.ones(1, 1, 8, 8), CFG, dt=2.0)
    assert float(m.age[..., :4, :].max()) == 0.0          # merged -> reset
    assert float(m.age[..., 4:, :].min()) == 2.0          # missed -> +dt
    assert float(m.trust[..., :4, :].min()) > 0.9         # merged -> ~1
    assert abs(float(m.trust[..., 4:, :].mean()) - 0.3 * 0.9) < 1e-5


def test_reset_clears_memory():
    m = _mem(trust=0.7)
    m.reset()
    assert float(m.trust.max()) == 0.0
    assert float(m.rgb.abs().max()) == 0.0
    assert float(m.age.max()) == 0.0


def test_detach_breaks_graph():
    rgb = torch.rand(1, 3, 4, 4, requires_grad=True)
    m = RevealMemory(rgb, torch.rand(1, 1, 4, 4), torch.zeros(1, 1, 4, 4))
    d = m.detach()
    assert not d.rgb.requires_grad


def test_warp_identity_preserves_memory():
    m = _mem(trust=0.6)
    rgb0, trust0 = m.rgb.clone(), m.trust.clone()
    m.warp(torch.eye(3).unsqueeze(0))
    assert torch.allclose(m.rgb, rgb0, atol=1e-4)
    assert torch.allclose(m.trust, trust0, atol=1e-4)


def test_warp_border_fades_trust_at_invalid_regions():
    # A homography that shifts content off one edge -> revealed border has ~0 trust.
    m = RevealMemory(torch.rand(1, 3, 24, 24), torch.ones(1, 1, 24, 24), torch.zeros(1, 1, 24, 24))
    off = torch.zeros(1, 4, 2)
    off[:, :, 0] = 0.4  # push everything right -> left edge becomes out-of-bounds
    m.warp(four_point_to_homography(off))
    assert float(m.trust[..., :, 0].mean()) < 0.5   # faded near the newly-revealed edge
    assert float(m.trust.max()) <= 1.0 + 1e-6


def test_age_decay_half_life():
    hl = 30.0
    assert abs(float(age_decay(torch.tensor(0.0), hl)) - 1.0) < 1e-6
    assert abs(float(age_decay(torch.tensor(hl), hl)) - 0.5) < 1e-6
    assert abs(float(age_decay(torch.tensor(2 * hl), hl)) - 0.25) < 1e-6


def test_composite_prefers_trustworthy_memory():
    # High memory trust + low current confidence -> output tracks memory.
    mem = RevealMemory(torch.ones(1, 3, 8, 8), torch.ones(1, 1, 8, 8), torch.zeros(1, 1, 8, 8))
    j = torch.zeros(1, 3, 16, 16)
    conf = torch.full((1, 1, 16, 16), 1e-3)
    out, stale = composite(j, conf, mem, CFG)
    assert float(out.mean()) > 0.9              # memory (ones) dominates
    assert out.shape == (1, 3, 16, 16)
    assert stale.shape == (1, 1, 16, 16)


def test_composite_prefers_current_when_memory_untrusted():
    mem = RevealMemory(torch.ones(1, 3, 8, 8), torch.zeros(1, 1, 8, 8), torch.zeros(1, 1, 8, 8))
    j = torch.zeros(1, 3, 16, 16)
    conf = torch.ones(1, 1, 16, 16)
    out, _ = composite(j, conf, mem, CFG)
    assert float(out.mean()) < 0.1              # current restoration (zeros) dominates
