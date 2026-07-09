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


# ---------------------------------------------------------------------------
# v2.1 anchor-frame regression tests
# ---------------------------------------------------------------------------
def _trans(tx: float) -> torch.Tensor:
    """Pure horizontal-translation homography (normalized coords)."""
    h = torch.eye(3).unsqueeze(0)
    h[0, 0, 2] = tx
    return h


def test_long_horizon_no_blur_invariance():
    """DECISIVE: static scene + identity camera for T=40 must NOT accumulate blur.

    The v2.0 design re-warped the buffer in place every frame (N frames = N bilinear
    resamples -> mush). The anchor-frame design resamples only the fresh observation,
    so the buffer converges to the (single) observation and stays there. Buffer must
    stay within 1e-3 of the constant observation after 40 high-confidence updates.
    """
    torch.manual_seed(0)
    obs = torch.rand(1, 3, 16, 16)
    mem = RevealMemory.seed(obs, seed_trust=0.5, margin=1.0)   # no margin -> exact
    assert float((mem.rgb - obs).abs().max()) < 1e-3          # seed is a clean copy
    for _ in range(40):
        mem.compose(torch.eye(3).unsqueeze(0))                # identity camera
        mem.update(obs, torch.ones(1, 1, 16, 16), torch.ones(1, 1, 16, 16), CFG, dt=1.0)
    assert float((mem.rgb - obs).abs().max()) < 1e-3, float((mem.rgb - obs).abs().max())
    assert float((mem.read_view().rgb - obs).abs().max()) < 1e-3


def test_buffer_is_not_resampled_each_frame():
    """Anchor buffer identity: the stored buffer never changes under an identity camera.

    Contrast with the old design where each frame re-warped the buffer; here the buffer
    tensor is bit-stable across frames (only the fresh observation is resampled).
    """
    torch.manual_seed(1)
    obs = torch.rand(1, 3, 12, 12)
    mem = RevealMemory.seed(obs, seed_trust=0.5, margin=1.0)
    snap = mem.rgb.clone()
    for _ in range(20):
        mem.compose(torch.eye(3).unsqueeze(0))
        mem.update(obs, torch.ones(1, 1, 12, 12), torch.ones(1, 1, 12, 12), CFG)
    assert float((mem.rgb - snap).abs().max()) < 1e-3


def test_panning_margin_recall_and_necessity():
    """Panning translation + margin: off-anchor-view content is stored and recalled.

    A target region lives OUTSIDE the frame-0 (anchor) view and is only ever seen after
    the camera pans. With ``margin >= 2`` the anchor buffer covers it (near-exact recall,
    high trust); with ``margin == 1`` those writes fall outside the buffer and are
    dropped (no recall). Aligner homographies are injected (translation-only) to isolate
    memory correctness from aligner quality.
    """
    torch.manual_seed(0)
    v = 48
    world = torch.rand(1, 3, v, 96)
    cfg = {**CFG, "reanchor_px": 10.0, "decay_keep": 0.995}   # disable rebase for isolation

    def ox(t: int) -> int:
        return min(2 * t, 24)

    def txc(t: int) -> float:
        return -2.0 * ox(t) / (v - 1)                          # cumulative anchor->cur

    def run(margin: float):
        mem = RevealMemory.seed(world[:, :, :, 0:v], 0.3, margin=margin)
        prev = txc(0)
        for t in range(1, 31):
            frame = world[:, :, :, ox(t):ox(t) + v]
            mem.compose(_trans(txc(t) - prev))                # frame-to-frame delta
            prev = txc(t)
            mem.update(frame, torch.ones(1, 1, v, v), torch.ones(1, 1, v, v), cfg)
        view = mem.read_view()
        gt = world[:, :, :, 50:58]                            # world cols 50..58
        recall = view.rgb[:, :, :, 26:34]                     # same cols in the frame-30 view
        return (recall - gt).abs().mean().item(), view.trust[:, :, :, 26:34].mean().item()

    err_big, trust_big = run(2.0)
    err_none, trust_none = run(1.0)
    assert err_big < 5e-3 and trust_big > 0.5, (err_big, trust_big)   # exact recall w/ margin
    assert err_none > 5 * err_big and trust_none < 0.1, (err_none, trust_none)  # margin matters


def test_maybe_reanchor_counts_and_recenters_on_drift():
    """Large cumulative drift triggers a single-resample re-anchor: count>0, H->identity."""
    torch.manual_seed(2)
    mem = RevealMemory.seed(torch.rand(1, 3, 24, 24), 0.5, margin=1.5)
    mem.compose(_trans(0.8))                                   # big pan -> corner drift > 0.7
    n = mem.maybe_reanchor(CFG, torch.ones(1, 1))              # high trust, drift triggers
    assert n == 1
    assert torch.allclose(mem.H, torch.eye(3).unsqueeze(0), atol=1e-5)   # reset to identity


def test_maybe_reanchor_on_trust_collapse():
    """Alignment-trust collapse (< t_lo) triggers a re-anchor even without drift."""
    mem = RevealMemory.seed(torch.rand(1, 3, 20, 20), 0.6, margin=1.5)
    trust0 = mem.trust.clone()
    n = mem.maybe_reanchor({**CFG, "t_lo": 0.2, "rebase_decay": 0.9}, torch.zeros(1, 1))
    assert n == 1
    assert float(mem.trust.max()) <= float(trust0.max()) + 1e-6   # trust decayed by rebase


def test_detach_detaches_buffers_and_H():
    rgb = torch.rand(1, 3, 8, 8, requires_grad=True)
    mem = RevealMemory(rgb, torch.rand(1, 1, 8, 8), torch.zeros(1, 1, 8, 8))
    mem.H = mem.H.clone().requires_grad_(True)
    d = mem.detach()
    assert not d.rgb.requires_grad and not d.H.requires_grad
