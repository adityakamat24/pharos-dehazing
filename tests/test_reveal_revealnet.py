"""Integration tests for RevealNet over the real PharosNet (WS-v2A, DESIGN.md §9d)."""
import pathlib
import sys

import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pharos.config import load_config  # noqa: E402
from pharos.contracts import PharosOutput  # noqa: E402
from pharos.models import PharosNet  # noqa: E402
from pharos.models.reveal import RevealNet  # noqa: E402

CFG_PATH = ROOT / "configs" / "base.yaml"


def _build(**reveal_cfg) -> RevealNet:
    cfg = load_config(CFG_PATH)
    return RevealNet(PharosNet(cfg.model), reveal_cfg or None)


def _check(out: PharosOutput, b: int, h: int, w: int) -> None:
    assert isinstance(out, PharosOutput)
    o = out.output.detach()
    assert o.shape == (b, 3, h, w)
    assert torch.isfinite(o).all()
    assert float(o.min()) >= -1e-5 and float(o.max()) <= 1.0 + 1e-5
    assert out.confidence.shape == (b, 1, h, w)
    assert out.grid.shape[:2] == (b, 12)
    for key in ("staleness", "memory_trust", "align_trust", "j_restored"):
        assert key in out.aux, key
    assert out.aux["staleness"].shape == (b, 1, h, w)
    assert out.aux["memory_trust"].shape == (b, 1, h, w)


def test_forward_chains_state_across_six_frames():
    net = _build().eval()
    state = None
    x = torch.rand(2, 3, 64, 96)
    for _ in range(6):
        out = net(x, state=state)
        _check(out, 2, 64, 96)
        assert isinstance(out.state, dict) and {"inner", "memory", "anchor"} <= set(out.state)
        state = out.state


def test_first_frame_seeds_memory_from_restoration():
    net = _build().eval()
    out = net(torch.rand(1, 3, 48, 48), state=None)
    _check(out, 1, 48, 48)
    # No alignment on the first frame -> align_trust is all zero, staleness ~0 (age 0).
    assert float(out.aux["align_trust"].detach().abs().max()) == 0.0
    assert float(out.aux["staleness"].detach().max()) == 0.0


def test_non_divisible_sizes_and_batches():
    net = _build().eval()
    for (b, h, w) in [(1, 50, 70), (2, 33, 45), (1, 17, 19)]:
        out = net(torch.rand(b, 3, h, w))
        _check(out, b, h, w)


def test_memory_recall_under_moving_occluder():
    # Static random background, an opaque square sweeping across, camera identity.
    # After N frames the composite under the (now) occluded band must be closer to
    # the true background than the raw occluded frame is (memory recall works).
    torch.manual_seed(1)
    net = _build().eval()
    hs, ws, sq = 40, 40, 10
    bg = torch.rand(1, 3, hs, ws)

    def occluded(cx: int):
        f = bg.clone()
        x0, x1 = max(0, cx - sq // 2), min(ws, cx + sq // 2)
        f[:, :, 12:12 + sq, x0:x1] = 0.0
        return f, (x0, x1)

    state = None
    positions = list(range(4, 14))  # occluder moves right each frame
    with torch.no_grad():
        for cx in positions:
            f, _ = occluded(cx)
            out = net(f, state=state)
            state = out.state
    f_last, (x0, x1) = occluded(positions[-1])
    reg = (slice(None), slice(None), slice(12, 12 + sq), slice(x0, x1))
    comp_err = (out.output[reg] - bg[reg]).abs().mean()
    occ_err = (f_last[reg] - bg[reg]).abs().mean()
    assert comp_err < occ_err, (float(comp_err), float(occ_err))


def test_new_module_param_budget_under_0_8m():
    net = _build()
    inner = sum(p.numel() for p in net.inner.parameters())
    total = sum(p.numel() for p in net.parameters())
    new = total - inner
    print(f"\nRevealNet new-module params: {new:,} ({new / 1e6:.4f}M)")
    assert new < 0.8e6, new


def test_backward_through_whole_revealnet():
    net = _build()
    x = torch.rand(1, 3, 48, 48, requires_grad=True)
    o1 = net(x, state=None)
    o2 = net(x, state=o1.state)  # second frame exercises the aligner
    loss = o2.output.mean() + o2.confidence.mean() + o2.aux["memory_trust"].mean()
    loss.backward()
    grads = [p.grad for p in net.parameters() if p.grad is not None]
    assert grads, "no gradients produced"
    assert all(torch.isfinite(g).all() for g in grads)
    assert x.grad is not None and torch.isfinite(x.grad).all()


def test_amp_safe_cpu_chain():
    net = _build().eval()
    state = None
    x = torch.rand(1, 3, 48, 48)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        for _ in range(3):
            out = net(x, state=state)
            state = out.state
    assert torch.isfinite(out.output.float()).all()
    assert out.output.shape == (1, 3, 48, 48)


def test_reparameterize_passthrough_matches():
    net = _build()
    net.train()
    s = None
    for _ in range(2):  # populate inner BN running stats
        o = net(torch.rand(1, 3, 48, 48), state=s)
        s = o.state
    net.eval()
    x = torch.rand(1, 3, 48, 64)
    a = net(x)
    net.reparameterize()
    b = net(x)
    assert torch.allclose(a.output, b.output, atol=1e-4), float((a.output - b.output).abs().max())


def test_reveal_cfg_overrides_are_applied():
    net = _build(mem_res=64, half_life=5.0, seed_trust=0.25)
    assert net.mem_res == 64
    assert net.cfg["half_life"] == 5.0
    assert net.seed_trust == 0.25


def test_satisfies_pharos_model_contract():
    net = _build().eval()
    # Runs through the exact contract call signature and exposes reparameterize().
    out = net(torch.rand(1, 3, 40, 40), None, None)
    assert isinstance(out, PharosOutput)
    assert callable(net.reparameterize)
    net.reparameterize()
