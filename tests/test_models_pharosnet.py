"""Tests for pharos.models.pharosnet (WS-A)."""
import pathlib
import sys

import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pharos.config import load_config  # noqa: E402
from pharos.contracts import PharosOutput  # noqa: E402
from pharos.models import PharosNet  # noqa: E402

CFG_PATH = ROOT / "configs" / "base.yaml"


def _model(**overrides):
    cfg = load_config(CFG_PATH, overrides or None)
    return PharosNet(cfg.model)


def _check_output(out: PharosOutput, b: int, h: int, w: int):
    assert isinstance(out, PharosOutput)
    o = out.output.detach()
    conf = out.confidence.detach()
    deg = {k: v.detach() for k, v in out.deg.items()}
    assert o.shape == (b, 3, h, w)
    assert float(o.min()) >= -1e-5 and float(o.max()) <= 1.0 + 1e-5
    assert conf.shape == (b, 1, h, w)
    assert float(conf.min()) > 0.0 and float(conf.max()) <= 1.0 + 1e-5
    assert out.grid.shape[:2] == (b, 12)
    assert deg["beta"].shape == (b, 1)
    assert deg["airlight"].shape == (b, 3)
    assert deg["sigma"].shape == (b, 1)
    assert deg["domain_logits"].shape == (b, 3)
    assert float(deg["beta"].min()) >= 0.0 and float(deg["sigma"].min()) >= 0.0
    assert 0.0 <= float(deg["airlight"].min()) and float(deg["airlight"].max()) <= 1.0


def test_param_count_within_budget():
    net = _model()
    n = sum(p.numel() for p in net.parameters())
    print(f"\nPharosNet param count: {n:,} ({n / 1e6:.3f}M)")
    assert 1.5e6 <= n < 4.0e6, n


def test_forward_image_mode():
    # temporal=False -> image mode, state stays None
    net = _model(**{"model.temporal": False}).eval()
    x = torch.rand(2, 3, 64, 96)
    out = net(x)
    _check_output(out, 2, 64, 96)
    assert out.state is None
    assert out.t_hat is not None and out.t_hat.shape[:2] == (2, 1)


def test_forward_video_mode_state_threading():
    net = _model().eval()  # temporal=True by default
    f1 = torch.rand(1, 3, 72, 64)
    r1 = net(f1, state=None)
    _check_output(r1, 1, 72, 64)
    assert r1.state is not None  # temporal=True initializes a fresh state
    r2 = net(f1, state=r1.state)
    _check_output(r2, 1, 72, 64)
    assert r2.state is not None


def test_scene_cut_reset_in_full_model():
    net = _model().eval()
    r1 = net(torch.zeros(1, 3, 48, 48), state=None)
    # radically different frame -> scene cut inside temporal module (must not error)
    r2 = net(torch.ones(1, 3, 48, 48), state=r1.state)
    _check_output(r2, 1, 48, 48)


def test_non_divisible_sizes():
    net = _model().eval()
    for h, w in [(50, 70), (33, 45), (17, 19)]:
        out = net(torch.rand(1, 3, h, w))
        assert out.output.shape == (1, 3, h, w)


def test_reparameterize_equivalence_full_model():
    net = _model()
    # populate BN running stats, then compare eval-mode outputs before/after folding.
    net.train()
    for _ in range(2):
        net(torch.rand(2, 3, 64, 96))
    net.eval()
    x = torch.rand(2, 3, 48, 80)
    o1 = net(x)
    net.reparameterize()
    o2 = net(x)
    assert torch.allclose(o1.output, o2.output, atol=1e-4), float((o1.output - o2.output).abs().max())
    assert torch.allclose(o1.grid, o2.grid, atol=1e-4)


def test_severity_gate_passthrough_static():
    j = torch.rand(2, 3, 8, 8)
    frame = torch.rand(2, 3, 8, 8)
    zero = torch.zeros(2, 1)
    one = torch.ones(2, 1)
    assert torch.allclose(PharosNet.severity_gate(j, frame, zero), frame, atol=1e-6)
    assert torch.allclose(PharosNet.severity_gate(j, frame, one), j, atol=1e-6)


def test_gate_below_beta_lo_passes_input_through():
    # Force predicted beta ~ 0 (below beta_lo) -> alpha=0 -> output == input.
    net = _model().eval()
    with torch.no_grad():
        net.deg_head.head_beta.weight.zero_()
        net.deg_head.head_beta.bias.fill_(-30.0)  # softplus(-30) ~ 0
    x = torch.rand(1, 3, 40, 56)
    with torch.no_grad():
        out = net(x)
    assert float(out.aux["alpha"].max()) < 1e-5
    assert torch.allclose(out.output, x, atol=1e-5)


def test_forward_amp_safe_cpu():
    net = _model().eval()
    x = torch.rand(1, 3, 48, 48)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        out = net(x)
    assert torch.isfinite(out.output.float()).all()
    assert out.output.shape == (1, 3, 48, 48)


def test_backward_runs():
    net = _model()
    x = torch.rand(1, 3, 48, 48, requires_grad=True)
    out = net(x)
    loss = out.output.mean() + out.confidence.mean() + out.deg["beta"].mean()
    loss.backward()
    grads = [p.grad for p in net.parameters() if p.requires_grad]
    assert any(g is not None for g in grads)
    assert all(torch.isfinite(g).all() for g in grads if g is not None)
    assert x.grad is not None and torch.isfinite(x.grad).all()


def test_external_cond_supported():
    net = _model(**{"model.cond_dim": 8}).eval()
    x = torch.rand(1, 3, 40, 40)
    out = net(x, cond=torch.rand(1, 8))
    _check_output(out, 1, 40, 40)
