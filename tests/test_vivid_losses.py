"""CPU tests for VividLoss (WS-vivid): finite total, self-managed discriminator
update, GAN warmup ramp, confidence NLL reuse, clip last-frame reduction, and
state_dict round-trip. No GPU, no network (LPIPS weight kept 0 so the possibly
network-touching build never runs).
"""
from __future__ import annotations

import math

import torch

from pharos.contracts import PharosOutput
from pharos.losses.vivid_losses import VividLoss

B, T, H, W = 2, 3, 64, 64
HL = 16  # low-res logvar side (mirrors the model's confidence head)


def _cfg(**vivid_over):
    vivid = dict(l1=1.0, lpips=0.0, gan=0.02, conf=0.05, gan_warmup=4, disc_lr=1e-3)
    vivid.update(vivid_over)
    vivid.setdefault("disc", {"base_ch": 32})  # small patch D for fast CPU tests
    return {"loss": {"vivid": vivid}}


def _img_out(conf=True, logvar=True):
    aux = {"logvar": torch.randn(B, 1, HL, HL)} if logvar else {}
    return PharosOutput(
        output=torch.rand(B, 3, H, W, requires_grad=True),
        confidence=torch.rand(B, 1, H, W).clamp(0.05, 1.0) if conf else None,
        grid=torch.rand(B, 12, 2, 4, 4),
        state=None,
        deg={"beta": torch.rand(B, 1)},
        aux=aux,
    )


def _img_batch(paired=True):
    return {
        "hazy": torch.rand(B, 3, H, W),
        "clean": torch.rand(B, 3, H, W) if paired else None,
        "domain": torch.randint(0, 3, (B,)),
        "clip": False,
        "meta": [{} for _ in range(B)],
    }


def _clip_out():
    return PharosOutput(
        output=torch.rand(B, T, 3, H, W, requires_grad=True),
        confidence=torch.rand(B, T, 1, H, W).clamp(0.05, 1.0),
        grid=torch.rand(B, T, 12, 2, 4, 4),
        state=None,
        deg={"beta": torch.rand(B, T, 1)},
        aux={"logvar": torch.randn(B, T, 1, HL, HL)},
    )


def _clip_batch():
    return {
        "hazy": torch.rand(B, T, 3, H, W),
        "clean": torch.rand(B, T, 3, H, W),
        "domain": torch.randint(0, 3, (B,)),
        "clip": True,
        "meta": [{} for _ in range(B)],
    }


# --------------------------------------------------------------------------
def test_finite_total_paired_batch():
    loss = VividLoss(_cfg())
    total, log = loss(_img_out(), _img_batch(), teachers=None)
    assert total.shape == ()
    assert math.isfinite(float(total.detach()))
    for k in ("l1", "lpips", "gan", "conf", "d", "gan_w", "total"):
        assert k in log and math.isfinite(log[k])
    assert total.requires_grad  # generator graph is live (anchors on out.output)


def test_discriminator_weights_change_after_call():
    loss = VividLoss(_cfg())
    before = [p.detach().clone() for p in loss.disc.parameters()]
    loss(_img_out(), _img_batch(), teachers=None)  # runs one internal D step
    changed = any(not torch.equal(b, a) for b, a in zip(before, loss.disc.parameters()))
    assert changed, "discriminator optimizer did not step (weights unchanged)"


def test_discriminator_skips_when_unpaired():
    loss = VividLoss(_cfg())
    before = [p.detach().clone() for p in loss.disc.parameters()]
    total, log = loss(_img_out(), _img_batch(paired=False), teachers=None)
    # no clean GT -> no real sample -> D not updated, all paired terms zero.
    assert all(torch.equal(b, a) for b, a in zip(before, loss.disc.parameters()))
    assert log["l1"] == 0.0 and log["conf"] == 0.0 and log["gan"] == 0.0
    assert math.isfinite(float(total))


def test_gan_warmup_factor_ramps():
    loss = VividLoss(_cfg(gan_warmup=4))
    loss._step = 0
    assert loss._warmup_factor() == 0.0
    loss._step = 2
    assert abs(loss._warmup_factor() - 0.5) < 1e-6
    loss._step = 4
    assert loss._warmup_factor() == 1.0
    loss._step = 99
    assert loss._warmup_factor() == 1.0  # clamped at target


def test_gan_weight_ramps_across_calls():
    loss = VividLoss(_cfg(gan_warmup=4))
    weights = []
    for _ in range(6):
        _, log = loss(_img_out(), _img_batch(), teachers=None)
        weights.append(log["gan_w"])
    assert weights[0] == 0.0  # step 0 -> no adversarial pressure yet
    assert weights == sorted(weights)  # monotonic non-decreasing
    assert abs(weights[-1] - loss.w["gan"]) < 1e-9  # reaches target after warmup


def test_no_warmup_activates_gan_immediately():
    loss = VividLoss(_cfg(gan=0.1, gan_warmup=0))
    _, log = loss(_img_out(), _img_batch(), teachers=None)
    assert abs(log["gan_w"] - 0.1) < 1e-9


def test_conf_term_nonzero_with_logvar():
    loss = VividLoss(_cfg(gan=0.0, conf=1.0))
    _, log = loss(_img_out(logvar=True), _img_batch(), teachers=None)
    assert log["conf"] != 0.0 and math.isfinite(log["conf"])


def test_conf_term_zero_without_confidence():
    loss = VividLoss(_cfg())
    _, log = loss(_img_out(conf=False, logvar=False), _img_batch(), teachers=None)
    assert log["conf"] == 0.0


def test_reduce_last_uses_last_frame():
    out = _clip_out()
    clean = torch.rand(B, T, 3, H, W)
    out_last, clean_last, logvar_last = VividLoss._reduce_last(out, clean)
    assert out_last.shape == (B, 3, H, W)
    assert torch.equal(out_last, out.output[:, -1])
    assert clean_last.shape == (B, 3, H, W)
    assert torch.equal(clean_last, clean[:, -1])
    assert logvar_last.shape == (B, 1, HL, HL)
    assert torch.equal(logvar_last, out.aux["logvar"][:, -1])


def test_clip_batch_runs_and_is_finite():
    loss = VividLoss(_cfg())
    total, log = loss(_clip_out(), _clip_batch(), teachers=None)
    assert math.isfinite(float(total.detach()))
    assert log["l1"] != 0.0  # last-frame Charbonnier evaluated
    assert log["conf"] != 0.0


def test_generator_total_backprops():
    loss = VividLoss(_cfg(gan=0.1, gan_warmup=0))  # gan active from step 0
    out = _img_out()
    total, log = loss(out, _img_batch(), teachers=None)
    assert log["gan_w"] == 0.1
    total.backward()
    assert out.output.grad is not None and torch.isfinite(out.output.grad).all()


def test_state_dict_roundtrip_restores_disc_and_optimizer():
    loss1 = VividLoss(_cfg())
    loss1(_img_out(), _img_batch(), teachers=None)  # step D -> populates Adam state
    sd = loss1.state_dict()
    assert set(sd) >= {"disc", "d_opt", "step"}

    loss2 = VividLoss(_cfg())
    assert any(
        not torch.equal(a, b)
        for a, b in zip(loss1.disc.parameters(), loss2.disc.parameters())
    )  # fresh random init differs before load

    loss2.load_state_dict(sd)
    for a, b in zip(loss1.disc.parameters(), loss2.disc.parameters()):
        assert torch.equal(a, b)  # discriminator weights restored exactly
    assert loss2._step == loss1._step

    s1 = loss1.d_opt.state_dict()["state"]
    s2 = loss2.d_opt.state_dict()["state"]
    assert len(s2) == len(s1) and len(s2) > 0  # Adam momentum restored
    for k in s1:
        assert torch.allclose(s1[k]["exp_avg"], s2[k]["exp_avg"])
        assert torch.allclose(s1[k]["exp_avg_sq"], s2[k]["exp_avg_sq"])
