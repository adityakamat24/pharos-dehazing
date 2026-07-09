"""CPU tests for PharosLoss: finite outputs, term dict, graceful-zero behavior."""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from pharos.contracts import PharosOutput
from pharos.losses import PharosLoss
from pharos.losses import losses as L

B, H, W = 2, 32, 32
GD, GS = 8, 16  # grid depth, grid size


# ---- fake teachers -------------------------------------------------------
class _FakeDepth:
    def __call__(self, img):
        b = img.shape[0]
        return torch.rand(b, 1, 24, 24)


class _FakeDet:
    def __call__(self, img):
        # shape-consistent features derived from the input
        return [F.adaptive_avg_pool2d(img, 8), F.adaptive_avg_pool2d(img, 4)]


class _FakeFlow:
    def __call__(self, a, b):
        return torch.zeros(a.shape[0], 2, a.shape[-2], a.shape[-1])


class _Teachers:
    def __init__(self, depth=None, detector=None, flow=None):
        self.depth = depth
        self.detector = detector
        self.flow = flow


def _deg():
    return {
        "beta": torch.rand(B, 1),
        "airlight": torch.rand(B, 3),
        "sigma": torch.rand(B, 1),
        "domain_logits": torch.randn(B, 3),
    }


def _image_output(with_aux=True):
    aux = {}
    if with_aux:
        aux["lowres_feats"] = torch.rand(B, 16, 24, 24)
    return PharosOutput(
        output=torch.rand(B, 3, H, W),
        confidence=torch.rand(B, 1, H, W).clamp(0.05, 1.0),
        grid=torch.rand(B, 12, GD, GS, GS),
        state=None,
        deg=_deg(),
        aux=aux,
    )


def _image_batch():
    return {
        "hazy": torch.rand(B, 3, H, W),
        "clean": torch.rand(B, 3, H, W),
        "domain": torch.randint(0, 3, (B,)),
        "clip": False,
        "meta": {"beta": torch.rand(B), "airlight": torch.rand(B, 3), "sigma": torch.rand(B)},
    }


def _cfg(every_n=1):
    return {
        "loss": {"rec": 1.0, "freq": 0.1, "conf": 0.05, "depth": 0.1, "det": 0.05, "temp": 0.5, "phys": 0.1},
        "teachers": {"detector": {"every_n": every_n}},
    }


TERMS = ["rec", "freq", "conf", "depth", "det", "temp", "phys", "total"]


def test_image_loss_finite_full_teachers():
    loss = PharosLoss(_cfg(every_n=1))
    teachers = _Teachers(_FakeDepth(), _FakeDet(), _FakeFlow())
    total, log = loss(_image_output(), _image_batch(), teachers)
    assert total.shape == ()
    assert math.isfinite(float(total))
    for k in TERMS:
        assert k in log and math.isfinite(log[k])
    # with teachers + clean present, several terms should be strictly positive
    assert log["rec"] > 0 and log["depth"] > 0 and log["det"] > 0 and log["phys"] > 0


def test_clip_loss_finite_and_temporal_active():
    loss = PharosLoss(_cfg(every_n=1))
    teachers = _Teachers(_FakeDepth(), _FakeDet(), _FakeFlow())
    T = 3
    out = _image_output()
    out.aux["outputs"] = torch.rand(B, T, 3, H, W)
    out.aux["grids"] = torch.rand(B, T, 12, GD, GS, GS)
    batch = _image_batch()
    batch["clip"] = True
    batch["clean"] = torch.rand(B, T, 3, H, W)  # clip GT
    total, log = loss(out, batch, teachers)
    assert math.isfinite(float(total))
    assert log["temp"] > 0  # grid smoothness + flow-warp photometric


def test_all_terms_zero_when_inputs_missing():
    loss = PharosLoss(_cfg())
    teachers = _Teachers(None, None, None)  # all disabled
    out = PharosOutput(
        output=torch.rand(B, 3, H, W),
        confidence=torch.rand(B, 1, H, W).clamp(0.05, 1.0),
        grid=torch.rand(B, 12, GD, GS, GS),
        state=None,
        deg=_deg(),
        aux={},
    )
    batch = {"hazy": torch.rand(B, 3, H, W), "clean": None, "clip": False, "meta": {}}
    total, log = loss(out, batch, teachers)
    assert float(total) == 0.0
    for k in ["rec", "freq", "conf", "depth", "det", "temp", "phys"]:
        assert log[k] == 0.0


def test_disabled_weights_zero_out_total():
    cfg = _cfg()
    for k in cfg["loss"]:
        cfg["loss"][k] = 0.0
    loss = PharosLoss(cfg)
    teachers = _Teachers(_FakeDepth(), _FakeDet(), _FakeFlow())
    total, log = loss(_image_output(), _image_batch(), teachers)
    assert float(total) == 0.0  # all weights zero -> total zero regardless of terms


def test_det_every_n_gating():
    loss = PharosLoss(_cfg(every_n=3))
    teachers = _Teachers(None, _FakeDet(), None)
    batch = _image_batch()
    vals = [loss(_image_output(with_aux=False), batch, teachers)[1]["det"] for _ in range(3)]
    # only the 3rd call (counter % 3 == 0) computes a nonzero detection loss
    assert vals[0] == 0.0 and vals[1] == 0.0 and vals[2] > 0.0


def test_missing_aux_uses_grid_fallback_for_depth():
    loss = PharosLoss(_cfg())
    teachers = _Teachers(_FakeDepth(), None, None)
    total, log = loss(_image_output(with_aux=False), _image_batch(), teachers)
    assert math.isfinite(float(total))
    assert log["depth"] >= 0.0  # falls back to pooled grid, no crash


# ---------------------------------------------------------------------------
# item 3 — CSF-weighted L_freq
# ---------------------------------------------------------------------------
def test_csf_mask_shape_and_midband_peak():
    loss = PharosLoss(_cfg())
    h, w = 32, 32
    m = loss._csf_mask(h, int(w), torch.device("cpu"), torch.float32)
    assert m.shape == (1, 1, h, w // 2 + 1)
    assert abs(float(m.max()) - 1.0) < 1e-5      # normalized to peak 1
    assert float(m[0, 0, 0, 0]) < 0.5            # DC strongly down-weighted
    # peak sits at a mid radial frequency (not DC, not Nyquist corner)
    idx = int(m.flatten().argmax())
    r, c = divmod(idx, w // 2 + 1)
    fy = float(torch.fft.fftfreq(h)[r])
    fx = float(torch.fft.rfftfreq(w)[c])
    radius = (fy * fy + fx * fx) ** 0.5 / 0.5
    assert 0.15 < radius < 0.85, radius


def test_csf_freq_toggle_changes_loss_but_both_finite():
    out, batch = _image_output(), _image_batch()
    teachers = _Teachers(None, None, None)
    on = PharosLoss(_cfg())                       # csf_freq default True
    cfg_off = _cfg()
    cfg_off["loss"]["csf_freq"] = False
    off = PharosLoss(cfg_off)
    lo_on = on(out, batch, teachers)[1]["freq"]
    lo_off = off(out, batch, teachers)[1]["freq"]
    assert math.isfinite(lo_on) and math.isfinite(lo_off)
    assert lo_on != lo_off                        # CSF actually reweights


# ---------------------------------------------------------------------------
# item 4 — JND-weighted L_rec
# ---------------------------------------------------------------------------
def test_jnd_weight_range():
    clean = torch.rand(B, 3, H, W)
    w = L._jnd_weight(clean, jnd_scale=1.0)
    assert w.shape == (B, 1, H, W)
    # jnd_scale=1 with JND normalized to [0,1] -> weights in [0.5, 1]
    assert float(w.min()) >= 0.5 - 1e-4 and float(w.max()) <= 1.0 + 1e-4


def test_jnd_rec_reduces_to_plain_charbonnier_when_off():
    cfg_off = _cfg()
    cfg_off["loss"]["jnd_rec"] = False
    loss = PharosLoss(cfg_off)
    out, batch = _image_output(), _image_batch()
    rec = loss._rec(out, batch["clean"], torch.device("cpu"))
    plain = torch.sqrt((out.output - batch["clean"]) ** 2 + loss.charb_eps ** 2).mean()
    assert torch.allclose(rec, plain, atol=1e-7)


def test_jnd_rec_on_is_finite_and_differs():
    on, off_cfg = PharosLoss(_cfg()), _cfg()
    off_cfg["loss"]["jnd_rec"] = False
    off = PharosLoss(off_cfg)
    out, batch = _image_output(), _image_batch()
    r_on = on._rec(out, batch["clean"], torch.device("cpu"))
    r_off = off._rec(out, batch["clean"], torch.device("cpu"))
    assert math.isfinite(float(r_on)) and float(r_on) != float(r_off)


# ---------------------------------------------------------------------------
# item 5 — split backscatter beta supervision
# ---------------------------------------------------------------------------
def test_phys_supervises_beta_bs():
    loss = PharosLoss(_cfg())
    deg = _deg()
    deg["beta_bs"] = torch.rand(B, 1)
    out = PharosOutput(
        output=torch.rand(B, 3, H, W), confidence=torch.rand(B, 1, H, W).clamp(0.05, 1.0),
        grid=torch.rand(B, 12, GD, GS, GS), state=None, deg=deg, aux={},
    )
    meta = {"beta": torch.rand(B), "beta_bs": torch.rand(B), "airlight": torch.rand(B, 3),
            "sigma": torch.rand(B)}
    with_bs = loss._phys(out, {"meta": meta, "domain": torch.randint(0, 3, (B,))}, torch.device("cpu"))
    # dropping beta_bs from meta must lower the phys loss (one fewer supervised term)
    meta_no = {k: v for k, v in meta.items() if k != "beta_bs"}
    without_bs = loss._phys(out, {"meta": meta_no, "domain": torch.randint(0, 3, (B,))}, torch.device("cpu"))
    assert math.isfinite(float(with_bs)) and float(with_bs) != float(without_bs)
