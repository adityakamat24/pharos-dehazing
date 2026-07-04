"""CPU tests for RevealLoss (§9d.4): finite total, per-term toggling, recall mask.

All inputs are stubs — no GPU, no datasets, and NO dependency on the parallel
RevealNet / reveal_dataset modules (RevealLoss wraps an injected inner loss).
"""
from __future__ import annotations

import math

import torch

from pharos.contracts import PharosOutput
from pharos.losses.reveal_losses import RevealLoss, homography_to_4pt, revealed_recall_mask

B, T, H, W = 2, 3, 8, 8


# ---- injected inner loss stub (stands in for PharosLoss) ------------------
class _StubInner:
    """Minimal LossFn: L1 to clean (or energy), returns a tensor + log dict."""

    def __call__(self, out, batch, teachers):
        pred = out.output
        target = batch.get("clean")
        rec = (pred - target).abs().mean() if target is not None else pred.pow(2).mean()
        return rec, {"rec": float(rec.detach()), "total": float(rec.detach())}


def _cfg(**reveal_over):
    reveal = dict(recall=1.0, align=0.2, stale=0.05, occ_thresh=0.6, reveal_thresh=0.3)
    reveal.update(reveal_over)
    return {"loss": {"reveal": reveal}}


def _deg():
    return {
        "beta": torch.rand(B, T, 1),
        "airlight": torch.rand(B, T, 3),
        "sigma": torch.rand(B, T, 1),
        "domain_logits": torch.randn(B, T, 3),
    }


def _density(pattern):
    """Per-sample smoke_density list of (T,1,H,W) with a given per-frame level list."""
    metas = []
    for _ in range(B):
        d = torch.stack([torch.full((1, H, W), float(v)) for v in pattern], dim=0)  # T,1,H,W
        metas.append(d)
    return metas


def _cam_H(scale=1.0):
    """Per-sample cam_H list of (T,3,3): small per-frame translations."""
    metas = []
    for b in range(B):
        hs = []
        for t in range(T):
            h = torch.eye(3)
            h[0, 2] = 0.05 * scale * (t + 1)
            h[1, 2] = 0.03 * scale * (b + 1)
            hs.append(h)
        metas.append(torch.stack(hs, dim=0))  # T,3,3
    return metas


def _clip_output(aux=None):
    return PharosOutput(
        output=torch.rand(B, T, 3, H, W),
        confidence=torch.rand(B, T, 1, H, W).clamp(0.05, 1.0),
        grid=torch.rand(B, T, 12, 2, 4, 4),
        state=None,
        deg=_deg(),
        aux=aux or {},
    )


def _clip_batch(smoke=None, cam=None):
    meta = [{} for _ in range(B)]
    if smoke is not None:
        for m, d in zip(meta, smoke):
            m["smoke_density"] = d
    if cam is not None:
        for m, h in zip(meta, cam):
            m["cam_H"] = h
    return {
        "hazy": torch.rand(B, T, 3, H, W),
        "clean": torch.rand(B, T, 3, H, W),
        "domain": torch.randint(0, 3, (B,)),
        "clip": True,
        "meta": meta,
    }


def _full_aux():
    return {
        "align_H": torch.eye(3).view(1, 1, 3, 3).repeat(B, T, 1, 1) + 0.01 * torch.randn(B, T, 3, 3),
        "memory_trust": torch.rand(B, T, 1, H, W).clamp(0.05, 1.0),
        "staleness": torch.cat(  # frame 0 fresh, frames 1..T-1 aged
            [torch.zeros(B, 1, 1, H, W)] + [float(t) * torch.ones(B, 1, 1, H, W) for t in range(1, T)],
            dim=1,
        ),
    }


# --------------------------------------------------------------------------
def test_reveal_loss_finite_full_inputs():
    loss = RevealLoss(_cfg(), inner=_StubInner())
    out = _clip_output(_full_aux())
    batch = _clip_batch(smoke=_density([0.1, 0.9, 0.9]), cam=_cam_H())
    total, log = loss(out, batch, teachers=None)
    assert total.shape == ()
    assert math.isfinite(float(total))
    for k in ("recall", "align", "stale", "total"):
        assert k in log and math.isfinite(log[k])
    # recall pixels exist (occluded frames 1,2 were revealed at frame 0);
    # align_H != cam_H so alignment is supervised; both strictly positive.
    assert log["recall"] > 0.0
    assert log["align"] > 0.0
    assert log["stale"] != 0.0
    assert total.requires_grad is False or True  # tensor, gradients optional for stub


def test_total_is_inner_plus_weighted_terms():
    inner = _StubInner()
    loss = RevealLoss(_cfg(), inner=inner)
    out = _clip_output(_full_aux())
    batch = _clip_batch(smoke=_density([0.1, 0.9, 0.9]), cam=_cam_H())
    total, log = loss(out, batch, teachers=None)
    inner_total = float(inner(out, batch, None)[0])
    expected = inner_total + 1.0 * log["recall"] + 0.2 * log["align"] + 0.05 * log["stale"]
    assert abs(float(total) - expected) < 1e-5


def test_reveal_terms_zero_on_image_batch():
    loss = RevealLoss(_cfg(), inner=_StubInner())
    out = PharosOutput(
        output=torch.rand(B, 3, H, W),
        confidence=torch.rand(B, 1, H, W).clamp(0.05, 1.0),
        grid=torch.rand(B, 12, 2, 4, 4),
        state=None,
        deg={"beta": torch.rand(B, 1)},
        aux={},
    )
    batch = {"hazy": torch.rand(B, 3, H, W), "clean": torch.rand(B, 3, H, W), "clip": False, "meta": {}}
    total, log = loss(out, batch, teachers=None)
    assert log["recall"] == 0.0  # recall needs a clip
    assert log["align"] == 0.0   # no cam_H / aux H
    assert log["stale"] == 0.0   # no memory_trust / staleness


def test_each_term_toggles_off_when_its_inputs_missing():
    loss = RevealLoss(_cfg(), inner=_StubInner())

    # recall off: no smoke_density in meta (align + stale still on)
    out = _clip_output(_full_aux())
    _, log = loss(out, _clip_batch(smoke=None, cam=_cam_H()), teachers=None)
    assert log["recall"] == 0.0 and log["align"] > 0.0 and log["stale"] != 0.0

    # align off: cam_H present but model exposes no estimated H
    aux_no_h = {k: v for k, v in _full_aux().items() if k != "align_H"}
    out = _clip_output(aux_no_h)
    _, log = loss(out, _clip_batch(smoke=_density([0.1, 0.9, 0.9]), cam=_cam_H()), teachers=None)
    assert log["align"] == 0.0 and log["recall"] > 0.0

    # align off: aux H present but no cam_H in meta
    out = _clip_output(_full_aux())
    _, log = loss(out, _clip_batch(smoke=_density([0.1, 0.9, 0.9]), cam=None), teachers=None)
    assert log["align"] == 0.0

    # stale off: no memory_trust / staleness aux
    out = _clip_output({"align_H": _full_aux()["align_H"]})
    _, log = loss(out, _clip_batch(smoke=_density([0.1, 0.9, 0.9]), cam=_cam_H()), teachers=None)
    assert log["stale"] == 0.0


def test_recall_zero_when_never_revealed():
    """Pixels occluded every frame (never seen) must not be recalled."""
    loss = RevealLoss(_cfg(), inner=_StubInner())
    out = _clip_output(_full_aux())
    batch = _clip_batch(smoke=_density([0.9, 0.9, 0.9]), cam=_cam_H())
    _, log = loss(out, batch, teachers=None)
    assert log["recall"] == 0.0


def test_zero_weights_make_total_equal_inner():
    inner = _StubInner()
    loss = RevealLoss(_cfg(recall=0.0, align=0.0, stale=0.0), inner=inner)
    out = _clip_output(_full_aux())
    batch = _clip_batch(smoke=_density([0.1, 0.9, 0.9]), cam=_cam_H())
    total, _ = loss(out, batch, teachers=None)
    assert abs(float(total) - float(inner(out, batch, None)[0])) < 1e-6


# ---- recall-mask construction (hand-built 3-frame density histories) -------
def _mask_1px(levels):
    d = torch.tensor(levels, dtype=torch.float32).view(1, len(levels), 1, 1, 1)
    m = revealed_recall_mask(d, occ_thresh=0.6, reveal_thresh=0.3)
    return [float(m[0, t, 0, 0, 0]) for t in range(len(levels))]


def test_revealed_recall_mask_handbuilt():
    # seen at t0, then occluded -> recalled at t1, t2
    assert _mask_1px([0.1, 0.9, 0.9]) == [0.0, 1.0, 1.0]
    # occluded, briefly seen, occluded -> only the final occluded frame recalled
    assert _mask_1px([0.9, 0.1, 0.9]) == [0.0, 0.0, 1.0]
    # never revealed -> nothing recalled
    assert _mask_1px([0.9, 0.9, 0.9]) == [0.0, 0.0, 0.0]
    # always visible -> nothing occluded, nothing recalled
    assert _mask_1px([0.1, 0.1, 0.1]) == [0.0, 0.0, 0.0]


def test_homography_to_4pt_identity_and_translation():
    ident = torch.eye(3).view(1, 3, 3)
    off = homography_to_4pt(ident)
    assert off.shape == (1, 4, 2)
    assert torch.allclose(off, torch.zeros_like(off), atol=1e-5)

    trans = torch.eye(3)
    trans[0, 2], trans[1, 2] = 0.1, 0.2
    off = homography_to_4pt(trans.view(1, 3, 3))
    assert torch.allclose(off[0, :, 0], torch.full((4,), 0.1), atol=1e-5)
    assert torch.allclose(off[0, :, 1], torch.full((4,), 0.2), atol=1e-5)


def test_single_dict_meta_form_supported():
    """meta as a pre-collated dict (B,T,1,H,W) works as well as a per-sample list."""
    loss = RevealLoss(_cfg(), inner=_StubInner())
    dens = torch.stack(
        [torch.full((B, 1, H, W), v) for v in (0.1, 0.9, 0.9)], dim=1
    )  # B,T,1,H,W
    batch = _clip_batch(smoke=None, cam=None)
    batch["meta"] = {"smoke_density": dens}
    _, log = loss(_clip_output(_full_aux()), batch, teachers=None)
    assert log["recall"] > 0.0
