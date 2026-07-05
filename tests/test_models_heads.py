"""CPU tests for pharos.models.heads.DegradationHead (item 5: split backscatter beta).

Focus: the new ``beta_bs`` output channel, its FiLM-conditioning decision (kept OUT
of the cond vector to preserve PharosNet checkpoint shapes), and old-checkpoint
loadability via a separate small head.
"""
from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import torch

from pharos.models.heads import DegradationHead

IN_CH = 96


def test_deg_head_outputs_beta_bs():
    head = DegradationHead(IN_CH)
    feat = torch.rand(4, IN_CH, 8, 8)
    deg, cond = head(feat)
    for k in ("beta", "beta_bs", "airlight", "sigma", "domain_logits"):
        assert k in deg
    assert deg["beta_bs"].shape == (4, 1)
    assert float(deg["beta_bs"].detach().min()) >= 0.0  # softplus -> non-negative


def test_cond_dim_unchanged_by_beta_bs():
    # beta_bs must NOT be appended to the conditioning vector: that would change
    # cond_dim and break FiLM weight shapes / PharosNet checkpoint compat.
    head = DegradationHead(IN_CH, embed=32, domain_embed=8)
    assert head.cond_dim == 32 + 1 + 3 + 1 + 8
    _, cond = head(torch.rand(2, IN_CH, 8, 8))
    assert cond.shape == (2, head.cond_dim)


def test_old_checkpoint_loads_into_new_head():
    # simulate a pre-batch1 checkpoint = new head's state dict minus the beta_bs keys
    head = DegradationHead(IN_CH)
    full = head.state_dict()
    old = {k: v.clone() for k, v in full.items() if not k.startswith("head_beta_bs")}

    fresh = DegradationHead(IN_CH)
    missing, unexpected = fresh.load_state_dict(old, strict=False)
    assert set(missing) == {"head_beta_bs.weight", "head_beta_bs.bias"}
    assert list(unexpected) == []  # nothing stale -> old weights map exactly
    # every shared weight loaded byte-for-byte
    for k, v in old.items():
        assert torch.equal(fresh.state_dict()[k], v)


def test_deg_head_backward_runs():
    head = DegradationHead(IN_CH)
    feat = torch.rand(2, IN_CH, 8, 8, requires_grad=True)
    deg, _ = head(feat)
    (deg["beta"].mean() + deg["beta_bs"].mean()).backward()
    assert feat.grad is not None and torch.isfinite(feat.grad).all()
    assert head.head_beta_bs.weight.grad is not None
