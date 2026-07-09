"""CPU tests for pharos.engine.reveal_metrics (constructed oracle cases)."""
from __future__ import annotations

import math
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import torch

from pharos.engine.reveal_metrics import psnr_over_time, recall_curve, time_to_recover


def _gen(seed: int) -> torch.Generator:
    return torch.Generator().manual_seed(seed)


def _toy_case(t: int = 6, h: int = 8, w: int = 8):
    """Static scene; left half revealed early then occluded, right half the opposite.

    From frame >= t//2 the left half is occluded-now AND was-revealed-earlier, so it
    forms the recall region that only a model with memory can reconstruct.
    """
    gt_frame = 0.4 * torch.rand(3, h, w, generator=_gen(0))  # kept in [0, 0.4]
    gt = gt_frame.unsqueeze(0).repeat(t, 1, 1, 1)
    half = w // 2
    density = torch.zeros(t, 1, h, w)
    for i in range(t):
        if i < t // 2:
            density[i, :, :, :half] = 0.0  # left revealed
            density[i, :, :, half:] = 1.0  # right occluded
        else:
            density[i, :, :, :half] = 1.0  # left now occluded (recall region)
            density[i, :, :, half:] = 0.0  # right revealed
    return gt, density


def _memory_oracle(gt: torch.Tensor) -> torch.Tensor:
    return (gt + 1e-3).clamp(0, 1)  # remembers everything (tiny error)


def _no_memory_oracle(gt: torch.Tensor, density: torch.Tensor, thresh: float = 0.5) -> torch.Tensor:
    """Shows the true scene only where currently visible; occluded -> wrong smoke value."""
    out = gt.clone()
    occ = (density > thresh).expand_as(out)
    out[occ] = (gt[occ] + 0.5).clamp(0, 1)  # RMS error 0.5 >> tol on occluded pixels
    return out


# ---------------------------------------------------------------------------
# psnr_over_time
# ---------------------------------------------------------------------------
def test_psnr_over_time_length_and_perfect():
    gt, _ = _toy_case()
    pts = psnr_over_time(gt, gt)
    assert len(pts) == gt.shape[0]
    assert all(math.isinf(p) for p in pts)  # identical -> inf per frame


def test_psnr_over_time_orders_oracles():
    gt, density = _toy_case()
    mem = psnr_over_time(_memory_oracle(gt), gt)
    nomem = psnr_over_time(_no_memory_oracle(gt, density), gt)
    assert sum(mem) / len(mem) > sum(nomem) / len(nomem)


# ---------------------------------------------------------------------------
# recall_curve — the reveal metric
# ---------------------------------------------------------------------------
def test_recall_region_is_nonempty_late():
    gt, density = _toy_case()
    rc = recall_curve(_memory_oracle(gt), gt, density)
    # first frame has no "revealed earlier" history -> empty region (nan)
    assert math.isnan(rc["frac_correct"][0])
    # later frames must have a real recall region (left half became occluded)
    assert rc["frac_region"][-1] > 0.0


def test_recall_curve_rewards_memory_over_no_memory():
    gt, density = _toy_case()
    rc_mem = recall_curve(_memory_oracle(gt), gt, density)
    rc_nomem = recall_curve(_no_memory_oracle(gt, density), gt, density)
    # compare on the last frame where the recall region is populated
    assert rc_mem["frac_correct"][-1] > 0.9
    assert rc_nomem["frac_correct"][-1] < 0.1
    assert rc_mem["psnr_recall"][-1] > rc_nomem["psnr_recall"][-1]


def test_recall_curve_shape_validation():
    gt, density = _toy_case()
    import pytest

    with pytest.raises(ValueError):
        recall_curve(gt[:, :, :4], gt, density)  # spatial mismatch


# ---------------------------------------------------------------------------
# time_to_recover
# ---------------------------------------------------------------------------
def test_time_to_recover_orders_oracles():
    gt, density = _toy_case()
    mem = time_to_recover(_memory_oracle(gt), gt, density)
    nomem = time_to_recover(_no_memory_oracle(gt, density), gt, density)
    assert mem["mean_frac_correct"] > nomem["mean_frac_correct"]
    assert mem["mean_recall_psnr"] > nomem["mean_recall_psnr"]
    assert mem["final_frac_correct"] > 0.9 and nomem["final_frac_correct"] < 0.1


def test_time_to_recover_reveal_time_reasonable():
    gt, density = _toy_case(t=6)
    scal = time_to_recover(_memory_oracle(gt), gt, density)
    # every pixel is revealed at some frame -> finite mean first-reveal time in [0, T)
    assert 0.0 <= scal["reveal_time_mean"] < 6.0
    assert not math.isnan(scal["recall_region_frac_final"])
