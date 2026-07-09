"""Tests for pharos.models.temporal (WS-A)."""
import pathlib
import sys

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from pharos.models.temporal import ConvGRUCell, TemporalModule  # noqa: E402


def test_convgru_cell_shapes_and_bounds():
    cell = ConvGRUCell(8, 6).eval()
    x = torch.rand(2, 8, 5, 5)
    h = torch.zeros(2, 6, 5, 5)
    h1 = cell(x, h)
    assert h1.shape == (2, 6, 5, 5)
    assert torch.isfinite(h1).all()


def _inputs(b=1, grid_ch_depth=(12, 8), feat_ch=16, gh=16):
    coeffs, d = grid_ch_depth
    grid = torch.rand(b, coeffs, d, gh, gh)
    feat = torch.rand(b, feat_ch, 8, 8)
    conf = torch.rand(b, 1, 32, 32)
    return grid, feat, conf


def test_temporal_state_threading_shapes():
    mod = TemporalModule(grid_ch=12 * 8, feat_ch=16).eval()
    grid, feat, conf = _inputs()
    frame = torch.rand(1, 3, 24, 24)
    sm1, st1 = mod(grid, feat, conf, frame, None)
    assert sm1.shape == grid.shape
    assert set(st1.keys()) == {"h", "ema", "hist"}
    sm2, st2 = mod(grid, feat, conf, frame, st1)
    assert sm2.shape == grid.shape


def test_scene_cut_resets_state():
    torch.manual_seed(0)
    mod = TemporalModule(grid_ch=12 * 8, feat_ch=16, scene_thresh=0.5).eval()
    g1, feat, conf = _inputs()
    frame_a = torch.zeros(1, 3, 24, 24)
    _, st1 = mod(g1, feat, conf, frame_a, None)

    g2, _, _ = _inputs()
    frame_b = torch.ones(1, 3, 24, 24)  # histogram maximally different -> scene cut
    sm_cut, _ = mod(g2, feat, conf, frame_b, st1)
    sm_fresh, _ = mod(g2, feat, conf, frame_b, None)
    # On a cut the state is reset, so continuing == a fresh start.
    assert torch.allclose(sm_cut, sm_fresh, atol=1e-6)


def test_no_scene_cut_uses_history():
    torch.manual_seed(0)
    mod = TemporalModule(grid_ch=12 * 8, feat_ch=16, scene_thresh=0.5).eval()
    g1, feat, conf = _inputs()
    frame = torch.rand(1, 3, 24, 24)
    _, st1 = mod(g1, feat, conf, frame, None)
    g2, _, _ = _inputs()
    sm_cont, _ = mod(g2, feat, conf, frame, st1)  # same frame -> no cut, blends history
    sm_fresh, _ = mod(g2, feat, conf, frame, None)
    assert not torch.allclose(sm_cont, sm_fresh, atol=1e-4)
