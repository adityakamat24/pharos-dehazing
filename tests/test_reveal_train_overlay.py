"""Tests for pharos.rt.reveal_overlay (pure numpy/cv2 staleness rendering)."""
from __future__ import annotations

import numpy as np

from pharos.rt.reveal_overlay import (
    apply_staleness_shade,
    draw_staleness_contours,
    render_staleness,
    staleness_hud_line,
    staleness_stats,
)


def _frame(h: int = 60, w: int = 80) -> np.ndarray:
    return np.random.randint(0, 256, size=(h, w, 3), dtype=np.uint8)


def _staleness(h: int = 60, w: int = 80) -> np.ndarray:
    """Half fresh (0 s), half remembered (ramping seconds)."""
    st = np.zeros((h, w), dtype=np.float32)
    st[:, w // 2:] = np.linspace(0.5, 12.0, w - w // 2, dtype=np.float32)[None, :]
    return st


def test_apply_staleness_shade_shape_dtype():
    frame = _frame()
    out = apply_staleness_shade(frame, _staleness())
    assert out.dtype == np.uint8 and out.shape == frame.shape


def test_shade_leaves_fresh_pixels_untouched():
    frame = _frame(20, 20)
    st = np.zeros((20, 20), dtype=np.float32)  # nothing remembered
    out = apply_staleness_shade(frame, st)
    assert np.array_equal(out, frame)


def test_shade_changes_remembered_pixels():
    frame = np.full((20, 20, 3), 200, dtype=np.uint8)
    st = np.full((20, 20), 12.0, dtype=np.float32)  # all old memory
    out = apply_staleness_shade(frame, st, max_seconds=10.0)
    assert not np.array_equal(out, frame)  # desaturated + tinted everywhere


def test_shade_resizes_mismatched_map():
    frame = _frame(60, 80)
    st = _staleness(30, 40)  # half res
    out = apply_staleness_shade(frame, st)
    assert out.shape == frame.shape and out.dtype == np.uint8


def test_draw_contours_shape_and_noop():
    frame = _frame()
    out = draw_staleness_contours(frame, _staleness(), levels=(2.0, 5.0, 10.0))
    assert out.dtype == np.uint8 and out.shape == frame.shape
    # no pixel reaches any level -> unchanged copy
    noop = draw_staleness_contours(frame, np.zeros(frame.shape[:2], np.float32))
    assert np.array_equal(noop, frame) and noop is not frame


def test_draw_contours_actually_draws():
    frame = np.zeros((40, 40, 3), dtype=np.uint8)
    st = np.zeros((40, 40), dtype=np.float32)
    st[10:30, 10:30] = 6.0  # a block older than the 5s level
    out = draw_staleness_contours(frame, st, levels=(5.0,), label=False)
    assert int(out.max()) > 0  # a contour line was drawn


def test_staleness_stats_and_hud_line():
    st = np.zeros((10, 10), dtype=np.float32)
    st[:, 5:] = 4.0  # half the view remembered, oldest 4s
    pct, oldest = staleness_stats(st)
    assert abs(pct - 50.0) < 1e-3
    assert abs(oldest - 4.0) < 1e-6
    line = staleness_hud_line(st)
    assert line == "memory: 50% of view, oldest 4s"


def test_staleness_stats_empty():
    st = np.zeros((8, 8), dtype=np.float32)
    pct, oldest = staleness_stats(st)
    assert pct == 0.0 and oldest == 0.0


def test_render_staleness_composes():
    frame = _frame()
    st = _staleness()
    conf = np.random.rand(*frame.shape[:2]).astype(np.float32)
    out = render_staleness(frame, st, confidence=conf, conf_threshold=0.5)
    assert out.dtype == np.uint8
    assert out.ndim == 3 and out.shape[2] == 3
    assert out.shape[:2] == frame.shape[:2]


def test_render_staleness_minimal_flags():
    frame = _frame()
    out = render_staleness(frame, _staleness(), contours=False, hud=False)
    assert out.dtype == np.uint8 and out.shape == frame.shape
