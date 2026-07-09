"""Tests for pharos.rt.overlay (pure numpy/cv2)."""
from __future__ import annotations

import numpy as np

from pharos.rt.overlay import (
    ViewState,
    apply_confidence_tint,
    compose,
    confidence_heatmap,
    draw_hud,
    side_by_side,
    split_view,
)


def _frame(h: int = 60, w: int = 80) -> np.ndarray:
    return np.random.randint(0, 256, size=(h, w, 3), dtype=np.uint8)


def _conf(h: int = 60, w: int = 80) -> np.ndarray:
    return np.random.rand(h, w).astype(np.float32)


def _result(h: int = 60, w: int = 80) -> dict:
    return {
        "output_bgr": _frame(h, w),
        "confidence": _conf(h, w),
        "deg": {"domain_name": "smoke", "domain": 1, "beta": 0.42, "sigma": 0.12},
        "gate_alpha": 0.66,
        "timings": {"fps_avg": 31.5, "fps": 30.0},
    }


def test_apply_confidence_tint_shape_dtype():
    frame = _frame()
    out = apply_confidence_tint(frame, _conf(), threshold=0.5)
    assert out.dtype == np.uint8
    assert out.shape == frame.shape
    assert frame.shape == _frame().shape  # sanity


def test_apply_confidence_tint_resizes_mismatched_conf():
    frame = _frame(60, 80)
    conf = _conf(30, 40)  # half res
    out = apply_confidence_tint(frame, conf)
    assert out.shape == frame.shape and out.dtype == np.uint8


def test_apply_confidence_tint_actually_tints_low_conf():
    frame = np.zeros((10, 10, 3), dtype=np.uint8)
    conf = np.zeros((10, 10), dtype=np.float32)  # all below threshold -> all tinted red
    out = apply_confidence_tint(frame, conf, threshold=0.5, alpha=0.5, color=(0, 0, 255))
    assert out[..., 2].max() > 0  # red channel raised
    # High-confidence region should be untouched.
    conf2 = np.ones((10, 10), dtype=np.float32)
    out2 = apply_confidence_tint(frame, conf2, threshold=0.5)
    assert int(out2.max()) == 0


def test_confidence_heatmap():
    out = confidence_heatmap(_conf(40, 50))
    assert out.dtype == np.uint8
    assert out.shape == (40, 50, 3)


def test_draw_hud():
    frame = _frame()
    out = draw_hud(frame, _result(), fps=29.9)
    assert out.dtype == np.uint8 and out.shape == frame.shape


def test_draw_hud_empty_is_noop_copy():
    frame = _frame()
    out = draw_hud(frame, result=None, fps=None)
    assert out.shape == frame.shape
    assert np.array_equal(out, frame) and out is not frame


def test_side_by_side():
    a, b = _frame(60, 80), _frame(40, 50)
    out = side_by_side(a, b)
    assert out.dtype == np.uint8
    assert out.shape[0] == a.shape[0]
    assert out.shape[1] > a.shape[1]  # wider than a single frame


def test_split_view():
    a, b = _frame(60, 80), _frame(60, 80)
    out = split_view(a, b, split=0.5)
    assert out.dtype == np.uint8 and out.shape == a.shape


def test_compose_all_view_modes():
    frame = _frame()
    result = _result()
    for view in (
        ViewState(overlay=True),
        ViewState(overlay=False),
        ViewState(overlay=True, confidence_view=True),
        ViewState(overlay=True, split=True),
    ):
        out = compose(frame, result, view, threshold=0.5, fps=30.0)
        assert out.dtype == np.uint8
        assert out.ndim == 3 and out.shape[2] == 3
