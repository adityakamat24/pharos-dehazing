"""Staleness rendering for the RevealNet demo (pure numpy / cv2, DESIGN.md §9d.3).

RevealNet composites the current restoration with remembered content and emits a
*staleness map*: seconds since each pixel was last directly confirmed (0 = fresh
this frame, larger = older memory). These helpers visualise it in the same style
as :mod:`pharos.rt.overlay`:

- :func:`apply_staleness_shade` — subtle desaturation + blue tint on remembered
  pixels (staleness > 0), scaled by age so the oldest memory reads coldest/greyest.
- :func:`draw_staleness_contours` — thin iso-age contour lines at 2 s / 5 s / 10 s.
- :func:`staleness_hud_line` / :func:`staleness_stats` — the ``memory: X% of view,
  oldest Ys`` HUD string and its underlying numbers.
- :func:`render_staleness` — composes the three (optionally with the standard
  confidence tint) into one display frame.

All functions take and return uint8 HxWx3 BGR frames and never touch torch.
"""
from __future__ import annotations

from typing import Optional, Sequence

import cv2
import numpy as np

from pharos.rt.overlay import apply_confidence_tint, draw_hud

# BGR. A desaturated cold blue for remembered pixels; contours a brighter cyan.
_MEMORY_TINT = (200, 130, 40)
_CONTOUR_COLOR = (255, 210, 120)
_DEFAULT_LEVELS = (2.0, 5.0, 10.0)


def _resize_map(m: np.ndarray, hw: tuple[int, int]) -> np.ndarray:
    if m.shape[:2] != hw:
        m = cv2.resize(m, (hw[1], hw[0]), interpolation=cv2.INTER_LINEAR)
    return m


def apply_staleness_shade(
    frame_bgr: np.ndarray,
    staleness: np.ndarray,
    max_seconds: float = 10.0,
    max_desat: float = 0.6,
    max_tint: float = 0.35,
    tint: Sequence[int] = _MEMORY_TINT,
) -> np.ndarray:
    """Desaturate + blue-tint remembered pixels (``staleness > 0``) by age.

    ``staleness`` is a float HxW map in seconds (resized to the frame if needed).
    The shade strength ramps linearly from 0 at ``staleness=0`` to full at
    ``staleness>=max_seconds``. Fresh pixels (staleness 0) are left untouched.
    Returns a new uint8 HxWx3 BGR frame (input is not mutated).
    """
    h, w = frame_bgr.shape[:2]
    st = _resize_map(staleness.astype(np.float32), (h, w))
    remembered = (st > 0.0).astype(np.float32)
    age = np.clip(st / max(float(max_seconds), 1e-6), 0.0, 1.0) * remembered  # HxW in [0,1]

    base = frame_bgr.astype(np.float32)
    gray = base.mean(axis=2, keepdims=True)  # HxWx1 luminance proxy
    d = (max_desat * age)[..., None]
    desat = base * (1.0 - d) + gray * d
    t = (max_tint * age)[..., None]
    tinted = desat * (1.0 - t) + np.asarray(tint, dtype=np.float32) * t

    out = np.where((remembered > 0.0)[..., None], tinted, base)
    return np.clip(out, 0, 255).astype(np.uint8)


def draw_staleness_contours(
    frame_bgr: np.ndarray,
    staleness: np.ndarray,
    levels: Sequence[float] = _DEFAULT_LEVELS,
    color: Sequence[int] = _CONTOUR_COLOR,
    thickness: int = 1,
    label: bool = True,
) -> np.ndarray:
    """Draw thin contour lines at each staleness isolevel (seconds).

    Each level's contour bounds the region at least that old. Returns a new frame
    (input copied); a no-op copy when no pixel reaches the smallest level.
    """
    h, w = frame_bgr.shape[:2]
    st = _resize_map(staleness.astype(np.float32), (h, w))
    out = frame_bgr.copy()
    col = tuple(int(c) for c in color)
    for lv in sorted(levels):
        binary = (st >= float(lv)).astype(np.uint8) * 255
        if int(binary.max()) == 0:
            continue
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        cv2.drawContours(out, contours, -1, col, thickness, cv2.LINE_AA)
        if label:
            pt = max(contours, key=cv2.contourArea)[0][0]
            cv2.putText(out, f"{lv:g}s", (int(pt[0]) + 2, int(pt[1]) - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, col, 1, cv2.LINE_AA)
    return out


def staleness_stats(staleness: np.ndarray) -> tuple[float, float]:
    """Return ``(percent_of_view_remembered, oldest_seconds)`` for a staleness map."""
    st = staleness.astype(np.float32)
    remembered = st > 0.0
    pct = 100.0 * float(remembered.mean()) if remembered.size else 0.0
    oldest = float(st[remembered].max()) if bool(remembered.any()) else 0.0
    return pct, oldest


def staleness_hud_line(staleness: np.ndarray) -> str:
    """Format the ``memory: X% of view, oldest Ys`` HUD line."""
    pct, oldest = staleness_stats(staleness)
    return f"memory: {pct:.0f}% of view, oldest {oldest:.0f}s"


def render_staleness(
    frame_bgr: np.ndarray,
    staleness: np.ndarray,
    confidence: Optional[np.ndarray] = None,
    *,
    max_seconds: float = 10.0,
    contours: bool = True,
    levels: Sequence[float] = _DEFAULT_LEVELS,
    hud: bool = True,
    conf_threshold: Optional[float] = None,
) -> np.ndarray:
    """Compose the full staleness overlay for the demo.

    Applies the age shade, then (optionally) the standard confidence tint when
    both ``confidence`` and ``conf_threshold`` are given, the iso-age contours,
    and the ``memory:`` HUD line. Returns a uint8 HxWx3 BGR frame.
    """
    out = apply_staleness_shade(frame_bgr, staleness, max_seconds=max_seconds)
    if confidence is not None and conf_threshold is not None:
        out = apply_confidence_tint(out, confidence, threshold=float(conf_threshold))
    if contours:
        out = draw_staleness_contours(out, staleness, levels=levels)
    if hud:
        out = draw_hud(out, extra_lines=[staleness_hud_line(staleness)])
    return out
