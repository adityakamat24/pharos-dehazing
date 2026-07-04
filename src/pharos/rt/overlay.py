"""Confidence + status rendering for the Pharos demo (pure numpy / cv2).

All functions take and return uint8 HxWx3 BGR frames and never touch torch, so they
are trivially unit-testable. The demo composes them via :func:`compose` driven by a
:class:`ViewState` toggled with keyboard shortcuts.

Rendering pieces:
- :func:`apply_confidence_tint` — semi-transparent red where confidence < threshold.
- :func:`confidence_heatmap` — colourised trust map (the ``c`` view).
- :func:`draw_hud` — small HUD: estimated density beta, domain, gate alpha, FPS.
- :func:`side_by_side` / :func:`split_view` — before/after comparison layouts.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import cv2
import numpy as np

_RED = (0, 0, 255)  # BGR
_WHITE = (255, 255, 255)
_PANEL = (20, 20, 20)
_FONT = cv2.FONT_HERSHEY_SIMPLEX


@dataclass
class ViewState:
    """Toggle state for the interactive demo (keys: o / c / b)."""

    overlay: bool = True
    confidence_view: bool = False
    split: bool = False


def _match_hw(img: np.ndarray, hw: tuple[int, int]) -> np.ndarray:
    if img.shape[:2] != hw:
        img = cv2.resize(img, (hw[1], hw[0]), interpolation=cv2.INTER_LINEAR)
    return img


def apply_confidence_tint(
    frame_bgr: np.ndarray,
    confidence: np.ndarray,
    threshold: float = 0.5,
    alpha: float = 0.45,
    color: Sequence[int] = _RED,
) -> np.ndarray:
    """Blend a translucent ``color`` where ``confidence < threshold`` (untrustworthy regions).

    ``confidence`` is a float HxW map in [0, 1]; it is resized to the frame if needed.
    Returns a new uint8 HxWx3 BGR frame (input is not mutated).
    """
    h, w = frame_bgr.shape[:2]
    conf = confidence.astype(np.float32)
    if conf.shape[:2] != (h, w):
        conf = cv2.resize(conf, (w, h), interpolation=cv2.INTER_LINEAR)
    mask = (conf < float(threshold))[..., None]
    base = frame_bgr.astype(np.float32)
    tint = np.empty_like(base)
    tint[:] = np.asarray(color, dtype=np.float32)
    blended = base * (1.0 - alpha) + tint * alpha
    out = np.where(mask, blended, base)
    return np.clip(out, 0, 255).astype(np.uint8)


def confidence_heatmap(confidence: np.ndarray, colormap: int = cv2.COLORMAP_TURBO) -> np.ndarray:
    """Colourise a float HxW confidence map in [0, 1] into a uint8 BGR heatmap.

    Warmer = higher confidence (more trustworthy).
    """
    conf = np.clip(confidence.astype(np.float32), 0.0, 1.0)
    u8 = (conf * 255.0).round().astype(np.uint8)
    return cv2.applyColorMap(u8, colormap)


def draw_hud(
    frame_bgr: np.ndarray,
    result: Optional[dict] = None,
    fps: Optional[float] = None,
    extra_lines: Optional[Sequence[str]] = None,
    scale: float = 0.5,
) -> np.ndarray:
    """Draw a small translucent HUD panel (top-left) with degradation + speed stats.

    ``result`` is a :meth:`StreamingRestorer.restore` dict; missing keys are skipped.
    Returns a new frame (input is copied).
    """
    lines: list[str] = []
    if fps is not None:
        lines.append(f"FPS {fps:5.1f}")
    if result is not None:
        deg = result.get("deg", {})
        if "domain_name" in deg:
            lines.append(f"domain {deg['domain_name']}")
        if "beta" in deg:
            lines.append(f"beta(density) {deg['beta']:.3f}")
        if "sigma" in deg:
            lines.append(f"sigma(nonhom) {deg['sigma']:.3f}")
        if "gate_alpha" in result:
            lines.append(f"gate alpha {result['gate_alpha']:.2f}")
    if extra_lines:
        lines.extend(extra_lines)
    if not lines:
        return frame_bgr.copy()

    out = frame_bgr.copy()
    thickness = max(1, int(round(scale * 2.6)))
    sizes = [cv2.getTextSize(t, _FONT, scale, thickness)[0] for t in lines]
    line_h = max(s[1] for s in sizes) + max(6, int(10 * scale))
    pad = max(4, int(8 * scale))
    panel_w = max(s[0] for s in sizes) + 2 * pad
    panel_h = line_h * len(lines) + pad
    panel_w = min(panel_w, out.shape[1])
    panel_h = min(panel_h, out.shape[0])

    roi = out[:panel_h, :panel_w].astype(np.float32)
    panel = np.empty_like(roi)
    panel[:] = np.asarray(_PANEL, dtype=np.float32)
    out[:panel_h, :panel_w] = (roi * 0.4 + panel * 0.6).astype(np.uint8)

    y = pad + sizes[0][1]
    for text in lines:
        cv2.putText(out, text, (pad, y), _FONT, scale, _WHITE, thickness, cv2.LINE_AA)
        y += line_h
    return out


def _label(img: np.ndarray, text: str, scale: float = 0.6) -> None:
    thickness = max(1, int(round(scale * 2.6)))
    (tw, th), _ = cv2.getTextSize(text, _FONT, scale, thickness)
    pad = 6
    cv2.rectangle(img, (0, 0), (tw + 2 * pad, th + 2 * pad), _PANEL, -1)
    cv2.putText(img, text, (pad, th + pad), _FONT, scale, _WHITE, thickness, cv2.LINE_AA)


def side_by_side(
    before_bgr: np.ndarray,
    after_bgr: np.ndarray,
    labels: Optional[tuple[str, str]] = ("Input", "Restored"),
    gap: int = 4,
) -> np.ndarray:
    """Horizontally concatenate two frames (after is resized to before's height)."""
    h = before_bgr.shape[0]
    after = _match_hw(after_bgr, (h, int(after_bgr.shape[1] * h / after_bgr.shape[0])))
    left, right = before_bgr.copy(), after.copy()
    if labels is not None:
        _label(left, labels[0])
        _label(right, labels[1])
    if gap > 0:
        sep = np.zeros((h, gap, 3), dtype=np.uint8)
        return np.concatenate([left, sep, right], axis=1)
    return np.concatenate([left, right], axis=1)


def split_view(
    before_bgr: np.ndarray,
    after_bgr: np.ndarray,
    split: float = 0.5,
    draw_line: bool = True,
    labels: Optional[tuple[str, str]] = ("Input", "Restored"),
) -> np.ndarray:
    """Single-frame split: left ``split`` fraction is ``before``, right is ``after``."""
    h, w = before_bgr.shape[:2]
    after = _match_hw(after_bgr, (h, w))
    x = int(np.clip(split, 0.0, 1.0) * w)
    out = after.copy()
    out[:, :x] = before_bgr[:, :x]
    if draw_line and 0 < x < w:
        cv2.line(out, (x, 0), (x, h), _WHITE, 1, cv2.LINE_AA)
    if labels is not None:
        _label(out, labels[0])
        thickness = 2
        (tw, th), _ = cv2.getTextSize(labels[1], _FONT, 0.6, thickness)
        pad = 6
        rx = w - tw - 2 * pad
        cv2.rectangle(out, (rx, 0), (w, th + 2 * pad), _PANEL, -1)
        cv2.putText(out, labels[1], (rx + pad, th + pad), _FONT, 0.6, _WHITE, thickness, cv2.LINE_AA)
    return out


def compose(
    frame_bgr: np.ndarray,
    result: dict,
    view: ViewState,
    threshold: float = 0.5,
    fps: Optional[float] = None,
) -> np.ndarray:
    """Assemble the display frame from the input, a restore result, and the view toggles.

    ``result`` is a :meth:`StreamingRestorer.restore` dict. Returns uint8 BGR.
    """
    output = result["output_bgr"]
    if view.split:
        canvas = split_view(frame_bgr, output)
    elif view.confidence_view:
        canvas = _match_hw(confidence_heatmap(result["confidence"]), output.shape[:2])
    else:
        canvas = output.copy()

    if view.overlay:
        if not view.confidence_view and not view.split:
            canvas = apply_confidence_tint(canvas, result["confidence"], threshold=threshold)
        eff_fps = fps if fps is not None else result.get("timings", {}).get("fps_avg")
        canvas = draw_hud(canvas, result, fps=eff_fps)
    return canvas
