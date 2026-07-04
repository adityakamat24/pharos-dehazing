"""Streaming / batch inference wrapper for a trained :class:`PharosModel`.

``StreamingRestorer`` wraps *any* object satisfying ``pharos.contracts.PharosModel``
(the real ``pharos.models.pharosnet.PharosNet`` or a stub) and turns it into a
frame-by-frame, causal video restorer plus a batch/image API. All the annoying
plumbing lives here: BGR<->RGB, uint8<->float, padding/unpadding, device & dtype
placement, half precision (``model.half`` + autocast), reparameterize-on-load,
recurrent state threading across frames, per-scene reset, and honest per-frame
timing (with CUDA synchronisation).

The real model lives in a *parallel* workstream and is imported lazily by name via
:func:`load_model`; this module never imports ``pharos.models`` at import time so it
stays usable (and testable) with only a stub model.
"""
from __future__ import annotations

import warnings
from collections import deque
from pathlib import Path
from time import perf_counter
from typing import Any, Optional, Union

import cv2
import numpy as np
import torch

from pharos.contracts import DOMAIN_NAMES, PharosModel, PharosOutput

# ---------------------------------------------------------------------------
# Stateless pre / post helpers (reused by bench.py and export.py).
# ---------------------------------------------------------------------------


def pad_to_multiple(x: torch.Tensor, multiple: int) -> tuple[torch.Tensor, tuple[int, int]]:
    """Reflect-pad the last two dims (H, W) up to a multiple of ``multiple``.

    Returns the padded tensor and the original ``(H, W)`` so it can be undone with a
    plain slice. ``multiple <= 1`` is a no-op. Falls back to replicate/constant padding
    for inputs too small for reflection padding.
    """
    h, w = x.shape[-2], x.shape[-1]
    if multiple <= 1:
        return x, (h, w)
    ph = (multiple - h % multiple) % multiple
    pw = (multiple - w % multiple) % multiple
    if ph == 0 and pw == 0:
        return x, (h, w)
    mode = "reflect" if (h > 1 and w > 1 and ph < h and pw < w) else "replicate"
    x = torch.nn.functional.pad(x, (0, pw, 0, ph), mode=mode)
    return x, (h, w)


def bgr_to_tensor(
    frame_bgr: np.ndarray,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
    pad_multiple: int = 1,
) -> tuple[torch.Tensor, tuple[int, int]]:
    """uint8 HxWx3 BGR frame -> float ``1,3,H,W`` RGB tensor in [0, 1] on ``device``.

    Returns the (optionally padded) tensor and the original ``(H, W)`` for unpadding.
    """
    if frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3:
        raise ValueError(f"expected HxWx3 BGR frame, got shape {frame_bgr.shape}")
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    t = torch.from_numpy(np.ascontiguousarray(rgb)).to(device)
    t = t.permute(2, 0, 1).unsqueeze(0).to(dtype).div_(255.0)
    t, orig_hw = pad_to_multiple(t, pad_multiple)
    return t, orig_hw


def tensor_to_bgr(output: torch.Tensor, orig_hw: Optional[tuple[int, int]] = None) -> np.ndarray:
    """float ``1,3,H,W`` (or ``3,H,W``) RGB tensor in [0, 1] -> uint8 HxWx3 BGR frame."""
    t = output.detach()
    if t.dim() == 4:
        t = t[0]
    if orig_hw is not None:
        t = t[:, : orig_hw[0], : orig_hw[1]]
    # Out-of-place ops only: `t` may be an inference-mode tensor consumed here.
    arr = (t.float().clamp(0.0, 1.0) * 255.0).round().permute(1, 2, 0).cpu().numpy().astype(np.uint8)
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def confidence_to_map(conf: torch.Tensor, orig_hw: Optional[tuple[int, int]] = None) -> np.ndarray:
    """``1,1,H,W`` (or ``1,H,W`` / ``H,W``) confidence tensor -> float32 HxW map in [0, 1]."""
    t = conf.detach()
    while t.dim() > 2:
        t = t[0]
    if orig_hw is not None:
        t = t[: orig_hw[0], : orig_hw[1]]
    return t.float().clamp(0.0, 1.0).cpu().numpy().astype(np.float32)


def smoothstep(x: float, lo: float, hi: float) -> float:
    """Hermite smoothstep in [0, 1]; matches DESIGN §3.7 severity gate."""
    if hi <= lo:
        return 1.0 if x >= hi else 0.0
    t = min(1.0, max(0.0, (x - lo) / (hi - lo)))
    return t * t * (3.0 - 2.0 * t)


# ---------------------------------------------------------------------------
# Lazy model loader (real model lives in a parallel workstream).
# ---------------------------------------------------------------------------


def _import_pharosnet() -> Any:
    """Late-import ``PharosNet`` so this module never hard-depends on pharos.models."""
    try:
        from pharos.models import PharosNet  # type: ignore
        return PharosNet
    except Exception:
        from pharos.models.pharosnet import PharosNet  # type: ignore
        return PharosNet


def _build_from_cfg(pharosnet_cls: Any, cfg: Any) -> Any:
    """Construct a PharosNet from a config, trying the plausible constructor shapes.

    The exact constructor signature is owned by WS-A; we try the config object, then
    its ``model`` sub-tree, then a no-arg fallback so this keeps working whatever the
    final signature turns out to be.
    """
    for attempt in (lambda: pharosnet_cls(cfg), lambda: pharosnet_cls(cfg.model), lambda: pharosnet_cls()):
        try:
            return attempt()
        except Exception:
            continue
    # Re-raise the most informative failure.
    return pharosnet_cls(cfg)


def load_model(
    ckpt_path_or_cfg: Union[str, Path, dict, Any],
    device: torch.device | str = "cpu",
    model_cfg: Any = None,
) -> PharosModel:
    """Load a ``PharosModel`` from a checkpoint path or build one from a config.

    - ``str``/``Path``: treated as a checkpoint. Config is taken from ``model_cfg`` or
      from a ``config``/``cfg`` key inside the checkpoint; weights from a
      ``model``/``state_dict``/``ema``/``ema_model`` key (or the checkpoint itself).
    - anything else: treated as a config object and passed to ``PharosNet``.

    Checkpoint layout is owned by WS-D; this loader is deliberately permissive and uses
    ``strict=False`` so it survives minor key drift. Returns an ``eval()`` model.
    """
    pharosnet_cls = _import_pharosnet()
    if isinstance(ckpt_path_or_cfg, (str, Path)):
        ckpt = torch.load(str(ckpt_path_or_cfg), map_location="cpu", weights_only=False)
        if not isinstance(ckpt, dict):
            raise ValueError(f"unexpected checkpoint type: {type(ckpt)}")
        cfg = model_cfg or ckpt.get("config") or ckpt.get("cfg")
        if cfg is None:
            raise ValueError("no config in checkpoint; pass model_cfg=load_config(...)")
        model = _build_from_cfg(pharosnet_cls, cfg)
        state = None
        for key in ("ema_model", "ema", "model", "state_dict"):
            if key in ckpt and isinstance(ckpt[key], dict):
                state = ckpt[key]
                break
        if state is None:
            state = ckpt
        # The engine's EMA saves an envelope {"decay": float, "shadow": {param: tensor}}.
        if "shadow" in state and isinstance(state["shadow"], dict):
            state = state["shadow"]
        missing, unexpected = model.load_state_dict(state, strict=False)  # type: ignore[union-attr]
        n_params = sum(1 for _ in model.state_dict())  # type: ignore[union-attr]
        if missing and len(missing) >= n_params:
            raise ValueError(
                f"load_model: no checkpoint keys matched the model ({len(unexpected)} unexpected); "
                f"refusing to run with untrained weights. Checkpoint keys: {sorted(ckpt)[:8]}"
            )
        if missing or unexpected:
            warnings.warn(
                f"load_model: {len(missing)} missing / {len(unexpected)} unexpected keys", stacklevel=2
            )
    else:
        model = _build_from_cfg(pharosnet_cls, ckpt_path_or_cfg)
    model.to(device)  # type: ignore[union-attr]
    model.eval()  # type: ignore[union-attr]
    return model  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Streaming restorer.
# ---------------------------------------------------------------------------


def _resolve_device(device: torch.device | str | None) -> torch.device:
    if device is None or device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


class StreamingRestorer:
    """Wrap a ``PharosModel`` for causal streaming and batch/image restoration.

    Per-frame API: :meth:`restore` (threads recurrent state) returns a plain dict of
    numpy/native values so callers never touch tensors. Use :meth:`reset` on a scene
    change, or enable ``auto_scene_cut`` to reset automatically via a low-res histogram
    distance (DESIGN §3.6). :meth:`restore_image` / :meth:`restore_folder` provide the
    stateless (image-mode) path.
    """

    def __init__(
        self,
        model: PharosModel,
        device: torch.device | str | None = "auto",
        half: bool = False,
        pad_multiple: int = 8,
        reparameterize: bool = True,
        gate: Optional[dict] = None,
        fps_window: int = 30,
        auto_scene_cut: bool = False,
        scene_cut_thresh: float = 0.5,
    ) -> None:
        self.device = _resolve_device(device)
        self.pad_multiple = int(pad_multiple)
        self.gate_lo = float((gate or {}).get("beta_lo", 0.15))
        self.gate_hi = float((gate or {}).get("beta_hi", 0.45))
        self.auto_scene_cut = auto_scene_cut
        self.scene_cut_thresh = float(scene_cut_thresh)

        # Half precision is only meaningful on CUDA; silently fall back on CPU.
        self.half = bool(half)
        if self.half and self.device.type != "cuda":
            warnings.warn("half precision requested on non-CUDA device; using float32", stacklevel=2)
            self.half = False
        self.dtype = torch.float16 if self.half else torch.float32

        if reparameterize:
            fn = getattr(model, "reparameterize", None)
            if callable(fn):
                try:
                    fn()
                except Exception as e:  # reparameterize is best-effort; never fatal
                    warnings.warn(f"reparameterize() failed: {e}", stacklevel=2)

        self.model = model
        to = getattr(model, "to", None)
        if callable(to):
            model.to(self.device)  # type: ignore[union-attr]
        if self.half:
            half_fn = getattr(model, "half", None)
            if callable(half_fn):
                model.half()  # type: ignore[union-attr]
        eval_fn = getattr(model, "eval", None)
        if callable(eval_fn):
            model.eval()  # type: ignore[union-attr]
        # nn.Module is callable (runs hooks); a bare protocol object exposes only forward.
        self._call = model if callable(model) else model.forward

        self.state: Optional[Any] = None
        self.frame_idx = 0
        self._prev_hist: Optional[np.ndarray] = None
        self._times: deque[float] = deque(maxlen=max(1, int(fps_window)))

    # -- internals ---------------------------------------------------------

    def _sync(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.synchronize()

    def _forward(self, frame_t: torch.Tensor, state: Optional[Any]) -> PharosOutput:
        autocast_dtype = torch.float16 if self.half else torch.float32
        with torch.inference_mode():
            with torch.autocast(device_type=self.device.type, dtype=autocast_dtype, enabled=self.half):
                return self._call(frame_t, state, None)

    @staticmethod
    def _scalar(t: Any) -> float:
        if isinstance(t, torch.Tensor):
            return float(t.float().flatten()[0].item())
        return float(t)

    @staticmethod
    def _vec(t: Any) -> list[float]:
        if isinstance(t, torch.Tensor):
            return [float(v) for v in t.float().flatten().tolist()]
        return [float(v) for v in t]

    def _unpack_deg(self, deg: dict) -> dict:
        beta = self._scalar(deg["beta"]) if "beta" in deg else 0.0
        sigma = self._scalar(deg["sigma"]) if "sigma" in deg else 0.0
        airlight = self._vec(deg["airlight"]) if "airlight" in deg else [0.0, 0.0, 0.0]
        logits = self._vec(deg["domain_logits"]) if "domain_logits" in deg else [0.0, 0.0, 0.0]
        domain = int(np.argmax(logits)) if logits else 0
        return {
            "beta": beta,
            "sigma": sigma,
            "airlight": airlight,
            "domain_logits": logits,
            "domain": domain,
            "domain_name": DOMAIN_NAMES.get(domain, str(domain)),
        }

    def _record_fps(self, total_ms: float) -> float:
        self._times.append(total_ms)
        avg = sum(self._times) / len(self._times)
        return 1000.0 / avg if avg > 0 else 0.0

    def _scene_cut(self, frame_bgr: np.ndarray) -> bool:
        """Cheap low-res grayscale histogram distance for automatic scene-cut reset."""
        small = cv2.resize(frame_bgr, (64, 64), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        hist = cv2.calcHist([gray], [0], None, [32], [0, 256]).flatten()
        hist = hist / (hist.sum() + 1e-6)
        cut = False
        if self._prev_hist is not None:
            # Bhattacharyya-style distance in [0, 1].
            bc = float(np.sum(np.sqrt(self._prev_hist * hist)))
            cut = (1.0 - bc) > self.scene_cut_thresh
        self._prev_hist = hist
        return cut

    def _run(self, frame_bgr: np.ndarray, state: Optional[Any]) -> tuple[dict, Optional[Any]]:
        t0 = perf_counter()
        frame_t, orig_hw = bgr_to_tensor(frame_bgr, self.device, self.dtype, self.pad_multiple)
        self._sync()
        t1 = perf_counter()
        out = self._forward(frame_t, state)
        self._sync()
        t2 = perf_counter()

        output_bgr = tensor_to_bgr(out.output, orig_hw)
        confidence = confidence_to_map(out.confidence, orig_hw)
        deg = self._unpack_deg(out.deg)
        t3 = perf_counter()

        pre_ms = (t1 - t0) * 1000.0
        infer_ms = (t2 - t1) * 1000.0
        post_ms = (t3 - t2) * 1000.0
        total_ms = (t3 - t0) * 1000.0
        gate_alpha = self._scalar(out.aux["gate_alpha"]) if "gate_alpha" in out.aux else smoothstep(
            deg["beta"], self.gate_lo, self.gate_hi
        )
        result = {
            "output_bgr": output_bgr,
            "confidence": confidence,
            "deg": deg,
            "gate_alpha": gate_alpha,
            "timings": {
                "pre_ms": pre_ms,
                "infer_ms": infer_ms,
                "post_ms": post_ms,
                "total_ms": total_ms,
                "fps": 1000.0 / total_ms if total_ms > 0 else 0.0,
                "fps_avg": self._record_fps(total_ms),
            },
        }
        return result, out.state

    # -- public API --------------------------------------------------------

    def reset(self) -> None:
        """Clear recurrent state (scene change / new clip). Leaves the FPS meter intact."""
        self.state = None
        self.frame_idx = 0
        self._prev_hist = None

    def restore(self, frame_bgr: np.ndarray) -> dict:
        """Restore one streaming frame, threading recurrent state across calls.

        ``frame_bgr``: uint8 HxWx3 BGR. Returns
        ``{output_bgr(uint8 HxWx3), confidence(float32 HxW), deg(dict), gate_alpha,
        timings}``.
        """
        if self.auto_scene_cut and self._scene_cut(frame_bgr):
            self.reset()
        result, new_state = self._run(frame_bgr, self.state)
        self.state = new_state
        self.frame_idx += 1
        return result

    def restore_image(self, frame_bgr: np.ndarray) -> dict:
        """Restore a single independent image in image mode (state stays ``None``)."""
        result, _ = self._run(frame_bgr, None)
        return result

    def restore_batch(self, frames: list[np.ndarray], stream: bool = False) -> list[dict]:
        """Restore a list of frames.

        ``stream=False`` (default): each frame is independent (image mode).
        ``stream=True``: thread state across the list (treat as one video/clip).
        """
        if stream:
            return [self.restore(f) for f in frames]
        return [self.restore_image(f) for f in frames]

    def restore_folder(
        self,
        in_dir: Union[str, Path],
        out_dir: Optional[Union[str, Path]] = None,
        exts: tuple[str, ...] = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"),
        stream: bool = False,
    ) -> list[dict]:
        """Restore every image in ``in_dir`` (sorted). Optionally write outputs to ``out_dir``.

        Returns one result dict per image with an added ``path`` (and ``out_path`` when
        saved). Files that fail to decode are skipped with a warning.
        """
        in_dir = Path(in_dir)
        paths = sorted(p for p in in_dir.iterdir() if p.suffix.lower() in exts)
        if out_dir is not None:
            out_dir = Path(out_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
        if stream:
            self.reset()
        results: list[dict] = []
        for p in paths:
            frame = cv2.imread(str(p), cv2.IMREAD_COLOR)
            if frame is None:
                warnings.warn(f"could not read image {p}", stacklevel=2)
                continue
            res = self.restore(frame) if stream else self.restore_image(frame)
            res["path"] = str(p)
            if out_dir is not None:
                op = out_dir / p.name
                cv2.imwrite(str(op), res["output_bgr"])
                res["out_path"] = str(op)
            results.append(res)
        return results
