"""ONNX export + parity check + TensorRT instructions for a reparameterized PharosNet.

The deployed model is exported in **image mode** (state=None): a small wrapper turns the
:class:`PharosOutput` dataclass into a plain ``(output, confidence)`` tensor tuple so the
tracer/exporter is happy. Both a static-HW and a dynamic-HW variant are supported.

**Video-mode (recurrent) export** is attempted only when the model's recurrent ``state``
is a single ``Tensor`` (so it can become an optional ONNX input/output). If ``state`` is an
opaque structure (dict/tuple/list — the likely shape of a ConvGRU hidden + grid EMA) it
cannot be expressed as a flat ONNX I/O without model-side cooperation, so we raise
:class:`ExportUnsupported` with a precise reason. Trilinear bilateral-grid **slicing uses
``grid_sample``**, which the TorchScript exporter supports only from opset >= 16 — hence the
default ``opset=17``; on an older opset the export would fail there and the reason is
surfaced verbatim.

``onnx`` / ``onnxruntime`` may not be installed; every path that needs them degrades to a
clear, catchable error (:class:`ExportDependencyMissing`) or a skip dict.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn as nn

from pharos.contracts import PharosModel


class ExportError(RuntimeError):
    """Base class for export problems."""


class ExportDependencyMissing(ExportError):
    """Raised when onnx/onnxscript is not installed (export is cleanly skippable)."""


class ExportUnsupported(ExportError):
    """Raised when a requested export variant cannot be represented in ONNX."""


def _is_missing_dep(exc: BaseException) -> bool:
    if isinstance(exc, (ImportError, ModuleNotFoundError)):
        return True
    msg = str(exc).lower()
    return "not installed" in msg or "onnxscript" in msg or "no module named" in msg


class _ExportWrapper(nn.Module):
    """Adapt ``PharosModel.forward`` (dataclass out) to a tensor-tuple returning module."""

    def __init__(self, model: PharosModel, video_mode: bool = False) -> None:
        super().__init__()
        self.model = model  # type: ignore[assignment]
        self.video_mode = video_mode

    def forward(self, frame: torch.Tensor, state: Optional[torch.Tensor] = None):  # type: ignore[override]
        m = self.model
        out = m(frame, state, None) if callable(m) else m.forward(frame, state, None)
        if self.video_mode:
            return out.output, out.confidence, out.state
        return out.output, out.confidence


def _prepare(model: PharosModel, reparameterize: bool, device: str) -> PharosModel:
    fn = getattr(model, "eval", None)
    if callable(fn):
        model.eval()  # type: ignore[union-attr]
    if reparameterize:
        rep = getattr(model, "reparameterize", None)
        if callable(rep):
            try:
                rep()
            except Exception:  # noqa: BLE001  (best-effort; some stubs have no branches)
                pass
    to = getattr(model, "to", None)
    if callable(to):
        model.to(device)  # type: ignore[union-attr]
    return model


def export_onnx(
    model: PharosModel,
    out_path: str | Path,
    resolution: tuple[int, int] = (256, 256),
    dynamic: bool = False,
    video_mode: bool = False,
    opset: int = 17,
    device: str = "cpu",
    reparameterize: bool = True,
) -> Path:
    """Export a reparameterized model to ONNX (image mode by default).

    ``resolution`` is ``(H, W)``. ``dynamic=True`` marks H/W as dynamic axes.
    ``video_mode=True`` additionally threads the recurrent ``state`` as an optional I/O and
    only works if that state is a single ``Tensor`` (else :class:`ExportUnsupported`).

    Raises :class:`ExportDependencyMissing` if ``onnx``/``onnxscript`` is missing (so the
    caller can cleanly skip) and :class:`ExportUnsupported` for un-exportable variants.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    model = _prepare(model, reparameterize, device)

    h, w = int(resolution[0]), int(resolution[1])
    dummy = torch.rand(1, 3, h, w, device=device)

    state_dummy: Optional[torch.Tensor] = None
    if video_mode:
        with torch.inference_mode():
            probe = model(dummy, None, None) if callable(model) else model.forward(dummy, None, None)
        st = probe.state
        if not isinstance(st, torch.Tensor):
            raise ExportUnsupported(
                "video-mode ONNX export needs a single-Tensor recurrent state; got "
                f"{type(st).__name__}. Export image-mode ONNX and keep recurrence in the runtime, "
                "or have the model expose a flat-tensor state."
            )
        state_dummy = torch.zeros_like(st)

    wrapper = _ExportWrapper(model, video_mode=video_mode).eval().to(device)

    # Validation forward — exercises the whole export path even when onnx is absent.
    with torch.inference_mode():
        args_probe = (dummy,) if not video_mode else (dummy, state_dummy)
        wrapper(*args_probe)

    input_names = ["frame"] + (["state_in"] if video_mode else [])
    output_names = ["output", "confidence"] + (["state_out"] if video_mode else [])
    dynamic_axes: Optional[dict[str, dict[int, str]]] = None
    if dynamic:
        dynamic_axes = {"frame": {2: "H", 3: "W"}, "output": {2: "H", 3: "W"}, "confidence": {2: "H", 3: "W"}}

    args = (dummy,) if not video_mode else (dummy, state_dummy)
    try:
        torch.onnx.export(
            wrapper,
            args,
            str(out_path),
            input_names=input_names,
            output_names=output_names,
            opset_version=opset,
            dynamic_axes=dynamic_axes,
            do_constant_folding=True,
            dynamo=False,
        )
    except ExportError:
        raise
    except Exception as e:  # noqa: BLE001
        if _is_missing_dep(e):
            raise ExportDependencyMissing(
                f"ONNX export needs the 'onnx' package (pip install onnx): {e}"
            ) from e
        raise
    return out_path


def export_all(
    model: PharosModel,
    out_dir: str | Path,
    resolution: tuple[int, int] = (1080, 1920),
    try_video: bool = True,
    device: str = "cpu",
) -> dict[str, Any]:
    """Export the standard set: static-HW and dynamic-HW image mode, plus video mode if feasible.

    Returns a dict describing each variant with ``path`` or a ``skipped`` reason; never raises
    for a merely-missing dependency (records it instead).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    h, w = resolution
    variants: dict[str, Any] = {}

    def _try(name: str, **kw: Any) -> None:
        try:
            p = export_onnx(model, out_dir / f"pharos_{name}.onnx", **kw)
            variants[name] = {"exported": True, "path": str(p)}
        except (ExportDependencyMissing, ExportUnsupported) as e:
            variants[name] = {"exported": False, "reason": str(e)}
        except Exception as e:  # noqa: BLE001
            variants[name] = {"exported": False, "reason": f"{type(e).__name__}: {e}"}

    _try("static", resolution=(h, w), dynamic=False, device=device)
    _try("dynamic", resolution=(h, w), dynamic=True, device=device)
    if try_video:
        _try("video", resolution=(h, w), dynamic=False, video_mode=True, device=device)
    return variants


def onnx_parity_check(
    model: PharosModel,
    onnx_path: str | Path,
    resolution: tuple[int, int] = (256, 256),
    device: str = "cpu",
    seed: int = 0,
) -> dict[str, Any]:
    """Compare torch vs ONNX Runtime on one random input; report max abs diff on ``output``.

    Returns ``{"available": False, "reason": ...}`` if onnxruntime is missing.
    """
    try:
        import onnxruntime as ort  # type: ignore
    except Exception as e:  # noqa: BLE001
        return {"available": False, "reason": f"onnxruntime not installed ({e})"}

    h, w = int(resolution[0]), int(resolution[1])
    x = torch.rand(1, 3, h, w, generator=torch.Generator().manual_seed(seed))

    model = _prepare(model, reparameterize=True, device=device)
    with torch.inference_mode():
        out = model(x.to(device), None, None) if callable(model) else model.forward(x.to(device), None, None)
    torch_out = out.output.detach().float().cpu().numpy()
    torch_conf = out.confidence.detach().float().cpu().numpy()

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    iname = sess.get_inputs()[0].name
    ort_outs = sess.run(None, {iname: x.numpy()})
    ort_out = ort_outs[0]

    import numpy as np

    diff_out = float(np.abs(torch_out - ort_out).max())
    result: dict[str, Any] = {
        "available": True,
        "max_abs_diff_output": diff_out,
        "output_shape": list(torch_out.shape),
    }
    if len(ort_outs) > 1:
        result["max_abs_diff_confidence"] = float(np.abs(torch_conf - ort_outs[1]).max())
    return result


def trtexec_command(
    onnx_path: str | Path,
    engine_path: Optional[str | Path] = None,
    fp16: bool = True,
    int8: bool = False,
    min_hw: tuple[int, int] = (720, 1280),
    opt_hw: tuple[int, int] = (1080, 1920),
    max_hw: tuple[int, int] = (2160, 3840),
    dynamic: bool = False,
) -> str:
    """Return the ``trtexec`` command line to build/benchmark a TensorRT engine from ONNX."""
    onnx_path = Path(onnx_path)
    engine_path = Path(engine_path) if engine_path else onnx_path.with_suffix(".engine")
    parts = [f"trtexec --onnx={onnx_path}", f"--saveEngine={engine_path}"]
    if fp16:
        parts.append("--fp16")
    if int8:
        parts.append("--int8")
    if dynamic:
        parts.append(f"--minShapes=frame:1x3x{min_hw[0]}x{min_hw[1]}")
        parts.append(f"--optShapes=frame:1x3x{opt_hw[0]}x{opt_hw[1]}")
        parts.append(f"--maxShapes=frame:1x3x{max_hw[0]}x{max_hw[1]}")
    return " ".join(parts)


def build_tensorrt_engine(
    onnx_path: str | Path,
    engine_path: Optional[str | Path] = None,
    fp16: bool = True,
    int8: bool = False,
) -> dict[str, Any]:
    """Best-effort TensorRT engine build. If tensorrt is unavailable, return instructions.

    Full engine building is best done with ``trtexec`` (returned in ``instructions``); this
    function reports whether the TensorRT Python package is importable and, if not, exactly
    how to build the engine offline.
    """
    cmd = trtexec_command(onnx_path, engine_path, fp16=fp16, int8=int8)
    try:
        import tensorrt as trt  # type: ignore
    except Exception as e:  # noqa: BLE001
        return {"built": False, "reason": f"tensorrt not installed ({e})", "instructions": cmd}
    return {
        "built": False,
        "reason": f"tensorrt {getattr(trt, '__version__', '?')} importable; "
        "use trtexec for a reproducible engine build",
        "instructions": cmd,
    }
