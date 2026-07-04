"""Honest FPS / latency benchmark for Pharos (DESIGN §7).

For each resolution in ``cfg.bench.resolutions`` and each available runtime mode
(PyTorch FP32, PyTorch FP16, ONNX Runtime, TensorRT) we run ``cfg.bench.frames`` frames
at batch 1 after a warmup, and report median / P95 latency and FPS for both the
**model only** and the **full pipeline** (pre/post included). Timing uses
``perf_counter`` with CUDA synchronisation. Results are written as JSON + a markdown
table under ``out_root/bench/``.

Modes whose dependency is missing (onnxruntime / tensorrt not installed) are reported
with ``available: false`` and a reason — for TensorRT the exact ``trtexec`` command is
included so it can be run externally.
"""
from __future__ import annotations

import copy
import json
import platform
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any, Optional, Sequence

import numpy as np
import torch

from pharos.contracts import PharosModel
from pharos.rt.infer import StreamingRestorer, _resolve_device, bgr_to_tensor, tensor_to_bgr

ALL_MODES = ("torch_fp32", "torch_fp16", "onnxruntime", "tensorrt")


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------


def count_params(model: Any) -> int:
    """Total number of parameters, or 0 for a model with no ``parameters()``."""
    fn = getattr(model, "parameters", None)
    if not callable(fn):
        return 0
    return int(sum(p.numel() for p in fn()))


def _stats(times_ms: Sequence[float]) -> dict[str, float]:
    arr = np.asarray(list(times_ms), dtype=np.float64)
    if arr.size == 0:
        return {"median_ms": 0.0, "p95_ms": 0.0, "mean_ms": 0.0, "min_ms": 0.0, "fps": 0.0, "frames": 0}
    median = float(np.median(arr))
    return {
        "median_ms": median,
        "p95_ms": float(np.percentile(arr, 95)),
        "mean_ms": float(arr.mean()),
        "min_ms": float(arr.min()),
        "fps": float(1000.0 / median) if median > 0 else 0.0,
        "frames": int(arr.size),
    }


def _random_frame(res_wh: Sequence[int], seed: Optional[int] = None) -> np.ndarray:
    """uint8 HxWx3 BGR random frame for a ``(W, H)`` resolution (config order)."""
    w, h = int(res_wh[0]), int(res_wh[1])
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8)


def _skip(res_wh: Sequence[int], mode: str, reason: str, **extra: Any) -> dict:
    entry = {
        "resolution": [int(res_wh[0]), int(res_wh[1])],
        "mode": mode,
        "available": False,
        "reason": reason,
    }
    entry.update(extra)
    return entry


# ---------------------------------------------------------------------------
# per-mode benchmarks
# ---------------------------------------------------------------------------


def _bench_torch(
    model: PharosModel,
    res_wh: Sequence[int],
    frames: int,
    warmup: int,
    device: torch.device,
    half: bool,
    gate: Optional[dict],
    pad_multiple: int,
) -> dict:
    mode = "torch_fp16" if half else "torch_fp32"
    if half and device.type != "cuda":
        return _skip(res_wh, mode, "fp16 requires CUDA")
    # Deepcopy for fp16 so we never mutate the shared fp32 model to half in place.
    m = copy.deepcopy(model) if half else model
    restorer = StreamingRestorer(
        m, device=device, half=half, pad_multiple=pad_multiple, reparameterize=True, gate=gate
    )
    frame = _random_frame(res_wh, seed=0)
    restorer.reset()
    for _ in range(max(0, warmup)):
        restorer.restore(frame)
    infer_ms: list[float] = []
    total_ms: list[float] = []
    pre_ms: list[float] = []
    post_ms: list[float] = []
    for _ in range(frames):
        t = restorer.restore(frame)["timings"]
        infer_ms.append(t["infer_ms"])
        total_ms.append(t["total_ms"])
        pre_ms.append(t["pre_ms"])
        post_ms.append(t["post_ms"])
    return {
        "resolution": [int(res_wh[0]), int(res_wh[1])],
        "mode": mode,
        "available": True,
        "model_only": _stats(infer_ms),
        "full_pipeline": _stats(total_ms),
        "pre_ms_mean": float(np.mean(pre_ms)) if pre_ms else 0.0,
        "post_ms_mean": float(np.mean(post_ms)) if post_ms else 0.0,
    }


def _bench_onnx(
    model: PharosModel,
    res_wh: Sequence[int],
    frames: int,
    warmup: int,
    device: torch.device,
    pad_multiple: int,
) -> dict:
    try:
        import onnxruntime as ort  # type: ignore
    except Exception as e:  # noqa: BLE001
        return _skip(res_wh, "onnxruntime", f"onnxruntime not installed ({e})")

    from pharos.rt.export import export_onnx  # local import; optional path

    w, h = int(res_wh[0]), int(res_wh[1])
    tmpdir = Path(tempfile.mkdtemp(prefix="pharos_bench_"))
    onnx_path = tmpdir / f"pharos_{w}x{h}.onnx"
    try:
        export_onnx(model, onnx_path, resolution=(h, w), dynamic=False, device="cpu")
    except Exception as e:  # noqa: BLE001  (any export failure -> skip with reason)
        return _skip(res_wh, "onnxruntime", f"onnx export failed: {e}")

    avail = set(ort.get_available_providers())
    providers = [p for p in ("CUDAExecutionProvider", "CPUExecutionProvider") if p in avail]
    sess = ort.InferenceSession(str(onnx_path), providers=providers or None)
    iname = sess.get_inputs()[0].name
    on_gpu = "CUDAExecutionProvider" in (sess.get_providers() or [])

    x = np.random.default_rng(0).random((1, 3, h, w), dtype=np.float32)
    frame = _random_frame(res_wh, seed=0)

    def run_model_only() -> None:
        sess.run(None, {iname: x})

    def run_pipeline() -> None:
        t, orig = bgr_to_tensor(frame, "cpu", torch.float32, pad_multiple)
        out = sess.run(None, {iname: t.numpy()})[0]
        tensor_to_bgr(torch.from_numpy(out), orig)

    model_only = _timeit(run_model_only, frames, warmup, sync=on_gpu)
    pipeline = _timeit(run_pipeline, frames, warmup, sync=on_gpu)
    return {
        "resolution": [w, h],
        "mode": "onnxruntime",
        "available": True,
        "providers": sess.get_providers(),
        "model_only": _stats(model_only),
        "full_pipeline": _stats(pipeline),
    }


def _bench_trt(model: PharosModel, res_wh: Sequence[int]) -> dict:
    from pharos.rt.export import trtexec_command

    w, h = int(res_wh[0]), int(res_wh[1])
    onnx_hint = f"pharos_{w}x{h}.onnx"
    cmd = trtexec_command(onnx_hint, f"pharos_{w}x{h}_fp16.engine", fp16=True)
    try:
        import tensorrt  # type: ignore  # noqa: F401
    except Exception as e:  # noqa: BLE001
        return _skip(
            res_wh,
            "tensorrt",
            f"tensorrt not installed ({e}); export ONNX then build/benchmark with trtexec",
            instructions=cmd,
        )
    # tensorrt is importable but full in-process engine build/benchmark is out of scope
    # here (and needs a static ONNX per resolution). Point at trtexec, the honest path.
    return _skip(
        res_wh,
        "tensorrt",
        "tensorrt importable; use trtexec for the engine build + benchmark",
        instructions=cmd,
    )


def _timeit(fn: Any, frames: int, warmup: int, sync: bool) -> list[float]:
    def _sync() -> None:
        if sync and torch.cuda.is_available():
            torch.cuda.synchronize()

    for _ in range(max(0, warmup)):
        fn()
    _sync()
    out: list[float] = []
    for _ in range(frames):
        _sync()
        t0 = perf_counter()
        fn()
        _sync()
        out.append((perf_counter() - t0) * 1000.0)
    return out


# ---------------------------------------------------------------------------
# orchestration + reporting
# ---------------------------------------------------------------------------


def run_benchmark(
    model: PharosModel,
    cfg: Any,
    *,
    out_dir: Optional[str | Path] = None,
    modes: Optional[Sequence[str]] = None,
    frames: Optional[int] = None,
    warmup: int = 10,
    device: torch.device | str | None = None,
    resolutions: Optional[Sequence[Sequence[int]]] = None,
    pad_multiple: int = 8,
    save: bool = True,
    tag: str = "",
) -> dict:
    """Benchmark ``model`` across resolutions and modes; write JSON + markdown.

    ``cfg`` supplies ``bench.resolutions`` / ``bench.frames`` and (optionally)
    ``model.gate`` and ``out_root``. ``frames`` / ``warmup`` / ``resolutions`` / ``modes``
    override the config for quick runs (tests pass a tiny ``frames`` + ``warmup``).
    Returns the full report dict.
    """
    dev = _resolve_device(device)
    res_list = resolutions or [list(r) for r in cfg.bench.resolutions]
    n_frames = int(frames if frames is not None else cfg.bench.frames)
    mode_list = list(modes) if modes is not None else list(ALL_MODES)
    gate = None
    try:
        gate = dict(cfg.model.gate)
    except Exception:  # noqa: BLE001  (gate is optional)
        gate = None

    report: dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "device": dev.type,
        "gpu": torch.cuda.get_device_name(0) if dev.type == "cuda" else platform.processor() or "cpu",
        "torch_version": torch.__version__,
        "params": count_params(model),
        "params_millions": round(count_params(model) / 1e6, 4),
        "frames": n_frames,
        "warmup": warmup,
        "batch": 1,
        "tag": tag,
        "results": [],
    }

    for res in res_list:
        for mode in mode_list:
            if mode == "torch_fp32":
                entry = _bench_torch(model, res, n_frames, warmup, dev, False, gate, pad_multiple)
            elif mode == "torch_fp16":
                entry = _bench_torch(model, res, n_frames, warmup, dev, True, gate, pad_multiple)
            elif mode == "onnxruntime":
                entry = _bench_onnx(model, res, n_frames, warmup, dev, pad_multiple)
            elif mode == "tensorrt":
                entry = _bench_trt(model, res)
            else:
                entry = _skip(res, mode, f"unknown mode '{mode}'")
            report["results"].append(entry)

    if save:
        out_dir = Path(out_dir) if out_dir is not None else _default_out_dir(cfg)
        out_dir.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d_%H%M%S")
        suffix = f"_{tag}" if tag else ""
        json_path = out_dir / f"bench_{stamp}{suffix}.json"
        md_path = out_dir / f"bench_{stamp}{suffix}.md"
        json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        md_path.write_text(render_markdown(report), encoding="utf-8")
        report["json_path"] = str(json_path)
        report["md_path"] = str(md_path)
    return report


def _default_out_dir(cfg: Any) -> Path:
    try:
        return Path(cfg.out_root) / "bench"
    except Exception:  # noqa: BLE001
        return Path("runs") / "bench"


def render_markdown(report: dict) -> str:
    """Render a benchmark report dict as a markdown summary + table."""
    lines: list[str] = []
    lines.append("# Pharos FPS Benchmark")
    lines.append("")
    lines.append(f"- GPU / device: **{report.get('gpu')}** ({report.get('device')})")
    lines.append(f"- torch: {report.get('torch_version')}")
    lines.append(f"- params: {report.get('params_millions')} M ({report.get('params')})")
    lines.append(
        f"- frames: {report.get('frames')} (warmup {report.get('warmup')}), batch {report.get('batch')}"
    )
    lines.append(f"- timestamp: {report.get('timestamp')}")
    lines.append("")
    lines.append("| Resolution | Mode | Avail | Model med ms | Model P95 ms | Model FPS "
                 "| Pipe med ms | Pipe P95 ms | Pipe FPS |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for r in report.get("results", []):
        res = r.get("resolution", ["?", "?"])
        res_s = f"{res[0]}x{res[1]}"
        if not r.get("available", False):
            lines.append(f"| {res_s} | {r.get('mode')} | no | - | - | - | - | - | - |")
            continue
        mo = r.get("model_only", {})
        fp = r.get("full_pipeline", {})
        lines.append(
            f"| {res_s} | {r.get('mode')} | yes "
            f"| {mo.get('median_ms', 0):.2f} | {mo.get('p95_ms', 0):.2f} | {mo.get('fps', 0):.1f} "
            f"| {fp.get('median_ms', 0):.2f} | {fp.get('p95_ms', 0):.2f} | {fp.get('fps', 0):.1f} |"
        )
    # Notes for skipped modes (e.g. TensorRT trtexec instructions).
    notes = [r for r in report.get("results", []) if not r.get("available", False) and r.get("instructions")]
    if notes:
        lines.append("")
        lines.append("## Skipped modes")
        seen: set[str] = set()
        for r in notes:
            key = r.get("mode", "")
            if key in seen:
                continue
            seen.add(key)
            lines.append(f"- **{key}**: {r.get('reason')}")
            lines.append(f"  - `{r.get('instructions')}`")
    return "\n".join(lines) + "\n"
