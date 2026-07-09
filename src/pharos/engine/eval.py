"""TriHaze fixed-weights evaluation protocol (DESIGN §7).

One checkpoint, no per-set tuning. Produces, for a single model:

* paired metrics (PSNR / SSIM / LPIPS-if-available) per eval set;
* no-reference metrics on unpaired sets (a documented *simplified* NIQE; BRISQUE
  via OpenCV's ``cv2.quality`` module when its model files are present; FADE is
  optional and skipped with a note);
* clear-frame no-harm check (clean inputs -> output should equal input,
  PSNR >= 45 dB);
* temporal warp-error on video clips (teacher flow when available, else a
  documented frame-difference proxy);
* a detection-mAP hook left as a clearly-marked stub.

Results are returned as a dict and, when ``out_dir`` is given, written as a single
JSON plus a Markdown summary table under ``out_root/<exp>/eval/``.

CLI:
    python -m pharos.engine.eval --config configs/<exp>.yaml --ckpt <path>
"""
from __future__ import annotations

import argparse
import gc
import json
import warnings
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, DataLoader

from ..config import Config, load_config
from . import metrics as M
from .train import Deps, pharos_collate
from .utils import load_checkpoint, move_batch_to_device

CLEAR_NO_HARM_DB = 45.0

__all__ = ["evaluate", "niqe_simplified", "brisque_score", "detection_map"]


# ---------------------------------------------------------------------------
# no-reference metrics
# ---------------------------------------------------------------------------


def _to_gray(img: torch.Tensor) -> torch.Tensor:
    if img.dim() == 3:
        img = img.unsqueeze(0)
    w = torch.tensor([0.299, 0.587, 0.114], device=img.device, dtype=img.dtype)
    return (img * w[None, :, None, None]).sum(1, keepdim=True)


def niqe_simplified(img: torch.Tensor) -> float:
    """A *simplified*, self-contained naturalness score (NOT calibrated NIQE).

    Real NIQE (Mittal 2013) scores an image against a multivariate-Gaussian model
    fitted to a corpus of pristine natural patches; that pristine model is not
    bundled here. This proxy instead measures how far the image's MSCN-coefficient
    statistics deviate from the natural-image regularity (unit variance, near-zero
    excess kurtosis after local normalization). Lower = more natural. It is stable
    and monotone-ish for ranking, but is not comparable to published NIQE numbers.
    """
    gray = _to_gray(img).float()
    win = M._gaussian_window(7, 7 / 6.0, 1, gray.device, gray.dtype)
    pad = 3
    mu = F.conv2d(gray, win, padding=pad, groups=1)
    mu2 = F.conv2d(gray * gray, win, padding=pad, groups=1)
    sigma = (mu2 - mu * mu).clamp_min(1e-8).sqrt()
    mscn = (gray - mu) / (sigma + 1e-3)
    m = mscn.flatten()
    var = m.var(unbiased=False).item()
    mean4 = (m.pow(4).mean()).item()
    kurt = mean4 / (var ** 2 + 1e-12) - 3.0  # excess kurtosis
    return float(((var - 1.0) ** 2 + 0.1 * kurt ** 2) ** 0.5)


def _to_uint8_bgr(img: torch.Tensor) -> np.ndarray:
    x = img.detach().float().clamp(0, 1).cpu()
    if x.dim() == 4:
        x = x[0]
    arr = (x.permute(1, 2, 0).numpy() * 255).astype(np.uint8)  # HWC RGB
    return arr[:, :, ::-1].copy()  # BGR for OpenCV


def brisque_score(
    img: torch.Tensor,
    model_path: Optional[str] = None,
    range_path: Optional[str] = None,
) -> Optional[float]:
    """BRISQUE via OpenCV ``cv2.quality`` if available *and* model files provided.

    Returns None (with a warning) when the module or its trained model/range YAML
    files are unavailable — those ship separately from opencv-python.
    """
    try:
        import cv2

        if not hasattr(cv2, "quality") or model_path is None or range_path is None:
            return None
        bgr = _to_uint8_bgr(img)
        score = cv2.quality.QualityBRISQUE_compute(bgr, model_path, range_path)
        return float(score[0])
    except Exception as e:  # pragma: no cover - env dependent
        warnings.warn(f"BRISQUE unavailable ({e}); skipping.")
        return None


def detection_map(model, loader, teachers, device) -> dict[str, Any]:
    """STUB (DESIGN §7): frozen-YOLO detection mAP on dehazed vs raw RTTS.

    TODO interface: run the frozen detection teacher (``teachers.detector``) over
    both raw inputs and model outputs, decode boxes, and compute mAP against RTTS
    annotations. Requires the detector teacher (WS-C) and RTTS labels; returns a
    placeholder until those land.
    """
    return {
        "status": "not_implemented",
        "todo": "frozen-YOLO mAP(dehazed) vs mAP(raw) on RTTS; needs teachers.detector + labels",
    }


# ---------------------------------------------------------------------------
# per-section runners
# ---------------------------------------------------------------------------


@torch.no_grad()
def _model_out(model, hazy: torch.Tensor) -> torch.Tensor:
    return model(hazy, state=None).output


def _cap_size(t: Optional[torch.Tensor], max_side: int) -> Optional[torch.Tensor]:
    """Downscale so the long side <= max_side (bilinear, antialiased).

    Full-res slicing materializes coeffs*depth channels at output resolution;
    multi-megapixel eval images (NH-HAZE is >4K) spike several GB and OOM'd a
    live run. Metrics are computed at the capped size (consistent within-run
    tracking; the final benchmark protocol can tile at native res).
    """
    if t is None or max_side <= 0:
        return t
    h, w = t.shape[-2], t.shape[-1]
    if max(h, w) <= max_side:
        return t
    s = max_side / max(h, w)
    return F.interpolate(t, size=(int(h * s), int(w * s)), mode="bilinear",
                         align_corners=False, antialias=True)


@torch.no_grad()
def _run_paired(model, loader, device, lpips_model, max_batches: Optional[int],
                max_side: int = 2048) -> dict[str, float]:
    psnr = ssim = lp = 0.0
    n = 0
    lp_ok = 0
    for i, batch in enumerate(loader):
        if max_batches and i >= max_batches:
            break
        batch = move_batch_to_device(batch, device)
        if batch.get("clean") is None:
            continue
        hazy = _cap_size(batch["hazy"], max_side)
        out = _model_out(model, hazy)
        clean = _cap_size(batch["clean"], max_side)
        b = out.shape[0]
        psnr += M.psnr(out, clean) * b
        ssim += M.ssim(out, clean) * b
        n += b
        if lpips_model is not None and lpips_model.available:
            try:
                lp += lpips_model(out, clean) * b
                lp_ok += b
            except Exception:
                pass
    res: dict[str, Any] = {
        "psnr": psnr / n if n else float("nan"),
        "ssim": ssim / n if n else float("nan"),
        "n": n,
    }
    res["lpips"] = (lp / lp_ok) if lp_ok else None
    return res


@torch.no_grad()
def _run_noref(model, loader, device, max_batches: Optional[int]) -> dict[str, Any]:
    niqe = 0.0
    n = 0
    for i, batch in enumerate(loader):
        if max_batches and i >= max_batches:
            break
        batch = move_batch_to_device(batch, device)
        out = _model_out(model, batch["hazy"])
        for j in range(out.shape[0]):
            niqe += niqe_simplified(out[j])
            n += 1
    return {
        "niqe_simplified": niqe / n if n else float("nan"),
        "brisque": None,  # requires external cv2.quality model files (see brisque_score)
        "n": n,
        "note": "niqe_simplified is a documented proxy, not calibrated NIQE",
    }


def _cap_clip(clip: torch.Tensor, max_side: int) -> torch.Tensor:
    """Apply :func:`_cap_size` framewise to a B,T,3,H,W clip."""
    b, t = clip.shape[0], clip.shape[1]
    capped = _cap_size(clip.flatten(0, 1), max_side)
    return capped.reshape(b, t, *capped.shape[1:])


@torch.no_grad()
def _clip_outputs(model, frames: torch.Tensor) -> torch.Tensor:
    """Run the recurrent model over a clip (B,T,3,H,W) -> outputs (B,T,3,H,W)."""
    state = None
    outs = []
    for t in range(frames.shape[1]):
        o = model(frames[:, t], state=state)
        state = o.state
        outs.append(o.output)
    return torch.stack(outs, dim=1)


@torch.no_grad()
def _run_temporal(model, loader, device, teachers, max_batches: Optional[int],
                  max_side: int = 1024, max_windows: int = 60) -> dict[str, Any]:
    """Temporal consistency on clips.

    Hard memory bounds: frames are capped to ``max_side`` (RAFT's correlation
    volume grows quadratically with pixels — full-res REVIDE OOMs a 24GB card)
    and at most ``max_windows`` clip windows are scored (statistically ample).
    """
    err = 0.0
    pairs = 0
    used_flow = False
    have_flow = teachers is not None and getattr(teachers, "flow", None) is not None
    limit = min(max_batches or max_windows, max_windows)
    for i, batch in enumerate(loader):
        if i >= limit:
            break
        batch = move_batch_to_device(batch, device)
        frames = batch["hazy"]
        if frames.dim() != 5:
            continue
        frames = _cap_clip(frames, max_side)
        clean = batch.get("clean")
        clean = _cap_clip(clean, max_side) if clean is not None and clean.dim() == 5 else clean
        outs = _clip_outputs(model, frames)
        for t in range(1, frames.shape[1]):
            flow = None
            if have_flow and clean is not None:
                try:
                    flow = teachers.flow(clean[:, t - 1], clean[:, t])
                    used_flow = True
                except Exception:
                    flow = None
            err += M.warp_error(outs[:, t - 1], outs[:, t], flow)
            pairs += 1
    return {
        "warp_error": err / pairs if pairs else float("nan"),
        "used_flow": used_flow,
        "pairs": pairs,
    }


@torch.no_grad()
def _clear_no_harm(model, loader, device, max_batches: Optional[int]) -> dict[str, Any]:
    """Feed clean frames as input; a good model must pass them through unharmed."""
    psnr = 0.0
    n = 0
    for i, batch in enumerate(loader):
        if max_batches and i >= max_batches:
            break
        batch = move_batch_to_device(batch, device)
        clean = batch.get("clean")
        if clean is None:
            continue
        inp = clean if clean.dim() == 4 else clean[:, 0]
        out = _model_out(model, inp)
        b = out.shape[0]
        psnr += M.psnr(out, inp) * b
        n += b
    val = psnr / n if n else float("nan")
    return {"psnr_out_vs_in": val, "threshold_db": CLEAR_NO_HARM_DB, "pass": bool(val >= CLEAR_NO_HARM_DB)}


# ---------------------------------------------------------------------------
# loader construction (factory-backed; injected loaders bypass this)
# ---------------------------------------------------------------------------


def _build_loaders(
    cfg: Config, deps: Deps, names: list[str], batch_size: int, device
) -> dict[str, DataLoader]:
    loaders: dict[str, DataLoader] = {}
    for name in names:
        try:
            ds_list = deps.build_datasets(cfg, [name], "eval")
            ds = ds_list[0] if len(ds_list) == 1 else ConcatDataset(ds_list)
            loaders[name] = DataLoader(
                ds,
                batch_size=batch_size,
                shuffle=False,
                num_workers=0,
                collate_fn=pharos_collate,
                pin_memory=device.type == "cuda",
            )
        except Exception as e:
            warnings.warn(f"eval set '{name}' unavailable ({e}); skipping.")
    return loaders


# ---------------------------------------------------------------------------
# main entry
# ---------------------------------------------------------------------------


def evaluate(
    model,
    cfg: Config,
    *,
    teachers: Optional[Any] = None,
    device: Optional[torch.device | str] = None,
    out_dir: Optional[str | Path] = None,
    step: int = 0,
    paired_loaders: Optional[dict[str, DataLoader]] = None,
    noref_loaders: Optional[dict[str, DataLoader]] = None,
    clip_loaders: Optional[dict[str, DataLoader]] = None,
    clear_loader: Optional[DataLoader] = None,
    lpips_model: Optional[M.LPIPS] = None,
    compute_lpips: bool = True,
    deps: Optional[Deps] = None,
    max_batches: Optional[int] = None,
) -> dict[str, Any]:
    """Run the TriHaze protocol and return a results dict.

    Loaders may be injected (dict name->DataLoader) for tests; anything not
    injected is built lazily from ``cfg`` via the WS factories and quietly skipped
    if the underlying dataset is missing.
    """
    device = torch.device(device) if device is not None else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    deps = deps or Deps()
    model = model.to(device)
    was_training = model.training
    model.eval()

    ds_cfg = cfg.get("datasets", {})
    ev_batch = int(cfg.get("eval", {}).get("batch", 1)) if isinstance(cfg.get("eval"), dict) else 1
    if paired_loaders is None:
        paired_loaders = _build_loaders(cfg, deps, list(ds_cfg.get("eval", [])), ev_batch, device)
    if noref_loaders is None:
        noref_loaders = _build_loaders(cfg, deps, list(ds_cfg.get("eval_noref", [])), ev_batch, device)
    if clip_loaders is None:
        clip_loaders = _build_loaders(cfg, deps, list(ds_cfg.get("eval_video", [])), 1, device)
    if lpips_model is None and compute_lpips:
        lpips_model = M.LPIPS(device=device)

    results: dict[str, Any] = {"step": int(step), "notes": []}
    ev = cfg.get("eval", {}) if isinstance(cfg.get("eval"), dict) else {}
    max_side = int(ev.get("max_side", 2048))

    def _purge() -> None:
        # Bound the allocator between sections; a full eval must never leave the
        # GPU too full for the next training step (has OOM'd a live run twice).
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    results["paired"] = {
        name: _run_paired(model, ld, device, lpips_model, max_batches, max_side=max_side)
        for name, ld in (paired_loaders or {}).items()
    }
    _purge()
    results["noref"] = {
        name: _run_noref(model, ld, device, max_batches)
        for name, ld in (noref_loaders or {}).items()
    }
    _purge()
    results["temporal"] = {
        name: _run_temporal(model, ld, device, teachers, max_batches,
                            max_side=int(ev.get("temporal_max_side", 1024)),
                            max_windows=int(ev.get("temporal_max_windows", 60)))
        for name, ld in (clip_loaders or {}).items()
    }
    _purge()

    # clear-frame no-harm: use explicit loader, else first paired loader with clean.
    ch_loader = clear_loader
    if ch_loader is None and paired_loaders:
        ch_loader = paired_loaders.get("sots_mix") or next(iter(paired_loaders.values()), None)
    if ch_loader is not None:
        results["clear_no_harm"] = _clear_no_harm(model, ch_loader, device, max_batches)
    else:
        results["clear_no_harm"] = {"note": "no clean loader available"}

    results["detection_map"] = detection_map(model, None, teachers, device)

    if not (lpips_model and lpips_model.available):
        results["notes"].append("LPIPS unavailable (package/weights); paired lpips=null")

    if out_dir is not None:
        _write_reports(Path(out_dir), results, step)
    if was_training:
        model.train()
    return results


def _write_reports(out_dir: Path, results: dict[str, Any], step: int) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"eval_step{step:06d}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=_json_default)
    md_path = out_dir / f"eval_step{step:06d}.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(_markdown_table(results))


def _json_default(o: Any):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    return str(o)


def _fmt(v: Any) -> str:
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v)


def _markdown_table(results: dict[str, Any]) -> str:
    lines = [f"# Pharos eval (step {results.get('step', 0)})", ""]
    paired = results.get("paired", {})
    if paired:
        lines += ["## Paired metrics", "", "| set | PSNR | SSIM | LPIPS | n |", "|---|---|---|---|---|"]
        for name, m in paired.items():
            lines.append(
                f"| {name} | {_fmt(m.get('psnr'))} | {_fmt(m.get('ssim'))} | "
                f"{_fmt(m.get('lpips'))} | {m.get('n', 0)} |"
            )
        lines.append("")
    noref = results.get("noref", {})
    if noref:
        lines += ["## No-reference", "", "| set | NIQE* | BRISQUE | n |", "|---|---|---|---|"]
        for name, m in noref.items():
            lines.append(
                f"| {name} | {_fmt(m.get('niqe_simplified'))} | {_fmt(m.get('brisque'))} | "
                f"{m.get('n', 0)} |"
            )
        lines += ["", "\\* simplified proxy, not calibrated NIQE", ""]
    temporal = results.get("temporal", {})
    if temporal:
        lines += ["## Temporal", "", "| set | warp_error | used_flow | pairs |", "|---|---|---|---|"]
        for name, m in temporal.items():
            lines.append(
                f"| {name} | {_fmt(m.get('warp_error'))} | {m.get('used_flow')} | "
                f"{m.get('pairs', 0)} |"
            )
        lines.append("")
    ch = results.get("clear_no_harm", {})
    if "psnr_out_vs_in" in ch:
        lines += [
            "## Clear-frame no-harm",
            "",
            f"PSNR(out, in) = {_fmt(ch['psnr_out_vs_in'])} dB "
            f"(threshold {ch['threshold_db']}, pass={ch['pass']})",
            "",
        ]
    lines += ["## Detection mAP", "", f"`{results.get('detection_map', {}).get('status')}`", ""]
    return "\n".join(lines)


def _load_model_from_ckpt(model, ckpt_path: str, device) -> None:
    ck = load_checkpoint(ckpt_path, map_location=device)
    if "ema" in ck and ck["ema"].get("shadow"):
        model.load_state_dict(ck["ema"]["shadow"], strict=False)
    else:
        model.load_state_dict(ck["model"], strict=False)


def main(argv: Optional[list[str]] = None) -> None:
    from .train import default_build_model
    from .utils import parse_overrides

    ap = argparse.ArgumentParser(description="Pharos evaluation (TriHaze protocol)")
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--override", nargs="*", default=[])
    args = ap.parse_args(argv)

    cfg = load_config(args.config, parse_overrides(args.override))
    cfg.setdefault("exp_name", Path(args.config).stem)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = default_build_model(cfg).to(device)
    _load_model_from_ckpt(model, args.ckpt, device)
    out_dir = Path(cfg["out_root"]) / cfg["exp_name"] / "eval"
    res = evaluate(model, cfg, device=device, out_dir=out_dir)
    print(json.dumps(res.get("paired", {}), indent=2, default=_json_default))


if __name__ == "__main__":
    main()
