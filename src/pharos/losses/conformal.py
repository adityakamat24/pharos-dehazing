"""Post-hoc conformal calibration of the confidence head (DESIGN.md §5, N4).

The confidence head yields a predicted sigma per pixel (in the loss we use
sigma = 1 / confidence). Split-conformal prediction turns those raw sigmas into
a distribution-free coverage guarantee: given held-out pairs (sigma_i, |err_i|),
we find a scalar `q_alpha` such that the calibrated band `|err| <= q_alpha·sigma`
covers at least `1 - alpha` of the validation errors. `q_alpha` is stored in the
checkpoint meta and multiplies sigma at inference.

Score: r_i = |err_i| / sigma_i (nonconformity). With a finite-sample correction,
q_alpha is the ceil((n+1)(1-alpha))/n empirical quantile of {r_i}, guaranteeing
marginal coverage >= 1 - alpha.
"""
from __future__ import annotations

import argparse
from typing import Any, Iterable

import numpy as np
import torch


def calibrate(sigmas: Any, errors: Any, alpha: float = 0.1, eps: float = 1e-8) -> float:
    """Return the conformal scale `q_alpha` from (sigma, |error|) pairs.

    sigmas, errors: array-likes (numpy / torch / list) of matching length.
    """
    s = _to_1d_numpy(sigmas)
    e = _to_1d_numpy(errors)
    if s.shape != e.shape or s.size == 0:
        raise ValueError("sigmas and errors must be non-empty and the same length")
    scores = np.abs(e) / np.maximum(s, eps)
    n = scores.size
    # finite-sample corrected quantile level, clipped to [0, 1]
    level = min(1.0, np.ceil((n + 1) * (1.0 - alpha)) / n)
    return float(np.quantile(scores, level, method="higher"))


def coverage(sigmas: Any, errors: Any, q: float, eps: float = 1e-8) -> float:
    """Empirical coverage of the band |error| <= q·sigma."""
    s = _to_1d_numpy(sigmas)
    e = _to_1d_numpy(errors)
    return float(np.mean(np.abs(e) <= q * np.maximum(s, eps)))


@torch.no_grad()
def collect_pairs(
    model: Any,
    loader: Iterable[dict],
    device: str | torch.device = "cpu",
    max_pixels: int = 2_000_000,
    conf_eps: float = 1e-4,
) -> tuple[np.ndarray, np.ndarray]:
    """Run `model` over a val loader, gather (sigma, |error|) pixel pairs.

    `model` is any object with `.forward` (contracts.PharosModel); we deliberately
    do NOT import pharos.models. sigma = 1/confidence to match PharosLoss. Pairs
    are subsampled to at most `max_pixels` to bound memory.
    """
    device = torch.device(device)
    sig_chunks: list[np.ndarray] = []
    err_chunks: list[np.ndarray] = []
    total = 0
    for batch in loader:
        hazy = batch["hazy"].to(device)
        clean = batch.get("clean")
        if clean is None:
            continue
        clean = clean.to(device)
        if hazy.dim() == 5:  # clip -> use current (last) frame
            hazy, clean = hazy[:, -1], clean[:, -1]
        out = model.forward(hazy)
        err = (out.output - clean).abs().mean(dim=1, keepdim=True)
        conf = out.confidence
        if conf.shape[-2:] != err.shape[-2:]:
            conf = torch.nn.functional.interpolate(conf, size=err.shape[-2:], mode="bilinear")
        sigma = 1.0 / conf.clamp(conf_eps, 1.0)
        s = sigma.reshape(-1).float().cpu().numpy()
        e = err.reshape(-1).float().cpu().numpy()
        sig_chunks.append(s)
        err_chunks.append(e)
        total += s.size
        if total >= max_pixels:
            break
    if not sig_chunks:
        return np.empty(0), np.empty(0)
    sig = np.concatenate(sig_chunks)
    err = np.concatenate(err_chunks)
    if sig.size > max_pixels:  # uniform subsample
        idx = np.random.default_rng(0).choice(sig.size, size=max_pixels, replace=False)
        sig, err = sig[idx], err[idx]
    return sig, err


def calibrate_model(
    model: Any, loader: Iterable[dict], device: str | torch.device = "cpu", alpha: float = 0.1
) -> float:
    """Convenience: collect pairs over a loader then calibrate."""
    sig, err = collect_pairs(model, loader, device=device)
    return calibrate(sig, err, alpha=alpha)


def _to_1d_numpy(x: Any) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        x = x.detach().float().cpu().numpy()
    return np.asarray(x, dtype=np.float64).reshape(-1)


def main(argv: list[str] | None = None) -> None:
    """CLI over a saved .npz of {'sigmas','errors'} (produced from a val split).

    Model construction lives in the training workstream, so this entry calibrates
    from pre-dumped pairs; `calibrate_model(model, loader)` is the programmatic
    path when a model is available.
    """
    ap = argparse.ArgumentParser(description="Conformal calibration of Pharos confidence.")
    ap.add_argument("pairs", help="path to .npz with arrays 'sigmas' and 'errors'")
    ap.add_argument("--alpha", type=float, default=0.1)
    args = ap.parse_args(argv)

    data = np.load(args.pairs)
    q = calibrate(data["sigmas"], data["errors"], alpha=args.alpha)
    cov = coverage(data["sigmas"], data["errors"], q)
    print(f"q_alpha={q:.6f}  target_coverage={1 - args.alpha:.3f}  empirical_coverage={cov:.4f}")


if __name__ == "__main__":
    main()
