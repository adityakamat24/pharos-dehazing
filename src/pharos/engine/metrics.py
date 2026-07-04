"""Image / video quality metrics for the Pharos engine.

Pure-torch PSNR and SSIM (no external deps, CPU-testable), an optional LPIPS
wrapper (lazy import; degrades gracefully if the ``lpips`` package is absent),
and temporal warp-error utilities used by the TriHaze eval protocol (DESIGN §7).

All functions accept ``B,C,H,W`` float tensors in ``[0, 1]`` unless noted.
"""
from __future__ import annotations

import warnings
from typing import Optional

import torch
import torch.nn.functional as F

__all__ = [
    "psnr",
    "ssim",
    "LPIPS",
    "warp",
    "warp_error",
    "frame_diff",
]


def _as_bchw(x: torch.Tensor) -> torch.Tensor:
    if x.dim() == 3:
        x = x.unsqueeze(0)
    if x.dim() != 4:
        raise ValueError(f"expected B,C,H,W (or C,H,W) tensor, got shape {tuple(x.shape)}")
    return x


def psnr(x: torch.Tensor, y: torch.Tensor, *, max_val: float = 1.0) -> float:
    """Mean PSNR (dB) over the batch. Identical inputs return ``inf``.

    MSE is taken per image over C,H,W so a single black frame does not dominate.
    """
    x = _as_bchw(x).float()
    y = _as_bchw(y).float()
    if x.shape != y.shape:
        raise ValueError(f"shape mismatch {tuple(x.shape)} vs {tuple(y.shape)}")
    mse = ((x - y) ** 2).flatten(1).mean(dim=1)  # B
    out = torch.empty_like(mse)
    zero = mse == 0
    out[zero] = float("inf")
    nz = ~zero
    out[nz] = 10.0 * torch.log10((max_val ** 2) / mse[nz])
    return float(out.mean().item())


def _gaussian_window(window_size: int, sigma: float, channels: int, device, dtype) -> torch.Tensor:
    coords = torch.arange(window_size, device=device, dtype=dtype) - (window_size - 1) / 2.0
    g = torch.exp(-(coords ** 2) / (2.0 * sigma ** 2))
    g = g / g.sum()
    win2d = g[:, None] @ g[None, :]  # window_size x window_size
    return win2d.expand(channels, 1, window_size, window_size).contiguous()


def ssim(
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    max_val: float = 1.0,
    window_size: int = 11,
    sigma: float = 1.5,
) -> float:
    """Mean SSIM over the batch (Wang et al. 2004). Identical inputs return 1.0."""
    x = _as_bchw(x).float()
    y = _as_bchw(y).float()
    if x.shape != y.shape:
        raise ValueError(f"shape mismatch {tuple(x.shape)} vs {tuple(y.shape)}")
    c = x.shape[1]
    pad = window_size // 2
    win = _gaussian_window(window_size, sigma, c, x.device, x.dtype)
    c1 = (0.01 * max_val) ** 2
    c2 = (0.03 * max_val) ** 2

    def _filt(t: torch.Tensor) -> torch.Tensor:
        return F.conv2d(t, win, padding=pad, groups=c)

    mu_x, mu_y = _filt(x), _filt(y)
    mu_x2, mu_y2, mu_xy = mu_x * mu_x, mu_y * mu_y, mu_x * mu_y
    sig_x2 = _filt(x * x) - mu_x2
    sig_y2 = _filt(y * y) - mu_y2
    sig_xy = _filt(x * y) - mu_xy
    num = (2 * mu_xy + c1) * (2 * sig_xy + c2)
    den = (mu_x2 + mu_y2 + c1) * (sig_x2 + sig_y2 + c2)
    ssim_map = num / den
    return float(ssim_map.mean().item())


class LPIPS:
    """Lazy, optional LPIPS wrapper. ``available`` is False if ``lpips`` is missing.

    Usage::

        m = LPIPS(net="alex")
        if m.available:
            d = m(pred, target)   # both B,3,H,W in [0,1]; returns mean float
    """

    def __init__(self, net: str = "alex", device: Optional[torch.device | str] = None) -> None:
        self.net = net
        self.device = torch.device(device) if device is not None else torch.device("cpu")
        self._model = None
        self.available = False
        try:
            import lpips  # noqa: F401

            self.available = True
        except Exception as e:  # pragma: no cover - depends on env
            warnings.warn(f"LPIPS unavailable ({e}); LPIPS scores will be skipped.")

    def _ensure(self) -> None:
        if self._model is None:
            import lpips

            self._model = lpips.LPIPS(net=self.net, verbose=False).to(self.device).eval()
            for p in self._model.parameters():
                p.requires_grad_(False)

    @torch.no_grad()
    def __call__(self, x: torch.Tensor, y: torch.Tensor) -> float:
        if not self.available:
            raise RuntimeError("LPIPS backend not installed")
        self._ensure()
        x = _as_bchw(x).to(self.device).float()
        y = _as_bchw(y).to(self.device).float()
        # lpips expects inputs in [-1, 1].
        d = self._model(x * 2 - 1, y * 2 - 1)
        return float(d.mean().item())


def warp(img: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
    """Backward-warp ``img`` by ``flow`` (B,2,H,W, pixel units, dx=chan0, dy=chan1).

    Samples ``img`` at ``(x + flow_x, y + flow_y)``. Given ``flow`` prev->curr,
    ``warp(curr, flow)`` reconstructs an estimate of the previous frame.
    """
    img = _as_bchw(img).float()
    b, _, h, w = img.shape
    ys, xs = torch.meshgrid(
        torch.arange(h, device=img.device, dtype=img.dtype),
        torch.arange(w, device=img.device, dtype=img.dtype),
        indexing="ij",
    )
    base = torch.stack((xs, ys), dim=0)[None].expand(b, -1, -1, -1)  # B,2,H,W
    coords = base + flow
    # normalize to [-1, 1] for grid_sample
    coords_x = 2.0 * coords[:, 0] / max(w - 1, 1) - 1.0
    coords_y = 2.0 * coords[:, 1] / max(h - 1, 1) - 1.0
    grid = torch.stack((coords_x, coords_y), dim=-1)  # B,H,W,2
    return F.grid_sample(img, grid, mode="bilinear", padding_mode="border", align_corners=True)


def frame_diff(prev: torch.Tensor, curr: torch.Tensor) -> float:
    """Flow-free temporal-consistency proxy: mean absolute inter-frame difference."""
    prev = _as_bchw(prev).float()
    curr = _as_bchw(curr).float()
    return float((curr - prev).abs().mean().item())


def warp_error(
    prev: torch.Tensor,
    curr: torch.Tensor,
    flow: Optional[torch.Tensor] = None,
) -> float:
    """Temporal warp error between consecutive (restored) frames.

    With ``flow`` (prev->curr, from a clean-frame teacher), returns the mean L1
    between ``prev`` and ``warp(curr, flow)``. Without flow, falls back to the
    documented frame-difference proxy (DESIGN §7: "else frame-diff proxy").
    """
    if flow is None:
        return frame_diff(prev, curr)
    prev = _as_bchw(prev).float()
    curr = _as_bchw(curr).float()
    warped = warp(curr, flow)
    return float((warped - prev).abs().mean().item())
