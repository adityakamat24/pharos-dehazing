"""Flow teacher: frozen torchvision RAFT-small (training-time only).

Computes optical flow between two *clean* frames for the temporal warp loss.
Flow never runs at inference. Also exposes `flow_warp`, a grid_sample-based
backward-warp utility.
"""
from __future__ import annotations

import importlib.util

import torch
import torch.nn.functional as F


def _dep_available(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        return False


def flow_warp(img: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
    """Backward-warp `img` by `flow` (both B,·,H,W; flow is B,2,H,W in pixels).

    Samples `img` at (grid + flow): with flow a->b, `flow_warp(img_b, flow)`
    aligns frame b into frame a's coordinates. A zero flow returns `img`
    unchanged (align_corners=True, base grid built from linspace(-1, 1)).
    """
    b, _, h, w = img.shape
    device = img.device
    ys = torch.linspace(-1.0, 1.0, h, device=device)
    xs = torch.linspace(-1.0, 1.0, w, device=device)
    base_y, base_x = torch.meshgrid(ys, xs, indexing="ij")
    base = torch.stack((base_x, base_y), dim=0).unsqueeze(0)  # 1,2,H,W
    # convert pixel-space flow to normalized [-1,1] displacement
    norm = torch.tensor([2.0 / max(w - 1, 1), 2.0 / max(h - 1, 1)], device=device).view(1, 2, 1, 1)
    grid = base + flow * norm  # B,2,H,W
    grid = grid.permute(0, 2, 3, 1)  # B,H,W,2 (x,y)
    return F.grid_sample(img, grid, mode="bilinear", padding_mode="border", align_corners=True)


class FlowTeacher:
    """Lazy, frozen `raft_small(weights=DEFAULT)`; returns flow from A to B."""

    def __init__(self, device: str | torch.device = "cpu") -> None:
        self.device = torch.device(device)
        self.available: bool = _dep_available("torchvision")
        self._model = None
        self._loaded = False

    def _load(self) -> None:
        if self._loaded:
            return
        self._loaded = True
        if not self.available:
            return
        try:
            from torchvision.models.optical_flow import Raft_Small_Weights, raft_small

            model = raft_small(weights=Raft_Small_Weights.DEFAULT)
            model.eval().to(self.device)
            for p in model.parameters():
                p.requires_grad_(False)
            self._model = model
        except Exception:
            self.available = False
            self._model = None

    @staticmethod
    def _pad_to_multiple(x: torch.Tensor, m: int = 8) -> tuple[torch.Tensor, tuple[int, int]]:
        _, _, h, w = x.shape
        ph = (m - h % m) % m
        pw = (m - w % m) % m
        if ph or pw:
            x = F.pad(x, (0, pw, 0, ph), mode="replicate")
        return x, (h, w)

    @torch.no_grad()
    def __call__(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """a, b: B,3,H,W float in [0,1]. Returns B,2,H,W flow from a to b."""
        bsz, _, h, w = a.shape
        if not self._loaded:
            self._load()
        if not self.available or self._model is None:
            return torch.zeros((bsz, 2, h, w), device=a.device, dtype=a.dtype)
        # RAFT wants [-1,1] inputs and spatial dims divisible by 8. It is NOT
        # fp16-safe: run in an autocast-free fp32 island (we may be called from
        # inside the trainer's AMP context).
        with torch.autocast(device_type=self.device.type if hasattr(self.device, "type") else "cuda",
                            enabled=False):
            a_n = a.to(self.device).float() * 2.0 - 1.0
            b_n = b.to(self.device).float() * 2.0 - 1.0
            a_p, (oh, ow) = self._pad_to_multiple(a_n)
            b_p, _ = self._pad_to_multiple(b_n)
            preds = self._model(a_p, b_p)  # list of iterative predictions
            flow = preds[-1][:, :, :oh, :ow]
        return flow.to(a.dtype)
