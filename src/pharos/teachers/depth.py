"""Depth teacher: Depth Anything V2 Small (training-time prior only).

Wraps the HuggingFace `depth-anything/Depth-Anything-V2-Small-hf` checkpoint
(Apache-2.0). The model runs on the *clean* image of a pair (see DESIGN.md §4)
and is used for (a) depth-based haze synthesis and (b) the depth-structure
distillation loss. It never runs at inference.

Orientation contract (contracts.TeacherBundle.depth): the returned map is
relative depth normalized to [0, 1] where **higher = farther**. Depth Anything
emits an inverse-depth / disparity map (higher = *closer*), so we min-max
normalize per image and then invert (`1 - x`) to satisfy the contract.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F

# ImageNet normalization used by Depth Anything V2 preprocessing.
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)

DEFAULT_MODEL_ID = "depth-anything/Depth-Anything-V2-Small-hf"
DEFAULT_CACHE_DIR = "D:/dehazing_desmoking/data/weights"


def _dep_available(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        return False


class DepthTeacher:
    """Lazy, device-aware wrapper around Depth Anything V2 Small.

    The constructor never downloads anything: `.available` reflects only whether
    the `transformers` dependency is importable. The checkpoint is fetched (and
    cached under `cache_dir`) on the first `__call__`. If that load fails (no
    weights, no network), `.available` flips to False and subsequent calls return
    a benign zero map so training keeps running with the depth term disabled.
    """

    def __init__(
        self,
        device: str | torch.device = "cpu",
        cache_dir: str | Path = DEFAULT_CACHE_DIR,
        model_id: str = DEFAULT_MODEL_ID,
        input_size: int = 518,
        eps: float = 1e-6,
    ) -> None:
        self.device = torch.device(device)
        self.cache_dir = Path(cache_dir)
        self.model_id = model_id
        self.input_size = int(input_size)
        self.eps = float(eps)
        self.available: bool = _dep_available("transformers")
        self._model = None  # loaded lazily
        self._loaded = False

    # -- lazy loading ------------------------------------------------------
    def _load(self) -> None:
        if self._loaded:
            return
        self._loaded = True
        if not self.available:
            return
        try:
            from transformers import AutoModelForDepthEstimation

            self.cache_dir.mkdir(parents=True, exist_ok=True)
            model = AutoModelForDepthEstimation.from_pretrained(
                self.model_id, cache_dir=str(self.cache_dir)
            )
            model.eval().to(self.device)
            for p in model.parameters():
                p.requires_grad_(False)
            self._model = model
        except Exception:  # network / weights / version issues -> disable gracefully
            self.available = False
            self._model = None

    # -- preprocessing -----------------------------------------------------
    def _preprocess(self, img: torch.Tensor) -> torch.Tensor:
        """[0,1] B,3,H,W -> normalized, resized to a multiple of 14 (long side ~ input_size)."""
        _, _, h, w = img.shape
        scale = self.input_size / max(h, w)
        new_h = max(14, int(round(h * scale / 14)) * 14)
        new_w = max(14, int(round(w * scale / 14)) * 14)
        x = F.interpolate(img, size=(new_h, new_w), mode="bilinear", align_corners=False)
        mean = torch.tensor(_IMAGENET_MEAN, device=x.device).view(1, 3, 1, 1)
        std = torch.tensor(_IMAGENET_STD, device=x.device).view(1, 3, 1, 1)
        return (x - mean) / std

    # -- public API --------------------------------------------------------
    @torch.no_grad()
    def __call__(self, img: torch.Tensor, out_size: Optional[tuple[int, int]] = None) -> torch.Tensor:
        """img: B,3,H,W float in [0,1]. Returns B,1,h,w depth in [0,1], higher = farther."""
        b, _, h, w = img.shape
        out_hw = out_size if out_size is not None else (h, w)
        if not self._loaded:
            self._load()
        if not self.available or self._model is None:
            return torch.zeros((b, 1, out_hw[0], out_hw[1]), device=img.device, dtype=img.dtype)

        # fp32 island: the ViT is not reliably fp16-safe and we may be inside
        # the trainer's AMP autocast context.
        dev_type = self.device.type if hasattr(self.device, "type") else "cuda"
        with torch.autocast(device_type=dev_type, enabled=False):
            img = img.to(self.device).float()
            pixel_values = self._preprocess(img)
            out = self._model(pixel_values=pixel_values)
            depth = out.predicted_depth  # B,h',w' (disparity-like: higher = closer)
        if depth.dim() == 3:
            depth = depth.unsqueeze(1)
        depth = F.interpolate(depth.float(), size=out_hw, mode="bilinear", align_corners=False)

        # per-image min-max to [0,1], then invert so higher = farther.
        flat = depth.view(b, -1)
        mn = flat.min(dim=1).values.view(b, 1, 1, 1)
        mx = flat.max(dim=1).values.view(b, 1, 1, 1)
        depth = (depth - mn) / (mx - mn + self.eps)
        depth = 1.0 - depth
        return depth.to(img.dtype)
