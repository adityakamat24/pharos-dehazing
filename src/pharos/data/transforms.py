"""Shared image I/O and crop/flip/resize utilities for Pharos data pipelines.

Everything operates on float32 CHW tensors in [0,1] (single image) or paired lists
of such tensors that must receive an identical geometric transform. No GPU, no
external state; a `torch.Generator` is threaded through for reproducibility.
"""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
import torch
import torch.nn.functional as F

IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp")


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------
def list_images(folder: str | Path) -> list[Path]:
    """Sorted list of image files directly under `folder` (non-recursive)."""
    folder = Path(folder)
    if not folder.is_dir():
        return []
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in IMG_EXTS)


def list_images_recursive(folder: str | Path) -> list[Path]:
    """Sorted list of image files anywhere under `folder`."""
    folder = Path(folder)
    if not folder.is_dir():
        return []
    return sorted(p for p in folder.rglob("*") if p.suffix.lower() in IMG_EXTS)


def load_image(path: str | Path) -> torch.Tensor:
    """Load an image as a float32 RGB CHW tensor in [0,1]."""
    path = str(path)
    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"could not read image: {path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return to_tensor(rgb)


def to_tensor(arr: np.ndarray) -> torch.Tensor:
    """HWC uint8/float ndarray -> CHW float32 tensor in [0,1]."""
    if arr.ndim == 2:
        arr = arr[:, :, None]
    t = torch.from_numpy(np.ascontiguousarray(arr))
    if t.dtype == torch.uint8:
        t = t.float() / 255.0
    else:
        t = t.float()
    return t.permute(2, 0, 1).contiguous()


def to_numpy_u8(img: torch.Tensor) -> np.ndarray:
    """CHW float [0,1] tensor -> HWC uint8 RGB ndarray."""
    arr = (img.clamp(0, 1) * 255.0 + 0.5).to(torch.uint8)
    return arr.permute(1, 2, 0).contiguous().cpu().numpy()


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------
def resize(img: torch.Tensor, size: int | tuple[int, int], mode: str = "bilinear") -> torch.Tensor:
    """Resize a CHW tensor. `size` is an int (square) or (H, W)."""
    if isinstance(size, int):
        out_hw = (size, size)
    else:
        out_hw = size
    align = None if mode in ("nearest", "area") else False
    out = F.interpolate(img.unsqueeze(0), size=out_hw, mode=mode, align_corners=align)
    return out.squeeze(0)


def resize_shorter(img: torch.Tensor, target: int, mode: str = "bilinear") -> torch.Tensor:
    """Resize so the shorter side equals `target`, preserving aspect ratio."""
    _, h, w = img.shape
    if min(h, w) == target:
        return img
    scale = target / float(min(h, w))
    return resize(img, (max(1, round(h * scale)), max(1, round(w * scale))), mode=mode)


def _rand_int(high: int, generator: torch.Generator | None) -> int:
    if high <= 0:
        return 0
    return int(torch.randint(0, high + 1, (1,), generator=generator).item())


def paired_random_crop(
    imgs: Sequence[torch.Tensor], size: int, generator: torch.Generator | None = None
) -> list[torch.Tensor]:
    """Identical random crop of `size` x `size` applied to every tensor in `imgs`.

    If an image is smaller than `size` on either axis it is first resized up so the
    shorter side is `size` (aspect preserved), then cropped.
    """
    ref = imgs[0]
    _, h, w = ref.shape
    if h < size or w < size:
        imgs = [resize_shorter(im, size) for im in imgs]
        _, h, w = imgs[0].shape
    top = _rand_int(h - size, generator)
    left = _rand_int(w - size, generator)
    return [im[:, top : top + size, left : left + size] for im in imgs]


def random_crop(img: torch.Tensor, size: int, generator: torch.Generator | None = None) -> torch.Tensor:
    return paired_random_crop([img], size, generator)[0]


def center_crop(img: torch.Tensor, size: int) -> torch.Tensor:
    _, h, w = img.shape
    if h < size or w < size:
        img = resize_shorter(img, size)
        _, h, w = img.shape
    top = (h - size) // 2
    left = (w - size) // 2
    return img[:, top : top + size, left : left + size]


def paired_random_flip(
    imgs: Sequence[torch.Tensor], generator: torch.Generator | None = None, vertical: bool = False
) -> list[torch.Tensor]:
    """Random horizontal (and optional vertical) flip applied identically to all imgs."""
    out = list(imgs)
    if torch.rand(1, generator=generator).item() < 0.5:
        out = [torch.flip(im, dims=[2]) for im in out]
    if vertical and torch.rand(1, generator=generator).item() < 0.5:
        out = [torch.flip(im, dims=[1]) for im in out]
    return out


def make_lowres(img: torch.Tensor, lowres: int) -> torch.Tensor:
    """HDRNet-style global-context stream: downsample the *full* image to a square
    `lowres` x `lowres` tensor (so per-sample lowres tensors stack in a batch)."""
    return resize(img, lowres, mode="bilinear").clamp(0, 1)


def pad_to_multiple(img: torch.Tensor, multiple: int) -> tuple[torch.Tensor, tuple[int, int]]:
    """Reflect-pad a CHW tensor so H and W are multiples of `multiple`. Returns
    (padded, (orig_h, orig_w)) so the caller can crop back."""
    _, h, w = img.shape
    ph = (multiple - h % multiple) % multiple
    pw = (multiple - w % multiple) % multiple
    if ph == 0 and pw == 0:
        return img, (h, w)
    padded = F.pad(img.unsqueeze(0), (0, pw, 0, ph), mode="reflect").squeeze(0)
    return padded, (h, w)
