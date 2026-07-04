"""TensorBoard logging wrapper for the Pharos engine.

Degrades to a silent no-op if ``tensorboard`` is unavailable so training and
tests never hard-fail on a missing logging backend.
"""
from __future__ import annotations

import warnings
from pathlib import Path
from typing import Optional

import torch

__all__ = ["TBLogger"]


class TBLogger:
    """Thin wrapper over ``torch.utils.tensorboard.SummaryWriter``.

    Parameters
    ----------
    log_dir:
        Directory for event files. If None or the backend is missing, the logger
        is disabled and all methods become no-ops.
    """

    def __init__(self, log_dir: Optional[str | Path], enabled: bool = True) -> None:
        self.writer = None
        if not enabled or log_dir is None:
            return
        try:
            from torch.utils.tensorboard import SummaryWriter

            self.writer = SummaryWriter(log_dir=str(log_dir))
        except Exception as e:  # pragma: no cover - env dependent
            warnings.warn(f"TensorBoard disabled ({e}); scalars/images will not be logged.")
            self.writer = None

    @property
    def enabled(self) -> bool:
        return self.writer is not None

    def scalar(self, tag: str, value: float, step: int) -> None:
        if self.writer is not None:
            self.writer.add_scalar(tag, float(value), step)

    def scalars(self, prefix: str, values: dict[str, float], step: int) -> None:
        for k, v in values.items():
            self.scalar(f"{prefix}/{k}" if prefix else k, v, step)

    def image_panel(
        self,
        tag: str,
        panels: dict[str, torch.Tensor],
        step: int,
        max_items: int = 4,
    ) -> None:
        """Log a labelled row of image tensors (each ``B,C,H,W`` or ``B,1,H,W``).

        Confidence / single-channel maps are broadcast to 3 channels for display.
        """
        if self.writer is None:
            return
        try:
            import torchvision.utils as vutils
        except Exception:
            vutils = None
        for name, img in panels.items():
            if img is None:
                continue
            img = img.detach().float().cpu()
            if img.dim() == 3:
                img = img.unsqueeze(0)
            img = img[:max_items].clamp(0, 1)
            if img.shape[1] == 1:
                img = img.repeat(1, 3, 1, 1)
            if vutils is not None:
                grid = vutils.make_grid(img, nrow=img.shape[0])
                self.writer.add_image(f"{tag}/{name}", grid, step)
            else:  # pragma: no cover
                self.writer.add_images(f"{tag}/{name}", img, step)

    def flush(self) -> None:
        if self.writer is not None:
            self.writer.flush()

    def close(self) -> None:
        if self.writer is not None:
            self.writer.flush()
            self.writer.close()
            self.writer = None
