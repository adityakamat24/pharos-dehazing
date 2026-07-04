"""Detection teacher: frozen torchvision Faster R-CNN MobileNetV3 FPN.

Returns the list of FPN feature maps for an image batch, used by the
detection-consistency loss (features on the restored frame vs the clean GT).
Frozen, eval, no gradients. Training-time only.
"""
from __future__ import annotations

import importlib.util

import torch

# torchvision detectors expect ImageNet-normalized inputs.
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


def _dep_available(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        return False


class DetectionTeacher:
    """Lazy, frozen `fasterrcnn_mobilenet_v3_large_fpn(weights=DEFAULT)` backbone.

    `.available` reflects whether torchvision is importable at construction. The
    weights download on first `__call__`; if that fails, `.available` flips to
    False and calls return an empty list so the detection loss is skipped.
    """

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
            from torchvision.models.detection import (
                FasterRCNN_MobileNet_V3_Large_FPN_Weights,
                fasterrcnn_mobilenet_v3_large_fpn,
            )

            weights = FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT
            model = fasterrcnn_mobilenet_v3_large_fpn(weights=weights)
            model.eval().to(self.device)
            for p in model.parameters():
                p.requires_grad_(False)
            self._model = model
        except Exception:
            self.available = False
            self._model = None

    def _normalize(self, img: torch.Tensor) -> torch.Tensor:
        mean = torch.tensor(_IMAGENET_MEAN, device=img.device).view(1, 3, 1, 1)
        std = torch.tensor(_IMAGENET_STD, device=img.device).view(1, 3, 1, 1)
        return (img - mean) / std

    @torch.no_grad()
    def __call__(self, img: torch.Tensor) -> list[torch.Tensor]:
        """img: B,3,H,W float in [0,1]. Returns list of FPN feature maps (B,C,h,w)."""
        if not self._loaded:
            self._load()
        if not self.available or self._model is None:
            return []
        x = self._normalize(img.to(self.device))
        feats = self._model.backbone(x)  # OrderedDict of FPN maps
        return list(feats.values())

    # protocol alias used in some call sites / DESIGN §8
    det_feats = __call__

    @property
    def enabled(self) -> bool:
        return self.available
