"""TeacherBundle: assembles the training-time teachers from config.

Satisfies the `contracts.TeacherBundle` protocol: attributes `depth`,
`detector`, `flow` are callables when enabled+available, else None. Every
teacher lazy-loads on first use, so building a bundle never touches the network.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import torch

from .depth import DepthTeacher
from .detector import DetectionTeacher
from .flow import FlowTeacher


@dataclass
class TeacherBundle:
    """Container of optional teachers. None = disabled or dependency/weights missing.

    - depth(img B,3,H,W) -> B,1,h,w relative depth (higher = farther)
    - detector(img B,3,H,W) -> list of FPN feature maps
    - flow(a, b) -> B,2,H,W flow a->b
    """

    depth: Optional[Any] = None
    detector: Optional[Any] = None
    flow: Optional[Any] = None

    @classmethod
    def from_config(cls, cfg: Any, device: str | torch.device = "cpu") -> "TeacherBundle":
        """Build from `cfg.teachers` (see configs/base.yaml).

        A teacher is wired in only when its section has `enabled: true` and the
        constructed teacher reports `.available`. Unknown/missing sections are
        treated as disabled.
        """
        tcfg = _get(cfg, "teachers", {}) or {}

        depth = None
        if _enabled(tcfg, "depth"):
            weights_dir = _get(cfg, "data_root", None)
            kwargs: dict[str, Any] = {"device": device}
            if weights_dir:
                kwargs["cache_dir"] = f"{weights_dir}/weights"
            t = DepthTeacher(**kwargs)
            depth = t if t.available else None

        detector = None
        if _enabled(tcfg, "detector"):
            t = DetectionTeacher(device=device)
            detector = t if t.available else None

        flow = None
        if _enabled(tcfg, "flow"):
            t = FlowTeacher(device=device)
            flow = t if t.available else None

        return cls(depth=depth, detector=detector, flow=flow)


def build_teachers(cfg: Any, device: str | torch.device | None = None) -> TeacherBundle:
    """Factory used by the training engine (DESIGN §8 / engine Deps.build_teachers).

    Accepts the full config tree (from_config reads cfg.teachers and cfg.data_root).
    Teachers run on CUDA when available unless a device is given explicitly.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    return TeacherBundle.from_config(cfg, device=device)


def _get(obj: Any, key: str, default: Any = None) -> Any:
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _enabled(tcfg: Any, name: str) -> bool:
    section = _get(tcfg, name, None)
    if section is None:
        return False
    return bool(_get(section, "enabled", False))
