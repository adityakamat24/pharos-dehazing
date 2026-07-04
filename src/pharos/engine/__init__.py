"""Pharos training / evaluation engine (WS-D).

Public surface (lazily imported so ``python -m pharos.engine.train`` / ``.eval``
do not re-import the package's submodules eagerly):
    Trainer, Deps, pharos_collate, build_trainer_from_config  -- training with DI
    evaluate                                                  -- TriHaze eval
    metrics                                                   -- psnr/ssim/LPIPS/warp
"""
from __future__ import annotations

import importlib
from typing import Any

__all__ = [
    "Trainer",
    "Deps",
    "pharos_collate",
    "build_trainer_from_config",
    "evaluate",
    "metrics",
]

_LAZY = {
    "Trainer": ("pharos.engine.train", "Trainer"),
    "Deps": ("pharos.engine.train", "Deps"),
    "pharos_collate": ("pharos.engine.train", "pharos_collate"),
    "build_trainer_from_config": ("pharos.engine.train", "build_trainer_from_config"),
    "evaluate": ("pharos.engine.eval", "evaluate"),
    "metrics": ("pharos.engine.metrics", None),
}


def __getattr__(name: str) -> Any:
    if name not in _LAZY:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module, attr = _LAZY[name]
    mod = importlib.import_module(module)
    return mod if attr is None else getattr(mod, attr)
