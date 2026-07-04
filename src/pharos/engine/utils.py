"""Engine utilities: seeding, meters/timers, EMA, LR schedule, checkpoint I/O.

Small, dependency-light helpers shared by ``train.py`` and ``eval.py``. Nothing
here needs a GPU, dataset, or teacher import.
"""
from __future__ import annotations

import copy
import math
import os
import random
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch

__all__ = [
    "seed_everything",
    "AverageMeter",
    "Timer",
    "ModelEMA",
    "cosine_warmup_lambda",
    "parse_overrides",
    "move_batch_to_device",
    "resolve_run_dirs",
    "save_checkpoint",
    "load_checkpoint",
]


def seed_everything(seed: int, *, deterministic: bool = False) -> None:
    """Seed python / numpy / torch (CPU+CUDA). ``deterministic`` toggles cudnn."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed % (2 ** 32))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True


class AverageMeter:
    """Running mean of a scalar."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.sum = 0.0
        self.count = 0

    def update(self, value: float, n: int = 1) -> None:
        self.sum += float(value) * n
        self.count += n

    @property
    def avg(self) -> float:
        return self.sum / self.count if self.count else 0.0


class Timer:
    """Wall-clock timer; ``lap`` returns seconds since the previous lap/reset."""

    def __init__(self) -> None:
        self._t = time.perf_counter()

    def reset(self) -> None:
        self._t = time.perf_counter()

    def lap(self) -> float:
        now = time.perf_counter()
        dt = now - self._t
        self._t = now
        return dt


class ModelEMA:
    """Exponential moving average of a model's float params & buffers.

    Kept on CPU-or-GPU alongside the model. ``update`` after each optimizer step,
    ``copy_to`` to load EMA weights into a model for eval, and ``store``/``restore``
    to temporarily swap and put the training weights back.
    """

    def __init__(self, model: torch.nn.Module, decay: float = 0.999) -> None:
        self.decay = decay
        self.shadow = copy.deepcopy(model.state_dict())
        for k, v in self.shadow.items():
            if torch.is_tensor(v):
                self.shadow[k] = v.detach().clone()
        self._backup: Optional[dict[str, torch.Tensor]] = None

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        d = self.decay
        msd = model.state_dict()
        for k, v in self.shadow.items():
            if not torch.is_tensor(v):
                continue
            new = msd[k]
            if v.dtype.is_floating_point:
                v.mul_(d).add_(new.detach().to(v.device), alpha=1.0 - d)
            else:
                v.copy_(new.detach().to(v.device))

    def copy_to(self, model: torch.nn.Module) -> None:
        model.load_state_dict(self.shadow, strict=False)

    def store(self, model: torch.nn.Module) -> None:
        self._backup = copy.deepcopy(model.state_dict())

    def restore(self, model: torch.nn.Module) -> None:
        if self._backup is not None:
            model.load_state_dict(self._backup, strict=False)
            self._backup = None

    def state_dict(self) -> dict[str, Any]:
        return {"decay": self.decay, "shadow": self.shadow}

    def load_state_dict(self, sd: dict[str, Any]) -> None:
        self.decay = sd.get("decay", self.decay)
        self.shadow = sd["shadow"]


def cosine_warmup_lambda(warmup_iters: int, total_iters: int, min_ratio: float = 0.0):
    """LR multiplier: linear warmup 0->1 then cosine decay 1->min_ratio.

    Returns a callable suitable for ``torch.optim.lr_scheduler.LambdaLR``.
    """
    warmup_iters = max(int(warmup_iters), 0)
    total_iters = max(int(total_iters), warmup_iters + 1)

    def fn(step: int) -> float:
        if warmup_iters > 0 and step < warmup_iters:
            return (step + 1) / warmup_iters
        progress = (step - warmup_iters) / max(total_iters - warmup_iters, 1)
        progress = min(max(progress, 0.0), 1.0)
        return min_ratio + (1.0 - min_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))

    return fn


def _parse_scalar(raw: str) -> Any:
    """YAML-parse a scalar, rescuing forms YAML 1.1 misses (e.g. ``3e-4`` -> float).

    PyYAML's implicit float resolver requires a decimal point, so scientific
    notation without one is returned as a string; we retry int() then float().
    """
    import yaml

    v = yaml.safe_load(raw)
    if isinstance(v, str):
        s = v.strip()
        for cast in (int, float):
            try:
                return cast(s)
            except ValueError:
                continue
    return v


def parse_overrides(pairs: Optional[list[str]]) -> dict[str, Any]:
    """Parse ``["train.lr=3e-4", "train.amp=false"]`` into a dotted-key dict.

    Values are parsed as typed scalars (numbers/bools/None) via ``_parse_scalar``.
    """
    out: dict[str, Any] = {}
    for item in pairs or []:
        if "=" not in item:
            raise ValueError(f"override '{item}' must be key=value")
        key, raw = item.split("=", 1)
        out[key.strip()] = _parse_scalar(raw)
    return out


def move_batch_to_device(batch: dict, device: torch.device, *, channels_last: bool = False) -> dict:
    """Move a contract batch dict to ``device`` (tensors only; ``meta`` untouched)."""
    out: dict[str, Any] = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            v = v.to(device, non_blocking=True)
            if channels_last and v.dim() == 4:
                v = v.contiguous(memory_format=torch.channels_last)
            out[k] = v
        else:
            out[k] = v
    return out


def resolve_run_dirs(cfg) -> dict[str, Path]:
    """Compute and create the run directory tree under ``out_root/<exp_name>``."""
    out_root = Path(cfg["out_root"])
    exp = cfg.get("exp_name", "default")
    run = out_root / exp
    dirs = {
        "run": run,
        "ckpt": run / "ckpt",
        "tb": run / "tb",
        "eval": run / "eval",
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    return dirs


def save_checkpoint(path: str | Path, state: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, tmp)
    os.replace(tmp, path)


def load_checkpoint(path: str | Path, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    return torch.load(Path(path), map_location=map_location, weights_only=False)
