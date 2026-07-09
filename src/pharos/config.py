"""Config loading for Pharos. Single YAML tree + dot-key overrides.

Frozen contract: implementers use load_config()/Config and add keys under their
own top-level section in configs/base.yaml only via their experiment YAMLs.
"""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml


class Config(dict):
    """Dict with attribute access, recursively."""

    def __getattr__(self, name: str) -> Any:
        try:
            v = self[name]
        except KeyError as e:
            raise AttributeError(name) from e
        return Config(v) if isinstance(v, dict) else v

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value


def _deep_update(base: dict, upd: dict) -> dict:
    for k, v in upd.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_update(base[k], v)
        else:
            base[k] = v
    return base


def load_config(path: str | Path, overrides: dict[str, Any] | None = None) -> Config:
    """Load a YAML config. If it has an `_base_` key, merge onto that file first.

    overrides: flat dict with dot keys, e.g. {"train.lr": 3e-4}.
    """
    path = Path(path)
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    if "_base_" in cfg:
        base = load_config(path.parent / cfg.pop("_base_"))
        cfg = _deep_update(copy.deepcopy(dict(base)), cfg)
    if overrides:
        for dotkey, value in overrides.items():
            node = cfg
            *parents, leaf = dotkey.split(".")
            for p in parents:
                node = node.setdefault(p, {})
            node[leaf] = value
    return Config(cfg)
