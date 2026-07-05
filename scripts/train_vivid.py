"""Launcher for vivid-mode training (DESIGN.md N4 photography variant).

Thin wrapper over :class:`pharos.engine.train.Trainer` that injects the vivid
components through the engine's ``Deps`` seam without touching the frozen engine:

    model -> pharos.models.PharosNet (optionally initialised from the fine-tuned
             checkpoint via model.vivid_init; EMA shadow preferred, then model)
    loss  -> pharos.losses.vivid_losses.VividLoss (self-manages the discriminator
             and its own optimizer; the engine needs no changes)

Datasets use the standard v1 factory (default Deps) — vivid trains on the same
real+synthetic image mix as the fine-tune stage. VividLoss lives in this repo, so
it is imported normally; PharosNet is imported lazily with a clear error message.

    python scripts/train_vivid.py --config configs/vivid.yaml [--override k=v ...]
    python scripts/train_vivid.py --config configs/vivid.yaml --resume last
"""
from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path
from typing import Any, Optional

# Make the src-layout `pharos` package importable when run as a plain script.
_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


def _lazy(module: str, attr: str, *, why: str) -> Any:
    """Import ``module.attr`` lazily with a clear, actionable error message."""
    try:
        mod = importlib.import_module(module)
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(
            f"Could not import '{module}' ({e}). {why} Ensure that workstream is "
            f"present/merged before launching vivid training."
        ) from e
    obj = getattr(mod, attr, None)
    if obj is None:
        raise RuntimeError(f"'{module}' has no attribute '{attr}'. {why}")
    return obj


def _try_calls(fn, arg_sets: list[tuple]):
    """Call ``fn`` with each argument tuple, skipping TypeError; re-raise the last."""
    last: Optional[BaseException] = None
    for args in arg_sets:
        try:
            return fn(*args)
        except TypeError as e:
            last = e
    raise last  # type: ignore[misc]


# ---------------------------------------------------------------------------
# injected factories
# ---------------------------------------------------------------------------
def build_model(cfg: Any) -> Any:
    """Build the deployed PharosNet, optionally initialised from a checkpoint.

    ``model.vivid_init`` (a fine-tuned checkpoint path) loads the EMA shadow when
    present, else the raw model weights — mirrors train_reveal's inner_ckpt logic.
    """
    PharosNet = _lazy("pharos.models", "PharosNet", why="PharosNet is the deployed net.")
    model_cfg = cfg.get("model", {}) if hasattr(cfg, "get") else {}
    net = _try_calls(PharosNet, [(model_cfg,), (cfg,), ()])

    init = model_cfg.get("vivid_init") if isinstance(model_cfg, dict) else None
    if init:
        import torch

        ck = torch.load(init, map_location="cpu", weights_only=False)
        state = ck.get("ema", {}).get("shadow") if isinstance(ck.get("ema"), dict) else None
        state = state or ck.get("model") or ck
        net.load_state_dict(state, strict=True)
        print(f"[train_vivid] PharosNet initialised from {init}")
    return net


def build_loss(cfg: Any) -> Any:
    """Build the self-contained VividLoss (owns the discriminator + its optimizer)."""
    from pharos.losses.vivid_losses import VividLoss

    return VividLoss(cfg)


def make_deps():
    """Assemble the engine Deps container with the vivid factories injected.

    Only model + loss are injected; datasets use the standard v1 image factory.
    """
    from pharos.engine.train import Deps

    return Deps(build_model=build_model, build_loss=build_loss)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser("train_vivid", description="Pharos vivid-mode training launcher")
    p.add_argument("--config", default="configs/vivid.yaml", help="config YAML")
    p.add_argument("--override", nargs="*", default=[], help="dotted key=value overrides")
    p.add_argument("--resume", default=None, help="checkpoint path or 'last'")
    return p


def main(argv: Optional[list[str]] = None) -> None:
    from pharos.config import load_config
    from pharos.engine.train import Trainer
    from pharos.engine.utils import parse_overrides

    args = build_parser().parse_args(argv)
    cfg = load_config(args.config, parse_overrides(args.override))
    cfg.setdefault("exp_name", Path(args.config).stem)
    if args.resume:
        cfg["train"]["resume"] = args.resume
    trainer = Trainer(cfg, deps=make_deps())
    trainer.train()


if __name__ == "__main__":  # Windows-safe dataloader entry guard
    main()
