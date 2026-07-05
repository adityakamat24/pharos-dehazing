"""Launcher for RevealNet v2 training (DESIGN.md §9d).

Thin wrapper over :class:`pharos.engine.train.Trainer` that injects the v2
components through the engine's ``Deps`` seam without touching the frozen engine:

    model    -> pharos.models.reveal.RevealNet (wraps a PharosNet)
    loss     -> pharos.losses.reveal_losses.RevealLoss (wraps a PharosLoss)
    datasets -> pharos.data.reveal_dataset.build_reveal_dataset for the reveal
                video sets, pharos.data.build_dataset for the image stability mix.

RevealNet and build_reveal_dataset live in parallel workstreams and are imported
*lazily*: absence raises a clear ``RuntimeError`` at launch time (never at import
time), so unit tests that only touch the loss/config never require them.

    python scripts/train_reveal.py --config configs/reveal.yaml [--override k=v ...]
    python scripts/train_reveal.py --config configs/reveal.yaml --resume last
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
            f"present/merged before launching RevealNet training."
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
    """Build RevealNet, wrapping a PharosNet backbone (§9d architecture)."""
    RevealNet = _lazy("pharos.models.reveal", "RevealNet", why="RevealNet is the v2 model.")
    model_cfg = cfg.get("model", {}) if hasattr(cfg, "get") else {}
    reveal_cfg = model_cfg.get("reveal", {}) if isinstance(model_cfg, dict) else {}

    # Strategy 1: build a PharosNet backbone and hand it to RevealNet.
    try:
        PharosNet = _lazy("pharos.models", "PharosNet", why="PharosNet is the v1 backbone.")
        base = _try_calls(PharosNet, [(model_cfg,), (cfg,), ()])
        inner_ckpt = reveal_cfg.get("inner_ckpt") if isinstance(reveal_cfg, dict) else None
        if inner_ckpt:
            import torch

            ck = torch.load(inner_ckpt, map_location="cpu", weights_only=False)
            state = ck.get("ema", {}).get("shadow") if isinstance(ck.get("ema"), dict) else None
            state = state or ck.get("model") or ck
            base.load_state_dict(state, strict=True)
            print(f"[train_reveal] inner PharosNet initialized from {inner_ckpt}")
        if isinstance(reveal_cfg, dict) and reveal_cfg.get("freeze_inner"):
            # Train only the reveal machinery (~33k params): preserves the audited
            # v1 restoration quality exactly and converges far faster. Without
            # this, clip-heavy reveal training drifts the backbone (observed:
            # SmokeBench 20.7->18.7 by step 5k).
            n_frozen = 0
            for name, p in base.named_parameters():
                # conf_head stays trainable: the frozen confidence is miscalibrated
                # out-of-domain (reports ~0.95 under dense synthetic smoke), which
                # starves the memory arbitration. Restoration stays frozen.
                if reveal_cfg.get("train_conf_head", True) and name.startswith("conf_head"):
                    continue
                p.requires_grad = False
                n_frozen += 1
            # Pin eval mode: Trainer calls model.train() every step, which would
            # still update the frozen backbone's BatchNorm running stats (silent
            # drift). Instance-level no-op shadows nn.Module.train for `base`.
            base.eval()
            base.train = lambda mode=True: base  # type: ignore[method-assign]
            print(f"[train_reveal] inner backbone FROZEN ({n_frozen} param tensors, eval-pinned)")
        return _try_calls(RevealNet, [(base, reveal_cfg), (base, cfg), (base,)])
    except (RuntimeError, TypeError):
        pass
    # Strategy 2: let RevealNet build itself from the config directly.
    return _try_calls(RevealNet, [(model_cfg,), (cfg,), ()])


def build_loss(cfg: Any) -> Any:
    """Build RevealLoss wrapping a PharosLoss (both read the full cfg tree)."""
    from pharos.losses import PharosLoss
    from pharos.losses.reveal_losses import RevealLoss

    return RevealLoss(cfg, inner=PharosLoss(cfg))


def build_datasets(cfg: Any, names: list[str], split: str) -> list:
    """Route reveal video sets to build_reveal_dataset, images to build_dataset."""
    datasets_cfg = cfg.get("datasets", {}) if hasattr(cfg, "get") else {}
    video_names = set(datasets_cfg.get("train_video", []) or []) | set(
        datasets_cfg.get("eval_video", []) or []
    )
    out = []
    for name in names:
        # Only 'reveal_*' names belong to the reveal builder; other video sets
        # (synth_video, revide) use the standard v1 factory.
        if name in video_names and name.startswith("reveal"):
            build = _lazy(
                "pharos.data.reveal_dataset", "build_reveal_dataset",
                why="build_reveal_dataset yields the synthetic reveal video clips.",
            )
        else:
            build = _lazy("pharos.data", "build_dataset", why="build_dataset yields v1 image sets.")
        out.append(build(name, cfg, split=split))
    return out


def make_deps():
    """Assemble the engine Deps container with the reveal factories injected."""
    from pharos.engine.train import Deps

    return Deps(build_model=build_model, build_loss=build_loss, build_datasets=build_datasets)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser("train_reveal", description="RevealNet v2 training launcher")
    p.add_argument("--config", default="configs/reveal.yaml", help="config YAML")
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
