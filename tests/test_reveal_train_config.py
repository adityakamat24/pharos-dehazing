"""Tests: reveal.yaml loads through pharos.config; launcher wiring (no parallel deps)."""
from __future__ import annotations

import importlib.util
from pathlib import Path

from pharos.config import load_config

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "configs"
SCRIPTS = ROOT / "scripts"


def test_reveal_config_loads_with_expected_keys():
    cfg = load_config(CONFIGS / "reveal.yaml")
    # inherited from base.yaml
    assert cfg.model.lowres == 256
    assert cfg.loss.rec == 1.0
    assert cfg.teachers.depth.enabled is True
    # reveal-specific
    assert cfg.exp_name == "reveal"
    assert cfg.model.reveal.half_life == 4.0
    assert cfg.model.reveal.thresholds.merge_conf == 0.60
    assert cfg.train.clip_len == 8
    assert cfg.train.clip_batch == 4
    assert cfg.train.clip_period == 1
    assert cfg.train.iters == 60000
    assert cfg.train.lr == 2.0e-4
    assert "reveal_video" in cfg.datasets.train_video
    assert cfg.loss.reveal.recall == 1.0
    assert cfg.loss.reveal.align == 0.2
    assert cfg.loss.reveal.stale == 0.05
    assert cfg.loss.reveal.occ_thresh == 0.6


def test_reveal_config_override_merge():
    cfg = load_config(CONFIGS / "reveal.yaml", {"train.clip_len": 16, "train.lr": 1e-4})
    assert cfg.train.clip_len == 16  # dotted override wins (curriculum stage 2)
    assert cfg.train.lr == 1e-4
    assert cfg.exp_name == "reveal"  # untouched


def _load_launcher():
    spec = importlib.util.spec_from_file_location("train_reveal", SCRIPTS / "train_reveal.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_launcher_build_loss_returns_reveal_loss():
    """build_loss wraps a PharosLoss in a RevealLoss without touching parallel modules."""
    mod = _load_launcher()
    cfg = load_config(CONFIGS / "reveal.yaml")
    loss = mod.build_loss(cfg)
    from pharos.losses.reveal_losses import RevealLoss

    assert isinstance(loss, RevealLoss)
    assert callable(loss)
    assert loss.w == {"recall": 1.0, "align": 0.2, "stale": 0.05}


def test_launcher_parser_defaults():
    mod = _load_launcher()
    args = mod.build_parser().parse_args([])
    assert args.config == "configs/reveal.yaml"
    assert args.override == []
    assert args.resume is None


def test_launcher_make_deps_wires_factories():
    mod = _load_launcher()
    deps = mod.make_deps()
    assert deps.build_model is mod.build_model
    assert deps.build_loss is mod.build_loss
    assert deps.build_datasets is mod.build_datasets
