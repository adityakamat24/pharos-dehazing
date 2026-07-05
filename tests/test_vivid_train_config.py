"""Tests: vivid.yaml loads through pharos.config; train_vivid launcher wiring."""
from __future__ import annotations

import importlib.util
from pathlib import Path

from pharos.config import load_config

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "configs"
SCRIPTS = ROOT / "scripts"


def test_vivid_config_loads_with_expected_keys():
    cfg = load_config(CONFIGS / "vivid.yaml")
    # inherited from base.yaml
    assert cfg.model.lowres == 256
    assert cfg.loss.rec == 1.0
    assert cfg.teachers.depth.enabled is True
    # vivid-specific
    assert cfg.exp_name == "vivid"
    assert cfg.train.iters == 20000
    assert cfg.train.batch == 10
    assert cfg.train.lr == 1.0e-5
    assert cfg.train.clip_period == 50  # image-dominant
    assert cfg.loss.vivid.l1 == 1.0
    assert cfg.loss.vivid.lpips == 0.3
    assert cfg.loss.vivid.gan == 0.02
    assert cfg.loss.vivid.conf == 0.05
    assert cfg.loss.vivid.gan_warmup == 2000
    assert cfg.loss.vivid.disc_lr == 1.0e-4
    # datasets: finetune_real mix + NTIRE split (copied, not imported)
    assert cfg.datasets.ntire_split is True
    for name in ("nhhaze", "densehaze", "ohaze", "ihaze", "reside6k", "smokebench",
                 "synth_smoke", "clear_passthrough"):
        assert name in cfg.datasets.train


def test_vivid_config_override_merge():
    cfg = load_config(CONFIGS / "vivid.yaml", {"train.lr": 5e-6, "loss.vivid.gan": 0.05})
    assert cfg.train.lr == 5e-6
    assert cfg.loss.vivid.gan == 0.05
    assert cfg.exp_name == "vivid"  # untouched


def _load_launcher():
    spec = importlib.util.spec_from_file_location("train_vivid", SCRIPTS / "train_vivid.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_launcher_build_loss_returns_vivid_loss():
    mod = _load_launcher()
    cfg = load_config(CONFIGS / "vivid.yaml")
    loss = mod.build_loss(cfg)
    from pharos.losses.vivid_losses import VividLoss

    assert isinstance(loss, VividLoss)
    assert callable(loss)
    assert loss.w == {"l1": 1.0, "lpips": 0.3, "gan": 0.02, "conf": 0.05}
    assert loss.gan_warmup == 2000
    assert loss.disc_lr == 1.0e-4


def test_launcher_parser_defaults():
    mod = _load_launcher()
    args = mod.build_parser().parse_args([])
    assert args.config == "configs/vivid.yaml"
    assert args.override == []
    assert args.resume is None


def test_launcher_make_deps_wires_factories():
    mod = _load_launcher()
    deps = mod.make_deps()
    assert deps.build_model is mod.build_model
    assert deps.build_loss is mod.build_loss
    # datasets use the standard v1 factory (not overridden by the vivid launcher)
    from pharos.engine.train import default_build_datasets

    assert deps.build_datasets is default_build_datasets
