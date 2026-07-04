"""Tests for pharos.engine.train: train step (image+clip), ckpt roundtrip, config."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from pharos.config import Config, load_config  # noqa: E402
from pharos.engine.train import Trainer, pharos_collate  # noqa: E402
from pharos.engine.utils import parse_overrides  # noqa: E402
from test_engine_stubs import (  # noqa: E402
    StubClipDataset,
    StubImageDataset,
    StubLoss,
    StubModel,
    StubTeachers,
)

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


def _cfg(tmp_path: Path, **train_over) -> Config:
    train = dict(
        iters=8, lr=1e-3, weight_decay=0.0, batch=2, clip_batch=2, warmup_iters=0,
        amp=False, ema=0.9, ckpt_every=0, eval_every=0, log_every=0, img_every=0,
        num_workers=0, grad_accum=1, clip_period=2,
    )
    train.update(train_over)
    return Config({"seed": 0, "out_root": str(tmp_path), "exp_name": "t", "train": train})


def _loaders():
    img = DataLoader(StubImageDataset(n=8), batch_size=2, collate_fn=pharos_collate, drop_last=True)
    vid = DataLoader(StubClipDataset(n=6), batch_size=2, collate_fn=pharos_collate, drop_last=True)
    return img, vid


def _make_trainer(tmp_path: Path, **train_over) -> Trainer:
    img, vid = _loaders()
    return Trainer(
        _cfg(tmp_path, **train_over),
        model=StubModel(),
        image_loader=img,
        video_loader=vid,
        loss_fn=StubLoss(),
        teachers=StubTeachers(),
        device="cpu",
    )


def test_train_step_image_and_clip(tmp_path):
    tr = _make_trainer(tmp_path, clip_period=2)
    before = tr.model.detail.weight.detach().clone()
    # step 0 -> image modality, step 1 -> clip modality (clip_period=2)
    modalities = []
    for i in range(2):
        tr.step = i
        modalities.append(tr._pick_modality(i))
        scalars = tr.train_step()
        assert "loss" in scalars and torch.isfinite(torch.tensor(scalars["loss"]))
    assert "image" in modalities and "video" in modalities
    after = tr.model.detail.weight.detach()
    assert not torch.allclose(before, after), "optimizer did not update weights"


def test_full_train_loop_runs(tmp_path):
    tr = _make_trainer(tmp_path, iters=6, clip_period=2)
    tr.train()
    assert tr.step == 6
    assert (Path(tmp_path) / "t" / "ckpt" / "last.pt").exists()


def test_ema_updates(tmp_path):
    tr = _make_trainer(tmp_path)
    init = tr.ema.shadow["detail.weight"].clone()
    for i in range(3):
        tr.step = i
        tr.train_step()
    assert not torch.allclose(init, tr.ema.shadow["detail.weight"])


def test_checkpoint_roundtrip_preserves_step_and_ema(tmp_path):
    tr = _make_trainer(tmp_path)
    for i in range(3):
        tr.step = i
        tr.train_step()
    tr.step = 3
    ckpt = Path(tmp_path) / "ck.pt"
    tr.save(ckpt)

    tr2 = _make_trainer(tmp_path)
    assert tr2.step == 0
    tr2.load_state(ckpt)
    assert tr2.step == 3
    for k in tr.ema.shadow:
        if torch.is_tensor(tr.ema.shadow[k]):
            assert torch.allclose(tr.ema.shadow[k], tr2.ema.shadow[k]), k
    for k, v in tr.model.state_dict().items():
        assert torch.allclose(v, tr2.model.state_dict()[k]), k


def test_config_override_merge(tmp_path):
    cfg = load_config(CONFIGS / "overfit50.yaml", {"train.lr": 1e-4, "train.batch": 3})
    assert cfg.model.lowres == 256            # inherited from base.yaml
    assert cfg.train.iters == 2000            # from overfit50.yaml
    assert cfg.train.lr == 1e-4               # dotted override wins
    assert cfg.train.batch == 3
    assert cfg.exp_name == "overfit50"


def test_parse_overrides_typing():
    out = parse_overrides(["train.amp=false", "train.lr=3e-4", "x.y=5"])
    assert out["train.amp"] is False
    assert abs(out["train.lr"] - 3e-4) < 1e-12
    assert out["x.y"] == 5


def test_periodic_ckpt_and_eval_hook(tmp_path):
    img, vid = _loaders()
    calls = {"n": 0}

    def fake_eval():
        calls["n"] += 1
        return {"paired": {"x": {"psnr": 30.0, "ssim": 0.9}}}

    tr = Trainer(
        _cfg(tmp_path, iters=4, ckpt_every=2, eval_every=2, clip_period=2),
        model=StubModel(), image_loader=img, video_loader=vid,
        loss_fn=StubLoss(), teachers=StubTeachers(), device="cpu", eval_fn=fake_eval,
    )
    tr.train()
    ck = Path(tmp_path) / "t" / "ckpt"
    assert (ck / "ckpt_000002.pt").exists() and (ck / "ckpt_000004.pt").exists()
    assert calls["n"] == 2  # eval hook fired at steps 2 and 4


def test_grad_accumulation(tmp_path):
    tr = _make_trainer(tmp_path, grad_accum=2)
    before = tr.model.detail.weight.detach().clone()
    tr.step = 0
    tr.train_step()
    assert not torch.allclose(before, tr.model.detail.weight.detach())
