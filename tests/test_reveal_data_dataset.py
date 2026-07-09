"""CPU, no-network tests for pharos.data.reveal_dataset (tiny fake folders in tmp_path)."""
from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from pharos.config import Config
from pharos.contracts import BATCH_KEYS, DOMAIN_SMOKE
from pharos.data.datasets import pharos_collate
from pharos.data.reveal_dataset import RevealVideoDataset, build_reveal_dataset

try:
    import cv2
except Exception:  # pragma: no cover
    cv2 = None

pytestmark = pytest.mark.skipif(cv2 is None, reason="opencv required for image I/O")


def _write(path: Path, h: int = 48, w: int = 64, seed: int = 0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    img = (rng.random((h, w, 3)) * 255).astype(np.uint8)
    cv2.imwrite(str(path), img)


def _assert_reveal_sample(s: dict, t: int, cs: int, lowres: int) -> None:
    assert set(s.keys()) == set(BATCH_KEYS.keys())
    assert s["clip"] is True
    assert s["domain"] == DOMAIN_SMOKE
    assert s["hazy"].dtype == torch.float32
    assert s["hazy"].shape == (t, 3, cs, cs)
    assert s["clean"].shape == (t, 3, cs, cs)
    assert 0.0 <= float(s["hazy"].min()) and float(s["hazy"].max()) <= 1.0
    m = s["meta"]
    assert m["dataset"] and "reveal" in m and m["reveal"] is True
    assert m["smoke_density"].shape == (t, 1, cs, cs)
    assert m["transmission"].shape == (t, 1, cs, cs)
    assert m["cam_H"].shape == (t, 3, 3)
    assert m["full_lowres"].shape == (t, 3, lowres, lowres)
    assert m["airlight"].shape == (3,)


def test_static_stills(tmp_path):
    for i in range(3):
        _write(tmp_path / "clean" / f"c{i}.png", seed=i)
    ds = RevealVideoDataset(tmp_path / "clean", clip_len=6, crop=32, lowres=16, seed=0)
    assert len(ds) == 3  # one static clip per still
    _assert_reveal_sample(ds[0], t=6, cs=32, lowres=16)


def test_real_sequences(tmp_path):
    for seq in ("a", "b"):
        for f in range(5):
            _write(tmp_path / "clean" / seq / f"{f:02d}.png", seed=f)
    ds = RevealVideoDataset(tmp_path / "clean", clip_len=3, crop=32, lowres=16, seed=0)
    assert len(ds) == 2 * (5 - 3 + 1)
    _assert_reveal_sample(ds[0], t=3, cs=32, lowres=16)


def test_clip_len_configurable(tmp_path):
    _write(tmp_path / "clean" / "c0.png", seed=0)
    ds = RevealVideoDataset(tmp_path / "clean", clip_len=10, crop=0, lowres=16, seed=0)
    assert ds[0]["hazy"].shape[0] == 10


def test_full_resolution_when_no_crop(tmp_path):
    _write(tmp_path / "clean" / "c0.png", h=48, w=64, seed=0)
    ds = RevealVideoDataset(tmp_path / "clean", clip_len=4, crop=0, lowres=16, seed=0)
    s = ds[0]
    assert s["hazy"].shape == (4, 3, 48, 64)
    assert s["meta"]["smoke_density"].shape == (4, 1, 48, 64)


def test_deterministic_with_seed(tmp_path):
    _write(tmp_path / "clean" / "c0.png", seed=0)
    ds = RevealVideoDataset(tmp_path / "clean", clip_len=4, crop=0, lowres=16, seed=42)
    a, b = ds[0], ds[0]
    assert torch.allclose(a["hazy"], b["hazy"])
    assert torch.allclose(a["meta"]["cam_H"], b["meta"]["cam_H"])


def test_cam_H_first_frame_identity_after_crop(tmp_path):
    # crop conjugation preserves H_0 == identity (A^{-1} I A = I)
    _write(tmp_path / "clean" / "c0.png", seed=0)
    ds = RevealVideoDataset(tmp_path / "clean", clip_len=5, crop=32, lowres=16, seed=1)
    cam_H = ds[0]["meta"]["cam_H"]
    assert torch.allclose(cam_H[0], torch.eye(3), atol=1e-5)


def test_collate_batch(tmp_path):
    for i in range(4):
        _write(tmp_path / "clean" / f"c{i}.png", seed=i)
    ds = RevealVideoDataset(tmp_path / "clean", clip_len=4, crop=32, lowres=16, seed=0)
    dl = DataLoader(ds, batch_size=2, collate_fn=pharos_collate)
    b = next(iter(dl))
    assert set(b.keys()) == set(BATCH_KEYS.keys())
    assert b["clip"] is True
    assert b["hazy"].shape == (2, 4, 3, 32, 32)
    assert b["clean"].shape == (2, 4, 3, 32, 32)
    assert b["domain"].dtype == torch.long and b["domain"].tolist() == [DOMAIN_SMOKE, DOMAIN_SMOKE]
    assert b["meta"]["smoke_density"].shape == (2, 4, 1, 32, 32)
    assert b["meta"]["cam_H"].shape == (2, 4, 3, 3)
    assert b["meta"]["full_lowres"].shape == (2, 4, 3, 16, 16)


# ---------------------------------------------------------------------------
# factory
# ---------------------------------------------------------------------------
def _cfg(tmp_path) -> Config:
    return Config({
        "data_root": str(tmp_path),
        "model": {"lowres": 16},
        "train": {"crop": 32, "clip_len": 5},
        "seed": 3407,
    })


def test_build_reveal_dataset_train(tmp_path):
    for i in range(3):
        _write(tmp_path / "reside6k" / "train" / "GT" / f"g{i}.png", seed=i)
    ds = build_reveal_dataset("reveal_video", _cfg(tmp_path), "train")
    assert len(ds) == 3
    _assert_reveal_sample(ds[0], t=5, cs=32, lowres=16)


def test_build_reveal_dataset_eval_full_res(tmp_path):
    _write(tmp_path / "reside6k" / "train" / "GT" / "g0.png", h=48, w=64, seed=0)
    ds = build_reveal_dataset("reveal_video", _cfg(tmp_path), "eval")
    s = ds[0]
    assert s["hazy"].shape == (5, 3, 48, 64)  # eval: no crop


def test_build_reveal_dataset_unknown_name_raises(tmp_path):
    with pytest.raises(ValueError):
        build_reveal_dataset("nope", _cfg(tmp_path), "train")
