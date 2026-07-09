"""CPU, no-network tests for pharos.data.datasets (tiny fake folders in tmp_path)."""
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
from pharos.contracts import BATCH_KEYS, DOMAIN_HAZE, DOMAIN_SATELLITE, DOMAIN_SMOKE
from pharos.data.datasets import (
    ClearPassthroughDataset,
    PairedFolderDataset,
    SyntheticDataset,
    SynthVideoDataset,
    UnpairedDataset,
    VideoClipDataset,
    build_dataset,
    pharos_collate,
)

try:
    import cv2
except Exception:  # pragma: no cover
    cv2 = None

pytestmark = pytest.mark.skipif(cv2 is None, reason="opencv required for image I/O")


def _write(path: Path, h: int = 40, w: int = 50, seed: int = 0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    img = (rng.random((h, w, 3)) * 255).astype(np.uint8)
    cv2.imwrite(str(path), img)


# ---------------------------------------------------------------------------
# per-sample contract dict
# ---------------------------------------------------------------------------
def _assert_sample(sample: dict, clip: bool = False, paired: bool = True) -> None:
    assert set(sample.keys()) == set(BATCH_KEYS.keys())
    assert isinstance(sample["hazy"], torch.Tensor)
    assert sample["hazy"].dtype == torch.float32
    assert 0.0 <= float(sample["hazy"].min()) and float(sample["hazy"].max()) <= 1.0
    if paired:
        assert isinstance(sample["clean"], torch.Tensor)
        assert sample["clean"].shape == sample["hazy"].shape
    else:
        assert sample["clean"] is None
    assert isinstance(sample["domain"], int)
    assert sample["domain"] in (DOMAIN_HAZE, DOMAIN_SMOKE, DOMAIN_SATELLITE)
    assert isinstance(sample["clip"], bool) and sample["clip"] == clip
    assert isinstance(sample["meta"], dict)
    assert "full_lowres" in sample["meta"]
    ndim = 5 if clip else 4  # (T,)3,L,L stacked later
    lr = sample["meta"]["full_lowres"]
    assert lr.shape[-1] == lr.shape[-2]  # square lowres stream


# ---------------------------------------------------------------------------
# paired
# ---------------------------------------------------------------------------
def test_paired_stem(tmp_path):
    for i in range(4):
        _write(tmp_path / "hazy" / f"img{i}.png", seed=i)
        _write(tmp_path / "GT" / f"img{i}.png", seed=100 + i)
    ds = PairedFolderDataset(tmp_path / "hazy", tmp_path / "GT", domain=DOMAIN_HAZE,
                             crop=32, augment=True, lowres=64, name="p")
    assert len(ds) == 4
    _assert_sample(ds[0])
    assert ds[0]["hazy"].shape == (3, 32, 32)
    assert ds[0]["meta"]["full_lowres"].shape == (3, 64, 64)


def test_paired_prefix_match(tmp_path):
    # RESIDE style: hazy '0001_0.8_0.2.png' <-> clean '0001.png'
    for i in range(3):
        _write(tmp_path / "hazy" / f"{i:04d}_0.9_0.1.png", seed=i)
        _write(tmp_path / "GT" / f"{i:04d}.png", seed=50 + i)
    ds = PairedFolderDataset(tmp_path / "hazy", tmp_path / "GT", match="prefix", lowres=32, name="r")
    assert len(ds) == 3
    _assert_sample(ds[0])


def test_paired_suffix_match(tmp_path):
    # NH-HAZE style: single folder with '_hazy'/'_GT'
    for i in range(3):
        _write(tmp_path / "nh" / f"{i:02d}_hazy.png", seed=i)
        _write(tmp_path / "nh" / f"{i:02d}_GT.png", seed=20 + i)
    ds = PairedFolderDataset(tmp_path / "nh", match="suffix", lowres=32, name="nh")
    assert len(ds) == 3
    _assert_sample(ds[0])


def test_paired_num_match(tmp_path):
    # SateHaze1k thin/thick style: hyphen role suffixes -> matched by leading number
    for i in range(1, 4):
        _write(tmp_path / "input" / f"{i}-inputs.png", seed=i)
        _write(tmp_path / "target" / f"{i}-targets.png", seed=30 + i)
    ds = PairedFolderDataset(tmp_path / "input", tmp_path / "target", match="num", lowres=32)
    assert len(ds) == 3
    _assert_sample(ds[0])


def test_paired_auto_match_picks_best(tmp_path):
    # auto should recover the numeric pairing without being told the mode
    for i in range(1, 5):
        _write(tmp_path / "input" / f"{i}-inputs.png", seed=i)
        _write(tmp_path / "target" / f"{i}-targets.png", seed=40 + i)
    ds = PairedFolderDataset(tmp_path / "input", tmp_path / "target", lowres=32)  # default auto
    assert len(ds) == 4


def test_paired_cross_dir_suffix_prefix(tmp_path):
    # NTIRE Dense/O-HAZE style: 'NN_..._hazy' <-> 'NN_..._GT' across two dirs
    for i in range(1, 4):
        _write(tmp_path / "hazy" / f"{i:02d}_outdoor_hazy.png", seed=i)
        _write(tmp_path / "GT" / f"{i:02d}_outdoor_GT.png", seed=60 + i)
    ds = PairedFolderDataset(tmp_path / "hazy", tmp_path / "GT", lowres=32)  # default auto
    assert len(ds) == 3
    _assert_sample(ds[0])


def test_paired_full_resolution_when_no_crop(tmp_path):
    _write(tmp_path / "hazy" / "a.png", h=40, w=50)
    _write(tmp_path / "GT" / "a.png", h=40, w=50)
    ds = PairedFolderDataset(tmp_path / "hazy", tmp_path / "GT", crop=0, lowres=16)
    assert ds[0]["hazy"].shape == (3, 40, 50)


# ---------------------------------------------------------------------------
# unpaired
# ---------------------------------------------------------------------------
def test_unpaired(tmp_path):
    for i in range(3):
        _write(tmp_path / "rtts" / f"x{i}.png", seed=i)
    ds = UnpairedDataset(tmp_path / "rtts", lowres=32)
    assert len(ds) == 3
    _assert_sample(ds[0], paired=False)


# ---------------------------------------------------------------------------
# synthetic
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("domain,dom_id", [("haze", DOMAIN_HAZE), ("smoke", DOMAIN_SMOKE),
                                           ("satellite", DOMAIN_SATELLITE)])
def test_synthetic(tmp_path, domain, dom_id):
    for i in range(3):
        _write(tmp_path / "clean" / f"c{i}.png", seed=i)
    ds = SyntheticDataset(tmp_path / "clean", domain=domain, crop=32, lowres=32, seed=0)
    assert len(ds) == 3
    s = ds[0]
    _assert_sample(s)
    assert s["domain"] == dom_id
    assert "beta" in s["meta"] and "airlight" in s["meta"] and "sigma" in s["meta"]
    assert s["meta"]["airlight"].shape == (3,)


def test_synthetic_deterministic_with_seed(tmp_path):
    _write(tmp_path / "clean" / "c0.png", seed=0)
    ds = SyntheticDataset(tmp_path / "clean", domain="haze", crop=0, lowres=32, seed=42)
    a, b = ds[0], ds[0]
    assert torch.allclose(a["hazy"], b["hazy"])


# ---------------------------------------------------------------------------
# clear passthrough
# ---------------------------------------------------------------------------
def test_clear_passthrough_identity(tmp_path):
    for i in range(3):
        _write(tmp_path / "clean" / f"c{i}.png", seed=i)
    ds = ClearPassthroughDataset(tmp_path / "clean", crop=0, lowres=32, seed=1)
    s = ds[0]
    _assert_sample(s)
    assert torch.allclose(s["hazy"], s["clean"])  # identity target for the gate


# ---------------------------------------------------------------------------
# video
# ---------------------------------------------------------------------------
def test_video_clip_dataset(tmp_path):
    for seq in ("s1", "s2"):
        for f in range(5):
            _write(tmp_path / "hazy" / seq / f"{f:03d}.png", seed=f)
            _write(tmp_path / "gt" / seq / f"{f:03d}.png", seed=100 + f)
    ds = VideoClipDataset(tmp_path / "hazy", tmp_path / "gt", clip_len=3, crop=32, lowres=32)
    assert len(ds) == 2 * (5 - 3 + 1)
    s = ds[0]
    _assert_sample(s, clip=True)
    assert s["hazy"].shape == (3, 3, 32, 32)  # (T,3,H,W)
    assert s["clean"].shape == (3, 3, 32, 32)
    assert s["meta"]["full_lowres"].shape == (3, 3, 32, 32)


def test_synth_video_static_stills(tmp_path):
    for i in range(3):
        _write(tmp_path / "clean" / f"c{i}.png", seed=i)
    ds = SynthVideoDataset(tmp_path / "clean", clip_len=4, lowres=32, seed=0)
    assert len(ds) == 3  # one static clip per still
    s = ds[0]
    _assert_sample(s, clip=True)
    assert s["hazy"].shape[0] == 4
    # temporally smooth: adjacent frames of a static scene stay close
    adj = (s["hazy"][1:] - s["hazy"][:-1]).abs().mean()
    assert float(adj) < 0.05


def test_synth_video_real_sequences(tmp_path):
    for seq in ("a", "b"):
        for f in range(4):
            _write(tmp_path / "clean" / seq / f"{f}.png", seed=f)
    ds = SynthVideoDataset(tmp_path / "clean", clip_len=3, lowres=32, seed=0)
    assert len(ds) == 2 * (4 - 3 + 1)
    _assert_sample(ds[0], clip=True)


# ---------------------------------------------------------------------------
# collate / batch contract
# ---------------------------------------------------------------------------
def test_collate_paired_batch(tmp_path):
    for i in range(4):
        _write(tmp_path / "hazy" / f"i{i}.png", seed=i)
        _write(tmp_path / "GT" / f"i{i}.png", seed=10 + i)
    ds = PairedFolderDataset(tmp_path / "hazy", tmp_path / "GT", crop=32, lowres=32)
    dl = DataLoader(ds, batch_size=2, collate_fn=pharos_collate)
    b = next(iter(dl))
    assert set(b.keys()) == set(BATCH_KEYS.keys())
    assert b["hazy"].shape == (2, 3, 32, 32)
    assert b["clean"].shape == (2, 3, 32, 32)
    assert b["domain"].dtype == torch.long and b["domain"].shape == (2,)
    assert isinstance(b["clip"], bool) and b["clip"] is False
    assert isinstance(b["meta"], dict)
    assert b["meta"]["full_lowres"].shape == (2, 3, 32, 32)


def test_collate_unpaired_clean_none(tmp_path):
    for i in range(3):
        _write(tmp_path / "rtts" / f"x{i}.png", seed=i)
    ds = UnpairedDataset(tmp_path / "rtts", crop=32, lowres=32)
    dl = DataLoader(ds, batch_size=2, collate_fn=pharos_collate)
    b = next(iter(dl))
    assert b["clean"] is None
    assert b["hazy"].shape == (2, 3, 32, 32)


def test_collate_video_batch(tmp_path):
    for seq in ("s1", "s2"):
        for f in range(4):
            _write(tmp_path / "hazy" / seq / f"{f}.png", seed=f)
            _write(tmp_path / "gt" / seq / f"{f}.png", seed=9 + f)
    ds = VideoClipDataset(tmp_path / "hazy", tmp_path / "gt", clip_len=3, crop=32, lowres=32)
    dl = DataLoader(ds, batch_size=2, collate_fn=pharos_collate)
    b = next(iter(dl))
    assert b["clip"] is True
    assert b["hazy"].shape == (2, 3, 3, 32, 32)  # (B,T,3,H,W)
    assert b["clean"].shape == (2, 3, 3, 32, 32)


def test_collate_synthetic_meta_tensors(tmp_path):
    for i in range(4):
        _write(tmp_path / "clean" / f"c{i}.png", seed=i)
    ds = SyntheticDataset(tmp_path / "clean", domain="smoke", crop=32, lowres=32, seed=0)
    dl = DataLoader(ds, batch_size=2, collate_fn=pharos_collate)
    b = next(iter(dl))
    assert b["meta"]["beta"].shape[0] == 2
    assert b["meta"]["airlight"].shape == (2, 3)


# ---------------------------------------------------------------------------
# factory
# ---------------------------------------------------------------------------
def _cfg(tmp_path) -> Config:
    return Config({
        "data_root": str(tmp_path),
        "model": {"lowres": 32},
        "train": {"crop": 32, "clip_len": 3},
        "seed": 3407,
    })


def test_build_dataset_all_names(tmp_path):
    # give the synth/clear datasets a clean pool to draw from
    for i in range(3):
        _write(tmp_path / "reside6k" / "train" / "GT" / f"g{i}.png", seed=i)
        _write(tmp_path / "reside6k" / "train" / "hazy" / f"g{i}_1_2.png", seed=50 + i)
    cfg = _cfg(tmp_path)
    train_names = ["reside6k", "smokebench", "satehaze1k", "synth_haze", "synth_smoke",
                   "synth_satellite", "clear_passthrough"]
    eval_names = ["sots_mix", "nhhaze", "densehaze", "smokebench_test", "satehaze1k_test", "rice1"]
    video_names = ["revide", "synth_video"]
    for n in train_names + video_names:
        ds = build_dataset(n, cfg, "train")
        assert hasattr(ds, "__len__")
    for n in eval_names:
        ds = build_dataset(n, cfg, "eval")
        assert hasattr(ds, "__len__")
    ds = build_dataset("rtts", cfg, "eval")
    assert hasattr(ds, "__len__")


def test_build_dataset_reside6k_loads(tmp_path):
    for i in range(3):
        _write(tmp_path / "reside6k" / "train" / "GT" / f"g{i}.png", seed=i)
        _write(tmp_path / "reside6k" / "train" / "hazy" / f"g{i}.png", seed=50 + i)
    cfg = _cfg(tmp_path)
    ds = build_dataset("reside6k", cfg, "train")
    assert len(ds) == 3
    _assert_sample(ds[0])


def test_build_dataset_synth_haze_loads(tmp_path):
    for i in range(3):
        _write(tmp_path / "reside6k" / "train" / "GT" / f"g{i}.png", seed=i)
    cfg = _cfg(tmp_path)
    ds = build_dataset("synth_haze", cfg, "train")
    assert len(ds) == 3
    s = ds[0]
    _assert_sample(s)
    assert s["domain"] == DOMAIN_HAZE


def test_build_dataset_unknown_raises(tmp_path):
    with pytest.raises(ValueError):
        build_dataset("nope", _cfg(tmp_path), "train")
