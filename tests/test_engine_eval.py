"""Tests for pharos.engine.eval: TriHaze protocol output structure + no-ref proxy."""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from pharos.config import Config  # noqa: E402
from pharos.engine.eval import evaluate, niqe_simplified  # noqa: E402
from pharos.engine.train import pharos_collate  # noqa: E402
from test_engine_stubs import (  # noqa: E402
    StubClipDataset,
    StubImageDataset,
    StubModel,
    StubTeachers,
)


def _cfg(tmp_path: Path) -> Config:
    return Config({"out_root": str(tmp_path), "exp_name": "e", "datasets": {}})


def test_evaluate_json_structure(tmp_path):
    paired = {"stub": DataLoader(StubImageDataset(n=4), batch_size=2, collate_fn=pharos_collate)}
    noref = {"rtts": DataLoader(StubImageDataset(n=4), batch_size=2, collate_fn=pharos_collate)}
    clips = {"revide": DataLoader(StubClipDataset(n=2), batch_size=1, collate_fn=pharos_collate)}
    out_dir = tmp_path / "eval"

    res = evaluate(
        StubModel(),
        _cfg(tmp_path),
        teachers=StubTeachers(with_flow=True),
        device="cpu",
        out_dir=out_dir,
        step=42,
        paired_loaders=paired,
        noref_loaders=noref,
        clip_loaders=clips,
        compute_lpips=False,
    )

    for key in ("step", "paired", "noref", "temporal", "clear_no_harm", "detection_map", "notes"):
        assert key in res, key
    assert res["step"] == 42

    p = res["paired"]["stub"]
    assert p["n"] > 0
    assert math.isfinite(p["psnr"]) and 0.0 <= p["ssim"] <= 1.0
    assert p["lpips"] is None  # compute_lpips=False

    t = res["temporal"]["revide"]
    assert t["pairs"] > 0 and t["used_flow"] is True
    assert res["detection_map"]["status"] == "not_implemented"
    assert "psnr_out_vs_in" in res["clear_no_harm"]

    # reports written and valid
    jpath = out_dir / "eval_step000042.json"
    assert jpath.exists()
    loaded = json.loads(jpath.read_text(encoding="utf-8"))
    assert loaded["step"] == 42
    assert (out_dir / "eval_step000042.md").exists()


def test_clear_no_harm_uses_paired_when_no_clear_loader(tmp_path):
    paired = {"sots_mix": DataLoader(StubImageDataset(n=4), batch_size=2, collate_fn=pharos_collate)}
    res = evaluate(
        StubModel(), _cfg(tmp_path), device="cpu",
        paired_loaders=paired, compute_lpips=False,
    )
    assert "psnr_out_vs_in" in res["clear_no_harm"]
    assert isinstance(res["clear_no_harm"]["pass"], bool)


def test_niqe_simplified_finite():
    torch.manual_seed(0)
    val = niqe_simplified(torch.rand(3, 64, 64))
    assert math.isfinite(val) and val >= 0.0
