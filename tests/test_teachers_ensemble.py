"""CPU test for the restoration-ensemble pseudo-labeling scaffold (no network)."""
from __future__ import annotations

import json

import numpy as np

from pharos.teachers.ensemble import NoRefScorer, RestorationEnsemble, clahe_dehaze, identity


def _write_dummy_images(d, n=3):
    import cv2

    for i in range(n):
        img = (np.random.default_rng(i).integers(0, 255, (24, 32, 3))).astype(np.uint8)
        cv2.imwrite(str(d / f"img_{i}.png"), img)


def test_members_shape_preserving():
    img = (np.random.default_rng(0).integers(0, 255, (16, 16, 3))).astype(np.uint8)
    assert identity(img).shape == img.shape
    assert clahe_dehaze(img).shape == img.shape


def test_scorer_proxy_higher_for_sharper():
    scorer = NoRefScorer()
    assert scorer.method == "grad_contrast_proxy"
    flat = np.full((32, 32, 3), 128, dtype=np.uint8)
    rng = np.random.default_rng(0)
    textured = rng.integers(0, 255, (32, 32, 3)).astype(np.uint8)
    assert scorer.score(textured) > scorer.score(flat)


def test_ensemble_run_writes_best_and_manifest(tmp_path):
    din = tmp_path / "in"
    dout = tmp_path / "out"
    din.mkdir()
    _write_dummy_images(din, n=3)

    manifest = RestorationEnsemble().run(din, dout)
    assert manifest["num_images"] == 3
    assert (dout / "manifest.json").exists()
    saved = json.loads((dout / "manifest.json").read_text())
    assert saved["members"] == ["identity", "clahe"]
    for entry in saved["entries"]:
        assert entry["best_member"] in ("identity", "clahe")
        assert (dout / entry["output"].split("/")[-1].split("\\")[-1]).exists()
        assert set(entry["scores"].keys()) == {"identity", "clahe"}
