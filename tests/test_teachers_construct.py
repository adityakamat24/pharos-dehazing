"""CPU tests: teachers construct offline; .available reflects deps; bundle wiring."""
from __future__ import annotations

import importlib.util as _ilu

import torch

from pharos.teachers import DepthTeacher, DetectionTeacher, FlowTeacher, TeacherBundle


def _force_missing(monkeypatch, missing: str):
    real = _ilu.find_spec

    def fake(name, *a, **k):
        if name == missing:
            return None
        return real(name, *a, **k)

    # patch in every teacher module that imported find_spec via importlib.util
    for mod in ("pharos.teachers.depth", "pharos.teachers.detector", "pharos.teachers.flow"):
        monkeypatch.setattr(f"{mod}.importlib.util.find_spec", fake, raising=False)


def test_depth_available_false_when_transformers_missing(monkeypatch):
    _force_missing(monkeypatch, "transformers")
    t = DepthTeacher(device="cpu")
    assert t.available is False
    # unavailable teacher still returns a benign zero map of the requested size
    out = t(torch.rand(1, 3, 32, 32), out_size=(16, 16))
    assert out.shape == (1, 1, 16, 16)
    assert torch.count_nonzero(out) == 0


def test_detector_and_flow_available_false_when_torchvision_missing(monkeypatch):
    _force_missing(monkeypatch, "torchvision")
    det = DetectionTeacher(device="cpu")
    flow = FlowTeacher(device="cpu")
    assert det.available is False
    assert flow.available is False
    assert det(torch.rand(1, 3, 32, 32)) == []


def test_bundle_all_disabled_gives_none():
    cfg = {
        "teachers": {
            "depth": {"enabled": False},
            "detector": {"enabled": False},
            "flow": {"enabled": False},
        }
    }
    b = TeacherBundle.from_config(cfg, device="cpu")
    assert b.depth is None and b.detector is None and b.flow is None


def test_bundle_disabled_depth_when_unavailable(monkeypatch):
    _force_missing(monkeypatch, "transformers")
    cfg = {
        "data_root": "D:/dehazing_desmoking/data",
        "teachers": {
            "depth": {"enabled": True},
            "detector": {"enabled": False},
            "flow": {"enabled": False},
        },
    }
    b = TeacherBundle.from_config(cfg, device="cpu")
    assert b.depth is None  # enabled but transformers missing -> None
