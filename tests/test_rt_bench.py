"""Tests for pharos.rt.bench (CPU stub, tiny frame counts)."""
from __future__ import annotations

import json

from pharos.rt.bench import count_params, render_markdown, run_benchmark
from test_rt_stub import StubModel, make_config


def test_count_params_positive():
    assert count_params(StubModel()) > 0
    assert count_params(object()) == 0  # no parameters() -> 0


def test_run_benchmark_cpu_stub_emits_valid_json(tmp_path):
    model = StubModel()
    cfg = make_config(tmp_path, resolutions=[[64, 48]], frames=3)
    report = run_benchmark(
        model,
        cfg,
        out_dir=tmp_path / "bench",
        modes=["torch_fp32"],
        frames=3,
        warmup=1,
        resolutions=[[64, 48]],
        device="cpu",
    )
    assert report["results"], "expected at least one result"
    entry = report["results"][0]
    assert entry["mode"] == "torch_fp32"
    assert entry["available"] is True
    assert entry["model_only"]["frames"] == 3
    assert entry["full_pipeline"]["frames"] == 3
    assert entry["model_only"]["fps"] > 0

    # JSON file exists and round-trips.
    jp = report["json_path"]
    with open(jp, encoding="utf-8") as f:
        loaded = json.load(f)
    assert loaded["params"] == report["params"]
    assert loaded["results"][0]["mode"] == "torch_fp32"

    # Markdown file exists.
    assert report["md_path"].endswith(".md")
    with open(report["md_path"], encoding="utf-8") as f:
        assert "Benchmark" in f.read()


def test_run_benchmark_skips_missing_backends(tmp_path):
    model = StubModel()
    cfg = make_config(tmp_path, resolutions=[[48, 32]], frames=2)
    report = run_benchmark(
        model,
        cfg,
        out_dir=tmp_path / "bench",
        modes=["onnxruntime", "tensorrt"],
        frames=2,
        warmup=0,
        resolutions=[[48, 32]],
        device="cpu",
    )
    by_mode = {r["mode"]: r for r in report["results"]}
    # Neither onnxruntime nor tensorrt is installed in the CI/dev env => skipped.
    assert by_mode["onnxruntime"]["available"] is False
    assert "reason" in by_mode["onnxruntime"]
    assert by_mode["tensorrt"]["available"] is False
    assert "trtexec" in by_mode["tensorrt"]["instructions"]


def test_render_markdown_contains_table():
    report = {
        "gpu": "CPU",
        "device": "cpu",
        "torch_version": "x",
        "params": 10,
        "params_millions": 0.0,
        "frames": 3,
        "warmup": 1,
        "batch": 1,
        "timestamp": "t",
        "results": [
            {
                "resolution": [64, 48],
                "mode": "torch_fp32",
                "available": True,
                "model_only": {"median_ms": 1.0, "p95_ms": 2.0, "fps": 1000.0},
                "full_pipeline": {"median_ms": 2.0, "p95_ms": 3.0, "fps": 500.0},
            },
            {
                "resolution": [64, 48],
                "mode": "tensorrt",
                "available": False,
                "reason": "tensorrt not installed",
                "instructions": "trtexec --onnx=x.onnx",
            },
        ],
    }
    md = render_markdown(report)
    assert "| Resolution |" in md
    assert "torch_fp32" in md
    assert "trtexec" in md
