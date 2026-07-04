"""Tests for pharos.rt.export (onnx/onnxruntime optional; skip cleanly when absent)."""
from __future__ import annotations

import torch

from pharos.rt.export import (
    ExportDependencyMissing,
    ExportUnsupported,
    _ExportWrapper,
    build_tensorrt_engine,
    export_all,
    export_onnx,
    onnx_parity_check,
    trtexec_command,
)
from test_rt_stub import DictStateModel, StubModel


def test_export_wrapper_returns_tensor_tuple():
    # Exercises the core of the export path regardless of onnx availability.
    wrapper = _ExportWrapper(StubModel(), video_mode=False).eval()
    dummy = torch.rand(1, 3, 32, 32)
    with torch.inference_mode():
        out = wrapper(dummy)
    assert isinstance(out, tuple) and len(out) == 2
    output, confidence = out
    assert output.shape == (1, 3, 32, 32)
    assert confidence.shape == (1, 1, 32, 32)


def test_export_onnx_runs_or_skips_cleanly(tmp_path):
    model = StubModel()
    path = tmp_path / "stub.onnx"
    try:
        result = export_onnx(model, path, resolution=(32, 32))
    except ExportDependencyMissing as e:
        import pytest

        pytest.skip(f"onnx not installed: {e}")
    else:
        assert result.exists() and result.stat().st_size > 0


def test_export_dynamic_variant_or_skip(tmp_path):
    try:
        result = export_onnx(StubModel(), tmp_path / "dyn.onnx", resolution=(48, 64), dynamic=True)
    except ExportDependencyMissing:
        import pytest

        pytest.skip("onnx not installed")
    else:
        assert result.exists()


def test_video_mode_export_unsupported_for_dict_state(tmp_path):
    # DictStateModel returns a non-tensor recurrent state -> ExportUnsupported, raised
    # before any onnx dependency is needed (deterministic on all envs).
    import pytest

    with pytest.raises(ExportUnsupported):
        export_onnx(DictStateModel(), tmp_path / "vid.onnx", resolution=(32, 32), video_mode=True)


def test_export_all_records_each_variant(tmp_path):
    variants = export_all(StubModel(), tmp_path, resolution=(32, 32), try_video=True)
    assert set(variants) == {"static", "dynamic", "video"}
    for name, info in variants.items():
        # Either exported (onnx present) or a recorded reason (onnx absent) — never a crash.
        assert info["exported"] is True or "reason" in info


def test_onnx_parity_check_skips_without_onnxruntime(tmp_path):
    res = onnx_parity_check(StubModel(), tmp_path / "missing.onnx", resolution=(32, 32))
    # onnxruntime is not installed in this env -> clean skip dict.
    if res["available"]:
        assert "max_abs_diff_output" in res
    else:
        assert "reason" in res


def test_trtexec_command_string():
    cmd = trtexec_command("model.onnx", "model.engine", fp16=True)
    assert cmd.startswith("trtexec")
    assert "--onnx=model.onnx" in cmd
    assert "--saveEngine=model.engine" in cmd
    assert "--fp16" in cmd

    dyn = trtexec_command("m.onnx", dynamic=True)
    assert "--minShapes=frame:" in dyn and "--maxShapes=frame:" in dyn


def test_build_tensorrt_engine_returns_instructions():
    info = build_tensorrt_engine("model.onnx", fp16=True)
    assert info["built"] is False
    assert "instructions" in info and "trtexec" in info["instructions"]
