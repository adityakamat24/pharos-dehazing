"""Tests for pharos.rt.infer.StreamingRestorer (CPU, stub model, no GPU/display)."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from pharos.rt.infer import (
    StreamingRestorer,
    bgr_to_tensor,
    confidence_to_map,
    pad_to_multiple,
    tensor_to_bgr,
)
from test_rt_stub import StubModel


def _frame(h: int = 48, w: int = 64) -> np.ndarray:
    return np.random.randint(0, 256, size=(h, w, 3), dtype=np.uint8)


def test_state_threads_across_frames_and_reset_clears():
    model = StubModel()
    r = StreamingRestorer(model, device="cpu")

    assert r.state is None
    r.restore(_frame())
    # First forward saw state=None; state is now a tensor counter == 0.
    assert model.last_state_in is None
    assert isinstance(r.state, torch.Tensor)
    assert float(r.state.flatten()[0]) == 0.0

    r.restore(_frame())
    # Second forward must have received the state returned by the first frame.
    assert isinstance(model.last_state_in, torch.Tensor)
    assert float(r.state.flatten()[0]) == 1.0

    r.restore(_frame())
    assert float(r.state.flatten()[0]) == 2.0

    r.reset()
    assert r.state is None
    r.restore(_frame())
    assert float(r.state.flatten()[0]) == 0.0  # counter restarts after reset


def test_image_mode_does_not_touch_streaming_state():
    r = StreamingRestorer(StubModel(), device="cpu")
    r.restore(_frame())
    saved = float(r.state.flatten()[0])
    r.restore_image(_frame())  # image mode must not advance the streaming state
    assert float(r.state.flatten()[0]) == saved


def test_uint8_bgr_roundtrip_preserves_shape_and_range():
    frame = _frame(50, 70)
    r = StreamingRestorer(StubModel(), device="cpu")
    res = r.restore_image(frame)
    out = res["output_bgr"]
    assert out.dtype == np.uint8
    assert out.shape == frame.shape
    assert int(out.min()) >= 0 and int(out.max()) <= 255

    conf = res["confidence"]
    assert conf.dtype == np.float32
    assert conf.shape == (frame.shape[0], frame.shape[1])
    assert float(conf.min()) >= 0.0 and float(conf.max()) <= 1.0


def test_padding_roundtrips_for_non_multiple_resolution():
    # 50x70 is not a multiple of 8; output must still match input HxW after unpad.
    frame = _frame(50, 70)
    r = StreamingRestorer(StubModel(), device="cpu", pad_multiple=8)
    out = r.restore(frame)["output_bgr"]
    assert out.shape == frame.shape


def test_deg_and_timings_are_native_python():
    res = StreamingRestorer(StubModel(), device="cpu").restore(_frame())
    deg = res["deg"]
    assert isinstance(deg["beta"], float)
    assert isinstance(deg["sigma"], float)
    assert len(deg["airlight"]) == 3
    assert deg["domain"] in (0, 1, 2)
    assert isinstance(deg["domain_name"], str)
    assert isinstance(res["gate_alpha"], float)

    t = res["timings"]
    for k in ("pre_ms", "infer_ms", "post_ms", "total_ms", "fps", "fps_avg"):
        assert k in t and isinstance(t[k], float)


def test_half_falls_back_to_fp32_on_cpu():
    with pytest.warns(UserWarning):
        r = StreamingRestorer(StubModel(), device="cpu", half=True)
    assert r.half is False
    assert r.dtype == torch.float32


def test_reparameterize_on_load():
    model = StubModel()
    assert model.reparameterized is False
    StreamingRestorer(model, device="cpu", reparameterize=True)
    assert model.reparameterized is True


def test_restore_folder_reads_and_writes(tmp_path):
    in_dir = tmp_path / "in"
    out_dir = tmp_path / "out"
    in_dir.mkdir()
    import cv2

    for i in range(3):
        cv2.imwrite(str(in_dir / f"{i:03d}.png"), _frame(32, 40))

    r = StreamingRestorer(StubModel(), device="cpu")
    results = r.restore_folder(in_dir, out_dir)
    assert len(results) == 3
    for res in results:
        assert "path" in res and "out_path" in res
        assert (out_dir / __import__("pathlib").Path(res["path"]).name).exists()


def test_pre_post_helpers_roundtrip():
    frame = _frame(32, 48)
    t, hw = bgr_to_tensor(frame, "cpu", torch.float32, pad_multiple=1)
    assert t.shape == (1, 3, 32, 48) and hw == (32, 48)
    back = tensor_to_bgr(t, hw)
    # No model in between => exact roundtrip within uint8 rounding.
    assert back.shape == frame.shape
    assert np.abs(back.astype(int) - frame.astype(int)).max() <= 1

    padded, hw2 = pad_to_multiple(t, 8)
    assert padded.shape[-2] % 8 == 0 and padded.shape[-1] % 8 == 0 and hw2 == (32, 48)

    cmap = confidence_to_map(torch.rand(1, 1, 32, 48), (32, 48))
    assert cmap.shape == (32, 48) and cmap.dtype == np.float32
