"""Pharos real-time workstream (WS-E).

Deployment-side utilities that wrap a trained ``PharosModel`` for causal video and
single-image restoration, confidence overlays, honest FPS benchmarking, and ONNX
export. Nothing here depends on the training stack, datasets, or GPUs being present;
every module imports cleanly on CPU with no optional dependencies installed.
"""
from __future__ import annotations

from pharos.rt.infer import (
    StreamingRestorer,
    bgr_to_tensor,
    confidence_to_map,
    load_model,
    pad_to_multiple,
    tensor_to_bgr,
)

__all__ = [
    "StreamingRestorer",
    "load_model",
    "bgr_to_tensor",
    "tensor_to_bgr",
    "confidence_to_map",
    "pad_to_multiple",
]
