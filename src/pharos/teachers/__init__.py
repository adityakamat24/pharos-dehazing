"""Pharos training-time teachers (never run at inference).

- DepthTeacher    : Depth Anything V2 Small (relative depth, higher = farther)
- DetectionTeacher: frozen Faster R-CNN MobileNetV3 FPN feature maps
- FlowTeacher     : frozen RAFT-small optical flow (+ flow_warp utility)
- TeacherBundle   : config-driven container satisfying contracts.TeacherBundle
- RestorationEnsemble: phase-2 pseudo-labeling scaffold
"""
from __future__ import annotations

from .bundle import TeacherBundle
from .depth import DepthTeacher
from .detector import DetectionTeacher
from .ensemble import NoRefScorer, RestorationEnsemble, clahe_dehaze, default_registry, identity
from .flow import FlowTeacher, flow_warp

__all__ = [
    "DepthTeacher",
    "DetectionTeacher",
    "FlowTeacher",
    "flow_warp",
    "TeacherBundle",
    "RestorationEnsemble",
    "NoRefScorer",
    "default_registry",
    "identity",
    "clahe_dehaze",
]
