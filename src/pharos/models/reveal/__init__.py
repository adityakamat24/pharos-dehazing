"""RevealNet v2 package: confidence-weighted reveal accumulation (DESIGN.md §9d)."""

from .aligner import TieredAligner, four_point_to_homography, invert_homography, warp_grid
from .compositor import age_decay, composite
from .memory import RevealMemory
from .revealnet import RevealNet

__all__ = [
    "RevealNet",
    "TieredAligner",
    "RevealMemory",
    "composite",
    "age_decay",
    "warp_grid",
    "four_point_to_homography",
    "invert_homography",
]
