"""Restoration teacher ensemble (phase-2 pseudo-labeling scaffold).

Runs N restoration callables over a directory of real (unpaired) frames, scores
each candidate with a no-reference metric, and writes the best candidate plus a
JSON manifest. This is the runnable skeleton for DESIGN.md §4 / §1-N5: the first
member is a trivial identity, the second a classical CLAHE-based dehazer, and the
registry leaves TODO slots for the pretrained restorers (DehazeFormer /
MB-TaylorFormer / RIDCP) that get wired in once weights are staged.

Scoring: `NoRefScorer` prefers OpenCV's BRISQUE (`cv2.quality`) when its model
files are available (BRISQUE is lower = better, so we negate it to a higher =
better score). When they are not, it falls back to a documented proxy:
`gradient_energy * contrast` on the luminance channel — a dehazed frame is
sharper and higher-contrast than its hazy input, so a larger proxy = better.
All members and scorers operate on BGR uint8 arrays (OpenCV native).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np

# A restoration member maps a BGR uint8 image to a BGR uint8 image.
RestoreFn = Callable[[np.ndarray], np.ndarray]

_IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


# --------------------------------------------------------------------------
# Restoration members
# --------------------------------------------------------------------------
def identity(img: np.ndarray) -> np.ndarray:
    """Passthrough baseline (also the safe fallback candidate)."""
    return img


def clahe_dehaze(img: np.ndarray, clip_limit: float = 2.0, tiles: int = 8) -> np.ndarray:
    """Classical contrast-limited adaptive histogram equalization on L (LAB).

    A cheap, weights-free dehazer: haze compresses local contrast, CLAHE restores
    it. Not competitive with learned restorers, but it makes the pipeline
    end-to-end runnable before any model weights are staged.
    """
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    lch, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tiles, tiles))
    lch = clahe.apply(lch)
    return cv2.cvtColor(cv2.merge((lch, a, b)), cv2.COLOR_LAB2BGR)


def default_registry() -> dict[str, RestoreFn]:
    """Restoration members keyed by name. Extend with pretrained restorers.

    TODO(phase-2): register weight-backed restorers once staged under
    data/weights, e.g.
        reg["dehazeformer"]   = DehazeFormerRestorer(...)
        reg["mb_taylor"]      = MBTaylorFormerRestorer(...)
        reg["ridcp"]          = RIDCPRestorer(...)
    Each must expose the RestoreFn signature (BGR uint8 -> BGR uint8).
    """
    return {"identity": identity, "clahe": clahe_dehaze}


# --------------------------------------------------------------------------
# No-reference scoring (higher = better)
# --------------------------------------------------------------------------
class NoRefScorer:
    """No-reference quality score; higher is better."""

    def __init__(self, brisque_model: Optional[str] = None, brisque_range: Optional[str] = None) -> None:
        self._brisque = None
        if brisque_model and brisque_range and hasattr(cv2, "quality"):
            try:
                self._brisque = cv2.quality.QualityBRISQUE_create(brisque_model, brisque_range)
            except Exception:
                self._brisque = None

    @property
    def method(self) -> str:
        return "brisque" if self._brisque is not None else "grad_contrast_proxy"

    def score(self, img: np.ndarray) -> float:
        if self._brisque is not None:
            try:
                val = self._brisque.compute(img)[0]  # BRISQUE: lower = better
                return -float(val)
            except Exception:
                pass
        return self._proxy(img)

    @staticmethod
    def _proxy(img: np.ndarray) -> float:
        """gradient_energy * contrast on luminance; documented higher = better proxy."""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        grad_energy = float(np.mean(np.sqrt(gx * gx + gy * gy)))
        contrast = float(np.std(gray))
        return grad_energy * contrast


# --------------------------------------------------------------------------
# Ensemble driver
# --------------------------------------------------------------------------
class RestorationEnsemble:
    """Score each member's candidate per image, keep the best, write a manifest."""

    def __init__(
        self,
        members: Optional[dict[str, RestoreFn]] = None,
        scorer: Optional[NoRefScorer] = None,
    ) -> None:
        self.members = members if members is not None else default_registry()
        self.scorer = scorer if scorer is not None else NoRefScorer()

    def run(self, dir_in: str | Path, dir_out: str | Path) -> dict:
        """Process every image in `dir_in`, writing best candidates + manifest to `dir_out`.

        Returns the manifest dict (also written to `dir_out/manifest.json`).
        """
        dir_in = Path(dir_in)
        dir_out = Path(dir_out)
        dir_out.mkdir(parents=True, exist_ok=True)

        files = sorted(p for p in dir_in.iterdir() if p.suffix.lower() in _IMG_EXTS)
        entries: list[dict] = []
        for path in files:
            img = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if img is None:
                continue
            scores: dict[str, float] = {}
            best_name, best_img, best_score = "identity", img, float("-inf")
            for name, fn in self.members.items():
                try:
                    cand = fn(img)
                except Exception:
                    continue
                s = self.scorer.score(cand)
                scores[name] = s
                if s > best_score:
                    best_name, best_img, best_score = name, cand, s
            out_path = dir_out / path.name
            cv2.imwrite(str(out_path), best_img)
            entries.append(
                {
                    "input": str(path),
                    "output": str(out_path),
                    "best_member": best_name,
                    "best_score": best_score,
                    "scores": scores,
                }
            )

        manifest = {
            "scorer": self.scorer.method,
            "members": list(self.members.keys()),
            "num_images": len(entries),
            "entries": entries,
        }
        with open(dir_out / "manifest.json", "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
        return manifest
