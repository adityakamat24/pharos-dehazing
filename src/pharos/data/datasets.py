"""Dataset classes for Pharos. Every dataset yields the contract batch dict:

    {"hazy": FloatTensor 3HW (or T3HW for clips),
     "clean": FloatTensor same shape or None,
     "domain": int DOMAIN_* (collated to LongTensor B),
     "clip": bool,
     "meta": dict}

Use :func:`pharos_collate` as the DataLoader ``collate_fn`` (it handles None clean
targets and dict metas). :func:`build_dataset` is the factory covering every name
used in ``configs/base.yaml``.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Callable, Sequence

import warnings

import torch
from torch.utils.data import ConcatDataset, Dataset, Subset

from ..contracts import DOMAIN_HAZE, DOMAIN_NAMES, DOMAIN_SATELLITE, DOMAIN_SMOKE
from . import synthesis, transforms
from .degradations import RobustnessPipeline
from .transforms import list_images, list_images_recursive, load_image, make_lowres


# ---------------------------------------------------------------------------
# collate
# ---------------------------------------------------------------------------
def _collate_meta(metas: list[dict]) -> dict:
    keys: set[str] = set()
    for m in metas:
        keys.update(m.keys())
    out: dict[str, Any] = {}
    for k in keys:
        vals = [m.get(k) for m in metas]
        if all(isinstance(v, torch.Tensor) for v in vals) and len({v.shape for v in vals}) == 1:
            out[k] = torch.stack(vals)
        elif all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in vals):
            out[k] = torch.tensor(vals, dtype=torch.float32)
        else:
            out[k] = vals
    return out


def pharos_collate(samples: Sequence[dict]) -> dict:
    """Collate per-sample contract dicts into a batch. Handles clean=None (unpaired)
    and heterogeneous meta dicts."""
    hazy = torch.stack([s["hazy"] for s in samples])
    cleans = [s["clean"] for s in samples]
    clean = None if any(c is None for c in cleans) else torch.stack(cleans)
    domain = torch.tensor([int(s["domain"]) for s in samples], dtype=torch.long)
    clip = bool(samples[0].get("clip", False))
    meta = _collate_meta([s.get("meta", {}) for s in samples])
    return {"hazy": hazy, "clean": clean, "domain": domain, "clip": clip, "meta": meta}


# ---------------------------------------------------------------------------
# base mixin
# ---------------------------------------------------------------------------
class _PharosDataset(Dataset):
    """Shared crop/flip/lowres plumbing. Subclasses fill ``self`` config fields and
    implement ``_sample(idx) -> (hazy, clean_or_None, domain, meta)`` on full-res
    tensors; the base applies paired crop/flip and attaches ``meta['full_lowres']``.
    """

    crop: int = 0
    augment: bool = False
    lowres: int = 256
    seed: int | None = None
    name: str = "dataset"
    # Camera-realism degradations (JPEG/blockiness/ISO noise) applied to the
    # INPUT only. Real deployment footage (CCTV smoke) carries compression
    # artifacts under the haze; training on clean-camera inputs makes the model
    # amplify blocks into mud. Set by build_dataset for real paired sets when
    # cfg datasets.robustness_real is true (SyntheticDataset has its own path).
    input_robustness: Any = None

    def _gen(self, idx: int) -> torch.Generator | None:
        if self.seed is None:
            return None
        return torch.Generator().manual_seed(int(self.seed) + idx)

    def _finish(
        self,
        hazy: torch.Tensor,
        clean: torch.Tensor | None,
        domain: int,
        meta: dict,
        generator: torch.Generator | None,
    ) -> dict:
        # degrade the input BEFORE the lowres stream so both views match the camera
        if self.input_robustness is not None:
            hazy = self.input_robustness(hazy, generator)
        # global-context lowres stream computed from the *full* image (pre-crop)
        meta = dict(meta)
        meta["full_lowres"] = make_lowres(hazy, self.lowres)
        imgs = [hazy] if clean is None else [hazy, clean]
        if self.crop and self.crop > 0:
            imgs = transforms.paired_random_crop(imgs, self.crop, generator)
        if self.augment:
            imgs = transforms.paired_random_flip(imgs, generator)
        hazy2 = imgs[0]
        clean2 = None if clean is None else imgs[1]
        return {"hazy": hazy2, "clean": clean2, "domain": int(domain), "clip": False, "meta": meta}


# ---------------------------------------------------------------------------
# paired folder dataset (generic; per-dataset behavior via params)
# ---------------------------------------------------------------------------
_NUM_RE = re.compile(r"\d+")


def _num_key(stem: str) -> str | None:
    m = _NUM_RE.search(stem)
    return m.group(0) if m else None


def _pairs_by(
    hazy_files: list[Path], clean_files: list[Path], keyfn: Callable[[str], str | None]
) -> list[tuple[Path, Path]]:
    cmap: dict[str, Path] = {}
    for p in clean_files:
        k = keyfn(p.stem)
        if k is not None:
            cmap.setdefault(k, p)
    pairs: list[tuple[Path, Path]] = []
    for hp in hazy_files:
        k = keyfn(hp.stem)
        cp = cmap.get(k) if k is not None else None
        if cp is not None:
            pairs.append((hp, cp))
    return pairs


def _match_pairs(
    hazy_files: list[Path],
    clean_files: list[Path],
    mode: str = "auto",
) -> list[tuple[Path, Path]]:
    """Pair hazy/clean files across two dirs.

    Modes:
    * ``stem``   exact filename-stem match (RESIDE-6K GT/hazy, SmokeBench GT/LQ).
    * ``prefix`` match on the token before the first '_' (RESIDE ``0001_0.8_0.2`` <->
      ``0001``; NTIRE ``NN_..._hazy`` <-> ``NN_..._GT``).
    * ``num``    match on the leading number (SateHaze1k ``1-inputs`` <-> ``1-targets``).
    * ``auto``   (default) try all three and keep whichever pairs the most files.
    """
    if mode == "stem":
        return _pairs_by(hazy_files, clean_files, lambda s: s)
    if mode == "prefix":
        return _pairs_by(hazy_files, clean_files, lambda s: s.split("_")[0])
    if mode == "num":
        return _pairs_by(hazy_files, clean_files, _num_key)
    candidates = [
        _pairs_by(hazy_files, clean_files, lambda s: s),
        _pairs_by(hazy_files, clean_files, lambda s: s.split("_")[0]),
        _pairs_by(hazy_files, clean_files, _num_key),
    ]
    return max(candidates, key=len)


class PairedFolderDataset(_PharosDataset):
    """Generic paired dataset.

    Two supported layouts:
    * two directories (``hazy_dir`` + ``clean_dir``) matched by filename stem
      (``match='stem'``) or hazy-stem-prefix (``match='prefix'`` for RESIDE-style
      ``0001_0.8_0.2.png`` <-> ``0001.png``);
    * a single directory (``hazy_dir`` only) with suffix-based pairs
      (``match='suffix'``, e.g. ``01_hazy.png`` <-> ``01_GT.png``).
    """

    def __init__(
        self,
        hazy_dir: str | Path,
        clean_dir: str | Path | None = None,
        domain: int = DOMAIN_HAZE,
        match: str = "auto",
        hazy_suffix: str = "_hazy",
        clean_suffix: str = "_GT",
        recursive: bool = False,
        crop: int = 0,
        augment: bool = False,
        lowres: int = 256,
        seed: int | None = None,
        name: str = "paired",
    ) -> None:
        self.domain = int(domain)
        self.crop, self.augment, self.lowres, self.seed, self.name = crop, augment, lowres, seed, name
        lister = list_images_recursive if recursive else list_images
        hazy_dir = Path(hazy_dir)
        if match == "suffix":
            files = lister(hazy_dir)
            pairs: list[tuple[Path, Path]] = []
            for f in files:
                if hazy_suffix in f.name:
                    cand = f.with_name(f.name.replace(hazy_suffix, clean_suffix))
                    if cand.exists():
                        pairs.append((f, cand))
            self.pairs = pairs
        else:
            clean_dir = Path(clean_dir) if clean_dir is not None else hazy_dir
            hazy_files, clean_files = lister(hazy_dir), lister(clean_dir)
            self.pairs = _match_pairs(hazy_files, clean_files, match)

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> dict:
        hp, cp = self.pairs[idx]
        hazy = load_image(hp)
        clean = load_image(cp)
        meta = {"dataset": self.name, "hazy_path": str(hp), "clean_path": str(cp)}
        return self._finish(hazy, clean, self.domain, meta, self._gen(idx))


# ---------------------------------------------------------------------------
# unpaired dataset (RTTS / URHI): clean = None
# ---------------------------------------------------------------------------
class UnpairedDataset(_PharosDataset):
    def __init__(
        self,
        hazy_dir: str | Path,
        domain: int = DOMAIN_HAZE,
        recursive: bool = True,
        crop: int = 0,
        augment: bool = False,
        lowres: int = 256,
        seed: int | None = None,
        name: str = "unpaired",
    ) -> None:
        self.domain = int(domain)
        self.crop, self.augment, self.lowres, self.seed, self.name = crop, augment, lowres, seed, name
        lister = list_images_recursive if recursive else list_images
        self.files = lister(hazy_dir)

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> dict:
        hp = self.files[idx]
        hazy = load_image(hp)
        meta = {"dataset": self.name, "hazy_path": str(hp), "clean_path": None}
        return self._finish(hazy, None, self.domain, meta, self._gen(idx))


# ---------------------------------------------------------------------------
# synthetic dataset (clean folder + synthesis.py)
# ---------------------------------------------------------------------------
class SyntheticDataset(_PharosDataset):
    """Wrap a clean-image folder; synthesize haze/smoke/satellite on the fly and
    optionally apply robustness augmentations to the degraded input only."""

    def __init__(
        self,
        clean_dir: str | Path,
        domain: int | str = DOMAIN_HAZE,
        recursive: bool = True,
        crop: int = 0,
        augment: bool = False,
        lowres: int = 256,
        seed: int | None = None,
        robustness: RobustnessPipeline | None = None,
        depth_dir: str | Path | None = None,
        name: str | None = None,
        synth_kwargs: dict | None = None,
    ) -> None:
        if isinstance(domain, str):
            domain_id = {v: k for k, v in DOMAIN_NAMES.items()}[domain]
        else:
            domain_id = int(domain)
        self.domain = domain_id
        self.domain_name = DOMAIN_NAMES[domain_id]
        self.crop, self.augment, self.lowres, self.seed = crop, augment, lowres, seed
        self.name = name or f"synth_{self.domain_name}"
        self.robustness = robustness
        self.depth_dir = Path(depth_dir) if depth_dir else None
        self.synth_kwargs = synth_kwargs or {}
        lister = list_images_recursive if recursive else list_images
        self.files = lister(clean_dir)

    def __len__(self) -> int:
        return len(self.files)

    def _load_depth(self, clean_path: Path) -> torch.Tensor | None:
        if self.depth_dir is None:
            return None
        cand = self.depth_dir / (clean_path.stem + ".png")
        if cand.exists():
            d = load_image(cand)[:1]  # single channel
            return d
        return None

    def __getitem__(self, idx: int) -> dict:
        cp = self.files[idx]
        clean = load_image(cp)
        g = self._gen(idx)
        depth = self._load_depth(cp) if self.domain == DOMAIN_HAZE else None
        hazy, params = synthesis.synthesize(
            clean, self.domain_name, generator=g, depth=depth, **self.synth_kwargs
        )
        if self.robustness is not None:
            hazy = self.robustness(hazy, g)
        meta = {
            "dataset": self.name,
            "clean_path": str(cp),
            "beta": torch.tensor([params["beta"]], dtype=torch.float32),
            "beta_bs": torch.tensor([params.get("beta_bs", params["beta"])], dtype=torch.float32),
            "airlight": params["airlight"].float(),
            "sigma": torch.tensor([params["sigma"]], dtype=torch.float32),
            "synthetic": True,
        }
        return self._finish(hazy, clean, self.domain, meta, g)


# ---------------------------------------------------------------------------
# clear-passthrough dataset (identity target; feeds the severity gate)
# ---------------------------------------------------------------------------
class ClearPassthroughDataset(_PharosDataset):
    """Clean image as both input and target; domain sampled. Trains the severity
    gate to pass clear frames through untouched (output ~= input)."""

    def __init__(
        self,
        clean_dir: str | Path,
        recursive: bool = True,
        crop: int = 0,
        augment: bool = False,
        lowres: int = 256,
        seed: int | None = None,
        name: str = "clear_passthrough",
    ) -> None:
        self.crop, self.augment, self.lowres, self.seed, self.name = crop, augment, lowres, seed, name
        lister = list_images_recursive if recursive else list_images
        self.files = lister(clean_dir)

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> dict:
        cp = self.files[idx]
        clean = load_image(cp)
        g = self._gen(idx)
        domain = int(torch.randint(0, 3, (1,), generator=g).item())
        meta = {
            "dataset": self.name,
            "clean_path": str(cp),
            "beta": torch.tensor([0.0], dtype=torch.float32),
            "clear": True,
        }
        # identity: hazy == clean
        return self._finish(clean.clone(), clean, domain, meta, g)


# ---------------------------------------------------------------------------
# video helpers
# ---------------------------------------------------------------------------
def _discover_sequences(root: str | Path, flat_fallback: bool = False) -> list[list[Path]]:
    """Return a list of frame-path lists, one per sequence subfolder under root.
    A subfolder counts as a sequence if it directly contains >=1 image. If no
    subfolder sequences exist and ``flat_fallback`` is set, a flat folder of frames
    is returned as a single sequence."""
    root = Path(root)
    seqs: list[list[Path]] = []
    if not root.is_dir():
        return seqs
    for sub in sorted(p for p in root.iterdir() if p.is_dir()):
        frames = list_images(sub)
        if frames:
            seqs.append(frames)
    if not seqs and flat_fallback:
        flat = list_images(root)
        if flat:
            seqs.append(flat)
    return seqs


def _clip_windows(seqs: list[list[Path]], clip_len: int) -> list[list[int]]:
    """Enumerate (seq_index, start) windows as flat index lists into a paired store."""
    windows: list[tuple[int, int]] = []
    for si, frames in enumerate(seqs):
        n = len(frames)
        if n >= clip_len:
            for start in range(0, n - clip_len + 1):
                windows.append((si, start))
        elif n > 0:
            windows.append((si, 0))  # short sequence: will be padded by repetition
    return windows  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# real paired video (REVIDE)
# ---------------------------------------------------------------------------
class VideoClipDataset(_PharosDataset):
    """Paired video clips (e.g. REVIDE). ``hazy_root`` and ``clean_root`` each
    contain one subfolder per sequence with matching frame filenames. Yields a clip
    of ``clip_len`` consecutive frames: hazy/clean are (T,3,H,W); clip=True."""

    def __init__(
        self,
        hazy_root: str | Path,
        clean_root: str | Path,
        clip_len: int = 3,
        domain: int = DOMAIN_HAZE,
        crop: int = 0,
        augment: bool = False,
        lowres: int = 256,
        seed: int | None = None,
        name: str = "video",
    ) -> None:
        self.clip_len = clip_len
        self.domain = int(domain)
        self.crop, self.augment, self.lowres, self.seed, self.name = crop, augment, lowres, seed, name
        self.hazy_seqs = _discover_sequences(hazy_root, flat_fallback=True)
        self.clean_seqs = _discover_sequences(clean_root, flat_fallback=True)
        self.windows = _clip_windows(self.hazy_seqs, clip_len)

    def __len__(self) -> int:
        return len(self.windows)

    def _frames(self, seqs: list[list[Path]], si: int, start: int) -> list[Path]:
        frames = seqs[si]
        idxs = [min(start + k, len(frames) - 1) for k in range(self.clip_len)]
        return [frames[i] for i in idxs]

    def __getitem__(self, idx: int) -> dict:
        si, start = self.windows[idx]
        g = self._gen(idx)
        hazy_paths = self._frames(self.hazy_seqs, si, start)
        clean_si = min(si, len(self.clean_seqs) - 1) if self.clean_seqs else 0
        clean_paths = self._frames(self.clean_seqs, clean_si, start) if self.clean_seqs else None
        hazy = torch.stack([load_image(p) for p in hazy_paths])
        clean = torch.stack([load_image(p) for p in clean_paths]) if clean_paths else None
        meta = {"dataset": self.name, "hazy_paths": [str(p) for p in hazy_paths]}
        return self._finish_clip(hazy, clean, self.domain, meta, g)

    def _finish_clip(self, hazy, clean, domain, meta, generator) -> dict:
        meta = dict(meta)
        meta["full_lowres"] = torch.stack([make_lowres(f, self.lowres) for f in hazy])
        if self.crop and self.crop > 0:
            t = hazy.shape[0]
            frames = [hazy[i] for i in range(t)] + ([clean[i] for i in range(t)] if clean is not None else [])
            frames = transforms.paired_random_crop(frames, self.crop, generator)
            if self.augment:
                frames = transforms.paired_random_flip(frames, generator)
            hazy = torch.stack(frames[:t])
            clean = torch.stack(frames[t:]) if clean is not None else None
        return {"hazy": hazy, "clean": clean, "domain": int(domain), "clip": True, "meta": meta}


# ---------------------------------------------------------------------------
# synthetic video (temporally-coherent synthesis)
# ---------------------------------------------------------------------------
class SynthVideoDataset(_PharosDataset):
    """Temporally-coherent synthetic clips. ``clean_root`` may contain per-sequence
    subfolders of clean frames (real video) or a flat folder of stills (each still
    becomes a static clip with a drifting degradation field). Domains cycle over the
    requested set (default all three)."""

    def __init__(
        self,
        clean_root: str | Path,
        clip_len: int = 3,
        domains: Sequence[str] = ("haze", "smoke", "satellite"),
        crop: int = 0,
        augment: bool = False,
        lowres: int = 256,
        seed: int | None = None,
        robustness: RobustnessPipeline | None = None,
        name: str = "synth_video",
        synth_kwargs: dict | None = None,
    ) -> None:
        self.clip_len = clip_len
        self.domains = list(domains)
        self.crop, self.augment, self.lowres, self.seed, self.name = crop, augment, lowres, seed, name
        self.robustness = robustness
        # synthesize_clip only accepts the temporal-relevant knobs
        self.synth_kwargs = {
            k: v for k, v in (synth_kwargs or {}).items()
            if k in ("smoke_mode", "isp_aware", "beta_bs_ratio")
        }
        self.seqs = _discover_sequences(clean_root)
        if self.seqs:
            self.stills: list[Path] = []
            self.windows = _clip_windows(self.seqs, clip_len)
        else:
            # no real sequences: each still becomes its own static clip (drifting
            # degradation over a frozen scene) -> proper temporally-coherent synth video
            self.stills = list_images_recursive(clean_root)
            self.windows = [(i, 0) for i in range(len(self.stills))]  # type: ignore[misc]

    def __len__(self) -> int:
        return len(self.windows)

    def _clean_clip(self, si: int, start: int) -> torch.Tensor:
        if self.stills:
            frame = load_image(self.stills[si])
            return frame.unsqueeze(0).repeat(self.clip_len, 1, 1, 1)
        frames = self.seqs[si]
        idxs = [min(start + k, len(frames) - 1) for k in range(self.clip_len)]
        return torch.stack([load_image(frames[i]) for i in idxs])

    def __getitem__(self, idx: int) -> dict:
        si, start = self.windows[idx]
        g = self._gen(idx)
        clean = self._clean_clip(si, start)
        domain = self.domains[idx % len(self.domains)]
        hazy, params = synthesis.synthesize_clip(clean, domain, generator=g, **self.synth_kwargs)
        if self.robustness is not None:
            hazy = torch.stack([self.robustness(hazy[i], g) for i in range(hazy.shape[0])])
        dom_id = {v: k for k, v in DOMAIN_NAMES.items()}[domain]
        meta = {
            "dataset": self.name,
            "beta": torch.tensor([params["beta"]], dtype=torch.float32),
            "beta_bs": torch.tensor([params.get("beta_bs", params["beta"])], dtype=torch.float32),
            "airlight": params["airlight"].float(),
            "sigma": torch.tensor([params.get("sigma", 0.0)], dtype=torch.float32),
            "synthetic": True,
        }
        return self._finish_clip(hazy, clean, dom_id, meta, g)

    _finish_clip = VideoClipDataset._finish_clip


# ---------------------------------------------------------------------------
# path resolution + factory
# ---------------------------------------------------------------------------
def _first_existing_dir(*paths: Path) -> Path | None:
    for p in paths:
        if p.is_dir():
            return p
    return None


def _resolve_paired(root: Path, split: str) -> tuple[Path, Path | None, str] | None:
    """Probe common paired layouts under ``root`` for the given split. Returns
    (hazy_dir, clean_dir_or_None_for_suffix, match_mode) or None if nothing found."""
    hazy_names = ["hazy", "haze", "input", "cloud", "cloudy", "LR_Haze", "LQ", "data", "thick_fog"]
    clean_names = ["GT", "gt", "clean", "clear", "target", "label", "gt_image", "no_fog"]
    split_prefixes = [split, split.capitalize(), split.upper(), ""]
    if split == "test":
        split_prefixes = ["test", "Test", "TEST", "val", "Val", "SOTS", ""]

    for sp in split_prefixes:
        base = root / sp if sp else root
        if not base.is_dir():
            continue
        for hn in hazy_names:
            hdir = base / hn
            if not hdir.is_dir():
                continue
            for cn in clean_names:
                cdir = base / cn
                if cdir.is_dir():
                    match = "prefix" if list(hdir.glob("*_*")) else "stem"
                    return hdir, cdir, match
    # suffix layout: a single dir with _hazy/_GT pairs (searched recursively so
    # NH-HAZE's nested 'NH-HAZE/NN_hazy.png'+'NN_GT.png' folder is found)
    for d in [root] + [p for p in root.rglob("*") if p.is_dir()]:
        imgs = list_images(d)
        if any("_hazy" in p.name.lower() for p in imgs) and any("_gt" in p.name.lower() for p in imgs):
            return d, None, "suffix"
    return None


_HAZY_DIR_NAMES = {"hazy", "haze", "input", "cloud", "cloudy", "lr_haze", "lq", "thick_fog", "smoke"}
_CLEAN_DIR_NAMES = {"gt", "clean", "clear", "target", "label", "hr", "gt_image", "no_fog", "groundtruth"}


def _recursive_paired(root: Path, split: str) -> list[tuple[Path, Path]]:
    """Find (hazy_dir, clean_dir) sibling pairs anywhere under ``root`` and filter by
    split token in the path. Handles nested layouts (e.g. SateHaze1k
    ``Haze1k/Haze1k_thin/dataset/train/{input,target}`` or O-HAZE's named subfolder)."""
    root = Path(root)
    if not root.is_dir():
        return []
    all_dirs = [root] + [p for p in root.rglob("*") if p.is_dir()]
    pairs: list[tuple[Path, Path]] = []
    for d in all_dirs:
        if d.name.lower() in _HAZY_DIR_NAMES:
            try:
                sibs = {s.name.lower(): s for s in d.parent.iterdir() if s.is_dir()}
            except OSError:
                continue
            cdir = next((sibs[c] for c in _CLEAN_DIR_NAMES if c in sibs), None)
            if cdir is not None and cdir != d:
                pairs.append((d, cdir))

    def token(p: Path) -> str:
        s = str(p).lower()
        if "train" in s:
            return "train"
        if "test" in s:
            return "test"
        if "val" in s:
            return "val"
        return ""

    if split == "train":
        want = [p for p in pairs if token(p[0]) == "train"]
    else:
        want = [p for p in pairs if token(p[0]) in ("test", "val")]
    return want or pairs


def _resolve_clean_pool(cfg: Any, data_root: Path) -> Path:
    """Where synthetic/clear datasets draw clean images from."""
    data_cfg = cfg.get("data", {}) if hasattr(cfg, "get") else {}
    override = data_cfg.get("clean_pool") if isinstance(data_cfg, dict) else None
    candidates = [Path(override)] if override else []
    candidates += [
        data_root / "reside6k" / "train" / "GT",
        data_root / "reside6k" / "train" / "gt",
        data_root / "reside6k" / "train" / "clean",
        data_root / "clean",
        data_root / "reside6k",
    ]
    found = _first_existing_dir(*candidates)
    return found if found is not None else (data_root / "clean")


_DOMAIN_OF = {
    "reside6k": DOMAIN_HAZE,
    "sots_mix": DOMAIN_HAZE,
    "nhhaze": DOMAIN_HAZE,
    "densehaze": DOMAIN_HAZE,
    "ohaze": DOMAIN_HAZE,
    "ihaze": DOMAIN_HAZE,
    "smokebench": DOMAIN_SMOKE,
    "smokebench_test": DOMAIN_SMOKE,
    "satehaze1k": DOMAIN_SATELLITE,
    "satehaze1k_test": DOMAIN_SATELLITE,
    "rice1": DOMAIN_SATELLITE,
    "rice2": DOMAIN_SATELLITE,
}

# name -> (dataset dir under data_root, split)
_PAIRED_DIR = {
    "reside6k": ("reside6k", "train"),
    "sots_mix": ("reside6k", "test"),
    "smokebench": ("smokebench", "train"),
    "smokebench_test": ("smokebench", "test"),
    "satehaze1k": ("satehaze1k", "train"),
    "satehaze1k_test": ("satehaze1k", "test"),
    "rice1": ("rice/RICE1", "train"),
    "rice2": ("rice/RICE2", "train"),
    "nhhaze": ("nhhaze", "train"),
    "densehaze": ("densehaze", "train"),
    "ohaze": ("ohaze", "train"),
    "ihaze": ("ihaze", "train"),
}


def _cfg_get(cfg: Any, dotted: str, default: Any) -> Any:
    node = cfg
    for part in dotted.split("."):
        if isinstance(node, dict) and part in node:
            node = node[part]
        else:
            return default
    return node


def _synth_kwargs(cfg: Any) -> dict:
    """Extract the ``synthesis`` config block (turbulent smoke / ISP-aware haze /
    split-beta knobs) as kwargs to splat into synthesis.synthesize(_clip)."""
    s = _cfg_get(cfg, "synthesis", {}) or {}
    if not isinstance(s, dict):
        return {}
    keys = ("smoke_mode", "isp_aware", "beta_bs_ratio", "isp_gamma", "shot_photons")
    return {k: s[k] for k in keys if k in s}


class _RobustView(Dataset):
    """Substitute a neighboring sample when a file is unreadable.

    Real archives ship the occasional truncated/corrupt member (e.g. one bad
    SateHaze1k zip entry); a multi-hour training run must degrade gracefully
    instead of dying in a DataLoader worker. Eight consecutive failures still
    raise — that indicates a broken dataset, not a bad file.
    """

    def __init__(self, ds: Dataset, name: str = "") -> None:
        self.ds = ds
        self.name = name

    def __len__(self) -> int:
        return len(self.ds)  # type: ignore[arg-type]

    def __getitem__(self, idx: int) -> dict:
        n = len(self)
        last: Exception | None = None
        for hop in range(8):
            j = (idx + hop * 131) % n
            try:
                return self.ds[j]
            except (FileNotFoundError, OSError) as e:
                last = e
                warnings.warn(f"dataset {self.name}: unreadable sample {j} ({e}); substituting")
        raise RuntimeError(f"dataset {self.name}: 8 consecutive unreadable samples from {idx}") from last


def build_dataset(name: str, cfg: Any, split: str = "train") -> Dataset:
    """Factory covering every dataset name used in configs/base.yaml.

    ``split`` is 'train' or 'eval'/'test'. In train split, random crop (cfg.train.crop)
    and flips are enabled; in eval split, full-resolution images are returned (use a
    DataLoader with batch_size=1 and :func:`pharos_collate`).

    When ``cfg.data_subset`` is set (e.g. configs/overfit50.yaml), the dataset is
    capped to that many samples, chosen deterministically from ``cfg.seed``.
    """
    ds = _build_dataset_impl(name, cfg, split)
    # NTIRE convention split for the small flat real-haze sets (no train/test
    # dirs on disk): train = all but the last 5 pairs, eval = last 5. Off by
    # default so long-run eval curves stay comparable; REQUIRED (set
    # datasets.ntire_split: true) for any config that trains on these sets,
    # otherwise fine-tuning would train on the benchmark images.
    if bool(_cfg_get(cfg, "datasets.ntire_split", False)) and name in _NTIRE_FLAT:
        n = len(ds)
        if n > 5:
            idx = list(range(0, n - 5)) if split == "train" else list(range(n - 5, n))
            ds = Subset(ds, idx)
    cap = _cfg_get(cfg, "data_subset", None)
    if cap and len(ds) > int(cap):
        g = torch.Generator().manual_seed(int(_cfg_get(cfg, "seed", 0) or 0))
        idx = torch.randperm(len(ds), generator=g)[: int(cap)].tolist()
        ds = Subset(ds, idx)
    return _RobustView(ds, name=name)


_NTIRE_FLAT = {"nhhaze", "densehaze", "ohaze", "ihaze"}


def _build_dataset_impl(name: str, cfg: Any, split: str = "train") -> Dataset:
    data_root = Path(_cfg_get(cfg, "data_root", "data"))
    lowres = int(_cfg_get(cfg, "model.lowres", 256))
    crop = int(_cfg_get(cfg, "train.crop", 256))
    clip_len = int(_cfg_get(cfg, "train.clip_len", 3))
    seed = _cfg_get(cfg, "seed", None)
    is_train = split == "train"
    use_crop = crop if is_train else 0
    augment = is_train

    real_robustness = (
        RobustnessPipeline()
        if is_train and bool(_cfg_get(cfg, "datasets.robustness_real", False))
        else None
    )

    def _make_paired(hdir: Path, cdir: Path | None, match: str, domain: int, nm: str) -> Dataset:
        # single-dir suffix layout keeps its mode; two-dir layouts use robust auto-match
        eff = "suffix" if cdir is None else "auto"
        ds = PairedFolderDataset(
            hdir, cdir, domain=domain, match=eff, recursive=(cdir is None),
            crop=use_crop, augment=augment, lowres=lowres, seed=seed, name=nm,
        )
        ds.input_robustness = real_robustness
        return ds

    def _concat_recursive(root: Path, split_kind: str, domain: int, nm: str) -> Dataset | None:
        pairs = _recursive_paired(root, split_kind)
        if not pairs:
            return None
        parts = [_make_paired(h, c, "auto", domain, nm) for h, c in pairs]
        return ConcatDataset(parts) if len(parts) > 1 else parts[0]

    # ---- paired datasets ----
    if name in _PAIRED_DIR:
        subdir, def_split = _PAIRED_DIR[name]
        root = data_root / subdir
        domain = _DOMAIN_OF[name]
        split_kind = "train" if def_split == "train" else "test"
        # SateHaze1k (nested thin/moderate/thick) always goes through the recursive path
        if name in ("satehaze1k", "satehaze1k_test"):
            ds = _concat_recursive(data_root / "satehaze1k", split_kind, domain, "satehaze1k")
            return ds if ds is not None else _make_paired(
                root / "input", root / "target", "stem", domain, "satehaze1k")
        # 1) fast top-level probe
        res = _resolve_paired(root, def_split)
        if res is not None:
            hdir, cdir, match = res
            return _make_paired(hdir, cdir, match, domain, name)
        # 2) recursive fallback for nested archive layouts
        ds = _concat_recursive(root, split_kind, domain, name)
        if ds is not None:
            return ds
        # 3) empty dataset on the conventional path so import/len still work
        return _make_paired(root / "hazy", root / "GT", "stem", domain, name)

    # ---- synthetic (single-image) ----
    if name in ("synth_haze", "synth_smoke", "synth_satellite"):
        domain_name = name.split("_", 1)[1]
        clean_pool = _resolve_clean_pool(cfg, data_root)
        robustness = RobustnessPipeline() if is_train else None
        depth_dir = data_root / "reside6k" / "train" / "depth"
        return SyntheticDataset(
            clean_pool, domain=domain_name, crop=use_crop, augment=augment, lowres=lowres,
            seed=seed, robustness=robustness, depth_dir=depth_dir if depth_dir.is_dir() else None,
            name=name, synth_kwargs=_synth_kwargs(cfg),
        )

    if name == "clear_passthrough":
        clean_pool = _resolve_clean_pool(cfg, data_root)
        return ClearPassthroughDataset(
            clean_pool, crop=use_crop, augment=augment, lowres=lowres, seed=seed, name=name,
        )

    # ---- unpaired ----
    if name in ("rtts", "urhi"):
        root = data_root / name
        hdir = _first_existing_dir(root / "JPEGImages", root / "images", root) or root
        return UnpairedDataset(
            hdir, domain=DOMAIN_HAZE, recursive=True, crop=use_crop, augment=augment,
            lowres=lowres, seed=seed, name=name,
        )

    # ---- video ----
    if name == "revide":
        root = data_root / "revide"
        split_dir = _first_existing_dir(root / "Train", root / "train", root) or root
        hazy_root = _first_existing_dir(split_dir / "hazy", split_dir / "haze", split_dir / "Hazy") or (
            split_dir / "hazy"
        )
        clean_root = _first_existing_dir(split_dir / "gt", split_dir / "GT", split_dir / "clean") or (
            split_dir / "gt"
        )
        return VideoClipDataset(
            hazy_root, clean_root, clip_len=clip_len, domain=DOMAIN_HAZE, crop=use_crop,
            augment=augment, lowres=lowres, seed=seed, name="revide",
        )

    if name == "synth_video":
        clean_root = _resolve_clean_pool(cfg, data_root)
        robustness = RobustnessPipeline() if is_train else None
        return SynthVideoDataset(
            clean_root, clip_len=clip_len, crop=use_crop, augment=augment, lowres=lowres,
            seed=seed, robustness=robustness, name="synth_video", synth_kwargs=_synth_kwargs(cfg),
        )

    raise ValueError(f"unknown dataset name: {name!r}")
