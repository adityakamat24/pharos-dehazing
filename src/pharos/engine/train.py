"""Config-driven Trainer for Pharos (WS-D).

Builds model / datasets / teachers / loss via factory names with dependency
injection so tests can pass stubs. Supports mixed image + clip training, AMP,
AdamW + cosine-warmup schedule, grad clipping & accumulation, EMA, checkpointing
with resume, a periodic eval hook, and TensorBoard logging. Designed to fit a
6 GB RTX 3060 (channels_last, pin_memory, grad accumulation, configurable
Windows-safe dataloader workers).

Factory-name contract (only used when components are NOT injected — tests inject):
    model    : pharos.models.PharosNet(cfg.model)
    datasets : pharos.data.build_dataset(name, cfg)      -> torch Dataset
    loss     : pharos.losses.PharosLoss(cfg.loss)
    teachers : pharos.teachers.bundle.build_teachers(cfg.teachers)
Each factory falls back to being called with the full ``cfg`` (then no args) if
the section-scoped call raises ``TypeError``; a missing module raises a clear
``RuntimeError`` at runtime (never at import time).

CLI:
    python -m pharos.engine.train --config configs/overfit50.yaml [--override k=v ...]

Windows note: dataloader workers require the ``if __name__ == "__main__"`` guard
below (multiprocessing 'spawn'); ``persistent_workers`` is enabled only when
``num_workers > 0``.
"""
from __future__ import annotations

import argparse
import importlib
import math
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

import torch
from torch.utils.data import ConcatDataset, DataLoader

from ..config import Config, load_config
from ..contracts import PharosOutput
from .logging import TBLogger
from .utils import (
    AverageMeter,
    ModelEMA,
    Timer,
    cosine_warmup_lambda,
    load_checkpoint,
    move_batch_to_device,
    parse_overrides,
    resolve_run_dirs,
    save_checkpoint,
    seed_everything,
)

# ---------------------------------------------------------------------------
# Collation: turn per-sample contract dicts into a batched contract dict.
# ---------------------------------------------------------------------------


def pharos_collate(samples: list[dict]) -> dict:
    """Collate per-sample dicts into the batch contract (§8).

    Images -> hazy ``B,3,H,W``; clips -> hazy ``B,T,3,H,W`` with ``clip=True``.
    ``clean`` is None if any sample lacks it; ``meta`` is kept as a list.
    """
    clip = bool(samples[0].get("clip", False))
    hazy = torch.stack([s["hazy"] for s in samples])
    cleans = [s.get("clean") for s in samples]
    clean = torch.stack(cleans) if all(c is not None for c in cleans) else None
    domain = torch.as_tensor([int(s.get("domain", 0)) for s in samples], dtype=torch.long)
    meta = [s.get("meta", {}) for s in samples]
    return {"hazy": hazy, "clean": clean, "domain": domain, "clip": clip, "meta": meta}


# ---------------------------------------------------------------------------
# Default factories (importlib, defensive) + dependency-injection container.
# ---------------------------------------------------------------------------


def _import(name: str):
    try:
        return importlib.import_module(name)
    except Exception as e:  # pragma: no cover - integration path
        raise RuntimeError(
            f"Could not import '{name}' ({e}). This module belongs to a parallel "
            f"workstream; either ensure it is available or inject the component "
            f"directly into Trainer(...)."
        ) from e


def _construct(fn: Callable, cfg: Config, section: str):
    """Call ``fn`` with the section config, then full cfg, then no args."""
    candidates: list[tuple] = []
    if isinstance(cfg.get(section), dict):
        candidates.append((cfg[section],))
    candidates += [(cfg,), ()]
    last: Optional[BaseException] = None
    for args in candidates:
        try:
            return fn(*args)
        except TypeError as e:
            last = e
    raise last  # type: ignore[misc]


def default_build_model(cfg: Config):
    return _construct(getattr(_import("pharos.models"), "PharosNet"), cfg, "model")


def default_build_loss(cfg: Config):
    # PharosLoss reads both cfg.loss and cfg.teachers — it must receive the full
    # config tree. Passing only the loss section would make it silently fall back
    # to default weights and detector every_n.
    return getattr(_import("pharos.losses"), "PharosLoss")(cfg)


def default_build_teachers(cfg: Config):
    # build_teachers reads cfg.teachers AND cfg.data_root (weights cache) — full tree.
    return getattr(_import("pharos.teachers.bundle"), "build_teachers")(cfg)


def default_build_datasets(cfg: Config, names: list[str], split: str) -> list:
    # split must be passed explicitly: build_dataset defaults to 'train', which
    # would silently give eval sets random crops/flips.
    build = getattr(_import("pharos.data"), "build_dataset")
    return [build(name, cfg, split=split) for name in names]


@dataclass
class Deps:
    """Injectable component factories (defaults resolve real workstream modules)."""

    build_model: Callable[[Config], Any] = default_build_model
    build_loss: Callable[[Config], Any] = default_build_loss
    build_teachers: Callable[[Config], Any] = default_build_teachers
    build_datasets: Callable[[Config, list[str], str], list] = default_build_datasets


class _Cycle:
    """Infinitely cycle a DataLoader, reshuffling each epoch."""

    def __init__(self, loader: DataLoader) -> None:
        self.loader = loader
        self._it = iter(loader)

    def next(self) -> dict:
        try:
            return next(self._it)
        except StopIteration:
            self._it = iter(self.loader)
            return next(self._it)


def _stack_outputs(outs: list[PharosOutput]) -> PharosOutput:
    """Stack a list of per-frame PharosOutputs into a clip output (T dim at axis 1).

    Contract note for WS-C: clip batches produce tensors shaped ``B,T,...`` so
    ``PharosLoss`` can compute temporal terms; ``state`` is the last frame's state.
    """

    def stk(key: str):
        vals = [getattr(o, key) for o in outs]
        return torch.stack(vals, dim=1) if all(v is not None for v in vals) else None

    deg_keys = outs[0].deg.keys()
    deg = {k: torch.stack([o.deg[k] for o in outs], dim=1) for k in deg_keys}
    return PharosOutput(
        output=stk("output"),
        confidence=stk("confidence"),
        grid=stk("grid"),
        state=outs[-1].state,
        deg=deg,
        t_hat=stk("t_hat"),
        aux={},
    )


class Trainer:
    """Pharos training engine. Inject any of model/loaders/loss/teachers for tests."""

    def __init__(
        self,
        cfg: Config,
        *,
        model: Optional[torch.nn.Module] = None,
        image_loader: Optional[DataLoader] = None,
        video_loader: Optional[DataLoader] = None,
        loss_fn: Optional[Callable] = None,
        teachers: Optional[Any] = None,
        eval_fn: Optional[Callable] = None,
        device: Optional[torch.device | str] = None,
        deps: Optional[Deps] = None,
        logger: Optional[TBLogger] = None,
    ) -> None:
        self.cfg = cfg
        self.deps = deps or Deps()
        self.device = torch.device(device) if device is not None else _auto_device()
        seed_everything(int(cfg.get("seed", 0)))
        tcfg = cfg.train

        # --- components (inject or build) ---
        self.model = (model if model is not None else self.deps.build_model(cfg)).to(self.device)
        self.channels_last = bool(tcfg.get("channels_last", False)) and self.device.type == "cuda"
        if self.channels_last:
            self.model = self.model.to(memory_format=torch.channels_last)
        self.loss_fn = loss_fn if loss_fn is not None else self.deps.build_loss(cfg)
        self.teachers = teachers if teachers is not None else _maybe_build_teachers(self.deps, cfg)
        self.eval_fn = eval_fn

        self.image_loader = image_loader if image_loader is not None else self._build_image_loader()
        self.video_loader = video_loader if video_loader is not None else self._build_video_loader()
        self._img_iter = _Cycle(self.image_loader) if self.image_loader is not None else None
        self._vid_iter = _Cycle(self.video_loader) if self.video_loader is not None else None
        if self._img_iter is None:
            raise RuntimeError("no image training loader available (datasets.train empty?)")

        # --- optim / schedule / amp / ema ---
        self.total_iters = int(tcfg.get("iters", 1000))
        self.grad_accum = max(int(tcfg.get("grad_accum", 1)), 1)
        self.clip_period = max(int(tcfg.get("clip_period", 4)), 1)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=float(tcfg.get("lr", 3e-4)),
            weight_decay=float(tcfg.get("weight_decay", 0.01)),
            betas=(0.9, 0.999),
        )
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer,
            cosine_warmup_lambda(int(tcfg.get("warmup_iters", 0)), self.total_iters),
        )
        self.amp = bool(tcfg.get("amp", False)) and self.device.type == "cuda"
        self.scaler = torch.amp.GradScaler(self.device.type, enabled=self.amp)
        self.ema = ModelEMA(self.model, decay=float(tcfg.get("ema", 0.999)))

        self.step = 0
        self.dirs = resolve_run_dirs(cfg)
        self.logger = logger if logger is not None else TBLogger(self.dirs["tb"])
        self.log_every = int(tcfg.get("log_every", 50))
        self.img_every = int(tcfg.get("img_every", 500))
        self.ckpt_every = int(tcfg.get("ckpt_every", 2000))
        self.eval_every = int(tcfg.get("eval_every", 5000))

        if tcfg.get("resume"):
            self._maybe_resume(tcfg.get("resume"))

    # ------------------------------------------------------------------ build
    def _loader(self, names: list[str], split: str, batch_size: int) -> Optional[DataLoader]:
        if not names:
            return None
        datasets = self.deps.build_datasets(self.cfg, names, split)
        datasets = [d for d in datasets if d is not None]
        if not datasets:
            return None
        ds = datasets[0] if len(datasets) == 1 else ConcatDataset(datasets)
        nw = int(self.cfg.train.get("num_workers", 0))
        return DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=nw,
            collate_fn=pharos_collate,
            pin_memory=self.device.type == "cuda",
            drop_last=True,
            persistent_workers=nw > 0,
        )

    def _build_image_loader(self) -> Optional[DataLoader]:
        names = list(self.cfg.get("datasets", {}).get("train", []))
        return self._loader(names, "train", int(self.cfg.train.get("batch", 8)))

    def _build_video_loader(self) -> Optional[DataLoader]:
        names = list(self.cfg.get("datasets", {}).get("train_video", []))
        return self._loader(names, "train", int(self.cfg.train.get("clip_batch", 3)))

    # ------------------------------------------------------------------- step
    def _pick_modality(self, step: int) -> str:
        if self._vid_iter is None:
            return "image"
        return "video" if step % self.clip_period == self.clip_period - 1 else "image"

    def _forward_clip(self, batch: dict) -> PharosOutput:
        frames = batch["hazy"]  # B,T,3,H,W
        state = None
        outs: list[PharosOutput] = []
        for t in range(frames.shape[1]):
            o = self.model(frames[:, t], state=state)
            state = o.state
            outs.append(o)
        return _stack_outputs(outs)

    def _forward_loss(self, modality: str):
        it = self._vid_iter if modality == "video" else self._img_iter
        assert it is not None
        batch = it.next()
        # Reject non-finite inputs BEFORE the forward pass: a NaN input poisons
        # BatchNorm running statistics during the forward itself, which the
        # post-hoc loss guard cannot undo (train mode uses batch stats, so the
        # damage only surfaces at eval — silently).
        for _ in range(4):
            hz, cl = batch.get("hazy"), batch.get("clean")
            ok = torch.isfinite(hz).all() and (cl is None or torch.isfinite(cl).all())
            if ok:
                break
            metas = batch.get("meta")
            src = {m.get("dataset") for m in metas if isinstance(m, dict)} if isinstance(metas, list) else "?"
            warnings.warn(f"non-finite INPUT batch (datasets={src}); refetching.")
            batch = it.next()
        batch = move_batch_to_device(
            batch, self.device, channels_last=self.channels_last and modality == "image"
        )
        with torch.autocast(device_type=self.device.type, enabled=self.amp):
            if modality == "video":
                out = self._forward_clip(batch)
            else:
                out = self.model(batch["hazy"], state=None)
            loss, scalars = self.loss_fn(out, batch, self.teachers)
        hz = batch["hazy"]
        n = hz.shape[0] * (hz.shape[1] if modality == "video" else 1)
        return loss, dict(scalars), n, out, batch

    def train_step(self) -> dict[str, float]:
        """One optimizer step (with grad accumulation). Returns averaged scalars."""
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        agg: dict[str, float] = {}
        imgs = 0
        last_out = last_batch = last_modality = None
        did_backward = False
        for _ in range(self.grad_accum):
            modality = self._pick_modality(self.step)
            loss, scalars, n, out, batch = self._forward_loss(modality)
            if not bool(torch.isfinite(loss.detach())):
                # A poisoned batch (bad image, teacher blow-up) must not corrupt
                # the weights: name the bad terms, drop the step, keep training.
                bad = [k for k, v in scalars.items() if not math.isfinite(float(v))]
                self._nan_skips = getattr(self, "_nan_skips", 0) + 1
                healed = _sanitize_bn_stats(self.model, self.ema)
                warnings.warn(
                    f"non-finite loss at step {self.step} (modality={modality}, "
                    f"bad terms={bad or ['total']}); skipping batch "
                    f"({self._nan_skips} skipped so far; {healed} BN buffers healed)."
                )
                if self._nan_skips > 200:
                    raise RuntimeError("more than 200 non-finite batches; aborting run")
                continue
            self.scaler.scale(loss / self.grad_accum).backward()
            did_backward = True
            for k, v in scalars.items():
                agg[k] = agg.get(k, 0.0) + float(v) / self.grad_accum
            imgs += n
            last_out, last_batch, last_modality = out, batch, modality
        if not did_backward:
            self.optimizer.zero_grad(set_to_none=True)
            agg["_imgs"] = imgs
            return agg
        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.scheduler.step()
        self.ema.update(self.model)
        agg["_imgs"] = imgs
        self._last_out, self._last_batch, self._last_modality = last_out, last_batch, last_modality
        return agg

    # ------------------------------------------------------------------- loop
    def train(self) -> None:
        loss_meter = AverageMeter()
        timer = Timer()
        imgs_since_log = 0
        while self.step < self.total_iters:
            scalars = self.train_step()
            imgs_since_log += int(scalars.pop("_imgs", 0))
            loss_meter.update(scalars.get("loss", scalars.get("total", 0.0)))
            self.step += 1

            if self.log_every and self.step % self.log_every == 0:
                dt = timer.lap()
                thru = imgs_since_log / dt if dt > 0 else 0.0
                imgs_since_log = 0
                lr = self.optimizer.param_groups[0]["lr"]
                self.logger.scalars("train", scalars, self.step)
                self.logger.scalar("train/lr", lr, self.step)
                self.logger.scalar("train/imgs_per_s", thru, self.step)
                print(
                    f"[{self.step}/{self.total_iters}] loss={loss_meter.avg:.4f} "
                    f"lr={lr:.2e} {thru:.1f} img/s"
                )
                loss_meter.reset()
            if self.img_every and self.step % self.img_every == 0:
                self._log_images()
            if self.ckpt_every and self.step % self.ckpt_every == 0:
                self.save(self.dirs["ckpt"] / f"ckpt_{self.step:06d}.pt")
            if self.eval_every and self.step % self.eval_every == 0:
                self.run_eval()
        self.save(self.dirs["ckpt"] / "last.pt")
        self.logger.close()

    def _log_images(self) -> None:
        out, batch, modality = getattr(self, "_last_out", None), self._last_batch, self._last_modality
        if out is None or modality == "video":
            return
        self.logger.image_panel(
            "panels",
            {
                "input": batch["hazy"],
                "output": out.output,
                "confidence": out.confidence,
                "gt": batch.get("clean"),
            },
            self.step,
        )

    # ------------------------------------------------------------------- eval
    def run_eval(self) -> Optional[dict]:
        eval_fn = self.eval_fn or self._default_eval
        self.ema.store(self.model)
        self.ema.copy_to(self.model)
        self.model.eval()
        try:
            metrics = eval_fn()
            if metrics:
                self.logger.scalars("eval", _flatten_scalars(metrics), self.step)
            return metrics
        except Exception as e:
            warnings.warn(f"eval hook failed at step {self.step}: {e}")
            return None
        finally:
            self.ema.restore(self.model)
            self.model.train()

    def _default_eval(self) -> Optional[dict]:
        from .eval import evaluate

        return evaluate(
            self.model,
            self.cfg,
            teachers=self.teachers,
            device=self.device,
            out_dir=self.dirs["eval"],
            step=self.step,
        )

    # ------------------------------------------------------------ checkpoints
    def state_dict(self) -> dict[str, Any]:
        return {
            "step": self.step,
            "model": self.model.state_dict(),
            "ema": self.ema.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scaler": self.scaler.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "config": dict(self.cfg),
            "meta": {"conformal_scale": None},
        }

    def save(self, path: str | Path) -> None:
        save_checkpoint(path, self.state_dict())
        save_checkpoint(Path(path).parent / "last.pt", self.state_dict())

    def load_state(self, path: str | Path) -> None:
        ck = load_checkpoint(path, map_location=self.device)
        self.model.load_state_dict(ck["model"])
        self.ema.load_state_dict(ck["ema"])
        self.optimizer.load_state_dict(ck["optimizer"])
        self.scaler.load_state_dict(ck["scaler"])
        self.scheduler.load_state_dict(ck["scheduler"])
        self.step = int(ck["step"])

    def _maybe_resume(self, resume: Any) -> None:
        path = Path(resume) if isinstance(resume, (str, Path)) and str(resume) not in ("1", "true") \
            else self.dirs["ckpt"] / "last.pt"
        if Path(path).exists():
            self.load_state(path)
            print(f"resumed from {path} at step {self.step}")
        else:
            warnings.warn(f"resume requested but no checkpoint at {path}")


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _auto_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _sanitize_bn_stats(model: torch.nn.Module, ema=None) -> int:
    """Reset non-finite BatchNorm running statistics (model + EMA shadow).

    A single NaN forward pollutes BN running stats permanently (train mode uses
    batch stats, so training looks healthy while eval breaks); the EMA lerp then
    keeps NaN forever. Called whenever a non-finite loss is detected.
    """
    healed = 0
    with torch.no_grad():
        for m in model.modules():
            if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
                if m.running_mean is not None and not torch.isfinite(m.running_mean).all():
                    m.running_mean.zero_()
                    healed += 1
                if m.running_var is not None and not torch.isfinite(m.running_var).all():
                    m.running_var.fill_(1.0)
                    healed += 1
        shadow = getattr(ema, "shadow", None)
        if isinstance(shadow, dict):
            for k, v in shadow.items():
                if "running_mean" in k and torch.is_tensor(v) and not torch.isfinite(v).all():
                    v.zero_()
                    healed += 1
                elif "running_var" in k and torch.is_tensor(v) and not torch.isfinite(v).all():
                    v.fill_(1.0)
                    healed += 1
    return healed


def _maybe_build_teachers(deps: Deps, cfg: Config):
    try:
        return deps.build_teachers(cfg)
    except Exception as e:  # teachers are optional at train time (contract §8)
        warnings.warn(f"teachers unavailable ({e}); training without prior distillation.")
        return None


def _flatten_scalars(d: dict, prefix: str = "") -> dict[str, float]:
    out: dict[str, float] = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(_flatten_scalars(v, f"{key}/"))
        elif isinstance(v, (int, float)):
            out[key] = float(v)
    return out


def build_trainer_from_config(path: str, overrides: Optional[list[str]] = None) -> Trainer:
    cfg = load_config(path, parse_overrides(overrides))
    cfg.setdefault("exp_name", Path(path).stem)
    return Trainer(cfg)


def main(argv: Optional[list[str]] = None) -> None:
    ap = argparse.ArgumentParser(description="Pharos training")
    ap.add_argument("--config", required=True)
    ap.add_argument("--override", nargs="*", default=[], help="dotted key=value overrides")
    ap.add_argument("--resume", default=None, help="checkpoint path or 'last'")
    args = ap.parse_args(argv)
    cfg = load_config(args.config, parse_overrides(args.override))
    cfg.setdefault("exp_name", Path(args.config).stem)
    if args.resume:
        cfg["train"]["resume"] = args.resume
    trainer = Trainer(cfg)
    trainer.train()


if __name__ == "__main__":  # Windows-safe dataloader entry guard
    main()
