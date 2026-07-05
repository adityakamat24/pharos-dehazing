"""VividLoss: perceptual/adversarial polish for vivid-mode (DESIGN.md N4).

Vivid mode is a PHOTOGRAPHY-mode fine-tune of PharosNet that makes restorations
look dramatically crisper and more natural — specifically killing muddy/blotchy
amplified-noise regions where haze/smoke was dense. The deployed network is the
*same* PharosNet at the *same* inference cost; only the weights differ.

This loss is **self-contained** so the training engine needs no changes: it owns
the discriminator AND its own Adam optimizer. Each ``__call__`` on a paired batch:

  1. Updates the discriminator one step (hinge, real=clean, fake=out.output.detach();
     run in fp32 under ``autocast(enabled=False)`` for stability), then
  2. Returns the generator-side total::

        total =  w_l1    * Charbonnier(J, GT)          (anchor, keep 1.0)
               + w_lpips * LPIPS(J, GT)                 (lazy import; 0 if missing)
               + w_gan   * generator hinge on D(J)      (ramped 0->target over warmup)
               + w_conf  * PharosLoss._conf NLL         (keeps confidence calibrated
                                                         to the NEW error profile)

Every component degrades to exactly 0 when its inputs are missing (no clean GT,
no confidence/logvar, LPIPS unavailable, warmup w_gan=0). Clip batches are reduced
to the last frame (``[:, -1]``). Device placement is lazy (from ``out.output`` on
first call). ``state_dict``/``load_state_dict`` include the discriminator and its
optimizer so a run resumes the adversarial state.
"""
from __future__ import annotations

from typing import Any, Optional

import torch
import torch.nn.functional as F

from ..contracts import PharosOutput
from ..models.discriminator import build_discriminator


class VividLoss:
    """Implements contracts.LossFn; self-manages a discriminator + its optimizer."""

    def __init__(self, cfg: Any, disc: Optional[torch.nn.Module] = None) -> None:
        loss_cfg = _get(cfg, "loss", {}) or {}
        vivid = _get(loss_cfg, "vivid", {}) or {}
        self.w = {
            "l1": float(_get(vivid, "l1", 1.0)),
            "lpips": float(_get(vivid, "lpips", 0.3)),
            "gan": float(_get(vivid, "gan", 0.02)),
            "conf": float(_get(vivid, "conf", 0.05)),
        }
        self.gan_warmup = int(_get(vivid, "gan_warmup", 2000))
        self.disc_lr = float(_get(vivid, "disc_lr", 1e-4))
        self.relativistic = bool(_get(vivid, "relativistic", False))
        self.lpips_net = str(_get(vivid, "lpips_net", "alex"))
        self.charb_eps = 1e-3
        # Conditional (pix2pix-style) D: judges (input, output) PAIRS. An
        # unconditional D cannot distinguish haze from natural atmosphere, so
        # near-passthrough of a hazy input scores as 'real' and the generator
        # learns to under-dehaze (observed on NH-HAZE with the v1 vivid run).
        self.conditional = bool(_get(vivid, "conditional", True))

        # Discriminator + its OWN optimizer (engine never sees these). Adam betas
        # (0.5, 0.999) are the DCGAN/pix2pix default for adversarial stability.
        disc_cfg = dict(_get(vivid, "disc", {}) or {})
        if self.conditional:
            disc_cfg.setdefault("in_ch", 6)
        self.disc = disc if disc is not None else build_discriminator(disc_cfg)
        self.d_opt = torch.optim.Adam(self.disc.parameters(), lr=self.disc_lr, betas=(0.5, 0.999))

        # Reuse the exact heteroscedastic conf NLL from PharosLoss (compose, don't
        # copy) so the confidence head recalibrates to the new error profile.
        from .losses import PharosLoss  # lazy: keeps import light for stub tests

        self._pharos = PharosLoss(cfg)

        self._lpips: Optional[torch.nn.Module] = None  # lazy-built on first device move
        self._lpips_failed = False
        self._device: Optional[torch.device] = None
        self._step = 0  # internal GAN-warmup counter

    # ------------------------------------------------------------------ call
    def __call__(
        self, out: PharosOutput, batch: dict, teachers: Any
    ) -> tuple[torch.Tensor, dict[str, float]]:
        device = out.output.device
        self._to(device)

        out_last, clean_last, logvar_last = self._reduce_last(out, batch.get("clean"))
        have_pair = clean_last is not None
        hazy = batch.get("hazy")
        cond = None
        if self.conditional and torch.is_tensor(hazy):
            cond = (hazy[:, -1] if hazy.dim() == 5 else hazy).to(device)
            if cond.shape[-2:] != out_last.shape[-2:]:
                cond = F.interpolate(cond, size=out_last.shape[-2:], mode="bilinear",
                                     align_corners=False)

        # (1) discriminator step — self-managed, fp32, autocast OFF for stability.
        d_val = self._update_disc(out_last, clean_last, cond, device) if have_pair else 0.0

        # (2) generator-side total.
        w_gan = self.w["gan"] * self._warmup_factor()
        l1 = self._l1(out_last, clean_last, device)
        lpips_v = self._lpips_term(out_last, clean_last, device)
        gan = self._gen_gan(out_last, clean_last, cond, device) if (have_pair and w_gan > 0.0) else _z(device)
        conf = self._conf_term(out, out_last, clean_last, logvar_last, device)

        total = self.w["l1"] * l1 + self.w["lpips"] * lpips_v + w_gan * gan + self.w["conf"] * conf

        self._step += 1
        log = {
            "l1": float(l1.detach()),
            "lpips": float(lpips_v.detach()),
            "gan": float(gan.detach()),
            "conf": float(conf.detach()),
            "d": float(d_val),
            "gan_w": float(w_gan),
            "total": float(total.detach()),
        }
        return total, log

    # ------------------------------------------------------------- warmup
    def _warmup_factor(self) -> float:
        if self.gan_warmup <= 0:
            return 1.0
        return min(1.0, self._step / float(self.gan_warmup))

    # ------------------------------------------------------------ D update
    def _update_disc(self, fake: torch.Tensor, real: torch.Tensor,
                     cond: Optional[torch.Tensor], device) -> float:
        """One hinge step on the discriminator (own optimizer, fp32, no scaler)."""
        dev_type = device.type if hasattr(device, "type") else "cpu"
        with torch.autocast(device_type=dev_type, enabled=False):
            real_f = real.detach().float()
            fake_f = fake.detach().float()
            if cond is not None:
                c = cond.detach().float()
                real_f = torch.cat([c, real_f], dim=1)
                fake_f = torch.cat([c, fake_f], dim=1)
            d_real = self.disc(real_f)
            d_fake = self.disc(fake_f)
            if self.relativistic:  # relativistic-average hinge
                d_loss = (
                    F.relu(1.0 - (d_real - d_fake.mean())).mean()
                    + F.relu(1.0 + (d_fake - d_real.mean())).mean()
                )
            else:  # standard hinge
                d_loss = F.relu(1.0 - d_real).mean() + F.relu(1.0 + d_fake).mean()
            self.d_opt.zero_grad(set_to_none=True)
            d_loss.backward()
            self.d_opt.step()
        return float(d_loss.detach())

    # -------------------------------------------------------- generator GAN
    def _gen_gan(self, fake: torch.Tensor, clean_last: Optional[torch.Tensor],
                 cond: Optional[torch.Tensor], device) -> torch.Tensor:
        """Generator hinge on D(J). D params are frozen so the engine's backward
        (which runs over the returned total) does not accumulate grads on them."""
        for p in self.disc.parameters():
            p.requires_grad_(False)
        try:
            if cond is not None:
                c = cond.detach()
                fake = torch.cat([c.to(fake.dtype), fake], dim=1)
            d_fake = self.disc(fake)
            if self.relativistic and clean_last is not None:
                real_in = clean_last.detach()
                if cond is not None:
                    real_in = torch.cat([cond.detach().to(real_in.dtype), real_in], dim=1)
                d_real = self.disc(real_in).detach()
                g = (
                    F.relu(1.0 - (d_fake - d_real.mean())).mean()
                    + F.relu(1.0 + (d_real - d_fake.mean())).mean()
                )
            else:
                g = -d_fake.mean()
        finally:
            for p in self.disc.parameters():
                p.requires_grad_(True)
        return g

    # ------------------------------------------------------------- terms
    def _l1(self, out_last: torch.Tensor, clean_last: Optional[torch.Tensor], device) -> torch.Tensor:
        if clean_last is None:
            return _z(device)
        return _charbonnier(out_last, clean_last, self.charb_eps)

    def _lpips_term(
        self, out_last: torch.Tensor, clean_last: Optional[torch.Tensor], device
    ) -> torch.Tensor:
        if clean_last is None or self._lpips is None:
            return _z(device)
        dev_type = device.type if hasattr(device, "type") else "cpu"
        # LPIPS wants [-1, 1] 3-channel; run fp32 (autocast off) for stability.
        with torch.autocast(device_type=dev_type, enabled=False):
            x = out_last.float().clamp(0.0, 1.0) * 2.0 - 1.0
            y = clean_last.float().clamp(0.0, 1.0) * 2.0 - 1.0
            d = self._lpips(x, y)
        return d.mean()

    def _conf_term(
        self,
        out: PharosOutput,
        out_last: torch.Tensor,
        clean_last: Optional[torch.Tensor],
        logvar_last: Optional[torch.Tensor],
        device,
    ) -> torch.Tensor:
        if clean_last is None or out.confidence is None:
            return _z(device)
        conf = out.confidence
        conf_last = conf[:, -1] if (torch.is_tensor(conf) and conf.dim() == 5) else conf
        single = PharosOutput(
            output=out_last,
            confidence=conf_last,
            grid=out.grid,
            state=None,
            deg=out.deg,
            aux={"logvar": logvar_last} if logvar_last is not None else {},
        )
        return self._pharos._conf(single, clean_last, device)

    # ------------------------------------------------------------- devices
    def _to(self, device) -> None:
        if self._device is not None and torch.device(self._device) == torch.device(device):
            return
        self.disc.to(device)
        # Build LPIPS only when it can matter (weight > 0): skips the (possibly
        # network-touching) load entirely when perceptual loss is disabled.
        if self.w["lpips"] > 0.0 and self._lpips is None and not self._lpips_failed:
            self._lpips = _build_lpips(self.lpips_net, device)
            self._lpips_failed = self._lpips is None
        elif self._lpips is not None:
            self._lpips.to(device)
        self._device = torch.device(device)

    # ---------------------------------------------------------- checkpoint
    def state_dict(self) -> dict[str, Any]:
        return {
            "disc": self.disc.state_dict(),
            "d_opt": self.d_opt.state_dict(),
            "step": self._step,
        }

    def load_state_dict(self, sd: dict[str, Any]) -> None:
        if not isinstance(sd, dict):
            return
        if "disc" in sd:
            self.disc.load_state_dict(sd["disc"])
        if "d_opt" in sd:
            self.d_opt.load_state_dict(sd["d_opt"])
        self._step = int(sd.get("step", self._step))

    # --------------------------------------------------------------- utils
    @staticmethod
    def _reduce_last(
        out: PharosOutput, clean: Optional[torch.Tensor]
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Reduce a (possibly clip) output/GT to the single current frame.

        Clip tensors are ``B,T,...`` (dim 5 for images, aux logvar stacked to 5);
        image tensors keep their native dim. Returns ``(out_last, clean_last,
        logvar_last)`` with ``clean_last`` None on shape mismatch.
        """
        output = out.output
        is_clip = torch.is_tensor(output) and output.dim() == 5
        out_last = output[:, -1] if is_clip else output

        clean_last: Optional[torch.Tensor] = None
        if clean is not None and torch.is_tensor(clean):
            clean_last = clean[:, -1] if clean.dim() == 5 else clean
            if clean_last.dim() != out_last.dim() or clean_last.shape != out_last.shape:
                clean_last = None

        aux = out.aux if isinstance(out.aux, dict) else {}
        logvar = aux.get("logvar")
        logvar_last: Optional[torch.Tensor] = None
        if torch.is_tensor(logvar):
            logvar_last = logvar[:, -1] if (is_clip and logvar.dim() == 5) else logvar
        return out_last, clean_last, logvar_last


# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------
def _z(device) -> torch.Tensor:
    return torch.zeros((), device=device)


def _charbonnier(x: torch.Tensor, y: torch.Tensor, eps: float) -> torch.Tensor:
    return torch.sqrt((x - y) ** 2 + eps * eps).mean()


def _build_lpips(net: str, device) -> Optional[torch.nn.Module]:
    """Lazily build a frozen LPIPS metric; return None if the package/weights are
    unavailable (no network at train time) so the term degrades to 0."""
    try:
        import lpips as lpips_pkg  # type: ignore
    except Exception:
        return None
    try:
        model = lpips_pkg.LPIPS(net=net, verbose=False)
    except Exception:
        return None
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model.to(device)


def _get(obj: Any, key: str, default: Any = None) -> Any:
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)
