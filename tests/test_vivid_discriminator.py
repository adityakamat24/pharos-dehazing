"""CPU tests for the vivid-mode discriminators (WS-vivid).

Forward shapes, spectral-norm presence, param budget (~1-2M for the patch D),
and the build_discriminator config factory. No GPU, no network.
"""
from __future__ import annotations

import torch

from pharos.models.discriminator import (
    PatchDiscriminator,
    UNetDiscriminator,
    build_discriminator,
    count_params,
)

B, H, W = 2, 64, 64


def test_patch_discriminator_forward_shape():
    d = PatchDiscriminator()
    x = torch.rand(B, 3, H, W)
    y = d(x)
    # patch-logits map: batch preserved, single channel, spatially downsampled.
    assert y.dim() == 4
    assert y.shape[0] == B and y.shape[1] == 1
    assert y.shape[-1] < W and y.shape[-2] < H
    assert torch.isfinite(y).all()


def test_patch_discriminator_param_budget():
    d = PatchDiscriminator()  # defaults: base_ch=64, n_layers=3, max_ch=256
    p = count_params(d)
    assert 1.0e6 < p < 2.0e6, f"patch D has {p} params, expected ~1-2M"


def test_patch_discriminator_has_spectral_norm():
    d = PatchDiscriminator()
    # parametrizations.spectral_norm stores the weight under a `parametrizations`
    # container: at least one conv must carry it.
    names = [n for n, _ in d.named_modules()]
    assert any("parametrizations" in n for n in names)


def test_patch_discriminator_configurable_size():
    small = PatchDiscriminator(base_ch=48, n_layers=3, max_ch=256)
    assert count_params(small) < count_params(PatchDiscriminator())


def test_unet_discriminator_forward_shape():
    d = UNetDiscriminator(base_ch=32)
    x = torch.rand(B, 3, H, W)
    y = d(x)
    # per-pixel logits: same spatial resolution as the input.
    assert y.shape == (B, 1, H, W)
    assert torch.isfinite(y).all()


def test_build_discriminator_factory():
    assert isinstance(build_discriminator({"type": "patch"}), PatchDiscriminator)
    assert isinstance(build_discriminator({"type": "unet"}), UNetDiscriminator)
    assert isinstance(build_discriminator(None), PatchDiscriminator)  # default
    assert isinstance(build_discriminator({}), PatchDiscriminator)


def test_build_discriminator_honours_dims():
    d = build_discriminator({"type": "patch", "base_ch": 32, "n_layers": 3, "max_ch": 128})
    assert count_params(d) < count_params(PatchDiscriminator())


def test_discriminator_backward_flows():
    d = PatchDiscriminator(base_ch=32)
    x = torch.rand(B, 3, H, W, requires_grad=True)
    d(x).mean().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
