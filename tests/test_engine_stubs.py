"""Shared CPU stubs for the WS-D engine tests (no GPU / network / datasets).

Also imported for its side effect of putting ``src/`` on ``sys.path`` so the
``pharos`` package resolves without an editable install. This file matches the
``test_engine_*`` glob but defines no test functions itself.
"""
from __future__ import annotations

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from torch.utils.data import Dataset  # noqa: E402

from pharos.contracts import PharosOutput  # noqa: E402

GRID_D, GRID_G = 2, 4  # tiny bilateral grid for the stub


class StubModel(nn.Module):
    """Minimal PharosModel: learnable, returns a contract-valid PharosOutput."""

    def __init__(self) -> None:
        super().__init__()
        self.detail = nn.Conv2d(3, 3, 3, padding=1)
        self.conf = nn.Conv2d(3, 1, 1)
        self.scale = nn.Parameter(torch.tensor(0.05))
        n_grid = 12 * GRID_D * GRID_G * GRID_G
        self.grid_lin = nn.Linear(3, n_grid)
        self.beta = nn.Linear(3, 1)
        self.air = nn.Linear(3, 3)
        self.sigma = nn.Linear(3, 1)
        self.dom = nn.Linear(3, 3)

    def forward(self, frame, state=None, cond=None) -> PharosOutput:
        b = frame.shape[0]
        pooled = frame.mean(dim=(2, 3))  # B,3
        out = (frame + self.scale * torch.tanh(self.detail(frame))).clamp(0, 1)
        conf = torch.sigmoid(self.conf(frame))
        grid = self.grid_lin(pooled).view(b, 12, GRID_D, GRID_G, GRID_G)
        deg = {
            "beta": self.beta(pooled),
            "airlight": self.air(pooled),
            "sigma": self.sigma(pooled),
            "domain_logits": self.dom(pooled),
        }
        return PharosOutput(output=out, confidence=conf, grid=grid, state=pooled, deg=deg)

    def reparameterize(self) -> None:  # no-op for the stub
        pass


class StubLoss:
    """Handles both image (B,3,H,W) and clip (B,T,3,H,W) outputs."""

    def __call__(self, out: PharosOutput, batch: dict, teachers):
        pred = out.output
        target = batch.get("clean")
        rec = F.l1_loss(pred, target) if target is not None else pred.pow(2).mean()
        loss = rec
        scalars = {"rec": float(rec.detach())}
        if batch.get("clip") and out.grid is not None and out.grid.dim() == 6:
            temp = (out.grid[:, 1:] - out.grid[:, :-1]).abs().mean()
            loss = loss + 0.1 * temp
            scalars["temp"] = float(temp.detach())
        scalars["loss"] = float(loss.detach())
        return loss, scalars


class StubImageDataset(Dataset):
    def __init__(self, n: int = 8, size: int = 32) -> None:
        self.n, self.size = n, size

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, i: int) -> dict:
        g = torch.Generator().manual_seed(i)
        return {
            "hazy": torch.rand(3, self.size, self.size, generator=g),
            "clean": torch.rand(3, self.size, self.size, generator=g),
            "domain": i % 3,
            "clip": False,
            "meta": {"name": "stub_img", "idx": i},
        }


class StubClipDataset(Dataset):
    def __init__(self, n: int = 6, t: int = 3, size: int = 32) -> None:
        self.n, self.t, self.size = n, t, size

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, i: int) -> dict:
        g = torch.Generator().manual_seed(100 + i)
        return {
            "hazy": torch.rand(self.t, 3, self.size, self.size, generator=g),
            "clean": torch.rand(self.t, 3, self.size, self.size, generator=g),
            "domain": 1,
            "clip": True,
            "meta": {"name": "stub_clip", "idx": i},
        }


class StubTeachers:
    """Contract TeacherBundle; ``with_flow`` enables a trivial flow teacher."""

    def __init__(self, with_flow: bool = False) -> None:
        self.depth = None
        self.detector = None
        self.flow = self._flow if with_flow else None

    @staticmethod
    def _flow(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return torch.zeros(a.shape[0], 2, a.shape[-2], a.shape[-1])
