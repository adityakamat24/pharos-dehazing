# Pharos — Real-Time, Confidence-Aware Dehazing & Desmoking

One tiny network (~2M params) that removes **atmospheric haze**, **dense fire smoke**, and
**satellite thin haze** in real time — for images and causal (streaming, no-future-frames) video —
and tells you, per pixel, **how much to trust its output**.

Built for three use-cases:
- **Seeing through smoke** — firefighting, search & rescue (bodycams, drones, CCTV)
- **Haze** — photography, driving assistance, public webcams
- **Satellite / remote sensing** — thin haze and cirrus over imagery

## Why it's different

All the heavy AI lives at **training time only** and is distilled away:

| Training-time teacher | What it teaches | Inference cost |
|---|---|---|
| Depth Anything V2 (on *clean* images) | scene structure → transmission | **zero** |
| Frozen object detector | restore what perception needs, not just pixels | **zero** |
| Optical flow (on *clean* frames) | temporal stability | **zero** |

The deployed student predicts a low-res **bilateral affine grid** (per-pixel affine color
transforms — the physics of scattering is affine in the input) sliced at full resolution, so compute
is decoupled from output resolution; a bounded detail branch restores texture. Because the primary
operator is an affine transform of the *real* input, the model **structurally cannot hallucinate
objects** — critical when a firefighter acts on the output. A calibrated **confidence overlay**
marks unreliable regions, and a severity gate passes already-clear frames through untouched.

See [DESIGN.md](DESIGN.md) for the full method spec, novelty analysis, and evaluation protocol.

## Layout

```
src/pharos/
  contracts.py      frozen interfaces        models/    PharosNet (grid, heads, temporal)
  config.py         YAML config tree         data/      datasets, synthesis, downloaders
  teachers/         training-only priors     losses/    loss stack + conformal calibration
  engine/           train / eval / metrics   rt/        streaming demo, FPS bench, export
configs/            base.yaml + experiments  scripts/   downloads, train wrappers, pod/
```

## Quickstart (Windows, RTX 3060 6GB)

```powershell
# env: .venv with CUDA torch is already set up in this repo
.venv\Scripts\python.exe -m pip install -e .[train,dev]
.venv\Scripts\python.exe scripts\download_datasets.py --only smokebench,reside6k,satehaze1k,rice
.venv\Scripts\python.exe -m pharos.engine.train --config configs\overfit50.yaml   # sanity
.venv\Scripts\python.exe -m pharos.rt.demo --source webcam --ckpt runs\<exp>\ckpt\latest.pth
.venv\Scripts\python.exe -m pharos.rt.bench --ckpt runs\<exp>\ckpt\latest.pth     # honest FPS
```

Full training runs on a rented GPU pod — see `scripts/pod/README.md`.

## Status

Research + design complete; implementation in progress. Benchmarks (fixed-weights TriHaze protocol:
SOTS, NH-HAZE, Dense-Haze, SmokeBench, SateHaze1k, RICE, REVIDE + FPS on named hardware) will be
published here once training completes.
