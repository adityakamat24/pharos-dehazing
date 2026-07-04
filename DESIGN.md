# Pharos — Real-Time, Confidence-Aware, Unified Dehazing & Desmoking

**Status:** design finalized 2026-07-03 from a 12-domain literature sweep (2023–2026) + adversarial novelty check.
This document is the binding spec for all implementation workstreams. Interface contracts live in
`src/pharos/contracts.py` and `configs/base.yaml` — do not change signatures without updating this doc.

---

## 1. Problem & novelty claims

Goal: one small network that removes atmospheric haze, dense smoke (fire scenes), and satellite thin
haze **in real time** (≥30 FPS @ 1080p on an RTX 3060, PyTorch FP16; more with TensorRT), for video
(causal/streaming, no future frames) and single images, with a **calibrated per-pixel confidence
output** so a human acting on the output (firefighter, SAR operator) knows which regions to trust.

Novelty (each verified open against 2024–2026 literature by an adversarial search agent):

- **N1 — Training-time-only foundation-prior distillation.** All existing depth-prior dehazers
  (PromptHaze AAAI'25; UDPNet arXiv:2601.06909; arXiv:2508.00698) run the depth model at inference.
  arXiv:2508.00698 names train-only distillation as unaddressed future work. Pharos runs
  Depth Anything V2 (and a frozen detector, and optical flow) **only during training**; the deployed
  student is prior-free.
- **N2 — Unified haze + smoke + satellite, real-time, one set of weights.** Only prior haze+smoke
  model is CIANet (2022): 0.2 FPS on CPU, no code. No all-in-one model covers this domain triple;
  none report FPS. Includes the first fixed-weights cross-domain eval protocol (§7).
- **N3 — Causal streaming desmoking with recurrent state + honest speed reporting.** SmokeBench
  (ACM MM'25, 9,975 real fire-smoke pairs) has no model, no temporal method, no FPS. Turtle
  (NeurIPS'24) shows causal ≥ bidirectional quality. Pharos: recurrent bilateral-grid state,
  FPS-at-resolution-on-named-GPU, ONNX/TensorRT/INT8 numbers.
- **N4 — Hallucination-bounded output + user-facing calibrated confidence + severity gate.**
  No dehazing/desmoking paper outputs user-facing confidence ("Looks Too Good To Be True",
  NeurIPS'24, proves the perception/uncertainty tradeoff; UDN AAAI'22 uses uncertainty only
  internally). Pharos's primary output is a per-pixel affine transform of the *input* — it can
  amplify existing signal but structurally cannot synthesize new objects; the small detail branch is
  magnitude-bounded; confidence is trained heteroscedastically and conformally calibrated; a
  continuous severity gate passes clear frames through (fixes the "From Fog to Failure" regression).
- **N5 (supporting) — Multi-teacher pseudo-labeling on real footage** (CORUN/QualiTeacher use
  single EMA teachers; MTKD's multi-teacher recipe exists only for SR).

## 2. Physics: one degradation family, continuous conditioning

Koschmieder: `I(x) = J(x)·t(x) + A·(1 − t(x))`. Inverting is per-pixel **affine** in I:
`J = I/t − A(1−t)/t`. Generalizations we must handle:

| Domain | t(x) structure | Airlight | Notes |
|---|---|---|---|
| Ground haze | `t = exp(−β·d(x))`, depth-driven | ~global, gray | ASM holds approximately |
| Dense smoke | non-homogeneous, turbulent, NOT depth-driven | **colored** (soot/material), spatially varying | multiply-scattered; ASM breaks (SmokeBench RTE model, CIANet color term) |
| Satellite | near-uniform per patch (no depth relation, no sky) | per-band wavelength-dependent | thin cirrus/haze veiling |

A spatially-varying per-pixel affine color transform `J(x) = M(x)·I(x) + b(x)` (M: 3×3, b: 3×1)
subsumes all three regimes. Pharos predicts this field as a **bilateral grid** (prediction cost
decoupled from output resolution — the pattern behind every published real-time 4K result:
Zheng CVPR'21 125 FPS@4K; 3D-LUT <2ms@4K; LiBrA-Net 25 FPS@4K video).
Conditioning is **continuous** (estimated density β̂, airlight color Â, non-homogeneity σ̂ + a small
learned domain embedding), not a discrete degradation classifier (they break on mixtures).

## 3. Student architecture (`PharosNet`, deployed)

Input: frame `I ∈ R^{B×3×H×W}` (+ optional recurrent `state`). All heads run on a low-res copy
(`base.yaml: model.lowres = 256`); only guidance + slicing + detail branch touch full res.

1. **Encoder** (low-res): 4 stages of `RepNAFBlock` = reparameterizable 3×3 convs (multi-branch at
   train → single 3×3 at inference, DEA-Net/RepVGG style) + SimpleGate + simplified channel
   attention (NAFNet). Channels ≈ [24, 48, 96, 96]. Wavelet (Haar) downsampling for stage 1.
2. **Degradation head**: pooled stage-3 features → MLP → `deg = {beta, airlight(3), sigma, domain_logits(3)}`.
   FiLM-modulates stages 3–4. Supervised on synthetic data; also the severity gate input.
3. **Bilateral grid head**: predicts `G ∈ R^{B×12×D×Gh×Gw}` (default D=8 guidance bins, Gh=Gw=16;
   12 = 3×4 affine). Slicing: full-res guidance map `g(x) ∈ [0,1]` from a 3-layer 1×1/3×3 conv on I;
   trilinear slice → per-pixel `M(x), b(x)`; coarse output `J0 = M·I + b`.
4. **Detail branch** (full res, tiny): 4 reparam 3×3 convs, 12 channels, input `concat(I, J0)` →
   residual `r = s · tanh(f)`, `J = clamp(J0 + r)`. `s` is a learned per-channel scale initialized 0.05
   — bounded hallucination by construction.
5. **Confidence head**: 1-channel log-variance at low res, upsampled → `conf ∈ (0,1]` per pixel
   (mapped from predicted error). Surfaced in the demo as an overlay.
6. **Temporal state** (video mode): ConvGRU over `[G, low-res feats]` (state ≈ 12×8×16×16 + C×32×32
   — a few KB). Confidence-weighted EMA on G for extra stability. Scene-cut reset via low-res
   histogram distance. Strictly causal.
7. **Severity gate**: continuous blend `out = α(β̂)·J + (1−α(β̂))·I`, α = smoothstep between
   `gate.beta_lo/beta_hi`. No hard switching.

Param budget: 1.5–3M. Trains in 6 GB (AMP, 256² crops + full-image-downsample for grid context,
HDRNet-style two-stream cropping).

## 4. Teacher stack (training only — never at inference)

- **Depth teacher**: Depth Anything V2 **Small** (Apache-2.0; larger variants are CC-BY-NC — research
  only) run on the **clean** image of each pair (sidesteps depth-in-haze unreliability,
  cf. DepthAnything-AC). Uses: (a) depth-based haze synthesis; (b) depth-structure distillation loss
  (feature-affinity vs depth-affinity at low res, §5).
- **Detection teacher**: frozen pretrained detector (yolov8n or yolov7-tiny). Detection-consistency
  loss between detector features/logits on `J` vs on clean GT (FriendNet pattern; auxiliary loss, NOT
  a shared trunk — D2SL shows tight coupling hurts).
- **Flow teacher** (video clips): RAFT-small (or teacher of choice) computed on **clean** frames →
  temporal warp loss on outputs. Flow never runs at inference (Turtle/TS-Mamba/RainMamba all show
  flow-free causal alignment wins at runtime).
- **Restoration teacher ensemble** (phase 2): pretrained DehazeFormer/MB-TaylorFormer/RIDCP +
  no-reference IQA agreement → pseudo-labels on real unpaired footage (RTTS/URHI/D-Fire frames).

## 5. Losses (`pharos/losses/`)

`L = L_rec + λ_freq·L_freq + λ_conf·L_conf + λ_depth·L_depth + λ_det·L_det + λ_temp·L_temp + λ_phys·L_phys`

- `L_rec`: Charbonnier(J, GT).
- `L_freq`: L1 on FFT amplitude (haze corrupts low-freq amplitude; MITNet/WaveDH lineage).
- `L_conf`: heteroscedastic NLL — `|J−GT|₁/σ + log σ` (per pixel; σ from confidence head).
  Post-training conformal calibration script produces a quantile scale (stored in checkpoint meta).
- `L_depth`: normalized feature-affinity matrix (student low-res feats) vs depth-affinity matrix
  (teacher depth on clean img), sampled pixel pairs; plus aux transmission head supervised by
  `exp(−β·d_teacher)` on synthetic data.
- `L_det`: L1 between frozen-detector multi-scale features on J vs on GT (subset of batches,
  `det_every_n`).
- `L_temp`: (clips) warp loss with clean-frame teacher flow + grid smoothness `‖G_t − G_{t−1}‖₁`
  weighted by (1 − scene-cut).
- `L_phys`: supervised `beta/airlight/sigma/domain` on synthetic data (known at synthesis time).

Defaults in `configs/base.yaml`; all λ overridable.

## 6. Data (`pharos/data/`, downloads to `D:\dehazing_desmoking\data\`, gitignored)

Local working set (<30 GB) — every loader yields the contract batch dict (§8):

| Set | Domain | Role | Source |
|---|---|---|---|
| RESIDE-6K (+SOTS test) | ground haze | base synthetic train/eval | Kaggle mirror / DehazeFormer GDrive |
| NH-HAZE'20, Dense-Haze, O/I-HAZE | real haze | fine-tune + eval | data.vision.ee.ethz.ch (direct) |
| SmokeBench | real fire smoke | core smoke train/eval | github.com/ncfjd/SmokeBench |
| SMOKE5K, D-Fire | smoke masks/frames | aux masks, pseudo-label pool | GitHub/GDrive |
| SateHaze1k, RICE1/2 | satellite | train/eval | Kaggle / GitHub (RICE zip 515 MB) |
| REVIDE | real haze video | temporal train/eval | GitHub links / Kaggle mirror |
| RTTS + URHI | real unpaired | no-ref eval + pseudo-labels | RESIDE-β page |

Pod-only (later): RESIDE-OTS full, HazeWorld (326k frames), RS-Haze.

**On-the-fly synthesis** (`pharos/data/synthesis.py`) from clean images:
ground haze = ASM with teacher depth (β ∈ [0.4, 3.0], gray-ish A);
smoke = Perlin/simplex multi-octave density + colored airlight (sampled soot/warm tints) +
optional fire-glow gain, non-depth-driven;
satellite = near-uniform t + per-channel wavelength bias.
**Robustness aug** (pipeline sweep finding): JPEG QF∈[30,95] and H.264-style blockiness on inputs
only, random exposure/WB jitter, ISO noise. Clear-passthrough samples (identity targets) for the gate.

## 7. Evaluation (`pharos/eval.py`) — the fixed-weights TriHaze protocol

One checkpoint, no per-set tuning: PSNR/SSIM/LPIPS on SOTS-mix, NH-HAZE, Dense-Haze, SmokeBench-test,
SateHaze1k(thin/mod/thick), RICE1; NIQE/FADE/BRISQUE on RTTS; detection mAP on RTTS (frozen YOLO,
dehazed vs raw); temporal warp-error on REVIDE test; clear-frame no-harm check (SOTS clean inputs →
output≈input, PSNR ≥ 45 dB); FPS benchmark: {720p, 1080p, 4K} × {PyTorch FP32/FP16, ONNX-RT, TensorRT
FP16/INT8 if available} on named GPU, batch 1, 100-frame median + P95 latency, JSON output.

## 8. Interface contracts (frozen — see `src/pharos/contracts.py`)

- `PharosNet.forward(frame, state=None, cond=None) -> PharosOutput` where `PharosOutput` is a
  TypedDict/dataclass: `output, confidence, grid, state, deg (dict), t_hat (optional)`.
- Batch dict from all datasets: `{"hazy": FloatTensor B3HW in [0,1], "clean": FloatTensor|None,
  "domain": LongTensor (0=haze,1=smoke,2=satellite), "meta": dict, "clip": bool}`;
  clips add T dim: B T 3 H W.
- Losses: `class PharosLoss: __call__(out: PharosOutput, batch, teachers: TeacherBundle) -> (loss, dict_of_scalars)`.
- Teachers: `TeacherBundle.depth(img)->B1hw`, `.det_feats(img)->list`, `.flow(a,b)->B2hw`; every
  teacher lazy-loads and is importable-optional (train must run with any subset disabled).
- Config: single YAML tree (`configs/base.yaml` + per-experiment overrides), loaded by
  `pharos.config.load_config(path, overrides)`. No argparse forests.

## 9. Repo layout & workstream ownership

```
src/pharos/
  contracts.py  config.py           # OWNED BY LEAD (frozen)
  models/       # WS-A: blocks.py (RepNAF, wavelet), grid.py (bilateral+slicing),
                #       heads.py, temporal.py, pharosnet.py
  data/         # WS-B: datasets.py, synthesis.py, degradations.py, transforms.py
  teachers/     # WS-C: depth.py, detector.py, flow.py, ensemble.py
  losses/       # WS-C: losses.py, conformal.py
  engine/       # WS-D: train.py, eval.py, metrics.py, logging.py
  rt/           # WS-E: infer.py, demo.py, bench.py, export.py
scripts/        # WS-B: download_*.py ; WS-D: run_train.ps1 ; pod/ (vast.ai) lead
configs/        # base.yaml (lead), experiment overrides (WS-D)
tests/          # each WS adds tests for its own modules
```

Rules for implementers: own only your directories; never edit `contracts.py`/`config.py`/`base.yaml`
(if a contract is wrong, note it in your final report); every module imports cleanly without GPUs,
teachers, or datasets present; add unit tests (CPU, tiny tensors) for your code; match repo style;
commit to your branch `ws/<name>`.

## 10. Milestones

M1 skeleton+contracts (lead) → M2 parallel workstreams (Opus, worktrees) → M3 merge + review + fix →
M4 local sanity (overfit 50 imgs; FPS bench on 3060; demo runs webcam/file) → M5 PR →
M6 (user-gated, billed) vast.ai full training → M7 pseudo-label round 2 + INT8/TensorRT + report.
