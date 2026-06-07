<div align="center">

<img src="assets/banner.png" alt="faster-trellis-plus-plus" width="680">

# ⚡ faster-trellis++

**Training-free acceleration for [microsoft/TRELLIS](https://github.com/microsoft/TRELLIS) image-to-3D — the exponential (DMD) forecast variant of [`faster-trellis`](https://github.com/Archerkattri/faster-trellis), in one line of code.**

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg?style=flat-square)](./LICENSE)
[![base: TRELLIS](https://img.shields.io/badge/base-microsoft%2FTRELLIS-555.svg?style=flat-square)](https://github.com/microsoft/TRELLIS)
[![arXiv: HiCache](https://img.shields.io/badge/arXiv-2508.16984%20HiCache-b31b1b.svg?style=flat-square)](https://arxiv.org/abs/2508.16984)
[![arXiv: Adaptive Guidance](https://img.shields.io/badge/arXiv-2312.12487%20Adaptive%20Guidance-b31b1b.svg?style=flat-square)](https://arxiv.org/abs/2312.12487)

`TRELLIS-image-large` · mesh + Gaussian + radiance-field · training-free · single RTX 5090 · MIT

</div>

## When to use this repo

These repos are **complementary accelerators, not competing solutions** — each speeds up a *different*
base generator, and the `+` / `++` suffix is a **method choice**, not a rival product. Pick by
**(1) which base model you run**, then **(2) which forecast basis you want**:

| base generator | `+` = HiCache (Hermite) | `++` = HiCache++ (DMD) |
|---|---|---|
| Hunyuan3D-2.1 | `hunyuan2.1-plus` | `hunyuan2.1-plus-plus` |
| Hunyuan3D-2 mini | `hunyuan2-plus` | `hunyuan2-plus-plus` |
| SAM 3D Objects | `sam3d-plus` | `sam3d-plus-plus` |
| Fast-SAM3D | `fastsam3d-plus` | `fastsam3d-plus-plus` |
| DiT-XL/2 (ImageNet) | `dit-plus` | `dit-plus-plus` |
| TRELLIS (v1) | `faster-trellis` | `faster-trellis-plus-plus` |
| TRELLIS.2-4B (v2) | `hermit-trellis2` | `hermit-trellis2-plus-plus` |

- **`+` (HiCache / scaled-Hermite):** the *published* polynomial velocity-forecast basis — conservative, reproduces the HiCache paper. Use it to deploy the established method.
- **`++` (HiCache++ / DMD exponential):** our Dynamic-Mode-Decomposition basis — *the same near-lossless quality at wider skip intervals*, where the polynomial diverges. Use it when you push the cache interval for more speed.
- **standalone / model-agnostic:** [`hicache-plus-plus`](https://github.com/Archerkattri/hicache-plus-plus) — the forecaster itself, to add DMD caching to *your own* diffusion/flow model.
- **`fast-trellis2`** = the TaylorSeer baseline fork (the upstream "Fast" accel) — the v2 reference point, not a HiCache variant.

> **This repo:** `faster-trellis-plus-plus` — **TRELLIS v1 × HiCache++ (DMD)** — carved-hybrid; 0.829 F@0.05 @ 2.76×, the most lossless at the deployed interval.

`faster-trellis++` is `microsoft/TRELLIS` image-to-3D with the same **training-free carved-hybrid
acceleration** as [`faster-trellis`](https://github.com/Archerkattri/faster-trellis) — but with the
sparse-structure velocity forecast on an **exponential Dynamic-Mode-Decomposition (DMD / Prony)
basis** instead of the Hermite polynomial. It forecasts and reuses the model's **final CFG-combined
velocity** so the sampler spends far fewer, cheaper network evaluations per asset — the weights,
decoders, and mesh + Gaussian + radiance-field outputs are the base model's, untouched.

```python
pipeline.enable_faster_mode()                                          # carved-hybrid, Hermite SS forecast (the faster-trellis default)
pipeline.enable_faster_mode("faster", hicache_kwargs={"backend": "dmd"})  # ← the exponential DMD forecast (HiCache++)
pipeline.enable_faster_mode("none")                                    # stock TRELLIS sampler (kill-switch)
```

**What's new vs `faster-trellis`.** Same carved-hybrid schedule, same token-carved SLaT stage — the
only change is the **forecast basis on the sparse-structure stage**:

- **HiCache++ (exponential DMD/Prony)** on the **sparse-structure** stage — forecasts the velocity
  with **Dynamic Mode Decomposition** instead of the dual-scaled Hermite polynomial. The exact
  solution of the diffusion feature-ODE is a sum of (damped/oscillatory) **exponentials**, not
  polynomials, so DMD is the natural basis: it stays lossless at **larger skip intervals** than the
  Hermite/Taylor polynomial bases, which diverge once the forecast horizon grows.
- **Token-carved SLaT** on the **structured-latent** stage — unchanged from `faster-trellis`: a
  learned-cadence temporal skip plus spatial **token carving** that recomputes only the
  high-frequency voxels each step.

**Relationship to the family.** [`faster-trellis`](https://github.com/Archerkattri/faster-trellis)
is the **HiCache (Hermite)** parent — it pairs Fast-TRELLIS's token-carving substrate with a Hermite
forecast on the sparse-structure stage and **beats Fast-TRELLIS on both speed and quality**.
`faster-trellis++` keeps that exact carved-hybrid and swaps the Hermite forecast for the exponential
DMD one. The DMD forecaster ships as a standalone library in
[`hicache-plus-plus`](https://github.com/Archerkattri/hicache-plus-plus); this repo is its TRELLIS-v1
integration, selectable with the sampler's `backend="dmd"`.

---

## Quickstart

```bash
git clone --recurse-submodules https://github.com/Archerkattri/faster-trellis-plus-plus
cd faster-trellis-plus-plus
# TRELLIS deps (CUDA toolchain required); see setup.sh for the full option list.
. ./setup.sh --new-env --basic --xformers --flash-attn --spconv --nvdiffrast
```

```python
from trellis.pipelines import TrellisImageTo3DPipeline

pipeline = TrellisImageTo3DPipeline.from_pretrained("microsoft/TRELLIS-image-large").cuda()
pipeline.enable_faster_mode("faster", hicache_kwargs={"backend": "dmd"})  # ← exponential DMD forecast

outputs = pipeline.run(image, formats=["mesh", "gaussian", "radiance_field"])
```

`backend="hermite"` (the default, with no `hicache_kwargs`) gives the original `faster-trellis`
behaviour; `backend="dmd"` selects the exponential forecast. The DMD snapshot-window length is the
`history` kwarg (default `6`).

Runnable scripts: `example.py` (`--mode {faster,none}`) and
`example_faster.py` (minimal, annotated).

> **Mesh output requires the FlexiCubes submodule.** `formats=["mesh"]` (the
> decoder used for F-score / CD evaluation) imports
> `trellis/representations/mesh/flexicubes`. If you cloned without
> `--recurse-submodules`, fetch it once with `git submodule update --init`,
> otherwise mesh decoding will raise a `flexicubes` import error. The
> `gaussian` and `radiance_field` formats do not need it.

<details>
<summary><b>RTX 50-series (sm_120) note</b></summary>

The default flash-attn / sparse-conv kernels may not ship sm_120 builds. Select the
spconv backend and its native algorithm:

```bash
export SPARSE_CONV_BACKEND=spconv
export SPCONV_ALGO=native
```
</details>

---

## Results

**TRELLIS v1, Toys4K mesh F-score@0.05** (n = 31 matched objects, sparse-structure stage —
swapping *only* the SS forecast basis Hermite→DMD, same carved-hybrid schedule + token-carved SLaT):

| variant | F-score@0.05 | speedup | vs vanilla |
|---|---:|---:|---:|
| vanilla (uncached) | 0.839 | 1.00× | — |
| Hermite parent ([`faster-trellis`](https://github.com/Archerkattri/faster-trellis)) | 0.825 | 2.82× | −0.014 |
| **HiCache++ (DMD) — this repo** | **0.829** | **2.76×** | **−0.010** |

At the deployed ~interval-3 schedule (2.8×) the exponential (DMD) basis is the **most lossless**
accelerator — it beats the Hermite parent by **+0.005** at matched speedup. The margin is modest at
this *low* skip (where the polynomial is still accurate); the DMD advantage **widens at larger
`GF_HICACHE_SS_INTERVAL`**, exactly where the exact-on-the-feature-ODE-class property pays off. For
the standalone forecaster benchmarks (DMD vs Hermite vs Taylor) and the full cross-model results,
see [`hicache-plus-plus`](https://github.com/Archerkattri/hicache-plus-plus).

---

## How it works

TRELLIS v1 samples a 3D asset in two flow-matching stages — **sparse structure** (the coarse
occupancy that fixes topology) then **structured latent** (the refined geometry). The velocity
field is smooth across diffusion steps and most tokens change slowly, so most model passes are
redundant. `faster-trellis++` exploits both: an **exponential DMD** forecast thins the
sparse-structure stage, and a **token-carved** SLaT sampler skips whole steps and recomputes only the
high-frequency voxels on the steps it does run.

<details>
<summary><b>① HiCache++ — exponential (DMD/Prony) velocity forecast</b> (removes skip-step network calls)</summary>

At each **compute** step the sampler runs the model and records the final CFG-combined velocity into
a short snapshot window (the last `history` compute steps). At a **skip** step the velocity is
forecast by **Dynamic Mode Decomposition** over those snapshots instead of calling the model:

```
# DMD identifies the linear propagator A from the velocity snapshots,
# its eigen-decomposition A = Φ Λ Φ⁻¹, and advances the modes k steps:
F̂_{t+k} ≈ Φ (Λᵏ · b)
```

**Why exponential over polynomial.** A diffusion feature/velocity trajectory is the solution of a
near-linear feature-ODE, whose **exact** solution class is a sum of (damped / oscillatory)
**exponentials** `Σ bⱼ λⱼᵏ`, not polynomials. DMD — the modern generalisation of Prony's method
(1795) — fits exactly that class: it recovers the modes `(Φ, Λ)` from the snapshots and is **exact on
an exponential series**, which the polynomial Hermite/Taylor bases are not. The polynomial forecasts
grow without bound as the skip horizon `k` increases (Taylor diverges fastest; the dual-scaled
Hermite contracts it but is still polynomial), so the exponential basis is what holds quality at the
**larger compute intervals** this variant targets. With too short a window DMD falls back to reusing
the last computed velocity. *(arXiv:2508.16984 for the HiCache/Hermite parent method; the DMD/Prony
basis is the `backend="dmd"` extension.)*
</details>

<details>
<summary><b>② Token-carved SLaT — recompute only the high-frequency voxels</b> (SLaT stage, unchanged from faster-trellis)</summary>

The structured-latent stage denoises a `SparseTensor` of voxel tokens, and most tokens change
slowly between steps. On each computed step we score every token by **spatial high-frequency
energy** (a 3D-FFT of the sparse-structure occupancy) together with its velocity magnitude and
motion, and recompute only the most active fraction — the **carving ratio**; the smoothest tokens
reuse their cached velocity under a staleness bound that forces a periodic full refresh. On top of
that a **learned-k delta cache** skips whole steps when the velocity field is locally linear
(`vₜ ≈ xₜ + Δ`). The pipeline hands the SS occupancy's frequency score to the SLaT sampler, so the
carving signal is the structure itself. *(Fast-TRELLIS token selection; carving ratio = `GF_CARVE_RATIO`.)*

**The savings multiply:** the SLaT sampler skips whole steps *and* carves tokens on the steps it
runs, while the DMD forecast independently thins the sparse-structure stage.
</details>

---

## Tuning

The shipped configuration is a single set of defaults; each is overridable by env var (e.g. to
trade a little speed for fidelity on a hard input). The forecast basis itself is the `backend`
kwarg to `enable_faster_mode` (`"hermite"` default, `"dmd"` for the exponential variant):

| knob | env | default | meaning |
|---|---|:--:|---|
| carving ratio | `GF_CARVE_RATIO` | `0.25` | fraction of SLaT tokens *skipped* per step (cached, not recomputed) |
| SS interval | `GF_HICACHE_SS_INTERVAL` | `3` | sparse-structure: compute 1 step, forecast `interval − 1` |
| SS first-enhance | `GF_HICACHE_FIRST_ENHANCE` | `2` | always compute the first N sparse-structure steps |

The DMD backend adds a `history` kwarg (snapshot-window length, default `6`) passed through
`hicache_kwargs`. Finer knobs live in `trellis/pipelines/samplers/faster_samplers.py`:
`GF_CARVE_THRESH` (`5.0`, SLaT delta-cache skip threshold) and — for the Hermite default — the SS
`sigma` (`0.5`, contraction σ∈(0,1)) / `max_order` (`1`).

---

## What's added on top of TRELLIS

The Microsoft TRELLIS model and decoder math are unchanged. The acceleration lives under
`trellis/pipelines/samplers/`: `hicache.py` (the velocity-forecast cache — both the Hermite
polynomial and the **DMD/Prony exponential** backends, selected by `backend=`) + `hicache_freq.py`
(the 3D-FFT token-frequency scoring that drives carving), `flow_euler_carved.py` (the token-carved
SLaT sampler), `adaptive_cfg.py`, and `faster_samplers.py` (the accelerated sampler classes), plus
the `enable_faster_mode()` API in `trellis/pipelines/trellis_image_to_3d.py`. The accelerators are
independent re-implementations of the cited methods.

The one decoder-file touch is import-only: `representations/mesh/cube2mesh.py` imports the
optional FlexiCubes submodule **lazily** (at mesh-decoder construction time) rather than at
module top level, so the package — and the gaussian / radiance-field paths — stay importable
when the submodule has not been fetched. Mesh-decoder geometry is identical when FlexiCubes is
present.

---

## Credits & license

| | |
|---|---|
| **TRELLIS** | Xiang et al., *Structured 3D Latents for Scalable and Versatile 3D Generation*, CVPR 2025 — [microsoft/TRELLIS](https://github.com/microsoft/TRELLIS) (MIT). `faster-trellis++` is a thin acceleration layer on their pipeline and weights. |
| **faster-trellis** | [Archerkattri/faster-trellis](https://github.com/Archerkattri/faster-trellis) — the HiCache (Hermite) parent this carved-hybrid is built on |
| **hicache-plus-plus** | [Archerkattri/hicache-plus-plus](https://github.com/Archerkattri/hicache-plus-plus) — the standalone DMD/Prony exponential-forecast library |
| **HiCache** | arXiv:2508.16984 — Hermite-polynomial velocity forecasting (the parent method the `++` extends) |
| **DMD / Prony** | Schmid, *Dynamic Mode Decomposition of numerical and experimental data* (JFM 2010); de Prony (1795) — the exponential-forecast basis |
| **Adaptive Guidance** | Castillo et al., arXiv:2312.12487 — unconditional-pass skipping |

MIT (see [`LICENSE`](LICENSE), [`NOTICE`](NOTICE)). Accelerations © 2026 Krishi Attri;
TRELLIS pipeline / architecture / weights © their authors.

**Krishi Attri** · krishiattriwork@gmail.com · [github.com/Archerkattri](https://github.com/Archerkattri)

<details>
<summary><b>BibTeX</b></summary>

```bibtex
@misc{attri2026fastertrellispp,
  title  = {faster-trellis++: Training-free DMD/Prony exponential-forecast acceleration for TRELLIS image-to-3D},
  author = {Krishi Attri}, year = {2026},
  howpublished = {\url{https://github.com/Archerkattri/faster-trellis-plus-plus}}
}
@inproceedings{xiang2025trellis,
  title     = {Structured 3D Latents for Scalable and Versatile 3D Generation},
  author    = {Xiang, Jianfeng and others}, booktitle = {CVPR}, year = {2025}
}
@article{hicache2025,
  title   = {HiCache: Training-free Acceleration of Diffusion Models via
             Hermite Polynomial Feature Forecasting},
  journal = {arXiv preprint arXiv:2508.16984}, year = {2025}
}
@article{schmid2010dmd,
  title   = {Dynamic mode decomposition of numerical and experimental data},
  author  = {Schmid, Peter J.},
  journal = {Journal of Fluid Mechanics}, volume = {656}, year = {2010}
}
@article{castillo2023adaptive,
  title   = {Adaptive Guidance: Training-free Acceleration of Conditional Diffusion Models},
  author  = {Castillo, Angela and others},
  journal = {arXiv preprint arXiv:2312.12487}, year = {2023}
}
```
</details>

## Weights & data

Model weights and demo/example assets are **not** committed to this repo — only the acceleration
architecture (code + integration). Download the base-model weights from the upstream project,
[microsoft/TRELLIS](https://github.com/microsoft/TRELLIS), per its instructions, and point the loader at them (see the code / upstream README). This
keeps the repository lightweight and avoids redistributing third-party weights.
