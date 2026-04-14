# akili_NeRF — SIMT full NeRF forward pass

A single Vortex SIMT-baseline benchmark that runs an entire NeRF forward pass
(ray-AABB intersection → stratified sampling → tiny MLP → alpha compositing)
in **one benchmark directory**, and reports per-stage cycle counts so the user
can cleanly separate the **GEMM part** (the MLP) from the **non-GEMM part**
(ray setup + compositing).

> **Scope**: SIMT only. No TCU, no sparse TCU. The benchmark exists as the
> baseline for a future `akili_NeRF_tcu` / `akili_NeRF_tcu_sp` variant that
> would replace the MLP stage with a WMMA-batched version.

## Why this benchmark exists

The existing `benchmarks/nerf_*/` directories were ported from the nerfacc
library and split the ray-sampling pipeline into **six separate binaries**,
one per CUDA kernel:

- `nerf_ray_aabb_intersect`, `nerf_traverse_grid`, `nerf_exclusive_prod`,
  `nerf_inclusive_sum`, `nerf_searchsorted`, `nerf_importance_sampling`

None of them include the MLP, because nerfacc itself **does not ship a CUDA
MLP** — in real nerfacc pipelines the MLP is a plain `torch.nn.Module` that
runs through cuBLAS (or through tiny-cuda-nn's `FullyFusedMLP` if the user
chose the ngp-style radiance field).

This benchmark closes that gap: it pairs nerfacc-style ray sampling and
compositing math with a hand-written first-principles NeRF MLP SIMT kernel,
so the whole forward pass runs in one `vxbin` under one host driver, and the
GEMM vs non-GEMM cycle ratio is directly observable.

## Reference sources

| Source | URL | Used for |
|---|---|---|
| **Mildenhall et al. 2020 — vanilla NeRF** (original paper/TF impl) | https://github.com/bmild/nerf | MLP architecture (8×256 + skip + direction branch), positional encoding `γ(p) = [sin(2ⁿπp), cos(2ⁿπp)]`, and the softplus(σ) + sigmoid(rgb) activation convention. |
| **nerfacc (Nerfstudio project)** | https://github.com/nerfstudio-project/nerfacc | CUDA ray-marching + volumetric rendering kernels under `nerfacc/cuda/csrc/`: `ray_aabb_intersect`, `traverse_grids`, `render_weight_from_density` / `exclusive_prod`, `accumulate_along_rays`. The canonical "vanilla NeRF using nerfacc" example is `examples/train_mlp_nerf.py` + `examples/radiance_fields/mlp.py` (class `VanillaNeRFRadianceField`). |
| **Repo-local CUDA reference** | `tests/bench_dir/bench/benchmarks/cuda_reference/nerfacc_bench.cu` | Standalone CUDA port of the nerfacc ray-AABB/traverse/exclusive-prod/inclusive-sum kernels (already in this repo — lines 35–153). The math in `kernel0_body` and `kernel2_body` matches it. |
| **nerf_pl (PyTorch reference)** | https://github.com/kwea123/nerf_pl | Clean Python implementation of the vanilla-NeRF MLP (`models/nerf.py`, class `NeRF`) for cross-checking the architecture. |
| **tiny-cuda-nn + instant-ngp** | https://github.com/NVlabs/tiny-cuda-nn (`src/fully_fused_mlp.cu`), used by https://github.com/NVlabs/instant-ngp | Reference for what a production fused-CUDA small-MLP looks like on GPUs. This benchmark does NOT copy code from tiny-cuda-nn — our SIMT kernel is a plain per-point GEMV loop — but it is cited here as the standard "how it would look on a real GPU" reference. |

**Caveat for readers**: nerfacc itself contains no CUDA NeRF MLP. The MLP
kernel in this benchmark (`kernel1_body` in `kernel.cpp`) is a first-
principles SIMT port of the **architecture** described in Mildenhall 2020,
not a translation of existing CUDA source. That is the core reason the six
existing `nerf_*/` tests lacked a GEMM component — and exactly the gap this
benchmark closes.

## Pipeline

```
  host inputs                          device kernels (1 vxbin, 3 spawns)
  ───────────                          ─────────────────────────────────────
  rays_o[n_rays*3]  ─┐
  rays_d[n_rays*3]  ─┤  Stage 1   ──► sample_pos[n_rays*n_samples*3]
  aabb[6]           ─┘  ray_setup      deltas[n_rays*n_samples]
                        (non-GEMM,
                         1 task/ray)     KCYC[SETUP,nt=N]
                                 │
                                 ▼
  W0..W3, B0..B3    ──► Stage 2   ──► sigmas[n_rays*n_samples]
   (2944 floats)        mlp_fwd        rgbs[n_rays*n_samples*3]
                        (GEMM,
                         1 task/pt)     KCYC[MLP,nt=N]
                                 │
                                 ▼
                         Stage 3   ──► image[n_rays*3]
                         composite
                        (non-GEMM,
                         1 task/ray)    KCYC[COMP,nt=N]
```

### Stage 1 — ray setup (non-GEMM)

One SIMT task per ray. Implements the three-slab AABB test exactly as in
`nerf_ray_aabb_intersect/kernel.cpp:31-64`, clamps the entry to `min_near`,
and then stratified-samples `n_samples` midpoints uniformly along
`[tmin, tmax]`. Rays that miss the AABB collapse to `step=0`, which yields
`alpha=0` downstream — no special-casing needed in stages 2 or 3.

### Stage 2 — tiny NeRF MLP (GEMM)

One SIMT task per `(ray, sample)` point → `n_rays * n_samples` tasks.
Each task:

1. **Positional encoding** with `L=4` frequency bands on `(x, y, z)`:
   `[x, y, z, sin(π·x), cos(π·x), sin(π·y), cos(π·y), sin(π·z), cos(π·z), ...]`
   → **27 input features**. (Vanilla NeRF uses L=10 → 60 features; we trim
   to L=4 so the input and the workload fit comfortably in simx.)
2. **4 hidden layers, width 64, ReLU** — standard `y = ReLU(W·x + b)` GEMVs.
   (Vanilla NeRF uses 8 × 256 with a skip at layer 5; we trim to 4 × 64,
   no skip. Same ops, fewer FMAs.)
3. **Output head**: a linear `32 → 4` layer producing `(σ_logit, r, g, b)`.
   Final activations are **softplus(σ)** and **sigmoid(rgb)** — matches
   Mildenhall 2020.

Weights are read directly from DRAM every task. Cache reuse is high
(~11.5 KB of total weights easily fits in L1/L2) and this is a SIMT baseline
where we are measuring the cost of the scalar approach — a TCU variant would
amortize weight loads at the warp level.

### Stage 3 — alpha compositing (non-GEMM)

One task per ray. Classical NeRF compositing:

```
T = 1
for s in [0 .. n_samples):
    alpha   = 1 - exp(-sigma[s] * delta[s])
    weight  = alpha * T
    pixel  += weight * rgb[s]
    T      *= (1 - alpha)
```

Same as `ngp_composite_fwd/kernel.cpp:11-55`, simplified to drop the
depth / weight-sum outputs and any early-termination threshold.

## Trimming rationale (tiny NeRF)

Vanilla NeRF (Mildenhall 2020) has:

- 8 hidden layers × 256 width, skip connection at layer 5
- A direction-dependent second head (29 → 128 → 3 for RGB)
- L=10 / L=4 positional encoding (60 / 24 dims)
- ≈600K parameters, tens of MB of activations per batch

That is too large for Vortex simx — it would run for hours. The trimmed
variant used here keeps **every operator identical** (Linear + ReLU + PE +
softplus + sigmoid + alpha compositing) but shrinks dimensions:

| Knob | Vanilla | akili_NeRF |
|---|---|---|
| Hidden layers | 8 | **4** |
| Hidden width | 256 | **64** |
| Skip connection | yes (layer 5) | **none** |
| Direction branch | yes | **dropped** (no view-dependence) |
| PE bands L_pos | 10 | **4** → 27 input dims |
| PE bands L_dir | 4 | **0** |
| Parameters | ~600K | **2944 floats (11.5 KB)** |

Because the ops don't change, the **ratio** of non-GEMM to GEMM cycles is
still representative of a "GEMM-heavy per-sample forward". The absolute
cycle counts scale roughly linearly in `n_rays * n_samples` and quadratically
in `MLP_W`, so the user can explore larger regimes by editing
`common.h` if they want to stress-test a TCU variant later.

## Cycle measurement

Each kernel's **device-side `main()`** brackets its single `vx_spawn_threads`
call with `vx_rdcycle()` and writes `(end - begin)` into
`kernel_arg_t::kernel_cycles` — the exact pattern `akili_attn/kernel.cpp:77-103`
uses. The host reads that field back between stages via
`read_back_cycles("TAG")` (see `akili_attn/main.cpp:124-129` for the helper
this is copied from) and prints:

```
KCYC[SETUP,nt=8]: <cycles>
KCYC[MLP,nt=8]:   <cycles>
KCYC[COMP,nt=8]:  <cycles>
KCYC[SUMMARY,nt=8]: non_gemm=<SETUP+COMP> gemm=<MLP> total=<sum> (gemm_frac=...)
```

`non_gemm = SETUP + COMP`, `gemm = MLP`. The final line is what answers the
user's question directly: **how much of the NeRF forward pass is GEMM work
vs. everything else** on a SIMT Vortex baseline.

## Build and run

### 1. Vortex simx baseline

SIMT baseline only — no TCU / no sparse CONFIGS. If the current Vortex
simx build isn't `NUM_THREADS=8`, rebuild it first:

```bash
cd /path/to/vortex/build
make -s
```

(A normal `make -s` with no `CONFIGS` override produces a SIMT build usable
by every `akili_*` SIMT baseline.)

### 2. Build the benchmark

```bash
cd tests/bench_dir/bench/benchmarks/akili_NeRF
make clean && make NUM_THREADS=8
```

Produces the host binary `akili_NeRF`, the kernel object `kernel.elf`, and
the shipping-format kernel `kernel.vxbin`.

### 3. Smoke run

```bash
make run-simx OPTS="-r 32 -s 8"
```

### 4. Medium run

```bash
make run-simx OPTS="-r 64 -s 16"
```

## CLI flags

| Flag | Meaning | Default |
|---|---|---|
| `-r <N>` | Number of rays | 32 |
| `-s <N>` | Samples per ray | 8 |
| `-k <path>` | Kernel binary | `kernel.vxbin` |

## Expected output

```
open device connection
akili_NeRF (SIMT) — n_rays=32 n_samples=8 MLP=4x32 PE_L=4
=== Stage 1: ray setup ===
KCYC[SETUP,nt=8]: ...
=== Stage 2: tiny MLP forward ===
KCYC[MLP,nt=8]:   ... (dominant)
=== Stage 3: alpha compositing ===
KCYC[COMP,nt=8]:  ...
KCYC[SUMMARY,nt=8]: non_gemm=... gemm=... total=... (gemm_frac=0.8x-0.9x)
PASSED!
```

MLP cycles should clearly dominate because each of the `n_rays * n_samples`
tasks runs ~4 kGEMV of scalar FMAs, while ray setup and compositing are
linear in samples with very cheap per-step math.

## Future extensions

- **`akili_NeRF_tcu`** — rewrite Stage 2 to batch the `n_rays * n_samples`
  tasks into a `[n_points, MLP_IN_DIM] x [MLP_IN_DIM, MLP_W]` GEMM using
  `vx_tensor.h` (`wmma_context<NT, fp16, fp32>`). Use the `sgemm_tcu`
  pattern. Stages 1 and 3 stay SIMT. This is where the bulk of the speedup
  will come from, because stage 2 is the only GEMM-heavy piece.
- **`akili_NeRF_tcu_sp`** — apply 2:4 structured sparsity to the MLP
  weights (`wmma_context<NT, fp16, fp32, true>`) via the existing
  `prune_2to4_matrix` / `compress_2to4_matrix` / `pack_metadata` utilities.

Both variants would land as new sibling directories, exactly like
`akili_attn` → `akili_attn_tcu` → `akili_attn_tcu_sp`.
