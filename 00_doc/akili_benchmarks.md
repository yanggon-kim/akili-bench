# Akili Benchmarks — User Guide

This document describes the fifteen `akili_*` benchmarks that ship with
Vortex Sparse TCU under `tests/bench_dir/bench/benchmarks/`. Each
benchmark is a **self-contained test directory** that builds a single
binary for one `(workload, hardware variant)` pair. Use these if you
want to run attention, FlashAttention, 2D convolution, or a full NeRF
forward pass on Vortex in SIMT mode, dense TCU mode, or 2:4 sparse TCU
mode — and compare the three against each other head-to-head at a
shape of your choice. The NeRF tests (`akili_NeRF*`) are documented in
Section 9.

All paths in this document are **relative to the Vortex source root**
(`vortex/`). No absolute paths, so everything works from
any clone.

For a repo-level simx sweep, `tests/bench_dir/akili-bench/run_akili_simx_matrix.sh`
discovers the top-level `akili_*` benchmarks, sweeps a configurable thread list
(`4,8,16,32` by default), and launches the selected benchmarks in parallel
within each thread-count suite. Its default workload profile is intentionally
fast; use `--attn-opts`, `--flash-opts`, `--cnn-opts`, `--nerf-opts`,
`--llama-opts`, or `--benchmark-opts akili_name="..."` when you want larger
manual shapes. By default it writes results to `~/results/` rather than inside
the repo, so benchmark output does not bloat the checkout. The runner also
creates a separate Vortex build tree per thread-count suite, so `nt=4`,
`nt=8`, `nt=16`, and `nt=32` do not accidentally share a mismatched simx
runtime.

Runner measurement caveat: the generated `results.csv` stores the final
`PERF:` line printed by each benchmark. That is fine for single-launch tests
such as `akili_cnn` / `akili_acccnn_tcu*` and the `akili_llama2*` family, but
it is not an end-to-end cycle metric for multi-stage benchmarks such as
`akili_attn*`, `akili_flash_tcu*`, and `akili_NeRF*`. For those, use the
benchmark-owned `KCYC[...]` stage counts and sum the stages when comparing
SIMT vs dense/sparse TCU.

---

## 1. What the fifteen tests are

### Attention  (unfused 3-stage: Q·Kᵀ → softmax → P·V)

| Directory | Hardware | What it runs |
|---|---|---|
| `tests/bench_dir/bench/benchmarks/akili_attn/`         | **SIMT**       | Three SIMT kernels (one thread per output cell) for QK, softmax, and PV. fp32 everywhere. |
| `tests/bench_dir/bench/benchmarks/akili_attn_tcu/`     | **Dense TCU**  | Q·Kᵀ and P·V via `sgemm_tcu`-style multi-block tile loops (fp16 input, fp32 accumulator). Softmax stays SIMT fp32. Host pre-packs Q as fp16 row-major, K/V as fp16 col-major. |
| `tests/bench_dir/bench/benchmarks/akili_attn_tcu_sp/`  | **Sparse TCU** | Same as dense, plus host 2:4 prune + compress + pack_metadata on **both** Q (before QK) and P (after softmax). Both GEMM stages run the sparse MMA path. |

### FlashAttention

| Directory | Hardware | What it runs |
|---|---|---|
| `tests/bench_dir/bench/benchmarks/akili_flash/`         | **SIMT**       | **Fused** online-softmax SIMT kernel (`flash_kernel_body<HEAD_DIM, BLOCK_SIZE_C>`) for `d ∈ {1,2,4,8,16}`. For other `d` values it falls through to the unfused 3-stage SIMT path — identical to `akili_attn`. |
| `tests/bench_dir/bench/benchmarks/akili_flash_tcu/`     | **Dense TCU**  | Same 3-stage dense-TCU pipeline as `akili_attn_tcu`. |
| `tests/bench_dir/bench/benchmarks/akili_flash_tcu_sp/`  | **Sparse TCU** | Same 3-stage sparse-TCU pipeline as `akili_attn_tcu_sp`. |

### CNN / conv2d  (shape-configurable, 3×3 by default, valid padding, stride 1)

| Directory | Hardware | What it runs |
|---|---|---|
| `tests/bench_dir/bench/benchmarks/akili_cnn/`           | **SIMT**       | Direct SIMT conv2d — one thread per output element, inner loop over `C_in × K × K`. |
| `tests/bench_dir/bench/benchmarks/akili_acccnn_tcu/`    | **Dense TCU**  | **im2col + GEMM** conv. Host computes im2col into an `[N_gemm × K_gemm]` fp16 col-major matrix and flattens the weights to `[M_gemm × K_gemm]` fp16 row-major. Kernel is a direct copy of `tests/regression/sgemm_tcu/kernel.cpp`. |
| `tests/bench_dir/bench/benchmarks/akili_acccnn_tcu_sp/` | **Sparse TCU** | Same im2col, but host prunes the flattened weight matrix 2:4 along `K_gemm`, compresses to half stride, and packs metadata. Kernel is a direct copy of `tests/regression/sgemm_tcu_sp/kernel.cpp`. |

Every benchmark prints per-stage cycle counts via `KCYC[...,nt=<NT>]`
and a final `PASSED!` or `FAILED!` after comparing the device output
against a fp32 CPU reference.

The repo-level matrix runner now also includes the three NeRF benchmarks:

| Workload | SIMT | Dense TCU | Sparse TCU (2:4) |
|---|---|---|---|
| **NeRF** | `benchmarks/akili_NeRF/` | `benchmarks/akili_NeRF_tcu/` | `benchmarks/akili_NeRF_tcu_sp/` |

NeRF uses `-r` for ray count and `-s` for samples per ray. The runner's
default sanity profile keeps the benchmark Makefile default of `-r 32 -s 8`.

---

## 2. Building and running

### 2.1 First-time Vortex build

From the Vortex source root:

```bash
cd vortex
mkdir -p build && cd build
../configure --xlen=32 --tooldir=$HOME/tools
./ci/toolchain_install.sh --all         # first time only
source ./ci/toolchain_env.sh             # every new shell
make -s                                  # full build, one-time
```

### 2.2 Per-shell environment

```bash
cd vortex/build
source ./ci/toolchain_env.sh
export VORTEX_DRIVER=simx
export LD_LIBRARY_PATH=$PWD/runtime:$LD_LIBRARY_PATH
```

### 2.3 Pick a warp width and rebuild `simx` + `runtime`

The simx binary bakes `NUM_THREADS` at compile time. Rebuild it for
the warp width you want (4, 8, 16, or 32). One `simx` build serves
all 9 akili_* benchmarks because `TCU_SPARSE_ENABLE` also turns on
the dense-TCU path:

```bash
# From vortex/build:
CONFIGS="-DNUM_THREADS=8 -DEXT_TCU_ENABLE -DTCU_SPARSE_ENABLE" \
  make -C sim/simx clean && \
CONFIGS="-DNUM_THREADS=8 -DEXT_TCU_ENABLE -DTCU_SPARSE_ENABLE" \
  make -C sim/simx -s

CONFIGS="-DNUM_THREADS=8 -DEXT_TCU_ENABLE -DTCU_SPARSE_ENABLE" \
  make -C runtime/simx clean && \
CONFIGS="-DNUM_THREADS=8 -DEXT_TCU_ENABLE -DTCU_SPARSE_ENABLE" \
  make -C runtime/simx -s
```

Substitute `-DNUM_THREADS=4/8/16/32` for other warp widths.

### 2.4 Build one benchmark

Each akili_* directory is independent. From the Vortex source root:

```bash
make -C tests/bench_dir/bench/benchmarks/akili_attn -s NUM_THREADS=8
```

Replace `akili_attn` with any of the fifteen dir names. The
`NUM_THREADS=8` argument must match the simx build's warp width.

### 2.5 Build all fifteen benchmarks at once

```bash
BENCH=tests/bench_dir/bench/benchmarks
for d in akili_attn akili_attn_tcu akili_attn_tcu_sp \
         akili_flash akili_flash_tcu akili_flash_tcu_sp \
         akili_cnn akili_acccnn_tcu akili_acccnn_tcu_sp; do
  make -C $BENCH/$d -s NUM_THREADS=8
done
```

### 2.6 Run one benchmark

```bash
cd tests/bench_dir/bench/benchmarks/akili_attn
./akili_attn -n 16 -d 16
```

---

## 3. Command-line flags

### Attention / FlashAttention

Used by all six `akili_attn*` and `akili_flash*` binaries:

| Flag | Meaning | Default |
|---|---|---|
| `-n N`    | Sequence length. In TCU modes this is padded up to `max(TM, TN, TK)`. | 16 |
| `-d D`    | Head dimension. In TCU modes this is padded up to `max(TK, TN)`.       | 16 |
| `-k file` | Override the `kernel.vxbin` path                                       | `kernel.vxbin` |

`N` and `d` accept any power-of-two values that survive the padding.
`akili_flash` automatically picks the fused online-softmax kernel
when `d ∈ {1, 2, 4, 8, 16}`; for larger `d` it falls through to the
unfused 3-stage SIMT path.

### CNN / conv2d

Used by `akili_cnn`, `akili_acccnn_tcu`, `akili_acccnn_tcu_sp`:

| Flag | Meaning | Default |
|---|---|---|
| `-c C_in`   | Input channels                    | 1 |
| `-o C_out`  | Output channels                   | 8 |
| `-h H`      | Input height                       | 28 |
| `-w W`      | Input width                        | 28 |
| `-s K_size` | Kernel spatial size (3 → 3×3)     | 3 |
| `-k file`   | Override `kernel.vxbin` path      | `kernel.vxbin` |

Stride is 1 and padding is 0 (valid). Output shape is
`(C_out, H - K + 1, W - K + 1)`.

---

## 4. Expected output

### Attention — SIMT / dense TCU / sparse TCU

```
akili_attn (SIMT) — N=16 d=16
=== Stage 1: S = Q @ K^T ===
KCYC[QK,nt=8]: 19492
=== Stage 2: P = softmax(S) ===
KCYC[SM,nt=8]: 679858
=== Stage 3: O = P @ V ===
KCYC[PV,nt=8]: 19019
PASSED!
```

### FlashAttention SIMT at `d ≤ 16` (fused single launch)

```
akili_flash (SIMT) — N=16 d=16
=== Fused SIMT flash (d=16) ===
KCYC[FUSED,nt=8]: 1020258
PASSED!
```

### FlashAttention SIMT at `d > 16` (unfused 3-stage)

```
akili_flash (SIMT) — N=32 d=32
=== Stage 1: S = Q @ K^T ===
KCYC[QK,nt=8]: 174218
...
PASSED!
```

### CNN — SIMT / dense TCU / sparse TCU

```
akili_cnn — C_in=16 C_out=16 H=32 W=32 K=3 → H_out=30 W_out=30
KCYC[CONV,nt=8]: 18112160
PASSED!
```

`FAILED!` means the device output didn't match the fp32 CPU reference
within the per-mode tolerance. Sparse TCU uses a looser tolerance
(`5e-2`) because 2:4 pruning is an intentional approximation.

---

## 5. Measured speedups (NT=8, simx, KMU launch API)

> **Measurement note — KMU era.** All 15 `akili_*` binaries migrated
> to the Vortex **KMU** (Kernel Management Unit) launch API in April
> 2026. On the old launch path each stage reported
> `KCYC[TAG,nt=N]: <total>` where `<total>` was the per-spawn duration
> measured with a single `vx_rdcycle()` bracket in the kernel's
> `int main()`. Under KMU, the kernel is a `__kernel void kernel_main`
> entry point that the hardware dispatches CTA-by-CTA — there is no
> single outer timer. Instead, each CTA writes its own cycle count to
> a per-block scratch buffer, and the host reads back
> `max(cycles[block])` per stage. Cycle numbers below are therefore
> **per-stage slowest-block cycle counts** summed across stages. They
> are **not directly comparable to any pre-KMU numbers** (historical
> tables in the git log). See `tests/regression/sgemm_tcu/main.cpp` and
> `kernel/include/vx_spawn2.h` for the current-era launch pattern.
>
> Raw CSV: `tests/bench_dir/bench/00_doc/speedup_results_akili.csv`.

### Attention

| Shape | `-n N` | `-d D` | SIMT (cyc) | Dense TCU (cyc) | Sparse TCU (cyc) | **dense/SIMT** | **sparse/dense** |
|---|---|---|---|---|---|---|---|
| S  | 16 | 16  |       719,809 |       712,941 |       729,202 | 1.01× | 0.98× |
| M  | 32 | 32  |     1,758,032 |     1,753,222 |     1,946,471 | 1.00× | 0.90× |
| L  | 64 | 128 |     1,735,437 |     1,732,536 |     1,891,587 | 1.00× | 0.92× |

### FlashAttention

| Shape | `-n N` | `-d D` | SIMT (cyc) | Dense TCU (cyc) | Sparse TCU (cyc) | **dense/SIMT** | **sparse/dense** |
|---|---|---|---|---|---|---|---|
| S  | 16 | 16  |     1,012,638 |       712,941 |       729,202 | **1.42×** | 0.98× |
| M  | 32 | 32  |     1,754,941 |     1,753,222 |     1,946,471 | 1.00× | 0.90× |
| L  | 64 | 128 |     1,738,913 |     1,732,536 |     1,891,587 | 1.00× | 0.92× |

The attention/flash totals look essentially flat across SIMT / dense
TCU / sparse TCU because **the softmax stage dominates** the per-CTA
max cycle count, and softmax runs the same SIMT kernel in all three
variants. To see the per-stage TCU speedup on the QK and PV GEMMs
alone, scrape the per-stage `KCYC[QK,nt=8]` / `KCYC[PV,nt=8]` lines
from each run — those stages do go ~3-30× faster on the TCU, but the
softmax anchor hides it in the sum.

### CNN / conv2d

| Shape  | `-c C_in` | `-o C_out` | `-h H` | `-s K` | GEMM (M × N × K) | SIMT (cyc) | Dense TCU (cyc) | Sparse TCU (cyc) | **dense/SIMT** | **sparse/dense** |
|---|---|---|---|---|---|---|---|---|---|---|
| mnist  | 1  | 8   | 28 | 3 |     8 × 680 × 16 |     6,156 |     1,773 |     1,807 | **3.47×** | 0.98× |
| small  | 16 | 16  | 32 | 3 |  16 × 904 × 144 |    40,721 |     9,560 |     7,847 | **4.26×** | **1.22×** |
| medium | 32 | 32  | 32 | 3 |  32 × 904 × 288 |    76,510 |    19,079 |    14,053 | **4.01×** | **1.36×** |

CNN shows the cleanest TCU gains because convolution is pure GEMM
and there is no softmax-like anchor dragging every variant's total
toward the same value. At `K_gemm=16` (mnist) the 2:4 metadata
overhead cancels the compression savings, exactly as before.
Above `K_gemm=144` sparse delivers a ~1.22-1.36× bonus on top of
dense — same story as under the old launch path, just with
different absolute cycle numbers.

### NeRF

| Shape  | `-r r` | `-s s` | SIMT (cyc) | Dense TCU (cyc) | Sparse TCU (cyc) | **dense/SIMT** | **sparse/dense** |
|---|---|---|---|---|---|---|---|
| smoke  | 32 | 8 |     5,664,898 |     3,847,782 |     3,831,630 | **1.47×** | 1.00× |

NeRF's end-to-end totals are Amdahl-limited: once the 4 MLP GEMMs
move to the TCU, **the positional-encoding + softplus + sigmoid
SIMT work becomes the dominant cost** of the pipeline. Report the
raw MLP GEMM speedup instead for the "what the TCU buys us" number —
at the smoke shape that is `4,911,858 (SIMT MLP) / 20,765 (dense TCU
MLP_GEMM) ≈ 236×`. The sparse MLP GEMM is 17,533 cycles (`1.18×`
over dense GEMM), but K is too small (32/64) for sparse to win
end-to-end at this shape. See Section 9 for the full NeRF story.

---

## 6. Reproducing a single CSV row

To reproduce any single row of the speedup table, build the matching
three binaries and run each one at the same CLI:

```bash
cd vortex
# Example: reproduce the attention XL row (N=64, d=512) at NT=8
BENCH=tests/bench_dir/bench/benchmarks
for d in akili_attn akili_attn_tcu akili_attn_tcu_sp; do
  make -C $BENCH/$d -s NUM_THREADS=8
done
export VORTEX_DRIVER=simx LD_LIBRARY_PATH=build/runtime
(cd $BENCH/akili_attn         && ./akili_attn         -n 64 -d 512)
(cd $BENCH/akili_attn_tcu     && ./akili_attn_tcu     -n 64 -d 512)
(cd $BENCH/akili_attn_tcu_sp  && ./akili_attn_tcu_sp  -n 64 -d 512)
```

Sum the `KCYC[...]` lines from each run to get the total cycle
count. The dense / sparse TCU binaries print per-stage QK/SM/PV
lines; sum all three to match the `dense_cycles` / `sparse_cycles`
column.

---

## 7. Recommended input sizes

Not every shape makes sense. These recommendations are based on
what actually runs in reasonable simx wall-clock time and what
shows the dense / sparse TCU benefits most clearly. `simx` is
cycle-approximate C++ simulation — 1 M device cycles ≈ 1 s of
simulator wall-clock on a modern host.

### Best configurations (sweet spots)

#### Hardware configuration used for these measurements

**The sweet spots below are specific to the Vortex hardware
configuration listed here.** Change any of these and the best
shapes may shift — cache size and the number of warps / banks
affect where the softmax anchor, memory pressure, and TCU tile
throughput cross over. If you're running on a different Vortex
build (more cores, bigger L1, L2 enabled, different warp width),
re-run the sweep.

| Knob | Value | Where it's set |
|---|---|---|
| Driver | `simx` (cycle-approximate C++ simulator) | `VORTEX_DRIVER=simx` + `LD_LIBRARY_PATH=<build>/runtime` |
| Clusters × cores × socket | `1 × 1 × 1` | `NUM_CLUSTERS`, `NUM_CORES`, `SOCKET_SIZE` (all default 1) |
| Warps per core | **4** | `NUM_WARPS` (default) |
| Threads per warp | **8** | `NUM_THREADS=8` — passed via `CONFIGS` at the simx rebuild |
| SIMD width | 8 | `SIMD_WIDTH = NUM_THREADS` |
| Issue width | 1 | `ISSUE_WIDTH = ceil(NUM_WARPS / 16)` → 1 |
| XLEN | 32 | `--xlen=32` at `../configure` time |
| L1 I-cache | 16 KB, 4-way, 64 B line | `ICACHE_SIZE=16384`, `ICACHE_NUM_WAYS=4` |
| L1 D-cache | 16 KB, 4-way, 64 B line, 8 banks, 16-entry MSHR | `DCACHE_SIZE=16384`, `DCACHE_NUM_WAYS=4`, `DCACHE_NUM_BANKS` (→ 8 at NT=8), `DCACHE_MSHR_SIZE=16` |
| L2 cache | **disabled** (direct-to-memory beyond L1D) | `L2_ENABLE` not defined |
| L3 cache | **disabled** | `L3_ENABLE` not defined |
| Local memory | 16 KB per core | `LMEM_ENABLE` + `LMEM_LOG_SIZE=14` (2^14 = 16384 bytes) |
| LSU lanes / blocks | 8 / 1 | `NUM_LSU_LANES = SIMD_WIDTH = 8`, `NUM_LSU_BLOCKS=1` |
| TCU | **enabled**, dense + 2:4 sparse | `EXT_TCU_ENABLE` + `TCU_SPARSE_ENABLE` at the simx rebuild |
| TCU lanes | 8 | `NUM_TCU_LANES = NUM_THREADS` |
| TCU fp16 tile (kernel-visible) | `TM = 8, TN = 8, TK = 16` elements | computed by `wmma_config_t<NUM_TCU_LANES, fp16, fp32>` |
| TCU registers per fragment | 8 | `TCU_NR = 8` (from `VX_tcu_pkg.sv`) |

To rebuild simx at exactly this configuration:

```bash
cd vortex/build
CONFIGS="-DNUM_THREADS=8 -DEXT_TCU_ENABLE -DTCU_SPARSE_ENABLE" \
  make -C sim/simx clean && \
CONFIGS="-DNUM_THREADS=8 -DEXT_TCU_ENABLE -DTCU_SPARSE_ENABLE" \
  make -C sim/simx -s

CONFIGS="-DNUM_THREADS=8 -DEXT_TCU_ENABLE -DTCU_SPARSE_ENABLE" \
  make -C runtime/simx clean && \
CONFIGS="-DNUM_THREADS=8 -DEXT_TCU_ENABLE -DTCU_SPARSE_ENABLE" \
  make -C runtime/simx -s
```

All 51 sweep runs in Section 5 of `sweet_spot_sweep.csv` used this
exact configuration.

**Why the hardware matters for the sweet-spot shapes**:

- **L1D = 16 KB + no L2** — the two attention/flash rebounds
  (`d=3072` peak, `d=4096` regression) are almost certainly
  L1D-capacity effects. At `n=64, d=3072` the fp16 Q + K_col
  buffers are `64 × 3072 × 2 bytes = 384 KB each`, ×2 for both
  matrices, which overflows the 16 KB L1D regardless — but the
  miss *rate* and bank conflict pattern changes as `d` grows
  past a specific threshold. A Vortex build with an L2 cache
  or a bigger L1D would likely push the dense/SIMT peak to a
  higher `d` and soften the `d=4096` dip.
- **Warps per core = 4** — SIMT performance on attention/flash is
  bottlenecked on softmax serialization; more warps would help
  SIMT hide that latency, which would *shrink* the dense/SIMT
  ratio at small/medium shapes. The sweet spot at `d=3072` would
  likely move down to `d=2048` with more warps.
- **Threads per warp = 8** — directly sets the TCU tile size
  (`TM = TN = 8`, `TK = 16` elements for fp16). Smaller `NT`
  shrinks the tiles and the sweet spot shifts toward smaller `d`;
  larger `NT` does the opposite. At `NT=16` or `NT=32` the
  absolute dense/SIMT numbers stay in the same ballpark but the
  sweet-spot shape changes.
- **TCU sparse metadata overhead is fixed per mma_sync call**,
  so the sparse/dense ratio is mostly sensitive to `K_gemm`
  (more k-steps amortize the overhead). That's why the CNN
  sparse sweet spot is at the largest `K_gemm` you can afford
  (`K_gemm ≥ 288` for 3×3 conv, `K_gemm = 800` for the 5×5 C4
  case).

If you want the sweet-spot numbers to transfer to a different
build, re-run the 11-candidate sweep above on your build and
re-pick the best-combined-score row per group. The shapes I
tested will still be reasonable candidates, but don't assume
the winner is the same.

#### Summary

A directed sweep across 11 candidate shapes (6 attention/flash +
5 cnn) × all three HW modes × NT=8 was run to pick a single "sweet
spot" per benchmark group — the shape that **simultaneously**
maximizes `dense/SIMT` and `sparse/dense`. Raw per-shape data is in
`tests/bench_dir/bench/00_doc/sweet_spot_sweep.csv`. Sweet spots are
chosen by combined score = `(dense/SIMT) × (sparse/dense)` so that
neither ratio is sacrificed to the other.

| Benchmark | **Sweet-spot shape** | **dense/SIMT** | **sparse/dense** | sparse/SIMT | Wall-clock (NT=8) |
|---|---|---|---|---|---|
| **attention** | `-n 64 -d 3072` | **7.10×** | **1.24×** | 8.76× | ~2 min |
| **flash** | `-n 64 -d 3072` | **6.99×** | **1.24×** | 8.63× | ~2 min |
| **cnn_group** | `-c 64 -o 64 -h 32 -w 32 -s 3` | **31.14×** | **1.32×** | 41.13× | ~9 min |

Flash tracks attention exactly because at `d ≥ 32` both fall through
to the same unfused 3-stage path and the TCU binaries run identical
kernels.

#### Candidates tested

All 17 rows below come from the NT=8 sweep. Combined score =
`dense/SIMT × sparse/dense`; the highest per group is the sweet
spot. Every row passed correctness (the CPU fp32 reference).

**Attention / FlashAttention**

| Label | `-n` | `-d` | SIMT | Dense | Sparse | dense/SIMT | sparse/dense | combined |
|---|---|---|---|---|---|---|---|---|
| A1 | 32 |  512 |  3.78 M |  1.24 M |  1.20 M | 3.06× | 1.03× | 3.15× |
| A2 | 32 | 1024 |  6.77 M |  1.48 M |  1.36 M | 4.58× | 1.09× | 4.98× |
| A3 | 64 | 1024 | 25.00 M |  4.53 M |  3.80 M | 5.52× | 1.19× | 6.58× |
| A4 | 64 | 2048 | 49.11 M |  7.58 M |  6.00 M | 6.48× | **1.26×** | 8.18× |
| **A5** | **64** | **3072** | **72.90 M** | **10.27 M** | **8.32 M** | **7.10×** | 1.24× | **8.76×** |
| A6 | 64 | 4096 | 96.71 M | 14.50 M | 11.87 M | 6.67× | 1.22× | 8.15× |

Flash numbers are within noise of attention at every row (at `d ≥ 32`
both run the same unfused 3-stage path on SIMT and identical TCU
kernels). See `sweet_spot_sweep.csv` for the flash rows separately.

**CNN / conv2d**

| Label | `-c C_in` | `-o C_out` | `-h H` | `-s K` | SIMT | Dense | Sparse | dense/SIMT | sparse/dense | combined |
|---|---|---|---|---|---|---|---|---|---|---|
| C1 | 16 | 16 | 32 | 3 |  18.11 M |  632.6 k |  545.0 k | 28.63× | 1.16× | 33.23× |
| C2 | 32 | 32 | 32 | 3 |  67.90 M |   2.28 M |   1.82 M | 29.72× | 1.26× | 37.37× |
| C3 | 64 | 32 | 32 | 3 | 131.17 M |   4.22 M |   3.19 M | **31.12×** | **1.32×** | 41.10× |
| C4 | 32 | 32 | 32 | **5** | 143.55 M |   4.96 M |   3.70 M | 28.97× | **1.34×** | 38.77× |
| **C5** | **64** | **64** | **32** | **3** | **262.88 M** |   **8.44 M** |   **6.39 M** | **31.14×** | **1.32×** | **41.13×** |

#### Trends observed during the sweep

- **Attention/flash dense/SIMT peaks at `d=3072`** (7.10×) and dips
  back at `d=4096` (6.67×). The A6 regression is ~6% relative to A5
  and repeats across attention and flash, so it isn't noise — most
  likely an L1/L2 cache pressure or TLB effect that shows up once
  both Q and K_col buffers exceed a certain threshold.
- **Attention/flash sparse/dense peaks at `d=2048` (1.263×), not
  at the dense/SIMT peak `d=3072` (1.235×).** If your demo is
  *specifically* about sparse speedup and you don't care about
  dense/SIMT, pick A4 (`-n 64 -d 2048`) instead of A5.
- **CNN dense/SIMT saturates near 31×** — going from `C_in=32`
  (C2) to `C_in=64` (C3/C5) barely nudges the ratio (29.72×
  → 31.1×). The TCU is already fully utilized at C2; the only
  remaining lever is sparse compression.
- **C4 (`-s 5` kernel) wins on sparse/dense alone (1.339×) —
  the strongest sparse ratio in the entire sweep**. Bigger kernel
  size inflates `K_gemm` from 288 (C2) to 800, amortizing sparse
  metadata overhead further. But dense/SIMT drops from 29.7× to
  29.0×, so C4 loses to C3/C5 on combined score. C4 is the pick if
  your demo is *sparse-only*.
- **C3 and C5 tie on combined score (~41.1×)** — C3 is the
  half-wall-clock alternative (SIMT ~4-5 min vs C5's ~9 min) at
  effectively identical ratios. Use C3 when you're iterating, C5
  for the headline numbers.

#### Best option per benchmark

Use these three commands together for a single balanced demo:

```bash
BENCH=tests/bench_dir/bench/benchmarks
export VORTEX_DRIVER=simx LD_LIBRARY_PATH=<build>/runtime

# Attention sweet spot (~2 min, dense/SIMT=7.10×, sparse/dense=1.24×)
(cd $BENCH/akili_attn         && ./akili_attn         -n 64 -d 3072)
(cd $BENCH/akili_attn_tcu     && ./akili_attn_tcu     -n 64 -d 3072)
(cd $BENCH/akili_attn_tcu_sp  && ./akili_attn_tcu_sp  -n 64 -d 3072)

# Flash sweet spot (same shape; dense/SIMT=6.99×, sparse/dense=1.24×)
(cd $BENCH/akili_flash         && ./akili_flash         -n 64 -d 3072)
(cd $BENCH/akili_flash_tcu     && ./akili_flash_tcu     -n 64 -d 3072)
(cd $BENCH/akili_flash_tcu_sp  && ./akili_flash_tcu_sp  -n 64 -d 3072)

# CNN sweet spot (~9 min, dense/SIMT=31.14×, sparse/dense=1.32×)
(cd $BENCH/akili_cnn            && ./akili_cnn            -c 64 -o 64 -h 32 -w 32 -s 3)
(cd $BENCH/akili_acccnn_tcu     && ./akili_acccnn_tcu     -c 64 -o 64 -h 32 -w 32 -s 3)
(cd $BENCH/akili_acccnn_tcu_sp  && ./akili_acccnn_tcu_sp  -c 64 -o 64 -h 32 -w 32 -s 3)
```

If you care only about one metric rather than the combined score:

| Priority | Recommendation |
|---|---|
| Max dense/SIMT (attn/flash) | A5 — `-n 64 -d 3072` (7.10×) |
| Max sparse/dense (attn/flash) | A4 — `-n 64 -d 2048` (1.263×) |
| Max dense/SIMT (cnn)   | C5 — `-c 64 -o 64 -h 32 -s 3` (31.14×) |
| Max sparse/dense (cnn) | C4 — `-c 32 -o 32 -h 32 -s 5` (1.339×) |
| Fastest CNN demo with strong ratios | C3 — `-c 64 -o 32 -h 32 -s 3` (31.12× / 1.32×, ~half C5's wall-clock) |

---

### Attention and FlashAttention — dense TCU wins (all three modes valid)

| Purpose | Shape | Why | ~Wall-clock (NT=8) |
|---|---|---|---|
| Quick sanity check   | `-n 16  -d 16`  | Fastest shape. All three modes finish in <1 s each. Use during development to confirm builds still pass. | < 5 s total |
| First real demo      | `-n 32  -d 32`  | Tiny enough to finish fast but flash drops to the unfused 3-stage path (d > 16), so flash = attention. | ~10 s total |
| **Dense TCU speedup**  | `-n 64  -d 128` | Dense TCU ~1.7× SIMT on attention / flash. Sparse vs dense still below 1.0× because PV's small K. | ~30 s total |
| **Strong dense win**   | `-n 64  -d 512` | Dense TCU ~4× SIMT. Sparse vs dense just crosses 1.0×. This is the shape you want for a dense-focused slide. | ~1 min total |
| Max realistic in simx | `-n 64  -d 1024` | Dense TCU ~5.6× SIMT (matches the v3 sweep). SIMT run alone is ~25 M cycles ≈ 25 s. Sparse/dense ≈ 1.14×. | ~2 min total |

### Attention and FlashAttention — sparse TCU wins

To see **sparse TCU beat dense TCU by a meaningful margin** you have
to push `n` and `d` together so the softmax and PV's inner-K stages
become a small fraction of total work. The first row below still has
a tractable SIMT reference; the two below it are **TCU-only** —
don't run the SIMT binaries at `n ≥ 128, d ≥ 2048` (they take 30+
min in simx because the `N² × d` SIMT grid explodes).

| Purpose | Shape | dense cyc | sparse cyc | **sparse/dense** | Modes to run | ~Wall-clock (NT=8) |
|---|---|---|---|---|---|---|
| **Sparse TCU win (SIMT still valid)** | `-n 64  -d 2048` |  7.48 M |  6.10 M | **1.23×** | all three (SIMT ~50 M cyc ≈ 1 min) | ~2-3 min total |
| Sparse demo, medium | `-n 128 -d 2048` | 25.59 M | 19.90 M | **1.29×** | TCU dense + TCU sparse only | ~1-2 min (TCU only) |
| **Strongest sparse (measured)** | `-n 256 -d 2048` | 98.88 M | 73.36 M | **1.35×** | TCU dense + TCU sparse only | ~5-10 min (TCU only) |

The dense/sparse cycle counts above come from the earlier v3 sparse
-PV sweep; every row passed correctness. To reproduce the TCU-only
rows, build **just** the two TCU binaries (`akili_attn_tcu` and
`akili_attn_tcu_sp`, or the flash equivalents) — skip the SIMT
binary at those shapes:

```bash
BENCH=tests/bench_dir/bench/benchmarks
make -C $BENCH/akili_attn_tcu    -s NUM_THREADS=8
make -C $BENCH/akili_attn_tcu_sp -s NUM_THREADS=8
(cd $BENCH/akili_attn_tcu    && ./akili_attn_tcu    -n 256 -d 2048)
(cd $BENCH/akili_attn_tcu_sp && ./akili_attn_tcu_sp -n 256 -d 2048)
```

### CNN / conv2d — dense AND sparse wins at the same shapes

CNN is the easiest workload to see both speedups at once: any shape
with `K_gemm ≥ 144` (which means `C_in × K × K ≥ 144`, so `C_in ≥ 16`
for 3×3 kernels) gets **both** the ~28-30× dense/SIMT win and the
~1.16-1.26× sparse/dense win in the same run.

| Purpose | Shape | K_gemm | **dense/SIMT** | **sparse/dense** | ~Wall-clock (NT=8) |
|---|---|---|---|---|---|
| Smoke test (mnist) | `-c 1  -o 8  -h 28 -w 28 -s 3` |  16 | **11.18×** | 0.89× (sparse loses — K_gemm too small for metadata to amortize) | ~5 s total |
| **Small: dense + sparse win** | `-c 16 -o 16 -h 32 -w 32 -s 3` | 144 | **28.63×** | **1.16×** | ~1-2 min total |
| **Recommended: strongest combo** | `-c 32 -o 32 -h 32 -w 32 -s 3` | 288 | **29.72×** | **1.26×** | ~5 min total |
| Upper end | `-c 32 -o 64 -h 32 -w 32 -s 3` | 288 | **29.71×** | **1.26×** | ~10 min total |

`-c 32 -o 32 -h 32 -w 32 -s 3` (the **Recommended** row) is the
single best shape for a combined dense-vs-SIMT + sparse-vs-dense
demo — big enough that both speedups show through clearly, small
enough that SIMT finishes in ~5 min.

**Avoid** `-c 64` or larger with SIMT — the C_in inner loop grows
linearly, SIMT wall-clock goes above 15 min quickly. Dense /
sparse TCU stay fast at any shape; use those directly if you want
to test very large C_in.

### Rules of thumb for picking shapes

- **Attention / flash — dense speedup (vs SIMT)**: pick `-n ≤ 64` so
  the SIMT binary still finishes in simx, and scale `-d` upward
  (128 → 512 → 1024) to raise the dense/SIMT ratio from ~1.7× to
  ~5.6×.
- **Attention / flash — sparse speedup (vs dense TCU)**: push `n`
  and `d` together so the softmax + PV anchors shrink as a fraction
  of total work. The practical shapes are `-n 64 -d 2048` (sparse
  /dense ≈ 1.23×, SIMT still tractable at ~1 min), `-n 128 -d 2048`
  (~1.29×, TCU-only), and `-n 256 -d 2048` (~1.35×, the strongest
  measured — also TCU-only). All three require you to run **only**
  `akili_*_tcu` and `akili_*_tcu_sp`, not the SIMT binary.
- **CNN — dense speedup (vs SIMT)**: any non-mnist shape gives
  28-30×; pick one that finishes in reasonable SIMT wall-clock
  (avoid `C_in ≥ 64`).
- **CNN — sparse speedup (vs dense TCU)**: pick `C_in × K × K ≥ 144`
  (at `K=3` that means `C_in ≥ 16`). Every shape above the
  threshold gets the full ~1.25× sparse benefit; mnist's
  `K_gemm = 16` is below the threshold and sparse loses to dense
  there. Unlike attention/flash, CNN hits both speedups at the
  same shape — `-c 32 -o 32 -h 32 -w 32 -s 3` is the single-shape
  sweet spot for a combined demo.
- **NT sweep**: all the numbers in Section 5 were at NT=8.
  Rebuild simx at NT=4/16/32 if you want to sweep warp width;
  the ratios stay in the same ballpark.

---

## 8. Where to look next

| File | Purpose |
|---|---|
| `tests/bench_dir/bench/00_doc/speedup_results_akili.csv` | Machine-readable version of Section 5's speedup tables (12 rows, one per (bench_group, shape_label)). |
| `tests/bench_dir/bench/00_doc/sweet_spot_sweep.csv` | Raw per-shape data for the Section 7 "Best configurations" sweep (17 rows: 6 attention + 6 flash + 5 cnn). Columns include cycles + ratios + combined score. |
| `tests/bench_dir/bench/00_doc/benchmark_status.md`       | Per-benchmark implementation history, build-config tables, and an end-to-end regression sanity-check script. |
| `tests/regression/sgemm/`                                | Reference SIMT GEMM test — the akili_cnn / akili_attn dispatch patterns are modeled on it. |
| `tests/regression/sgemm_tcu/`                            | Reference dense TCU GEMM. The akili_*_tcu TCU kernels are direct copies. |
| `tests/regression/sgemm_tcu_sp/`                         | Reference sparse TCU GEMM. The akili_*_tcu_sp kernels are direct copies. |

---

## 9. NeRF — full-pipeline benchmark (SIMT / dense TCU / sparse TCU)

Three benchmark directories implement a **full NeRF forward pass** in
one Vortex binary each. Unlike the attention/flash/CNN suite, which
compares one GEMM-heavy workload across three hardware variants, NeRF
exercises a realistic end-to-end pipeline: ray-AABB intersection →
stratified sampling → positional encoding → tiny-NeRF MLP
(4 hidden × 64) → softplus/sigmoid → alpha compositing.

Only the **4 MLP layer GEMMs** move to the TCU. Ray setup + positional
encoding + activations + compositing all stay on SIMT in all three
variants — so per-stage cycle comparisons isolate the tensor-core
effect on the GEMM portion cleanly.

| Directory | Hardware | MLP runtime (Y = W·X) |
|---|---|---|
| `tests/bench_dir/bench/benchmarks/akili_NeRF/`        | **SIMT**       | fp32, per-task GEMVs inlined into one kernel body |
| `tests/bench_dir/bench/benchmarks/akili_NeRF_tcu/`    | **Dense TCU**  | fp16 inputs, fp32 accumulators, `wmma_context<NT, fp16, fp32, false>`. 4 separate TCU spawns (one per layer). |
| `tests/bench_dir/bench/benchmarks/akili_NeRF_tcu_sp/` | **Sparse TCU** | 2:4-pruned weights, `wmma_context<NT, fp16, fp32, true>`. Host pre-prunes + compresses + packs metadata for each of the 4 weight matrices. |

### 9.1 Tiny NeRF architecture (used by all three variants)

| Knob | Value |
|---|---|
| Input encoding | Positional encoding (Mildenhall 2020), L=4 frequency bands → `3 + 3·2·4 = 27` features, padded to 32 for TCU tile alignment |
| Hidden layers | 4 |
| Hidden width (`MLP_W`) | **64** (multiple of `tileM=tileN=8`, `tileK=16` dense / 8 sparse) |
| Output head | 4 logits (1 σ + 3 RGB), padded to 8 for the final TCU layer |
| Activations | fp32 intermediate, ReLU after layers 0-2, softplus(σ) + sigmoid(rgb) after layer 3 |

### 9.2 CLI flags

All three binaries share the same interface:

| Flag | Meaning | Default |
|---|---|---|
| `-r <N>` | Number of rays | 32 |
| `-s <N>` | Samples per ray | 8 |
| `-k <path>` | Kernel binary | `kernel.vxbin` |

The total point count `n_points = -r × -s` must be a multiple of
`tileN = 8`. The default smoke shape (32×8 = 256) and the medium
shape (64×16 = 1024) both satisfy this.

### 9.3 Expected cycle output

Each binary prints the per-stage tags plus a summary line:

```
KCYC[SETUP,nt=8]:        <ray setup + PE>
KCYC[MLP_GEMM,nt=8]:     <sum of the 4 layer GEMMs>   (TCU variants only)
KCYC[MLP,nt=8]:          <single fused SIMT MLP>      (SIMT variant only)
KCYC[MLP_ACT,nt=8]:      <sum of SIMT act+cast spawns> (TCU variants only)
KCYC[COMP,nt=8]:         <alpha compositing>
KCYC[SUMMARY,nt=8]:      non_gemm=... gemm=... total=... (gemm_frac=...)
PASSED!
```

For the TCU variants, cycles are grouped as:
`non_gemm = SETUP + MLP_ACT + COMP`, `gemm = MLP_GEMM`.

### 9.4 Measured NeRF cycles (NT=8, simx, KMU launch API)

> **KMU-era measurement** — see the note at the top of Section 5.
> Numbers below are per-CTA slowest-block cycles per stage, summed
> across stages. Not comparable to any pre-KMU historical numbers.

#### Smoke: `-r 32 -s 8` (n_points = 256)

| Stage                      |        SIMT  |   Dense TCU |  Sparse TCU |
|----------------------------|-------------:|------------:|------------:|
| SETUP (ray + PE)           |        8,940 |   2,622,387 |   2,599,222 |
| MLP / MLP_GEMM             |    4,911,858 |      20,765 |      17,533 |
| MLP_ACT (act + cast)       |            — |   4,918,481*|   4,798,089*|
| COMP (alpha compositing)   |      744,100 |     742,558 |     746,601 |
| **total** (`KCYC[SUMMARY]`)|  **5,664,898** | **3,847,782** | **3,831,630** |

*Note: MLP_ACT in the TCU variants is dominated by the final softplus/sigmoid
stage (~3.3M of the 4.9M cycles), which runs on SIMT in all variants.*

> The SIMT variant has a tiny `SETUP` because it only does the slab
> test + stratified sampling (PE is folded into the per-point MLP
> kernel). The TCU variants have a **huge** `SETUP` because PE moved
> out of the MLP kernel — the fp16 input to the first TCU layer has
> to be prepared on SIMT beforehand, and that's where the sin/cos
> work now lives. This is one of the reasons the TCU end-to-end
> speedup is Amdahl-limited to ~1.47× despite the ~236× speedup on
> the raw MLP GEMMs.

### 9.5 Speedups

| Ratio | Smoke (`-r 32 -s 8`) |
|---|---:|
| **MLP GEMM speedup: dense TCU / SIMT** (`SIMT_MLP / dense_MLP_GEMM`) | **236.5×** |
| **MLP GEMM speedup: sparse TCU / dense TCU** | **1.18×** |
| End-to-end speedup: dense TCU / SIMT (total) | 1.472× |
| End-to-end speedup: sparse TCU / dense TCU (total) | 1.004× |

**How to read these numbers:**

- **Dense TCU gives a dramatic ~27× speedup on the MLP GEMM portion**
  — this is the headline result. The TCU turns the 4-layer fp32
  scalar MLP into a handful of WMMA tile operations, and the rest of
  the pipeline (PE, activations, compositing) continues to run on the
  same SIMT kernels.

- **End-to-end speedup is only ~1.33×-1.36×** because once the GEMMs
  are accelerated, the non-GEMM work (PE computation + softplus +
  sigmoid + compositing) becomes the dominant cost. This is classic
  Amdahl's law: a 27× speedup on 98% of the runtime gives an
  end-to-end ceiling of ~1 / (1 - 0.98 + 0.98/27) ≈ 25×, but the
  pipeline isn't purely GEMM — roughly half of the remaining work is
  transcendental math (sin/cos in PE, exp in softplus/sigmoid/alpha).

- **Sparse TCU over dense TCU is only ~1.02× on the MLP GEMMs** at
  these shapes. The MLP's K dimension is 32 (layer 0) or 64 (layers
  1-3), which is **below the K ≈ 256 threshold** where 2:4 structured
  sparsity starts to beat dense on fp16. Per the FPGA sweep in
  `MEMORY.md > "FPGA Sparse vs Dense Performance"`, fp16 sparse
  peaks at 1.86× around `K=1024` and crosses 1.0× near `K=256`. Below
  that, the per-tile metadata load overhead cancels the compute
  savings. A larger tiny-NeRF variant with `MLP_W=256` would show a
  clearer sparse win but take 10-20× longer in simx.

### 9.6 Recommended shapes for NeRF

| Purpose | Shape | Rationale | Wall-clock |
|---|---|---|---|
| Smoke test | `-r 32 -s 8` | Dense TCU PASSES at the smallest valid shape (256 points = 32 × tileN). ~1-2 min per variant. | ~1-2 min |
| End-to-end story | `-r 64 -s 16` | 4× the point count; the SIMT vs dense-TCU speedup ratio stays at ~27×, confirming it's not an artifact of the smallest shape. | ~5-7 min per variant |
| Going bigger | `-r 128 -s 32` | 4096 points; still passes, useful for FPGA runs but too slow for SIMT comparison in simx (>30 min). | — |

### 9.7 Rules of thumb for NeRF on TCU

- **The MLP GEMM speedup is the metric that matters for TCU
  comparison**. Report `SIMT_MLP / TCU_MLP_GEMM` if you want the
  "what does the tensor core buy us" number.
- **End-to-end total** is useful for wall-clock comparisons but will
  be pulled down by the PE + softplus/sigmoid cost, which is the
  same cost in all three variants.
- **Sparse wins want big K**. At `MLP_W=64` sparse/dense is ≈ 1.0×.
  Raise `MLP_W` to 128+ (edit `common.h` in all three directories)
  if you need the sparse path to show a visible speedup. The
  corresponding SIMT baseline re-run will be ~4× slower.
