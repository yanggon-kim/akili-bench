# Vortex Sparse TCU — Benchmark Implementation Status

Scope: three benchmark groups (`attention`, `flash`, `cnn_group`) ×
three execution modes (SIMT, dense TCU, sparse TCU 2:4).

As of the latest reorganization, **each (group × mode) pair has its
own self-contained benchmark directory** under
`tests/bench_dir/bench/benchmarks/`, following the
`tests/regression/sgemm*` pattern where `sgemm/`, `sgemm_tcu/`, and
`sgemm_tcu_sp/` are three independent directories. Each new directory
contains its own `main.cpp`, `kernel.cpp`, `common.h`, and `Makefile`
and builds a binary that runs only that one execution mode — no `-t`
dispatch, no cross-mode linkage.

All measurements in this document are from `simx` (cycle-approximate
C++ simulator). `rtlsim` runs at the same shapes take orders of
magnitude longer and were not used for this sweep set.

---

## 1. Current benchmark directory layout  (9 dirs)

```
tests/bench_dir/bench/benchmarks/
├── akili_attn/              ← attention, SIMT
├── akili_attn_tcu/          ← attention, dense TCU
├── akili_attn_tcu_sp/       ← attention, sparse TCU (2:4 on Q and P)
├── akili_flash/             ← flash, SIMT (fused for d≤16, unfused otherwise)
├── akili_flash_tcu/         ← flash, dense TCU
├── akili_flash_tcu_sp/      ← flash, sparse TCU (2:4 on Q and P)
├── akili_cnn/               ← cnn_group, SIMT conv2d
├── akili_acccnn_tcu/        ← cnn_group, dense TCU conv2d (im2col + GEMM)
└── akili_acccnn_tcu_sp/     ← cnn_group, sparse TCU conv2d
```

Each directory contains:

| File | Purpose |
|---|---|
| `common.h`  | `kernel_arg_t` definition, kernel stage IDs |
| `kernel.cpp` | Device-side kernel bodies for this single mode |
| `main.cpp` | Host-side harness: parse CLI, generate + upload data, dispatch, verify, print cycles |
| `Makefile` | Build rules. `PROJECT := <dir_name>`. CONFIGS reflect what the build needs (`ENABLE_TCU`, softfloat, etc.) |

### Historical (combined-binary) directories — preserved as-is

The original combined benchmarks are still in
`benchmarks/attention/`, `benchmarks/flash/`, and `benchmarks/conv_tcu/`.
They contain the full `v1 → v2 → v3 → v4` evolution and dispatch
between modes via a `-t <0|1|2>` flag. Nothing there is modified or
removed — they remain for history and cross-checking.

---

## 2. How to build and run each benchmark

### Build state assumed

```bash
cd vortex_sparse_tc/build
source ./ci/toolchain_env.sh               # set TOOLDIR + PATH

# Pick a warp width (4, 8, 16, 32) and build simx + runtime for it:
/tmp/rebuild_nt.sh 8                        # or whatever NT you want
```

All runs below require:

```bash
export VORTEX_DRIVER=simx
export LD_LIBRARY_PATH=<absolute path to vortex_sparse_tc/build>/runtime
```

### Attention trio

```bash
# SIMT attention
make -C tests/bench_dir/bench/benchmarks/akili_attn -s NUM_THREADS=8
cd tests/bench_dir/bench/benchmarks/akili_attn
./akili_attn -n 16 -d 16

# Dense TCU attention
make -C tests/bench_dir/bench/benchmarks/akili_attn_tcu -s NUM_THREADS=8
cd tests/bench_dir/bench/benchmarks/akili_attn_tcu
./akili_attn_tcu -n 16 -d 16

# Sparse TCU attention (2:4 on Q and P)
make -C tests/bench_dir/bench/benchmarks/akili_attn_tcu_sp -s NUM_THREADS=8
cd tests/bench_dir/bench/benchmarks/akili_attn_tcu_sp
./akili_attn_tcu_sp -n 16 -d 16
```

Attention CLI flags (identical in all 3 binaries):

| Flag | Meaning | Default |
|---|---|---|
| `-n N` | Sequence length (auto-padded to `max(TM, TN, TK)` in TCU modes) | 16 |
| `-d D` | Head dimension (auto-padded to `max(TK, TN)` in TCU modes) | 16 |
| `-k file` | Override `kernel.vxbin` path | `kernel.vxbin` |

Each prints per-stage `KCYC[{QK,SM,PV},nt=<NT>]: <cycles>` and a
final `PASSED!`/`FAILED!`.

### Flash trio

```bash
# SIMT flash (fused online-softmax for d∈{1,2,4,8,16}, unfused otherwise)
make -C tests/bench_dir/bench/benchmarks/akili_flash -s NUM_THREADS=8
cd tests/bench_dir/bench/benchmarks/akili_flash
./akili_flash -n 16 -d 16

# Dense TCU flash
make -C tests/bench_dir/bench/benchmarks/akili_flash_tcu -s NUM_THREADS=8
cd tests/bench_dir/bench/benchmarks/akili_flash_tcu
./akili_flash_tcu -n 16 -d 16

# Sparse TCU flash
make -C tests/bench_dir/bench/benchmarks/akili_flash_tcu_sp -s NUM_THREADS=8
cd tests/bench_dir/bench/benchmarks/akili_flash_tcu_sp
./akili_flash_tcu_sp -n 16 -d 16
```

Same `-n` / `-d` flags as attention. For `akili_flash` the fused
path is picked automatically when `d ∈ {1,2,4,8,16}`; `akili_flash_tcu`
and `akili_flash_tcu_sp` are identical to their attention
counterparts (3 stage kernel launches). Flash's fused SIMT prints
`KCYC[FUSED,...]` (single launch); all other modes print
per-stage `QK` / `SM` / `PV` cycles.

### CNN (cnn_group) trio — shape-configurable conv2d

```bash
# SIMT conv2d
make -C tests/bench_dir/bench/benchmarks/akili_cnn -s NUM_THREADS=8
cd tests/bench_dir/bench/benchmarks/akili_cnn
./akili_cnn -c 16 -o 16 -h 32 -w 32 -s 3

# Dense TCU conv2d (im2col + sgemm_tcu kernel)
make -C tests/bench_dir/bench/benchmarks/akili_acccnn_tcu -s NUM_THREADS=8
cd tests/bench_dir/bench/benchmarks/akili_acccnn_tcu
./akili_acccnn_tcu -c 16 -o 16 -h 32 -w 32 -s 3

# Sparse TCU conv2d (2:4 on the flattened weight matrix)
make -C tests/bench_dir/bench/benchmarks/akili_acccnn_tcu_sp -s NUM_THREADS=8
cd tests/bench_dir/bench/benchmarks/akili_acccnn_tcu_sp
./akili_acccnn_tcu_sp -c 16 -o 16 -h 32 -w 32 -s 3
```

CNN CLI flags (identical in all 3 binaries):

| Flag | Meaning | Default |
|---|---|---|
| `-c C_in`   | Input channels  | 1 |
| `-o C_out`  | Output channels | 8 |
| `-h H`      | Input height    | 28 |
| `-w W`      | Input width     | 28 |
| `-s K_size` | Kernel spatial size (3 for 3×3) | 3 |
| `-k file`   | Override `kernel.vxbin` path    | `kernel.vxbin` |

stride=1, padding=0 (valid). Output: `(C_out, H-K+1, W-K+1)`.
Prints `KCYC[CONV,nt=<NT>]: <cycles>` and `PASSED!`/`FAILED!`.

---

## 3. Vortex build configuration

All NT rebuilds go through `/tmp/rebuild_nt.sh <NT>`:

```bash
CONFIGS="-DNUM_THREADS=<NT> -DEXT_TCU_ENABLE -DTCU_SPARSE_ENABLE"
make -C sim/simx     clean && make -C sim/simx
make -C runtime/simx clean && make -C runtime/simx
```

The **simx build** is shared across all 9 benchmarks — one simx
binary serves SIMT, dense TCU, and sparse TCU modes, because
`TCU_SPARSE_ENABLE` includes both dense and sparse TCU support.

Per-benchmark CONFIGS vary so each binary only pulls in what it
actually needs:

| Dir | `NUM_THREADS` | `EXT_TCU_ENABLE` | softfloat link |
|---|---|---|---|
| `akili_attn`         | ✓ |   |   |
| `akili_attn_tcu`     | ✓ | ✓ |   |
| `akili_attn_tcu_sp`  | ✓ | ✓ | ✓ |
| `akili_flash`        | ✓ |   |   |
| `akili_flash_tcu`    | ✓ | ✓ |   |
| `akili_flash_tcu_sp` | ✓ | ✓ | ✓ |
| `akili_cnn`          | ✓ |   |   |
| `akili_acccnn_tcu`   | ✓ | ✓ |   |
| `akili_acccnn_tcu_sp`| ✓ | ✓ | ✓ |

`softfloat` is only required when the host calls
`vt::prune_2to4_matrix<fp16>` (which uses `rv_htof_s` internally),
so only the three sparse TCU variants pull it in.

Canonical compile-time macros (from earlier versions; unchanged):

| Macro | Purpose |
|---|---|
| `NUM_THREADS` | Warp width. Baselines at 8. |
| `EXT_TCU_ENABLE` | Enables the TCU ISA extension in simx/runtime. Required for dense & sparse TCU paths. |
| `TCU_SPARSE_ENABLE` | Enables the 2:4 sparse path in simx's TCU model + VX_tcu_meta SRAM. |
| `ENABLE_TCU` | C++ guard (Makefile-defined) that gates TCU kernel bodies in device source. Implied by presence of TCU kernels in the dir. |
| `NUM_TCU_LANES` | Host template parameter for `wmma_context<NT, ...>`. Set equal to `NUM_THREADS`. |

---

## 4. Implementation history (by benchmark group)

### attention  (currently: `akili_attn` / `akili_attn_tcu` / `akili_attn_tcu_sp` — v3 behavior)

| Version | What changed | Headline perf (N=64, d=1024, NT=8) |
|---|---|---|
| **v1** | Initial port: `grid_dim[0]=1` single-block TCU kernel with local-memory staging, per-element `f2h` on device, sparse PV runs the dense kernel. | dense/SIMT = **0.098×** (TCU 10× slower than SIMT) |
| **v2** | TCU kernels rewritten as direct copies of `tests/regression/sgemm_tcu/kernel.cpp`. Multi-block `grid_dim=(N/TN, N/TM)`, direct global→fragment loads, no local-memory staging. Host packs fp16 Q/K/V with K/V in col-major, converts P to fp16 after softmax. | dense/SIMT = **5.64×** (56× improvement vs v1); sparse/dense = 1.11× |
| **v3** (current) | New `kernel4_sparse_body` added — mirrors `kernel3_sparse_body` but with PV dims. Host prunes P to 2:4 between softmax and PV, uploads compressed P + metadata. `common.h` already had `meta_P_addr` reserved. | dense/SIMT = 5.64× (unchanged); sparse/dense = **1.14× at N=64, up to 1.35× at N=256, d=2048** |

### flash  (currently: `akili_flash` / `akili_flash_tcu` / `akili_flash_tcu_sp` — v3 behavior)

| Version | What changed | Headline perf (N=64, d=1024, NT=8) |
|---|---|---|
| **v1** | Copied attention's unfused 3-stage SIMT + TCU kernels. Original fused `flash_kernel_body` preserved but unused. | dense/SIMT = 0.098× |
| **v1.5** (user-asked restore) | Fused SIMT path brought back via `KID_FLASH_FUSED=7`. For `d ∈ {1,2,4,8,16}` host picks `flash_kernel_body<HEAD_DIM,BLOCK_SIZE_C>` (single launch, online softmax). For larger `d` falls through to unfused 3-stage. | At d=16 fused SIMT ≈ 1.01 M cycles vs 727 k unfused (per-row online softmax serialization cost is real) |
| **v2** | Same sgemm_tcu refactor as attention for `flash_qk_tcu` / `flash_pv_tcu`. Host fp16 packing + K/V col-major transpose. | dense/SIMT = **5.77×**; sparse/dense = 1.10× |
| **v3** (current) | `flash_pv_sparse` added; `flash/common.h` gained `meta_P_addr`; host-side P pruning identical to attention. | dense/SIMT = 5.77×; sparse/dense = **1.17× at N=64**, scales up with N |

### cnn_group  (currently: `akili_cnn` / `akili_acccnn_tcu` / `akili_acccnn_tcu_sp` — v4)

| Version | What it was |
|---|---|
| **v1 – v3** | Proxy: SIMT = real `acccnn` Fashion-MNIST pipeline; dense/sparse TCU = attention QK-stage cycles copied over. Honest, but not a real CNN conv measurement. |
| **v4** (current) | **Real conv2d via im2col + sgemm_tcu-style GEMM.** Host: `im2col` → `Icol [N_gemm × K_gemm]` fp16 col-major, `W_gemm [M_gemm × K_gemm]` fp16. For sparse: 2:4 prune on `W_gemm` + metadata via `pack_metadata`. Device: direct `sgemm_tcu` / `sgemm_tcu_sp` tile loop. Shape is CLI-configurable. |

---

## 5. Summary speedups (NT=8)

All numbers below come from the current `akili_*` trio sweep,
recorded in
`tests/bench_dir/bench/00_doc/speedup_results_akili.csv`. Twelve
rows total: 3 benchmark groups × 4 shapes. Simx, NT=8, fp16 inputs,
fp32 accumulator, one thread per output for SIMT, `sgemm_tcu`-style
multi-block tile dispatch for dense/sparse TCU.

### Attention

| Shape | `-n N` | `-d D` | SIMT (cyc) | Dense TCU | Sparse TCU | **dense/SIMT** | **sparse/dense** |
|---|---|---|---|---|---|---|---|
| S  | 16 | 16  |    718,369 |    706,251 |    726,927 | 1.02× | 0.97× |
| M  | 32 | 32  |  1,841,797 |  1,726,860 |  1,907,601 | 1.07× | 0.91× |
| L  | 64 | 128 |  6,016,719 |  3,556,948 |  3,766,620 | **1.69×** | 0.94× |
| XL | 64 | 512 | 12,858,096 |  3,171,153 |  3,003,852 | **4.06×** | **1.06×** |

### Flash

| Shape | `-n N` | `-d D` | SIMT (cyc) | Dense TCU | Sparse TCU | **dense/SIMT** | **sparse/dense** |
|---|---|---|---|---|---|---|---|
| S  | 16 | 16  | 1,020,258 (fused) | 706,251 | 726,927 | **1.45×** | 0.97× |
| M  | 32 | 32  |  1,833,746 |  1,726,860 |  1,907,601 | 1.06× | 0.91× |
| L  | 64 | 128 |  5,965,485 |  3,556,948 |  3,766,620 | **1.68×** | 0.94× |
| XL | 64 | 512 | 12,691,267 |  3,171,153 |  3,003,852 | **4.00×** | **1.06×** |

At `d = 16` `akili_flash` uses the fused online-softmax single-launch
kernel (~1.02 M cycles — slower in absolute terms than the unfused
3-stage path because each row is processed by a single thread). For
`d ≥ 32` flash falls through to the same unfused 3-stage SIMT path
as attention, and its numbers match attention within noise at every
shape. The dense / sparse TCU binaries run identical kernels between
attention and flash, so their cycle counts are also identical.

### CNN / conv2d

| Shape  | `-c C_in` | `-o C_out` | `-h H` | `-s K` | GEMM (M × N × K) | SIMT (cyc) | Dense TCU | Sparse TCU | **dense/SIMT** | **sparse/dense** |
|---|---|---|---|---|---|---|---|---|---|---|
| mnist  | 1  | 8  | 28 | 3 |    8 × 680 × 16 |       910,563 |      81,464 |      91,374 | **11.18×** | 0.89× |
| small  | 16 | 16 | 32 | 3 |  16 × 904 × 144 |    18,112,160 |     632,602 |     544,989 | **28.63×** | **1.16×** |
| medium | 32 | 32 | 32 | 3 |  32 × 904 × 288 |    67,898,024 |   2,284,659 |   1,816,719 | **29.72×** | **1.26×** |
| large  | 32 | 64 | 32 | 3 |  64 × 904 × 288 |   135,665,614 |   4,566,969 |   3,621,118 | **29.71×** | **1.26×** |

CNN shows the strongest ratios because conv is a pure GEMM with no
softmax anchor. At `mnist` the GEMM's `K_gemm = 16` is only one
tileK iteration, so the sparse metadata overhead exceeds the 2×
compression savings and sparse is slightly slower than dense there.
Every shape with `K_gemm ≥ 144` gets the expected 1.16-1.26×
sparse bonus and 28-30× dense-over-SIMT speedup.

---

## 6. Verification sanity checks (run after any edit)

```bash
cd vortex_sparse_tc/build
/tmp/rebuild_nt.sh 8
export VORTEX_DRIVER=simx LD_LIBRARY_PATH=$PWD/runtime

BENCH=tests/bench_dir/bench/benchmarks
for d in akili_attn akili_attn_tcu akili_attn_tcu_sp \
         akili_flash akili_flash_tcu akili_flash_tcu_sp; do
  make -C $BENCH/$d -s NUM_THREADS=8
  (cd $BENCH/$d && ./$d -n 16 -d 16) | grep -E 'PASSED|FAILED|KCYC'
done
for d in akili_cnn akili_acccnn_tcu akili_acccnn_tcu_sp; do
  make -C $BENCH/$d -s NUM_THREADS=8
  (cd $BENCH/$d && ./$d -c 16 -o 16 -h 32 -w 32 -s 3) | grep -E 'PASSED|FAILED|KCYC'
done
```

Every invocation must print `PASSED!`. Cycles will match the
"Summary speedups" table at the baseline shapes within noise.

---

## 7. Recommended input sizes

Not every shape makes sense to run. These recommendations pair each
benchmark group with shapes that actually finish in reasonable simx
wall-clock time **and** show the dense / sparse TCU speedups most
clearly. 1 M device cycles ≈ 1 s of simx wall-clock on a modern host.

### Attention and flash (`akili_attn*`, `akili_flash*`)

| Purpose | `-n N` | `-d D` | dense/SIMT | sparse/dense | ~Wall-clock (NT=8) |
|---|---|---|---|---|---|
| Smoke test          | 16 | 16  | 1.02× | 0.97× | < 5 s total |
| Warm-up / demo      | 32 | 32  | 1.07× | 0.91× | ~10 s total |
| **Real dense demo** | 64 | 128 | **1.69×** | 0.94× | ~30 s total |
| **Strong dense win**| 64 | 512 | **4.06×** | **1.06×** | ~1 min total |
| Max realistic in simx | 64 | 1024 | ~5.6× (from earlier v3 sweep) | ~1.14× | ~2 min total |

Rules:
- Keep `-n ≤ 64` when running the SIMT binary in simx — at `-n ≥ 128`
  the `N × N × d` grid blows up SIMT wall-clock above 15 min.
- Scale `-d` to grow the dense/SIMT ratio (`d = 128 → 512 → 1024`
  takes it from 1.7× → 4× → 5.6×).
- Sparse/dense stays below 1.0× until `-d ≥ 512`. The softmax and
  PV's small `K = n_attn` anchor it at smaller shapes. If you need
  sparse/dense > 1.2×, run **only** the TCU binaries
  (`akili_*_tcu` and `akili_*_tcu_sp`) at large `n` (128 or 256)
  and skip the SIMT comparison entirely — the v3 sweep hit sparse
  /dense ≈ 1.35× at `n=256, d=2048`.
- All three modes use the same simx build, so one `rebuild_nt.sh N`
  call configures everything.

### CNN (`akili_cnn`, `akili_acccnn_tcu`, `akili_acccnn_tcu_sp`)

| Purpose | `-c C_in` | `-o C_out` | `-h H` | `-s K` | GEMM K_gemm | dense/SIMT | sparse/dense | ~Wall-clock |
|---|---|---|---|---|---|---|---|---|
| Smoke test (mnist shape) | 1  | 8  | 28 | 3 |  16 | 11.2× | 0.89× | ~5 s total |
| **Small demo** | 16 | 16 | 32 | 3 | 144 | **28.6×** | **1.16×** | ~1-2 min total |
| **Recommended** | 32 | 32 | 32 | 3 | 288 | **29.7×** | **1.26×** | ~5 min total |
| Upper end | 32 | 64 | 32 | 3 | 288 | **29.7×** | **1.26×** | ~10 min total |

Rules:
- **`C_in × K × K ≥ 144`** is the break-even where sparse starts
  beating dense. At `K = 3` that means `C_in ≥ 16`. Below that
  (Fashion-MNIST mnist shape in particular) the sparse metadata
  overhead dominates and sparse is slightly slower than dense.
- Dense-over-SIMT stays in the 28-30× range for any non-mnist
  shape. The ratio is not sensitive to `C_out` or spatial dims.
- Avoid `C_in ≥ 64` with the SIMT binary — SIMT wall-clock grows
  linearly with `C_in`, above 15 min quickly. The TCU binaries
  handle bigger `C_in` comfortably.

### NT (warp width) sweep

All numbers in Section 5 are at NT=8. Other warp widths (4, 16, 32)
are fully supported — just rerun
`CONFIGS="-DNUM_THREADS=<NT> -DEXT_TCU_ENABLE -DTCU_SPARSE_ENABLE" make -C sim/simx clean && make -C sim/simx -s`
and rebuild all nine akili_* dirs with `NUM_THREADS=<NT>` to match.
Ratios stay in the same ballpark at all NTs.

---

## 8. Docs in this directory

The `tests/bench_dir/bench/00_doc/` dir contains exactly three files:

| File | Purpose |
|---|---|
| `akili_benchmarks.md`          | User-facing "how to build + run" guide. Start here if you just want to use the benchmarks. |
| `benchmark_status.md`          | This file — implementation history, per-benchmark code layout, build configuration, summary speedups, and recommended input sizes. |
| `speedup_results_akili.csv`    | Machine-readable version of Section 5's speedup tables (12 rows, one per (bench_group, shape_label)). |
