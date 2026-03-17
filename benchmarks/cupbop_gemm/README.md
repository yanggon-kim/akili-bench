# CuPBoP Tiled GEMM Benchmark

Portable CUDA tiled GEMM from [ColfaxResearch/cutlass-kernels](https://github.com/ColfaxResearch/cutlass-kernels)
(`cutlass-gemm`), compiled through the CuPBoP CUDA-to-Vortex pipeline.

Replaces SM90 features (TMA, GMMA, CUTLASS CollectiveBuilder) with
shared memory tiling and FMA loops.

## Algorithm

C = alpha * A * B + beta * C

Tiles A and B into shared memory blocks of size TILE_M x TILE_K and
TILE_K x TILE_N, accumulates partial sums via FMA, and applies the
alpha/beta epilogue scaling — the same computation flow as the original
CUTLASS 3.0 GEMM, without tensor cores.

## Test Cases

| Test | M | N | K | alpha | beta | Grid | What it tests |
|------|---|---|---|-------|------|------|---------------|
| square_8x8 | 8 | 8 | 8 | 1.0 | 0.0 | (1,1) | Basic single-tile GEMM |
| multi_tile_16x16 | 16 | 16 | 16 | 1.0 | 0.0 | (2,2) | K-dimension tiling loop |
| rect_alpha_beta | 8 | 8 | 16 | 2.0 | 0.5 | (1,1) | Epilogue alpha/beta scaling |

## CUDA Features Exercised

- Static shared memory (`__shared__ float tile[]`)
- 2D thread blocks (`threadIdx.x/y`, `blockIdx.x/y`)
- `__syncthreads()` barriers

## Build & Run

```bash
apptainer exec --nv \
  -B /projects/ci-runners/CuPBoP-Vortex/:/projects/ci-runners/CuPBoP-Vortex/ \
  /projects/ci-runners/CuPBoP-Vortex/tools/cupbop_env.sif \
  /bin/bash -c "
    cd /nethome/hkim358/project/CuPBoP_Vortex
    source ./ci/rg-ci-setup.sh 2>/dev/null
    export VORTEX_SCHEDULE_FLAG=2
    cd /nethome/hkim358/project/vortex-benchmarks/benchmarks/cupbop_gemm
    make run-simx
  "
```

## Results (2026-03-17)

All 3 tests PASS on CuPBoP-Vortex simx (NUM_WARPS=64). Max error < 4e-05.
