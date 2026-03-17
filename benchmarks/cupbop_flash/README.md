# CuPBoP Flash Attention Benchmark

Flash attention (from [tspeterkim/flash-attention-minimal](https://github.com/tspeterkim/flash-attention-minimal))
compiled through the **CuPBoP** CUDA-to-Vortex pipeline.

Unlike `benchmarks/flash/` which uses native Vortex APIs, this benchmark
compiles unmodified CUDA code through CuPBoP's kernel/host translators,
validating the full CUDA→LLVM IR→RISC-V translation pipeline.

## CUDA Features Exercised

- Dynamic shared memory (`extern __shared__ float sram[]`)
- `__syncthreads()` barriers
- 2D grid dispatch (`blockIdx.x/y`, `gridDim.y`)
- `expf()` math functions
- Online softmax (multi-tile flash attention algorithm)

## Test Cases

| Test | B | nh | N | d | Bc | Br | Tc | Tr | Grid | What it tests |
|------|---|-----|---|---|----|----|----|----|------|---------------|
| single_tile | 1 | 1 | 8 | 8 | 8 | 8 | 1 | 1 | (1,1) | Basic kernel correctness |
| multi_tile  | 1 | 1 | 16| 4 | 8 | 8 | 2 | 2 | (1,1) | Tiling / online softmax loop |
| multi_head  | 1 | 2 | 8 | 4 | 8 | 8 | 1 | 1 | (1,2) | 2D grid dispatch |

All tests verify GPU output against a CPU standard attention reference
(`softmax(QK^T / sqrt(d)) * V`).

## Build & Run

Requires the CuPBoP apptainer environment:

```bash
apptainer exec --nv \
  -B /projects/ci-runners/CuPBoP-Vortex/:/projects/ci-runners/CuPBoP-Vortex/ \
  /projects/ci-runners/CuPBoP-Vortex/tools/cupbop_env.sif \
  /bin/bash -c "
    cd /nethome/hkim358/project/CuPBoP_Vortex
    source ./ci/rg-ci-setup.sh 2>/dev/null
    export VORTEX_SCHEDULE_FLAG=2
    cd /nethome/hkim358/project/vortex-benchmarks/benchmarks/cupbop_flash
    make run-simx
  "
```

## Verification Results (2026-03-17)

All 3 tests **PASSED** on CuPBoP-Vortex (simx, NUM_WARPS=64):

- single_tile: max error 2.98e-08
- multi_tile:  max error 2.24e-08
- multi_head:  max error 3.73e-08

Verified against:
1. CPU standard attention (naive softmax(QK^T/√d)·V)
2. CPU flash attention simulation
3. Independent Python implementation
