# CuPBoP Flash Multi-Head Attention (FMHA) Benchmark

Portable CUDA FlashAttention-2 from [ColfaxResearch/cutlass-kernels](https://github.com/ColfaxResearch/cutlass-kernels)
(`fmha`), compiled through the CuPBoP CUDA-to-Vortex pipeline.

Replaces SM90 features (TMA, GMMA, cluster multicast, warp shuffle
softmax) with shared memory tiling, FMA dot products, and per-thread
online softmax.

## Algorithm

O = softmax(Q @ K^T / sqrt(d)) @ V

Implements FlashAttention-2's tiled forward pass with online softmax:
1. **GEMM-I:** Q_tile @ K_tile^T → S (attention scores)
2. **Online softmax:** row-max tracking, exp, row-sum accumulation
3. **GEMM-II:** P_tile @ V_tile → O (weighted output)
4. **Rescale:** Correct running softmax statistics across tiles

## Test Cases

| Test | B | nh | N | d | Tc | Tr | Grid | What it tests |
|------|---|-----|---|---|----|----|------|---------------|
| single_tile | 1 | 1 | 8 | 8 | 1 | 1 | (1,1) | Basic kernel correctness |
| multi_tile | 1 | 1 | 16 | 4 | 2 | 2 | (1,1) | Tiling / online softmax loop |
| multi_head | 1 | 2 | 8 | 4 | 1 | 1 | (1,2) | 2D grid dispatch |

## CUDA Features Exercised

- Dynamic shared memory (`extern __shared__ float sram[]`)
- `__syncthreads()` barriers
- 2D grid dispatch (`blockIdx.x/y`, `gridDim.y`)
- `expf()` math function

## Build & Run

```bash
apptainer exec --nv \
  -B /projects/ci-runners/CuPBoP-Vortex/:/projects/ci-runners/CuPBoP-Vortex/ \
  /projects/ci-runners/CuPBoP-Vortex/tools/cupbop_env.sif \
  /bin/bash -c "
    cd /nethome/hkim358/project/CuPBoP_Vortex
    source ./ci/rg-ci-setup.sh 2>/dev/null
    export VORTEX_SCHEDULE_FLAG=2
    cd /nethome/hkim358/project/vortex-benchmarks/benchmarks/cupbop_fmha
    make run-simx
  "
```

## Results (2026-03-17)

All 3 tests PASS on CuPBoP-Vortex simx (NUM_WARPS=64). Max error < 4e-08.
