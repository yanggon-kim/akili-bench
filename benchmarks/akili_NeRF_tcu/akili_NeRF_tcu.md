# akili_NeRF_tcu — dense TCU full-NeRF forward pass

This is the **dense TCU** variant of `akili_NeRF`. The benchmark runs
the entire NeRF forward pipeline — ray-AABB intersection → stratified
sampling → positional encoding → 4-layer tiny-NeRF MLP →
softplus/sigmoid → alpha compositing — with **only the MLP's 4 layer
GEMMs moved onto the Vortex tensor core**. Ray setup, activations and
compositing all stay on SIMT, identical to the SIMT baseline
`akili_NeRF`.

## Source files

| File | Role |
|---|---|
| `common.h` | Shared `kernel_arg_t` struct + compile-time MLP sizes (`MLP_W=64`, `PE_L=4`, `MLP_IN_DIM=27` padded to 32, `MLP_OUT_DIM=4` padded to 8). |
| `kernel.cpp` | Five kernels: `kernel_ray_setup` (SIMT, one task per point, writes PE features into `[32 × n_points]` fp16), `kernel_mlp_gemm` (dense TCU, 2-D grid over `(n_points/tileN, N_out/tileM)`), `kernel_mlp_act` (SIMT ReLU + bias + fp32→fp16 cast), `kernel_mlp_out_act` (SIMT softplus + sigmoid for the final layer), `kernel_composite` (SIMT alpha compositing). |
| `main.cpp` | Host driver: deterministic weight init with `srand(42)`, fp16 packing via `f2h_host`, per-layer GEMM dispatch, CPU reference verify at `atol=rtol=1e-2`. |
| `Makefile` | `CONFIGS = -DNUM_THREADS=... -DNUM_TCU_LANES=... -DEXT_TCU_ENABLE`. No softfloat needed. |

## GEMM formulation

Each MLP layer computes `Y = W · X` where:

- **A = W**, shape `[N_out × K_in]` fp16 row-major, **row-major load**
- **B = X**, shape `[K_in × n_points]` fp16 row-major (activations live
  feature-major throughout the pipeline), **row-major load**
- **C = Y**, shape `[N_out × n_points]` fp32 row-major, stored by
  `store_matrix_sync` with stride `n_points`

Different from `akili_attn_tcu` which loads K col-major to transpose
it — here the intermediate activation layout already matches what B
needs, so both fragA and fragB use the default row-major load. This
also positions the weight matrix (A) for sparsity in the `_sp` variant.

Grid per layer: `grid_dim = (n_points / tileN, N_out / tileM)`,
`block_dim = (NUM_THREADS, 1)`. Layer 0: `N_out=64, K_in=32`; layers
1-2: `N_out=64, K_in=64`; layer 3 (output head): `N_out=8, K_in=64`
(rows 0-3 are real outputs, 4-7 padded to match `tileM=8`).

## Dispatch order (10 spawns per run)

1. SIMT `kernel_ray_setup` → PE features + deltas
2. TCU `kernel_mlp_gemm` (layer 0) → fp32 scratch
3. SIMT `kernel_mlp_act` (layer 0) → ReLU + bias + fp16 cast
4. TCU `kernel_mlp_gemm` (layer 1) → fp32 scratch
5. SIMT `kernel_mlp_act` (layer 1)
6. TCU `kernel_mlp_gemm` (layer 2) → fp32 scratch
7. SIMT `kernel_mlp_act` (layer 2)
8. TCU `kernel_mlp_gemm` (layer 3, output head) → fp32 out_head
9. SIMT `kernel_mlp_out_act` → softplus(σ) + sigmoid(rgb) → sigmas/rgbs
10. SIMT `kernel_composite` → per-ray image RGB

Each spawn is bracketed by `vx_rdcycle()` via the device `main()`
switch (same pattern as `akili_attn_tcu/kernel.cpp:101-126`). The
host aggregates all 4 TCU spawns into `KCYC[MLP_GEMM]` and all 4 SIMT
activation spawns into `KCYC[MLP_ACT]`.

## Build & run

```bash
cd tests/bench_dir/bench/benchmarks/akili_NeRF_tcu
make clean && make NUM_THREADS=8
make run-simx OPTS="-r 32 -s 8"      # smoke
make run-simx OPTS="-r 64 -s 16"     # medium
```

## Measured speedup vs SIMT baseline

At NT=8, the dense TCU gives **~27× speedup on the MLP GEMM portion**
compared to `akili_NeRF`'s scalar MLP kernel (`SIMT_MLP / dense_MLP_GEMM`).
End-to-end speedup is ~1.33×-1.36× because the PE + softplus + sigmoid
SIMT work becomes the dominant cost after TCU acceleration — classic
Amdahl's law on a tiny MLP. See `00_doc/akili_benchmarks.md` Section 9
for the full 3-way table.

## Related

- SIMT baseline: `benchmarks/akili_NeRF/` (fp32 MLP inline in one
  kernel body)
- Sparse variant: `benchmarks/akili_NeRF_tcu_sp/` (2:4 weight
  sparsity on the 4 layer weight matrices)
- Reference TCU pattern: `tests/bench_dir/bench/benchmarks/akili_attn_tcu/kernel.cpp`
