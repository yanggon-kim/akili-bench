# akili_NeRF_tcu_sp — 2:4 sparse TCU full-NeRF forward pass

This is the **sparse TCU** variant of `akili_NeRF`. Same end-to-end
pipeline as `akili_NeRF_tcu` (ray-AABB → sampling → PE → 4-layer MLP
→ softplus/sigmoid → alpha compositing), but the **4 MLP weight
matrices run through the 2:4 structured sparsity pipeline**:

1. Host-side `prune_2to4_matrix<fp16>` selects 2 of every 4 fp16
   weights along the K dimension
2. `compress_2to4_matrix<fp16>` halves the weight storage and emits
   per-element masks
3. `pack_metadata` lays the masks out in the exact format the Vortex
   sparse TCU metadata SRAM expects

The device `kernel_mlp_gemm` then uses the 5-arg form of
`load_matrix_sync` (with a metadata pointer) and walks the A matrix
at **half stride** (`tileK/2`) per K-iteration.

## Source files

| File | Role |
|---|---|
| `common.h` | Same `kernel_arg_t` as the dense TCU variant plus `meta_cur_addr` for the current layer's packed metadata. |
| `kernel.cpp` | `kernel_mlp_gemm` replaces the dense load with the sparse 5-arg load and the `pMetaSp` walk. All other kernels (ray setup, activation, output activation, compositing) are byte-identical to the dense variant. |
| `main.cpp` | Host driver with a per-layer `prune_compress_layer` helper that runs `prune_2to4_matrix` → `compress_2to4_matrix` → `pack_metadata`. The **CPU reference uses the pruned fp32 weights** (fed back after pruning) so device vs CPU compare at `atol=rtol=1e-2`. |
| `Makefile` | `CONFIGS = -DNUM_THREADS=... -DNUM_TCU_LANES=... -DEXT_TCU_ENABLE` plus softfloat host-side linkage (needed by `prune_2to4_matrix`'s internal fp16 ops — matches `akili_attn_tcu_sp/Makefile`). |

## Sparse kernel diff vs dense variant

```cpp
// Dense:
for (int i = 0; i < (int)K_in; i += (int)tileK) {
  auto pTileA = pA + tile_row * K_in + i;
  auto pTileB = pB + i * n_pts + tile_col;
  tcu_ctx::load_matrix_sync(fragA, pTileA, K_in);        // 3-arg
  tcu_ctx::load_matrix_sync(fragB, pTileB, n_pts);
  tcu_ctx::mma_sync(fragC, fragA, fragB, fragC);
}

// Sparse:
auto pMetaSp = pMetaBase + blockIdx.y * num_k_tiles * per_k_tile_words;
auto pTileA = pA + tile_row * (K_in / 2);                // compressed stride
auto pTileB = pB + tile_col;
for (int i = 0; i < (int)K_in; i += (int)tileK) {
  sp_ctx::load_matrix_sync<row_major>(fragA, pTileA, K_in/2, nullptr, pMetaSp); // 5-arg
  sp_ctx::load_matrix_sync(fragB, pTileB, n_pts);
  sp_ctx::mma_sync(fragC, fragA, fragB, fragC);
  pMetaSp += per_k_tile_words;
  pTileA  += tileK / 2;                                  // half step
  pTileB  += tileK * n_pts;
}
```

This follows the `sparse_mma_loop` helper from
`akili_attn_tcu_sp/kernel.cpp:40-81` with one change: both fragA and
fragB are loaded row-major (the attention benchmark uses col-major on
fragB to transpose K implicitly, which NeRF doesn't need).

## Per-layer host preprocessing

For each of the 4 weight matrices in `main.cpp`:

```cpp
auto prune_compress_layer = [](std::vector<float>& h_W,           // updated in place
                               std::vector<uint16_t>& h_W_fp16,
                               std::vector<uint16_t>& h_Wc,
                               std::vector<uint32_t>& h_meta,
                               uint32_t M, uint32_t K) -> bool {
  // 1. fp32 -> fp16
  for (size_t i = 0; i < M*K; ++i) h_W_fp16[i] = f2h_host(h_W[i]);
  // 2. In-place 2:4 prune
  vt::prune_2to4_matrix<vt::fp16>(h_W_fp16.data(), M, K);
  // 3. Copy pruned values back to fp32 for CPU reference
  for (size_t i = 0; i < M*K; ++i) h_W[i] = h2f_host(h_W_fp16[i]);
  // 4. Compress to [M x K/2] + emit masks
  vt::compress_2to4_matrix<vt::fp16>(h_Wc.data(), h_W_fp16.data(), masks, M, K);
  // 5. Pack metadata into the RTL-specific format
  pack_metadata(h_meta, masks, M, K);
  return true;
};
```

The `pack_metadata` function is lifted verbatim from
`akili_attn_tcu_sp/main.cpp:42-107` — the formula is general and
works for any `(M, K)` shape.

## Build & run

```bash
cd tests/bench_dir/bench/benchmarks/akili_NeRF_tcu_sp
make clean && make NUM_THREADS=8
make run-simx OPTS="-r 32 -s 8"      # smoke
make run-simx OPTS="-r 64 -s 16"     # medium
```

## Measured speedup vs dense TCU

At NT=8 with `MLP_W=64`, sparse over dense on the MLP GEMMs is
**~1.02×** — essentially within noise. The reason is that the MLP's
K dimension is 32 (layer 0) or 64 (layers 1-3), which is well below
the `K ≈ 256` crossover at which 2:4 structured sparsity reliably
beats dense on fp16 (see `MEMORY.md > "FPGA Sparse vs Dense
Performance"`). For a visible sparse win, set `MLP_W=128` or larger
in `common.h` and re-run all three variants — the SIMT baseline will
take ~4× longer but the sparse ratio will climb toward the 1.3-1.5×
range observed in the attention/CNN sweeps.

The implementation **is correct and exercises the full sparse
pipeline**; the weak speedup is a function of the chosen MLP size,
not the sparse path.

## Related

- Dense TCU variant: `benchmarks/akili_NeRF_tcu/`
- SIMT baseline: `benchmarks/akili_NeRF/`
- Reference sparse TCU pattern: `tests/bench_dir/bench/benchmarks/akili_attn_tcu_sp/kernel.cpp`
- Host sparse pipeline: `kernel/include/tensor.h`
  (`prune_2to4_matrix`, `compress_2to4_matrix`)
