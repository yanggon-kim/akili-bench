#ifndef AKILI_WGMMA_DXA_LAYOUT_H
#define AKILI_WGMMA_DXA_LAYOUT_H

// Shared-memory stage layout of the DEEP_BUF WGMMA+DXA pipeline, used by both
// the kernel (wgmma_dxa_kernel.h) and the host (wgmma_dxa_host.h) so the
// launch's lmem_size always matches what the kernel indexes.
//
// PIPE_N stages, each holding KB consecutive K-tiles (CUTLASS BLOCK_K):
//   dense : A [cta_m][KB*tileK]      then B [xtileN][KB*tileK]  (K-major)
//   sparse: A [cta_m][KB*tileK/2]    then B [xtileN][KB*tileK]  (K-major),
//           B bank-row aligned; tileK is the sparse tile (tileK_sp under FULLK)

#include <stdint.h>

#ifndef PIPE_N
#define PIPE_N 2
#endif
#ifndef KB
#define KB 1
#endif

namespace akili {

constexpr uint32_t kWgPipeN = PIPE_N;
constexpr uint32_t kWgKB    = KB;
constexpr uint32_t kWgElem  = 2;  // fp16 inputs

constexpr uint32_t wg_round_up(uint32_t x, uint32_t a) {
  return ((x + a - 1) / a) * a;
}

constexpr uint32_t wg_bank_bytes() { return VX_CFG_NUM_THREADS * 4; }

constexpr uint32_t wg_dense_stage_bytes(uint32_t cta_m, uint32_t tile_k,
                                        uint32_t xtile_n) {
  return (cta_m + xtile_n) * kWgKB * tile_k * kWgElem;
}

constexpr uint32_t wg_sp_stage_b_off(uint32_t cta_m, uint32_t tile_k) {
  return wg_round_up(cta_m * (kWgKB * tile_k / 2) * kWgElem, wg_bank_bytes());
}

constexpr uint32_t wg_sp_stage_meta_off(uint32_t cta_m, uint32_t tile_k,
                                        uint32_t xtile_n) {
  return wg_sp_stage_b_off(cta_m, tile_k)
       + wg_round_up(kWgKB * tile_k * xtile_n * kWgElem, wg_bank_bytes());
}

// AKILI_META_DXA stages each stage's sparse metadata ([warps][KB tiles of
// NT words]) through DXA with A and B instead of a per-tile global TCU_LD.
constexpr uint32_t wg_sp_stage_meta_bytes(uint32_t cta_m, uint32_t xtile_m) {
#ifdef AKILI_META_DXA
  return wg_round_up((cta_m / xtile_m) * kWgKB * VX_CFG_NUM_THREADS * 4,
                     wg_bank_bytes());
#else
  (void)cta_m; (void)xtile_m;
  return 0;
#endif
}

constexpr uint32_t wg_sp_stage_bytes(uint32_t cta_m, uint32_t tile_k,
                                     uint32_t xtile_n, uint32_t xtile_m) {
  return wg_sp_stage_meta_off(cta_m, tile_k, xtile_n)
       + wg_sp_stage_meta_bytes(cta_m, xtile_m);
}

// Metadata descriptor slot paired with a sparse GEMM's A descriptor.
constexpr uint32_t wg_meta_desc(uint32_t desc_a) { return desc_a + 8; }

} // namespace akili

#endif
