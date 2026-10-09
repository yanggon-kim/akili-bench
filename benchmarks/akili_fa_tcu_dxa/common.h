#ifndef _COMMON_H_
#define _COMMON_H_

#include <stdint.h>

// Fused FlashAttention-2 forward: O = softmax(Q Kᵀ) V, one CTA per FA_BR query
// rows, streaming FA_BC-key K/V tiles. Matches akili_attn's semantics (no
// 1/sqrt(d) scale). Tile sizes are fixed by the 16 KB LMEM budget at d = 32.
// FA_SPARSE prunes Q 2:4 on the host so Q·Kᵀ runs on sparse WGMMA; P·V stays
// dense.
#define FA_D   32   // head dimension
#define FA_BR  32   // query rows per CTA (4 warps x xtileM)
#define FA_BC  32   // keys per K/V tile (xtileN at NRC=32)

typedef struct {
  uint64_t Q_addr;   // fp16 [N x d] row-major (FA_SPARSE: compressed [N x d/2])
  uint64_t K_addr;   // fp16 [N x d] row-major
  uint64_t V_addr;   // fp16 [N x d] row-major
  uint64_t O_addr;   // fp32 [N x d] row-major
  uint64_t meta_addr;  // FA_SPARSE: packed Q metadata
  uint32_t N;
  uint32_t _pad;
} kernel_arg_t;

#endif
