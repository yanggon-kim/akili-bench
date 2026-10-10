#ifndef _COMMON_H_
#define _COMMON_H_

#include <stdint.h>

// Fused FlashAttention forward, O = softmax(Q Kᵀ) V (no 1/sqrt(d) scale, as in
// akili_attn), tiled over the head dimension so d = 128 fits the 16 KB shared
// memory: one warp owns FA_BR query rows, K and V stream as FA_BC-key tiles,
// and the output accumulator lives in shared memory in FA_BC-wide blocks.
#ifndef FA_D
#define FA_D   128  // head dimension
#endif
#define FA_BR  8    // query rows per CTA (one warp x xtileM)
#define FA_BC  16   // keys per tile (= WGMMA accumulator width, NRC = 16)

typedef struct {
  uint64_t Q_addr;     // fp16 [N x d] row-major (FA_SPARSE: compressed [N x d/2])
  uint64_t K_addr;     // fp16 [N x d] row-major
  uint64_t V_addr;     // fp16 [N x d] row-major
  uint64_t O_addr;     // fp32 [N x d] row-major
  uint64_t meta_addr;  // FA_SPARSE: packed Q metadata
  uint32_t N;
  uint32_t _pad;
} kernel_arg_t;

#endif
