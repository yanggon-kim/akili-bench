#ifndef _COMMON_H_
#define _COMMON_H_

#include <stdint.h>

// Fused FlashAttention-2 forward on WMMA: O = softmax(Q Kᵀ) V (no 1/sqrt(d)
// scale, as in akili_attn). Each warp owns tileM query rows and streams
// FA_BC-key tiles; S, P and O live in that warp's LMEM slice. FA_SPARSE prunes
// Q 2:4 on the host so Q·Kᵀ runs on the sparse TCU path; P·V stays dense.
#ifndef FA_D
#define FA_D   32
#endif
#define FA_BR  32   // query rows per CTA (4 warps x tileM)
#define FA_BC  32   // keys per tile

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
