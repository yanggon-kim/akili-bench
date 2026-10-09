#pragma once
#include <stdint.h>

// FlashAttention forward (non-causal), O = softmax(Q K^T / sqrt(d)) V, per head.
#define FA_D 128          // head dim
#define FA_BM 48          // query rows per CTA on the TCU path (3 consumer groups x 16)
#define FA_SIMT_BM 32     // query rows per CTA on the FPU path
#define FA_SIMT_BN 32     // keys per block on the FPU path

typedef struct {
  uint64_t q, q_sc;       // TCU: quantized [H*N][d] (16-row blocked) ; FPU: fp32 [H*N][d]
  uint64_t k, k_sc;       // same layout as q
  uint64_t vt, vt_sc;     // TCU: quantized V^T [H*d][N] (128-row blocked) ; FPU: fp32 V [H*N][d]
  uint64_t o;             // fp32 [H*N][d]
  float    ts_q, ts_k, ts_v, ts_p;
  float    qk_scale;      // log2(e) / sqrt(d)
  uint32_t H, N;
} fa_args_t;
