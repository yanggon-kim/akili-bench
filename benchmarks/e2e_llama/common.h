#pragma once
#include <stdint.h>

// One Llama-3.2-1B decoder layer (all 16 layers share these shapes):
//   x1 = x + Wo . attn(rope(Wqkv . rmsnorm(x)))
//   y  = x1 + W2 . (silu(W1 . rmsnorm(x1)) * (W3 . rmsnorm(x1)))
// GQA 32 query / 8 key-value heads of 64, SwiGLU FFN 8192, llama3 RoPE scaling.
#define LL_DIM      2048
#define LL_HEADS    32
#define LL_KV_HEADS 8
#define LL_HEAD     64
#define LL_KV_DIM   (LL_KV_HEADS * LL_HEAD)
#define LL_QKV      (LL_DIM + 2 * LL_KV_DIM)
#define LL_FFN      8192

typedef struct {
  uint64_t x;              // fp32 [T][DIM] residual stream (updated in place)
  uint64_t gamma;          // fp32 [DIM] rmsnorm weight
  uint64_t src;            // fp32 [T][K] input of rmsnorm / quant_act
  uint64_t dst;            // fp32 [T][DIM] rmsnorm output
  uint64_t act, act_sc;    // quant_act output: blocked activation operand + scales
  uint64_t qkv;            // fp32 [T][QKV]
  uint64_t q;              // fp32 [T][DIM] rotated queries
  uint64_t kcache;         // fp32 [seqs][KV_DIM][max_ctx] (keys transposed)
  uint64_t vcache;         // fp32 [seqs][max_ctx][KV_DIM]
  uint64_t att;            // fp32 [T][HEADS][max_ctx] scores / probabilities
  uint64_t ao;             // fp32 [T][DIM] attention output
  uint64_t freq;           // fp32 [HEAD/2] llama3-scaled RoPE frequencies
  uint64_t pos;            // int32 [T] position of each row
  uint64_t seq;            // int32 [T] KV-cache sequence of each row
  float    ts_out;         // per-tensor scale of the quant_act output (fp8/nvfp4)
  float    attn_scale;     // log2(e) / sqrt(HEAD)
  uint32_t T, K, arb, max_ctx;
} llama_args_t;
