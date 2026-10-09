#pragma once
#include <stdint.h>

// Mixtral-style sparse MoE FFN layer with top-2 softmax gating:
//   router (fp32 SIMT) -> align/permute tokens per expert (tile-padded) ->
//   gather+quantize -> grouped GEMM1 (W1|W3, fused SwiGLU) -> grouped GEMM2
//   (W2, scaled by the gate weight) -> combine (sum of the top-k rows).
#define MOE_TOPK 2
#define MOE_MAX_E 16

typedef struct {
  uint64_t x;            // fp32 [T][d]
  uint64_t wg;           // fp32 router [E][d]
  uint64_t topk_id;      // u32 [T][K]
  uint64_t topk_w;       // fp32 [T][K]
  uint64_t sorted_tok;   // u32 [Mpad] token of each permuted row (0xffffffff = pad)
  uint64_t row_w;        // fp32 [Mpad] gate weight of each permuted row
  uint64_t pos;          // u32 [T][K] permuted row of (token, k)
  uint64_t tile_w13;     // u32 [Mpad/128] W13 row offset per M tile (0xffffffff = empty)
  uint64_t tile_w2;      // u32 [Mpad/128] W2 row offset per M tile
  uint64_t a, a_sc;      // gathered, quantized tokens [Mpad][d]
  uint64_t y_perm;       // fp32 [Mpad][d]
  uint64_t y;            // fp32 [T][d]
  float    ts_x;
  uint32_t T, d, ffn, E, mpad;
} moe_args_t;
