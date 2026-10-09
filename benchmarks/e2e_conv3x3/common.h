#pragma once

#include <stdint.h>

#define CONV_C 128
#define CONV_BM 128
#define CONV_BN 64
// Row blocking of the GEMM operands (CTA tile rows of the side they feed).
#ifdef E2E_FPU
#define CONV_ARB 128
#define CONV_BRB 128
#else
#define CONV_ARB CONV_BM
#define CONV_BRB CONV_BN
#endif

// ResNet basic block on an NHWC feature map:
//   h = relu(conv3x3(x, W1) + b1);  y = relu(conv3x3(h, W2) + b2 + x)
// Each conv is an explicit im2col pass followed by a GEMM
//   [P x 9C] . [C x 9C]^T, K ordered (ky, kx, ci).
typedef struct {
  uint64_t src;        // quant_in: fp32 [P][C];  im2col: packed words [P][C/epw]
  uint64_t src_sc;     // im2col: flat per-pixel scales [P][C/blk]
  uint64_t dst;        // quant_in / im2col / gemm output words (or fp32)
  uint64_t dst_sc;     // scales of dst (flat for quant_in/gemm, tiled for im2col)
  uint64_t a, a_sc;    // gemm: im2col operand words + scale tiles
  uint64_t w, w_sc;    // gemm: weight words [C][K/epw] + scale tiles
  uint64_t bias;       // gemm: fp32 [C]
  uint64_t resid;      // gemm: fp32 [P][C] residual, 0 if none
  float    ts_a;       // per-tensor scale of the A operand (fp8/nvfp4)
  float    ts_w;       // per-tensor scale of the weights
  float    ts_out;     // per-tensor scale used when quantizing the gemm output
  uint32_t H, W, C;
  uint32_t P;          // H * W
  uint32_t K;          // 9 * C
  uint32_t num_kt;     // K / KT (TCU) or K / 16 (SIMT)
  uint32_t slot;       // DXA slots of this gemm's weights (data, scales)
  uint32_t quant_out;  // gemm writes quantized NHWC (1) or fp32 (0)
} conv_args_t;
