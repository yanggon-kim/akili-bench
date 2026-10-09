#include "e2e_dev.h"
#include "common.h"

using namespace e2e;

// ---------------------------------------------------------------------------
// quant_in: fp32 NHWC -> packed operand words + flat per-pixel scales.
// ---------------------------------------------------------------------------
__kernel void quant_in(conv_args_t* __UNIFORM__ arg) {
  const float* x = reinterpret_cast<const float*>(arg->src);
  auto dst = reinterpret_cast<uint32_t*>(arg->dst);
  auto sc = reinterpret_cast<uint8_t*>(arg->dst_sc);
  const uint32_t runs = arg->P * arg->C / QRUN;
  for (uint32_t r = blockIdx.x * blockDim.x + threadIdx.x; r < runs; r += gridDim.x * blockDim.x) {
    uint32_t e0 = r * QRUN;
    quant_store_run(x + e0, QRUN, arg->ts_out, dst + e0 / F::epw,
                    IS_MX ? sc + e0 / BLK : nullptr);
  }
}

// ---------------------------------------------------------------------------
// im2col on packed operands: row p = (oy, ox), K = (ky, kx, ci). Pure word
// gather (zero padding at the border) into the CTA-blocked GEMM layout; MX
// scales move from the flat per-pixel layout into blocked scale tiles.
// ---------------------------------------------------------------------------
__kernel void im2col(conv_args_t* __UNIFORM__ arg) {
  auto src = reinterpret_cast<const uint32_t*>(arg->src);
  auto dst = reinterpret_cast<uint32_t*>(arg->dst);
  const uint32_t H = arg->H, W = arg->W;
  constexpr uint32_t C = CONV_C;
  constexpr uint32_t RW = C / F::epw;      // words per tap
  const uint32_t total = arg->P * 9 * RW;
  constexpr uint32_t nktw = 9 * RW / KT_WORDS;
  const uint32_t stride = gridDim.x * blockDim.x;
  for (uint32_t g = blockIdx.x * blockDim.x + threadIdx.x; g < total; g += stride) {
    uint32_t w = g % RW;
    uint32_t pt = g / RW;
    uint32_t tap = pt % 9, p = pt / 9;
    int32_t iy = (int32_t)(p / W) + (int32_t)(tap / 3) - 1;
    int32_t ix = (int32_t)(p % W) + (int32_t)(tap % 3) - 1;
    uint32_t v = 0;
    if (iy >= 0 && iy < (int32_t)H && ix >= 0 && ix < (int32_t)W)
      v = src[((uint32_t)iy * W + (uint32_t)ix) * RW + w];
    dst[blk_word(p, tap * RW + w, CONV_ARB, nktw)] = v;
  }
  if constexpr (IS_MX) {
    auto ssc = reinterpret_cast<const uint8_t*>(arg->src_sc);
    auto dsc = reinterpret_cast<uint8_t*>(arg->dst_sc);
    constexpr uint32_t BPT = C / BLK;    // scale blocks per tap
    const uint32_t stotal = arg->P * 9 * BPT;
    for (uint32_t g = blockIdx.x * blockDim.x + threadIdx.x; g < stotal; g += stride) {
      uint32_t b = g % BPT;
      uint32_t pt = g / BPT;
      uint32_t tap = pt % 9, p = pt / 9;
      int32_t iy = (int32_t)(p / W) + (int32_t)(tap / 3) - 1;
      int32_t ix = (int32_t)(p % W) + (int32_t)(tap % 3) - 1;
      uint8_t v = SC_ONE;
      if (iy >= 0 && iy < (int32_t)H && ix >= 0 && ix < (int32_t)W)
        v = ssc[((uint32_t)iy * W + (uint32_t)ix) * BPT + b];
      dsc[blk_sc(p, (tap * BPT + b) * F::blk, CONV_ARB, arg->num_kt)] = v;
    }
  }
}

#if E2E_TCU
// ---------------------------------------------------------------------------
// TCU GEMM with fused epilogue: scale, bias, (residual), relu, and either a
// quantized NHWC store (next conv's operand) or an fp32 store.
// ---------------------------------------------------------------------------
constexpr uint32_t GBM = CONV_BM, GBN = CONV_BN, CP = GBN + 1;
using G = tcu_gemm<GBM, GBN>;

__kernel void gemm(conv_args_t* __UNIFORM__ arg) {
  auto smem = reinterpret_cast<uint32_t*>(__local_mem());
  float* cs = reinterpret_cast<float*>(smem);
  const uint32_t m0 = blockIdx.y * GBM, n0 = blockIdx.x * GBN;
  operand_t A{reinterpret_cast<const uint32_t*>(arg->a), reinterpret_cast<const uint32_t*>(arg->a_sc), m0, 0, 0};
  operand_t B{reinterpret_cast<const uint32_t*>(arg->w), reinterpret_cast<const uint32_t*>(arg->w_sc), n0, 0, 0};
  G::run(smem, A, B, arg->num_kt, cs, CP);

  const float* bias = reinterpret_cast<const float*>(arg->bias);
  const float* resid = reinterpret_cast<const float*>(arg->resid);
  const float scale = arg->ts_a * arg->ts_w;
  const uint32_t C = arg->C;
  constexpr uint32_t RPR = GBN / QRUN;
  for (uint32_t run = threadIdx.x; run < GBM * RPR; run += blockDim.x) {
    uint32_t r = run / RPR, c = (run % RPR) * QRUN;
    float* v = cs + r * CP + c;
    uint32_t gr = m0 + r, gc = n0 + c;
    for (uint32_t i = 0; i < QRUN; ++i) {
      float t = v[i] * scale + bias[gc + i];
      if (resid) t += resid[gr * C + gc + i];
      v[i] = t > 0.0f ? t : 0.0f;
    }
    if (arg->quant_out) {
      quant_store_run(v, QRUN, arg->ts_out,
                      reinterpret_cast<uint32_t*>(arg->dst) + (gr * C + gc) / F::epw,
                      IS_MX ? reinterpret_cast<uint8_t*>(arg->dst_sc) + (gr * C + gc) / BLK : nullptr);
    } else {
      float* out = reinterpret_cast<float*>(arg->dst) + gr * C + gc;
      for (uint32_t i = 0; i < QRUN; ++i) out[i] = v[i];
    }
  }
}
#else
// ---------------------------------------------------------------------------
// FPU baseline GEMM (fp32 SIMT) with the same fused epilogue.
// ---------------------------------------------------------------------------
using S = simt_gemm<128, 128>;

__kernel void gemm(conv_args_t* __UNIFORM__ arg) {
  auto smem = reinterpret_cast<float*>(__local_mem());
  const uint32_t m0 = blockIdx.y * 128, n0 = blockIdx.x * 128;
  float acc[4][4] = {};
  operand_t A{reinterpret_cast<const uint32_t*>(arg->a), nullptr, m0, 0, 0};
  operand_t B{reinterpret_cast<const uint32_t*>(arg->w), nullptr, n0, 0, 0};
  S::run(acc, smem, A, B, arg->num_kt);
  const float* bias = reinterpret_cast<const float*>(arg->bias);
  const float* resid = reinterpret_cast<const float*>(arg->resid);
  float* out = reinterpret_cast<float*>(arg->dst);
  const uint32_t C = arg->C, r0 = S::ty(), c0 = S::tx();
  for (uint32_t i = 0; i < 4; ++i) {
    uint32_t gr = m0 + r0 + 32 * i;
    for (uint32_t j = 0; j < 4; ++j) {
      uint32_t gc = n0 + c0 + 32 * j;
      float t = acc[i][j] + bias[gc];
      if (resid) t += resid[gr * C + gc];
      out[gr * C + gc] = t > 0.0f ? t : 0.0f;
    }
  }
}
#endif
