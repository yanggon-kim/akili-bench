#include "e2e_linear.h"
#include "common.h"

using namespace e2e;

__kernel void linear(linear_args_t* __UNIFORM__ arg) {
  // Grouped GEMM: M tiles past the experts' padded segments are empty.
  if (arg->tile_w_row0 && reinterpret_cast<const uint32_t*>(arg->tile_w_row0)[blockIdx.y] == 0xffffffffu)
    return;
  linear_body(arg);
}

// Router: one thread per token. fp32 logits, softmax over experts, top-k,
// renormalized gate weights.
__kernel void router(moe_args_t* __UNIFORM__ arg) {
  const float* x = reinterpret_cast<const float*>(arg->x);
  const float* wg = reinterpret_cast<const float*>(arg->wg);
  const uint32_t d = arg->d, E = arg->E;
  for (uint32_t t = blockIdx.x * blockDim.x + threadIdx.x; t < arg->T; t += gridDim.x * blockDim.x) {
    float l[MOE_MAX_E];
    for (uint32_t e = 0; e < E; ++e) {
      float s = 0.0f;
      for (uint32_t k = 0; k < d; ++k) s = fma_dev(x[t * d + k], wg[e * d + k], s);
      l[e] = s;
    }
    float m = l[0];
    for (uint32_t e = 1; e < E; ++e) m = fmax_(m, l[e]);
    float sum = 0.0f;
    for (uint32_t e = 0; e < E; ++e) { l[e] = exp2_dev(1.44269504f * (l[e] - m)); sum += l[e]; }
    uint32_t id[MOE_TOPK];
    float w[MOE_TOPK], wsum = 0.0f;
    for (uint32_t k = 0; k < MOE_TOPK; ++k) {
      uint32_t best = 0;
      for (uint32_t e = 1; e < E; ++e) best = l[e] > l[best] ? e : best;
      id[k] = best;
      w[k] = l[best] / sum;
      wsum += w[k];
      l[best] = -1.0f;
    }
    for (uint32_t k = 0; k < MOE_TOPK; ++k) {
      reinterpret_cast<uint32_t*>(arg->topk_id)[t * MOE_TOPK + k] = id[k];
      reinterpret_cast<float*>(arg->topk_w)[t * MOE_TOPK + k] = w[k] / wsum;
    }
  }
}

// moe_align_block_size: one warp; lane e owns expert e. Counts the expert's
// assignments, places its segment (padded to the GEMM M tile) after the
// previous experts', and fills the permutation, row weights and tile table.
__kernel void align(moe_args_t* __UNIFORM__ arg) {
  const uint32_t* id = reinterpret_cast<const uint32_t*>(arg->topk_id);
  const float* tw = reinterpret_cast<const float*>(arg->topk_w);
  uint32_t* sorted = reinterpret_cast<uint32_t*>(arg->sorted_tok);
  float* roww = reinterpret_cast<float*>(arg->row_w);
  uint32_t* pos = reinterpret_cast<uint32_t*>(arg->pos);
  uint32_t* t13 = reinterpret_cast<uint32_t*>(arg->tile_w13);
  uint32_t* t2 = reinterpret_cast<uint32_t*>(arg->tile_w2);
  auto smem = reinterpret_cast<uint32_t*>(__local_mem());
  const uint32_t e = threadIdx.x, E = arg->E, n = arg->T * MOE_TOPK;
  uint32_t cnt = 0;
  if (e < E)
    for (uint32_t i = 0; i < n; ++i) cnt += (id[i] == e);
  if (e < E) smem[e] = (cnt + LIN_BM - 1) / LIN_BM * LIN_BM;
  __syncthreads();
  uint32_t start = 0;
  for (uint32_t j = 0; j < E && j < e; ++j) start += smem[j];
  if (e < E) {
    uint32_t r = start;
    for (uint32_t i = 0; i < n; ++i) {
      if (id[i] == e) {
        sorted[r] = i / MOE_TOPK;
        roww[r] = tw[i];
        pos[i] = r;
        ++r;
      }
    }
    for (; r < start + smem[e]; ++r) { sorted[r] = 0xffffffffu; roww[r] = 0.0f; }
    for (uint32_t tile = start / LIN_BM; tile < (start + smem[e]) / LIN_BM; ++tile) {
      t13[tile] = e * 2 * arg->ffn;
      t2[tile] = e * arg->d;
    }
  }
  __syncthreads();
  if (e == 0) {
    uint32_t total = 0;
    for (uint32_t j = 0; j < E; ++j) total += smem[j];
    for (uint32_t tile = total / LIN_BM; tile < arg->mpad / LIN_BM; ++tile) {
      t13[tile] = 0xffffffffu;
      t2[tile] = 0xffffffffu;
    }
  }
}

// Gather routed tokens into the permuted, quantized GEMM operand; one thread
// per (row, run of QRUN features). Padding rows are zero.
__kernel void gather_quant(moe_args_t* __UNIFORM__ arg) {
  const float* x = reinterpret_cast<const float*>(arg->x);
  const uint32_t* sorted = reinterpret_cast<const uint32_t*>(arg->sorted_tok);
  const uint32_t d = arg->d, runs = d / QRUN, nkt = d / KT;
  float z[QRUN];
  for (uint32_t i = 0; i < QRUN; ++i) z[i] = 0.0f;
  for (uint32_t g = blockIdx.x * blockDim.x + threadIdx.x; g < arg->mpad * runs; g += gridDim.x * blockDim.x) {
    const uint32_t r = g / runs, c0 = (g % runs) * QRUN;
    const uint32_t t = sorted[r];
    const float* src = (t == 0xffffffffu) ? z : x + t * d + c0;
    quant_store_run(src, QRUN, arg->ts_x,
                    reinterpret_cast<uint32_t*>(arg->a) + blk_word(r, c0 / F::epw, LIN_BM, nkt),
                    IS_MX ? reinterpret_cast<uint8_t*>(arg->a_sc) + blk_sc(r, c0, LIN_BM, nkt) : nullptr);
  }
}

// Combine: y[t] = sum_k y_perm[pos[t][k]] (gate weights already applied).
__kernel void combine(moe_args_t* __UNIFORM__ arg) {
  const float* yp = reinterpret_cast<const float*>(arg->y_perm);
  const uint32_t* pos = reinterpret_cast<const uint32_t*>(arg->pos);
  float* y = reinterpret_cast<float*>(arg->y);
  const uint32_t d = arg->d;
  for (uint32_t g = blockIdx.x * blockDim.x + threadIdx.x; g < arg->T * d; g += gridDim.x * blockDim.x) {
    const uint32_t t = g / d, c = g % d;
    float s = 0.0f;
    for (uint32_t k = 0; k < MOE_TOPK; ++k) s += yp[pos[t * MOE_TOPK + k] * d + c];
    y[g] = s;
  }
}
